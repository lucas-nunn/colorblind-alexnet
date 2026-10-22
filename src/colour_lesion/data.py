"""LAION-fMRI data, via the laion_fmri package: setup, downloads, shared images and brain responses.

Only small files are kept: per subject, the responses of its visual voxels to the shared images
(averaged over repetitions) are cached as one .npz in <work>/cache. The raw single-trial betas
(~1.2 GB per session) are only needed to build that cache.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import laion_fmri as lf
import numpy as np
from laion_fmri.download import download
from laion_fmri.splits import get_train_test_ids
from torch.utils.data import Dataset

from . import config

COCO_DIRS = ["COCO/unlabeled2017", "COCO/val2017", "coco/unlabeled2017", "coco/val2017"]


def in_colab():
    return "google.colab" in sys.modules


@dataclass(frozen=True)
class Dirs:
    data: Path   # LAION-fMRI downloads (the laion_fmri data folder; large)
    work: Path   # our cache/ and the COCO images (small)

    @property
    def cache(self):
        return self.work / "cache"

    @property
    def coco(self):
        return next((self.work / d for d in COCO_DIRS if (self.work / d).is_dir()), self.work / COCO_DIRS[0])


def init(data_dir=None, work_dir=None):
    """Set up the LAION-fMRI data folder and our work folder (defaults in config, Colab or local)."""
    colab = in_colab()
    if colab:
        from google.colab import drive
        drive.mount("/content/drive")
    data = Path(data_dir or (config.COLAB_DATA_DIR if colab else config.LOCAL_DATA_DIR))
    work = Path(work_dir or (config.COLAB_WORK_DIR if colab else config.LOCAL_WORK_DIR))
    if colab:
        # keep laion_fmri's settings and stimulus-access id on Drive across Colab sessions
        os.environ["XDG_CONFIG_HOME"] = str(work / ".config")
    data.mkdir(parents=True, exist_ok=True)
    (work / "cache").mkdir(parents=True, exist_ok=True)
    lf.dataset_initialize(str(data))
    dirs = Dirs(data, work)
    print(f"LAION-fMRI: {dirs.data}\nCache:      {dirs.cache}\nCOCO:       {dirs.coco}")
    return dirs


# ---------------------------------------------------------------- shared images

def shared_images():
    """The 1,121 regular images every subject saw (LAION-fMRI 'shared' pool), sorted by name, and
    the test fold (0-4) of each image in LAION-fMRI's predefined 5-fold split (random_0 ... random_4)."""
    fold_of = {}
    for k, split in enumerate(config.SPLITS):
        _, test = get_train_test_ids(split, pool="shared")
        fold_of.update({name: k for name in test})
    names = sorted(fold_of)
    return names, np.array([fold_of[n] for n in names])


class StimulusImages(Dataset):
    """LAION-fMRI stimulus images as shown in the scanner, with a transform (e.g. AlexNet's).

    The images are downloaded on first use, which asks you once to accept the
    LAION-fMRI Data Use Agreement.
    """

    def __init__(self, names, transform):
        lf.download_stimuli()
        self.stimuli = lf.load_stimuli()
        self.names = list(names)
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        return self.transform(self.stimuli.images.get(self.names[idx]))


# ---------------------------------------------------------------- brain responses

@dataclass
class SubjectData:
    betas: np.ndarray    # [n_images, n_voxels]; NaN rows for images this subject never saw
    names: list          # image names (rows)
    voxels: np.ndarray   # index of each voxel (column) into the subject's brain mask
    rois: dict           # {region: boolean mask over the voxels}
    n_reps: np.ndarray   # number of repetitions averaged per image

    def subset(self, names):
        rows = [self.names.index(n) for n in names]
        return SubjectData(self.betas[rows], list(names), self.voxels, self.rois, self.n_reps[rows])


def _download(subject, **filters):
    download(subject, desc="SingletrialBetas", n_jobs=4, **filters)


def _betas_file(dirs, subject, ses):
    return list((dirs.data / "derivatives" / "glmsingle-tedana" / subject / ses).glob(
        "func/*_stat-effect_desc-SingletrialBetas_statmap.nii.gz"))


def subject_betas(dirs, subject, names, fetch=True, delete_raw=False):
    """Responses of one subject's visual voxels (config.VISUAL_ROIS) to the shared images.

    Built once from the single-trial betas, then read from <work>/cache. Per session, each voxel's
    betas are z-scored over all trials of the session (as in NSD; Allen et al., 2022); then the
    repetitions of each shared image are averaged.
    fetch: download what is missing (one session at a time); delete_raw: remove each session's
    betas after use, so the peak disk use stays at about one session.
    """
    path = dirs.cache / f"{subject}_shared_betas.npz"
    if path.exists():
        f = np.load(path)
        return SubjectData(f["betas"], f["names"].tolist(), f["voxels"],
                           {r: f[f"roi_{r}"] for r in config.ROI_NAMES}, f["n_reps"])

    if fetch:   # trial tables, ROI masks, brain mask (small)
        _download(subject, suffix=["trials", "mask", "dseg"], include_anatomical=True)
    sub = lf.load_subject(subject)
    keep = sub.get_roi_mask(config.VISUAL_ROIS)   # boolean over the brain-mask voxels
    row_of = {n: i for i, n in enumerate(names)}
    total = np.zeros((len(names), keep.sum()))
    count = np.zeros((len(names), keep.sum()))

    for ses in sub.get_sessions():
        if fetch:   # this session's single-trial betas (~1.2 GB)
            _download(subject, ses=ses, suffix="statmap", stat="effect")
        elif not _betas_file(dirs, subject, ses):
            print(f"{subject} {ses}: betas not downloaded, skipped")
            continue
        betas = sub.get_betas(ses, mask=keep, streaming=True)          # [n_trials, n_voxels]
        betas = (betas - np.nanmean(betas, axis=0)) / np.nanstd(betas, axis=0)
        labels = sub.get_trial_info(ses)["label"].tolist()
        trials = [t for t, label in enumerate(labels) if label in row_of]
        rows = [row_of[labels[t]] for t in trials]
        np.add.at(total, rows, np.nan_to_num(betas[trials]))
        np.add.at(count, rows, ~np.isnan(betas[trials]))
        if delete_raw:
            for f in _betas_file(dirs, subject, ses):
                f.unlink()
        print(f"{subject} {ses}: {len(trials)} shared trials")

    with np.errstate(invalid="ignore"):
        mean = total / count
    seen = count.max(axis=1) > 0
    good = ~np.isnan(mean[seen]).any(axis=0)     # drop voxels without an estimate for a seen image
    voxels = np.flatnonzero(keep)[good]
    rois = {r: sub.get_roi_mask(q)[voxels] for r, q in config.ROIS.items()}
    data = SubjectData(mean[:, good].astype(np.float32), list(names), voxels, rois, count[:, good].max(axis=1))
    np.savez(path, betas=data.betas, names=np.array(names), voxels=voxels, n_reps=data.n_reps,
             **{f"roi_{r}": m for r, m in rois.items()})
    return data


def load_brain(dirs, subjects=None, **kwargs):
    """Shared-image responses of all subjects (default config.SUBJECTS), restricted to the images
    that every subject saw. Returns the image names, their folds, and {subject: SubjectData}.
    """
    subjects = config.SUBJECTS if subjects is None else subjects
    names, folds = shared_images()
    per_subject = {s: subject_betas(dirs, s, names, **kwargs) for s in subjects}
    common = [i for i, n in enumerate(names)
              if all(d.n_reps[i] > 0 for d in per_subject.values())]
    used = [names[i] for i in common]
    return used, folds[common], {s: d.subset(used) for s, d in per_subject.items()}


if __name__ == "__main__":
    # build the caches outside the notebook, e.g.:  python -m colour_lesion.data sub-03 sub-05 --delete-raw
    import argparse
    parser = argparse.ArgumentParser(description="Download LAION-fMRI betas and build the shared-image caches.")
    parser.add_argument("subjects", nargs="*", default=config.SUBJECTS)
    parser.add_argument("--delete-raw", action="store_true", help="remove each session's betas after use")
    args = parser.parse_args()
    dirs = init()
    for s in args.subjects:
        d = subject_betas(dirs, s, shared_images()[0], delete_raw=args.delete_raw)
        print(f"{s}: {int((d.n_reps > 0).sum())} images, {d.betas.shape[1]} voxels, "
              f"{ {r: int(m.sum()) for r, m in d.rois.items()} }")
