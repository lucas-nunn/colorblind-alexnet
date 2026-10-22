"""LAION-fMRI (local only, optional: pip install .[laion]), via its laion_fmri package.

5 subjects, 7T, single-trial GLMsingle betas; we use the 1,121 images all subjects saw and the
dataset's predefined 5-fold split. Only small files are kept: per subject, the responses of its
visual voxels to these images (averaged over repetitions) are cached as one .npz in <work>/cache.
Building a cache downloads that subject's betas (~35 GB, one session at a time).
"""

from dataclasses import dataclass
from pathlib import Path

import laion_fmri as lf
import numpy as np
from laion_fmri.download import download
from laion_fmri.splits import get_train_test_ids
from torch.utils.data import Dataset

from .brain import SubjectData

DATA_DIR = Path("/media/chuddy/Extreme SSD/data")                       # laion_fmri downloads (large)
WORK_DIR = Path("/media/chuddy/Extreme SSD/data/neuro-ai-intensive")    # cache/ (small)
SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
ROIS = {"EVC": "EVC", "hV4": "hV4", "LOC": "lobjects"}   # our regions -> LAION-fMRI ROI names
VISUAL_ROIS = ["laiongeneral", *ROIS.values()]          # voxels we keep
SPLITS = [f"random_{k}" for k in range(5)]
DENSITY = "fsaverage5"
MIN_COVER = 0.1   # see to_fsaverage


@dataclass(frozen=True)
class Dirs:
    data: Path   # laion_fmri data folder
    work: Path   # our cache/

    @property
    def cache(self):
        return self.work / "cache"


def init(data_dir=None, work_dir=None):
    dirs = Dirs(Path(data_dir or DATA_DIR), Path(work_dir or WORK_DIR))
    dirs.cache.mkdir(parents=True, exist_ok=True)
    lf.dataset_initialize(str(dirs.data))
    print(f"LAION-fMRI: {dirs.data}\nCache:      {dirs.cache}")
    return dirs


def shared_images():
    """The 1,121 regular images every subject saw (LAION-fMRI 'shared' pool), sorted by name, and
    the test fold (0-4) of each image in LAION-fMRI's predefined 5-fold split (random_0 ... random_4)."""
    fold_of = {}
    for k, split in enumerate(SPLITS):
        _, test = get_train_test_ids(split, pool="shared")
        fold_of.update({name: k for name in test})
    names = sorted(fold_of)
    return names, np.array([fold_of[n] for n in names])


class StimulusImages(Dataset):
    """LAION-fMRI stimulus images as shown in the scanner, with a transform (e.g. AlexNet's).
    Downloaded on first use, which asks you once to accept the LAION-fMRI Data Use Agreement."""

    def __init__(self, dirs, names, transform):
        lf.download_stimuli()
        self.stimuli = lf.load_stimuli()
        self.names = list(names)
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        return self.transform(self.stimuli.images.get(self.names[idx]))


def _download(subject, **filters):
    download(subject, desc="SingletrialBetas", n_jobs=4, **filters)


def _betas_file(dirs, subject, ses):
    return list((dirs.data / "derivatives" / "glmsingle-tedana" / subject / ses).glob(
        "func/*_stat-effect_desc-SingletrialBetas_statmap.nii.gz"))


def subject_betas(dirs, subject, names, fetch=True, delete_raw=False):
    """Responses of one subject's visual voxels (VISUAL_ROIS) to the shared images.

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
                           {r: f[f"roi_{r}"] for r in ROIS}, f["n_reps"])

    if fetch:   # trial tables, ROI masks, brain mask (small)
        _download(subject, suffix=["trials", "mask", "dseg"], include_anatomical=True)
    sub = lf.load_subject(subject)
    keep = sub.get_roi_mask(VISUAL_ROIS)   # boolean over the brain-mask voxels
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
    rois = {r: sub.get_roi_mask(q)[voxels] for r, q in ROIS.items()}
    data = SubjectData(mean[:, good].astype(np.float32), list(names), voxels, rois, count[:, good].max(axis=1))
    np.savez(path, betas=data.betas, names=np.array(names), voxels=voxels, n_reps=data.n_reps,
             **{f"roi_{r}": m for r, m in rois.items()})
    return data


def load_brain(dirs, subjects=None, **kwargs):
    """Shared-image responses of all subjects (default SUBJECTS), restricted to the images every
    subject saw. Returns the image names, their folds, and {subject: SubjectData}."""
    subjects = SUBJECTS if subjects is None else subjects
    names, folds = shared_images()
    per_subject = {s: subject_betas(dirs, s, names, **kwargs) for s in subjects}
    common = [i for i, n in enumerate(names) if all(d.n_reps[i] > 0 for d in per_subject.values())]
    used = [names[i] for i in common]
    return used, folds[common], {s: d.subset(used) for s, d in per_subject.items()}


def _ensure_freesurfer(sub, subject):
    # the projection needs the subject's white surfaces and sphere.reg from its FreeSurfer recon
    needed = [f"surf/{h}h.{x}" for h in "lr" for x in ("white", "sphere.reg")]
    if not (sub.has_freesurfer() and all((sub.get_freesurfer_dir() / f).exists() for f in needed)):
        download(subject, ses="01", desc="SingletrialBetas", suffix="trials", include_freesurfer=True)


def to_fsaverage(values, subject, data, dirs):
    """Per-voxel values {name: [n_voxels]} -> {name: {"L", "R": fsaverage5 values}}.

    One laion_fmri projection per subject for all maps (~10 s whatever their number). Each vertex
    gets the coverage-weighted average of the analysed voxels around it; vertices with < MIN_COVER
    of their weight on analysed voxels are NaN.
    """
    sub = lf.load_subject(subject)
    _ensure_freesurfer(sub, subject)
    full = np.zeros((len(values) + 1, sub.get_n_voxels()))
    full[:-1, data.units] = np.stack(list(values.values()))
    full[-1, data.units] = 1.0                                   # last row: coverage
    surf = sub.volume_to_surface(full, target="fsaverage", fsaverage_density=DENSITY)
    out = {name: {} for name in values}
    for h in "LR":
        weight = surf[h][-1]
        ok = weight > MIN_COVER
        for k, name in enumerate(values):
            out[name][h] = np.where(ok, surf[h][k] / np.where(ok, weight, 1.0), np.nan)
    return out


if __name__ == "__main__":
    # build the caches outside the notebook, e.g.:  python -m colour_lesion.laion sub-03 sub-05 --delete-raw
    import argparse
    parser = argparse.ArgumentParser(description="Download LAION-fMRI betas and build the shared-image caches.")
    parser.add_argument("subjects", nargs="*", default=SUBJECTS)
    parser.add_argument("--delete-raw", action="store_true", help="remove each session's betas after use")
    args = parser.parse_args()
    dirs = init()
    for s in args.subjects:
        d = subject_betas(dirs, s, shared_images()[0], delete_raw=args.delete_raw)
        print(f"{s}: {int((d.n_reps > 0).sum())} images, {d.betas.shape[1]} voxels, "
              f"{ {r: int(m.sum()) for r, m in d.rois.items()} }")
