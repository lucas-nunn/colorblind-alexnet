"""The course dataset (UvA_encodingtutorial): the Algonauts 2023 version of NSD (Allen et al., 2022),
8 subjects, the 872 images all of them saw, responses on fsaverage surface vertices.

    UvA_encodingtutorial/
        stimuli/                                  872 images (row i of the fMRI data = image i)
        fmri_data/subjXX/{lh,rh}_training_fmri.npy
        fmri_data/subjXX/roi_masks/

Colab: add the course folder to your own Drive as a shortcut in "My Drive"
(right-click the shared folder -> Organize -> Add shortcut -> My Drive).
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from . import config
from .brain import SubjectData

COLAB_DIR = Path("/content/drive/MyDrive/UvA_encodingtutorial")       # shortcut in My Drive
LOCAL_DIR = Path.home() / ".cache/neuro-ai-intensive/algonauts_view"  # same layout, rebuilt from Algonauts
SUBJECTS = [f"subj{s:02d}" for s in range(1, 9)]
N_HEMI = 163842   # fsaverage vertices per hemisphere

# our regions -> Algonauts masks
ROI_PARTS = {
    "EVC": ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d"],
    "hV4": ["hV4"],
    "LOC": ["lateral"],   # NSD has no LOC mask; the lateral stream contains LOC
}
# which Algonauts mask file each ROI lives in
ROI_CLASSES = {
    "prf-visualrois": ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"],
    "streams": ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"],
}


def init(course_dir=None):
    """Return the course folder (default: the My Drive shortcut on Colab, the local copy otherwise)."""
    colab = "google.colab" in sys.modules
    if colab:
        from google.colab import drive
        drive.mount("/content/drive")
    root = Path(course_dir or (COLAB_DIR if colab else LOCAL_DIR))
    required = [root / "stimuli"] + [root / "fmri_data" / s / f for s in SUBJECTS
                                     for f in ["lh_training_fmri.npy", "rh_training_fmri.npy", "roi_masks"]]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Course data not found under {root}. Add the shared folder UvA_encodingtutorial to My Drive "
            f"(right-click -> Organize -> Add shortcut -> My Drive), or set DATA_DIR. Missing: {missing[:3]}")
    print(f"Course data: {root}")
    return root


def _roi_mask(roi, hemisphere, subj_dir, space="challenge"):
    # boolean mask of one Algonauts ROI over the challenge vertices ("challenge") or all of fsaverage
    roi_class = next(c for c, rois in ROI_CLASSES.items() if roi in rois)
    masks = subj_dir / "roi_masks"
    class_map = np.load(masks / f"{hemisphere}h.{roi_class}_{space}_space.npy")
    mapping = np.load(masks / f"mapping_{roi_class}.npy", allow_pickle=True).item()   # {number: roi name}
    return class_map == list(mapping.keys())[list(mapping.values()).index(roi)]


def load_brain(root, subjects=None):
    """Responses of every subject's visual vertices (both hemispheres) to the 872 images, and
    5 random image folds (fixed seed). Returns the image names, their folds, {subject: SubjectData}."""
    subjects = SUBJECTS if subjects is None else subjects
    names = sorted(p.name for p in (root / "stimuli").iterdir())
    folds = np.random.default_rng(config.CV_SEED).permutation(len(names)) % config.N_FOLDS
    brain = {}
    for s in subjects:
        d = root / "fmri_data" / s
        betas = np.concatenate([np.load(d / f"{h}h_training_fmri.npy") for h in "lr"], axis=1)
        rois = {r: np.concatenate([np.any([_roi_mask(p, h, d) for p in parts], axis=0) for h in "lr"])
                for r, parts in ROI_PARTS.items()}
        on_fsaverage = [np.load(d / "roi_masks" / f"{h}h.all-vertices_fsaverage_space.npy").astype(bool) for h in "lr"]
        units = np.concatenate([np.flatnonzero(on_fsaverage[0]), N_HEMI + np.flatnonzero(on_fsaverage[1])])
        brain[s] = SubjectData(betas, names, units, rois, np.ones(len(names)))
    return names, folds, brain


class StimulusImages(Dataset):
    """The course stimuli with a transform (e.g. AlexNet's)."""

    def __init__(self, root, names, transform):
        self.paths = [root / "stimuli" / n for n in names]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("RGB"))


def to_fsaverage(values, subject, data, root):
    """Per-vertex values {name: [n_units]} -> {name: {"L", "R": fsaverage values}}; NaN off the visual vertices."""
    out = {}
    for name, v in values.items():
        full = np.full(2 * N_HEMI, np.nan)
        full[data.units] = v
        out[name] = {"L": full[:N_HEMI], "R": full[N_HEMI:]}
    return out
