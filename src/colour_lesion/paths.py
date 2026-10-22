"""Find the data, on Colab (Google Drive) or locally (external SSD).

The data root contains the Algonauts 2023 data (NSD) and COCO images:

    Drive  MyDrive/SHARE/NeuroAI Project/data/
               algonauts_2023_tutorial_data/subj01..08/   (shortcut)
               COCO/val2017/
    SSD    /media/chuddy/Extreme SSD/data/
               algonauts/train/subj01..08/
               neuro-ai-intensive/coco/unlabeled2017/ or .../val2017/

Each subject folder: roi_masks/, training_split/training_fmri/, training_split/training_images/.

Where we look for the data root, in this order: the folder passed to `resolve()`, the
environment variable NAI_DATA_ROOT, then the Drive or SSD default. NAI_COCO overrides the
COCO folder.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

ENV_VAR = "NAI_DATA_ROOT"
COCO_ENV_VAR = "NAI_COCO"
DRIVE = Path("/content/drive")
COLAB_DEFAULT = DRIVE / "MyDrive" / "SHARE" / "NeuroAI Project" / "data"
LOCAL_DEFAULT = Path("/media/chuddy/Extreme SSD/data")

# known places of the two datasets inside a data root (first match wins)
ALGONAUTS_DIRS = ["algonauts_2023_tutorial_data", "algonauts/train", "algonauts_2023_challenge_data"]
COCO_DIRS = ["COCO/unlabeled2017", "COCO/val2017",
             "neuro-ai-intensive/coco/unlabeled2017", "neuro-ai-intensive/coco/val2017"]


def in_colab():
    return "google.colab" in sys.modules


@dataclass(frozen=True)
class DataPaths:
    algonauts: Path   # folder with subj01 ... subj08
    coco: Path        # folder with COCO jpgs (colour localizer)

    def subject_dir(self, subj):
        return self.algonauts / f"subj{subj:02d}"

    def training_images(self, subj):
        return self.subject_dir(subj) / "training_split" / "training_images"

    def training_fmri(self, subj, hemisphere):
        return self.subject_dir(subj) / "training_split" / "training_fmri" / f"{hemisphere[0]}h_training_fmri.npy"

    def missing(self, subjects):
        """List the required files/folders that do not exist (empty list = all good)."""
        required = [self.coco]
        for subj in subjects:
            required += [self.training_images(subj), self.training_fmri(subj, "left"),
                         self.training_fmri(subj, "right"), self.subject_dir(subj) / "roi_masks"]
        return [p for p in required if not p.exists()]


def _first_existing(root, candidates):
    return next((root / c for c in candidates if (root / c).is_dir()), root / candidates[0])


def resolve(root=None, subjects=None):
    """Return the DataPaths for this machine, and check that all data is there."""
    from . import config

    subjects = config.SUBJECTS if subjects is None else subjects

    if in_colab():
        from google.colab import drive
        drive.mount(str(DRIVE))

    root = Path(root or os.environ.get(ENV_VAR) or (COLAB_DEFAULT if in_colab() else LOCAL_DEFAULT))
    coco = os.environ.get(COCO_ENV_VAR)
    data = DataPaths(algonauts=_first_existing(root, ALGONAUTS_DIRS),
                     coco=Path(coco) if coco else _first_existing(root, COCO_DIRS))

    missing = data.missing(subjects)
    if missing:
        listing = "\n  ".join(str(p) for p in missing[:10])
        raise FileNotFoundError(f"Data under {root} is incomplete (set DATA_ROOT to the folder with the "
                                f"Algonauts and COCO data). Missing:\n  {listing}")

    print(f"Algonauts: {data.algonauts}\nCOCO:      {data.coco}")
    return data
