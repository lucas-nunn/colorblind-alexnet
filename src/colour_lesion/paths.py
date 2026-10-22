"""Find the data folder, on Colab (Google Drive) or locally (external SSD).

The data folder must contain:

    stimuli/             the 872 NSD images (course dataset)
    fmri_data/subjXX/    fMRI responses and roi_masks (course dataset)
    coco/unlabeled2017/  COCO unlabeled2017 images (colour localizer)

Where we look, in this order:
    1. the folder passed to `resolve()`
    2. the environment variable NAI_DATA_ROOT
    3. on Colab: a folder in your Drive that contains stimuli/ and fmri_data/
    4. locally: LOCAL_DEFAULT (the external SSD)
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

ENV_VAR = "NAI_DATA_ROOT"
LOCAL_DEFAULT = Path("/media/chuddy/Extreme SSD/data/neuro-ai-intensive")
DRIVE = Path("/content/drive")


def in_colab():
    return "google.colab" in sys.modules


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @property
    def stimuli(self):
        return self.root / "stimuli"

    @property
    def fmri(self):
        return self.root / "fmri_data"

    @property
    def coco(self):
        return self.root / "coco" / "unlabeled2017"

    def subject_dir(self, subj):
        return self.fmri / f"subj{subj:02d}"

    def missing(self, subjects):
        """List the required files/folders that do not exist (empty list = all good)."""
        required = [self.stimuli, self.fmri]
        for subj in subjects:
            d = self.subject_dir(subj)
            required += [d / "lh_training_fmri.npy", d / "rh_training_fmri.npy", d / "roi_masks"]
        return [p for p in required if not p.exists()]


def _looks_like_data_root(p):
    return (p / "stimuli").is_dir() and (p / "fmri_data").is_dir()


def _search_drive():
    # Folders shared with you show up under .shortcut-targets-by-id once you
    # add a shortcut to them in "My Drive".
    found = []
    for base in [DRIVE / "MyDrive", DRIVE / "Shareddrives", DRIVE / ".shortcut-targets-by-id"]:
        if not base.is_dir():
            continue
        for pattern in ["*", "*/*", "*/*/*"]:
            found += [p for p in base.glob(pattern) if _looks_like_data_root(p)]
    return sorted(set(found))


def resolve(root=None, subjects=None):
    """Return the DataPaths for this machine, and check that the course data is there."""
    from . import config

    subjects = config.SUBJECTS if subjects is None else subjects

    if in_colab():
        from google.colab import drive
        drive.mount(str(DRIVE))

    if root is None:
        root = os.environ.get(ENV_VAR)

    if root is None and in_colab():
        candidates = _search_drive()
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected exactly one Drive folder with stimuli/ and fmri_data/, found {len(candidates)}: "
                f"{[str(c) for c in candidates]}. Pass the folder explicitly: paths.resolve('/content/drive/...')"
            )
        root = candidates[0]

    if root is None:
        root = LOCAL_DEFAULT

    data = DataPaths(Path(root))
    missing = data.missing(subjects)
    if missing:
        listing = "\n  ".join(str(p) for p in missing[:10])
        raise FileNotFoundError(f"Data folder {data.root} is incomplete. Missing:\n  {listing}")

    print(f"Data folder: {data.root}")
    return data
