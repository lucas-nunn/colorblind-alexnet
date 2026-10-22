"""Brain responses to the stimuli, the same for every dataset (see course.py and laion.py)."""

from dataclasses import dataclass

import numpy as np


@dataclass
class SubjectData:
    betas: np.ndarray    # [n_images, n_units] (units = surface vertices or voxels)
    names: list          # image names (rows)
    units: np.ndarray    # where each unit lives (used by the dataset's to_fsaverage)
    rois: dict           # {region: boolean mask over the units}
    n_reps: np.ndarray   # number of repetitions averaged per image

    def subset(self, names):
        rows = [self.names.index(n) for n in names]
        return SubjectData(self.betas[rows], list(names), self.units, self.rois, self.n_reps[rows])
