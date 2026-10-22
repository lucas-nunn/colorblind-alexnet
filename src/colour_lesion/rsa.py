"""Model-brain fit with RDMs (RSA; Kriegeskorte et al., 2008), within each subject.

A model's fit to a brain RDM is a linear regression of that RDM on the model's layer RDMs
(reweighting its layers; Khaligh-Razavi & Kriegeskorte, 2014); score = R².
Variance partitioning (commonality analysis; Groen et al., 2018) compares the R² of the healthy
model, a lesioned model and both together.
"""

import numpy as np
from scipy.spatial.distance import pdist

from . import config


def brain_rdms(brain, roi_names=config.ROI_NAMES):
    """Correlation-distance RDM per subject and region. brain: {subject: brain.SubjectData}
    -> [n_subjs, n_regions, RDM_len]"""
    return np.array([[pdist(d.betas[:, d.rois[r]], metric="correlation") for r in roi_names]
                     for d in brain.values()])


def regress(predictors, target):
    """Ordinary least squares of target on the predictor RDMs (+ intercept).
    predictors: [n_predictors, RDM_len]. Returns R² and the predictor weights."""
    X = np.column_stack([predictors.T, np.ones(len(target))])
    coef, *_ = np.linalg.lstsq(X, target, rcond=None)
    residual = target - X @ coef
    return 1 - residual @ residual / np.sum((target - target.mean()) ** 2), coef[:-1]


def _stack(rdms_by_layer):
    return np.stack(list(rdms_by_layer.values()))   # [n_layers, RDM_len]


def fit(rdms_by_layer, brain):
    """R² of each subject's region RDM regressed on the model's layer RDMs.

    brain: [n_subjs, n_regions, RDM_len]
    returns R² [n_subjs, n_regions] and layer weights [n_subjs, n_regions, n_layers]
    """
    layers = _stack(rdms_by_layer)
    n_subjs, n_regions, _ = brain.shape
    r2 = np.zeros((n_subjs, n_regions))
    weights = np.zeros((n_subjs, n_regions, len(layers)))
    for s in range(n_subjs):
        for reg in range(n_regions):
            r2[s, reg], weights[s, reg] = regress(layers, brain[s, reg])
    return r2, weights


def variance_partition(healthy_rdms, lesioned_rdms, brain):
    """Split the brain variance explained by the healthy and a lesioned model into three parts,
    from the R² of three regressions per subject and region (healthy layers, lesioned layers, both):

        unique healthy  = R²(both) - R²(lesioned)    what only the intact network explains
        shared          = R²(healthy) + R²(lesioned) - R²(both)
        unique lesioned = R²(both) - R²(healthy)     what only the lesioned network explains

    returns {part name: [n_subjs, n_regions]}
    """
    h, l = _stack(healthy_rdms), _stack(lesioned_rdms)
    r2 = {name: fit(dict(enumerate(layers)), brain)[0]
          for name, layers in [("healthy", h), ("lesioned", l), ("both", np.concatenate([h, l]))]}
    return {
        "unique healthy": r2["both"] - r2["lesioned"],
        "shared": r2["healthy"] + r2["lesioned"] - r2["both"],
        "unique lesioned": r2["both"] - r2["healthy"],
    }
