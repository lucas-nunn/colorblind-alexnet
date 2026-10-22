"""Model-brain fit with RDMs (RSA; Kriegeskorte et al., 2008), within each subject.

A model's fit to a brain RDM is a linear regression of that RDM on the model's layer RDMs
(reweighting its layers; Khaligh-Razavi & Kriegeskorte, 2014), cross-validated over images:
fitted on pairs of training images, R² on pairs of test images (Storrs et al., 2021).
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


def pair_masks(test_stim):
    """Which RDM entries (image pairs, in pdist order) are training and which are test pairs.
    Training pairs: both images are training images. Test pairs: both are test images.
    Pairs with one of each are not used, so test images never influence the fit."""
    i, j = np.triu_indices(len(test_stim), k=1)
    return ~test_stim[i] & ~test_stim[j], test_stim[i] & test_stim[j]


def cv_regress(predictors, targets, folds):
    """Cross-validated linear regression (+ intercept) of several target RDMs on the same predictors.

    predictors: [n_predictors, RDM_len]; targets: [n_targets, RDM_len]; folds: fold of every image.
    Per fold: least squares on the training pairs, R² = 1 - SSE/SST on the test pairs.
    Returns R² [n_targets] and weights [n_targets, n_predictors], averaged over folds.
    """
    X = np.column_stack([predictors.T, np.ones(predictors.shape[1])])
    Y = targets.T
    r2, weights = [], []
    for f in np.unique(folds):
        train, test = pair_masks(folds == f)
        coef, *_ = np.linalg.lstsq(X[train], Y[train], rcond=None)
        residual = Y[test] - X[test] @ coef
        r2.append(1 - (residual ** 2).sum(axis=0) / ((Y[test] - Y[test].mean(axis=0)) ** 2).sum(axis=0))
        weights.append(coef[:-1].T)
    return np.mean(r2, axis=0), np.mean(weights, axis=0)


def _stack(rdms_by_layer):
    return np.stack(list(rdms_by_layer.values()))   # [n_layers, RDM_len]


def fit(rdms_by_layer, brain, folds):
    """Cross-validated R² of each subject's region RDM regressed on the model's layer RDMs.

    brain: [n_subjs, n_regions, RDM_len]; folds: fold of every image
    returns R² [n_subjs, n_regions] and layer weights [n_subjs, n_regions, n_layers]
    """
    layers = _stack(rdms_by_layer)
    n_subjs, n_regions, rdm_len = brain.shape
    r2, weights = cv_regress(layers, brain.reshape(-1, rdm_len), folds)
    return r2.reshape(n_subjs, n_regions), weights.reshape(n_subjs, n_regions, len(layers))


def variance_partition(healthy_rdms, lesioned_rdms, brain, folds):
    """Split the brain variance explained by the healthy and a lesioned model into three parts,
    from the cross-validated R² of three regressions (healthy layers, lesioned layers, both):

        unique healthy  = R²(both) - R²(lesioned)    what only the intact network explains
        shared          = R²(healthy) + R²(lesioned) - R²(both)
        unique lesioned = R²(both) - R²(healthy)     what only the lesioned network explains

    With cross-validation, a part can be slightly negative; that means "about zero".
    returns {part name: [n_subjs, n_regions]}
    """
    h, l = _stack(healthy_rdms), _stack(lesioned_rdms)
    r2 = {name: fit(dict(enumerate(layers)), brain, folds)[0]
          for name, layers in [("healthy", h), ("lesioned", l), ("both", np.concatenate([h, l]))]}
    return {
        "unique healthy": r2["both"] - r2["lesioned"],
        "shared": r2["healthy"] + r2["lesioned"] - r2["both"],
        "unique lesioned": r2["both"] - r2["healthy"],
    }
