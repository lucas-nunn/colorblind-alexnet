"""Model-brain fit with RDMs (RSA; Kriegeskorte et al., 2008).

Model-brain fit is reweighted RSA (Khaligh-Razavi & Kriegeskorte, 2014; Storrs et al., 2021):
each model's prediction of a brain RDM is a non-negative weighted sum of its layer RDMs.
The weights are cross-validated over subjects and stimuli, so the score shows how well the
model generalises to a new subject and new images.

Variance partitioning (commonality analysis; Groen et al., 2018; Hebart et al., 2018) splits the
brain variance that the healthy and a lesioned model explain into unique and shared parts.
"""

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import nnls
from scipy.spatial.distance import pdist

from . import config


def brain_rdms(brain, roi_names=config.ROI_NAMES):
    """Correlation-distance RDM per subject and region. brain: {subject: data.SubjectData}
    -> [n_subjs, n_regions, RDM_len]"""
    return np.array([[pdist(d.betas[:, d.rois[r]], metric="correlation") for r in roi_names]
                     for d in brain.values()])


def pair_masks(test_stim):
    """Which RDM entries (stimulus pairs, in pdist order) are training and which are test pairs.

    Training pairs: both stimuli are training stimuli. Test pairs: both are test stimuli.
    Pairs with one of each are not used, so no test image ever influences the weights.
    """
    i, j = np.triu_indices(len(test_stim), k=1)
    return ~test_stim[i] & ~test_stim[j], test_stim[i] & test_stim[j]


def _nnls_gram(gram, xty):
    """Non-negative least squares min ||Xw - y|| (w >= 0), given only X'X and X'y.

    Same solution as nnls(X, y), but on a tiny n_layers x n_layers problem, which is
    much faster than working on the ~240k RDM entries every time.
    (With X'X = R'R and R'z = X'y:  ||Xw - y||^2 = ||Rw - z||^2 + constant.)
    """
    jitter = 1e-12 * np.trace(gram) / len(gram)   # keeps Cholesky stable for near-identical layers
    R = cholesky(gram + jitter * np.eye(len(gram)))
    z = solve_triangular(R, xty, trans="T")
    w, _ = nnls(R, z)
    return w


def fit_weights(layer_rdms, target):
    """Non-negative layer weights w such that w @ layer_rdms best predicts target (least squares).

    Both sides are mean-centred first, which is the same as fitting a free intercept.
    """
    X = layer_rdms - layer_rdms.mean(axis=1, keepdims=True)
    return _nnls_gram(X @ X.T, X @ (target - target.mean()))


def _cv_fold_r(layers, brain, folds):
    # held-out Pearson r per subject, fold and region (see reweighted_fit)
    n_subjs, n_regions, _ = brain.shape
    n_folds = len(np.unique(folds))
    brain_sum = brain.sum(axis=0)

    r = np.zeros((n_subjs, n_folds, n_regions))
    weights = np.zeros((n_subjs, n_folds, n_regions, len(layers)))
    for f in range(n_folds):
        train, test = pair_masks(folds == f)
        X = layers[:, train] - layers[:, train].mean(axis=1, keepdims=True)
        gram = X @ X.T
        for s in range(n_subjs):
            for reg in range(n_regions):
                # mean RDM of the other subjects, training pairs only
                target = (brain_sum[reg, train] - brain[s, reg, train]) / (n_subjs - 1)
                w = _nnls_gram(gram, X @ (target - target.mean()))
                weights[s, f, reg] = w
                if not w.any():
                    continue   # no layer helps: flat prediction, counts as r = 0
                prediction = w @ layers[:, test]
                r[s, f, reg] = np.corrcoef(prediction, brain[s, reg, test])[0, 1]
    return r, weights


def _stack(rdms_by_layer):
    return np.stack(list(rdms_by_layer.values()))   # [n_layers, RDM_len]


def reweighted_fit(rdms_by_layer, brain, folds):
    """Cross-validated reweighted RSA: one score per subject and region.

    For each held-out subject and each stimulus fold:
      1. fit the layer weights on the mean RDM of the other subjects, using only training stimuli
      2. predict the held-out subject's RDM for the test stimuli
      3. score = Pearson r between prediction and the held-out subject's RDM
         (0 if all weights are 0, i.e. no layer predicts the brain RDM)
    The score is averaged over folds.

    brain: [n_subjs, n_regions, RDM_len]; folds: fold number of every stimulus
    returns scores [n_subjs, n_regions] and weights [n_subjs, n_folds, n_regions, n_layers]
    """
    r, weights = _cv_fold_r(_stack(rdms_by_layer), brain, folds)
    return r.mean(axis=1), weights


def variance_partition(healthy_rdms, lesioned_rdms, brain, folds):
    """Split the brain variance explained by the healthy and a lesioned model into three parts.

    Three reweighted models are cross-validated exactly like reweighted_fit (same folds):
    healthy layers, lesioned layers, and both together. Variance explained = r^2 of the
    held-out prediction (negative r counts as 0). Then, per subject, fold and region:

        unique healthy  = r2(both) - r2(lesioned)    what only the intact network explains
        shared          = r2(healthy) + r2(lesioned) - r2(both)
        unique lesioned = r2(both) - r2(healthy)     what only the lesioned network explains

    Small negative parts can happen with cross-validation and mean "about zero".
    returns {part name: [n_subjs, n_regions]}, averaged over folds
    """
    h, l = _stack(healthy_rdms), _stack(lesioned_rdms)
    r2 = {}
    for name, layers in [("healthy", h), ("lesioned", l), ("both", np.concatenate([h, l]))]:
        r, _ = _cv_fold_r(layers, brain, folds)
        r2[name] = np.clip(r, 0, None) ** 2

    parts = {
        "unique healthy": r2["both"] - r2["lesioned"],
        "shared": r2["healthy"] + r2["lesioned"] - r2["both"],
        "unique lesioned": r2["both"] - r2["healthy"],
    }
    return {name: v.mean(axis=1) for name, v in parts.items()}
