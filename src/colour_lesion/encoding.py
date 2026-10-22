"""Voxelwise encoding models with (banded) kernel ridge regression (himalaya; Dupré la Tour et al., 2022).

Each vertex's response to the images is predicted from a model layer's activations.
A model with one feature space is ridge regression; a joint model with several feature
spaces is banded ridge regression, where every feature space gets its own regularisation
(Nunez-Elizalde et al., 2019). The joint model's R^2 is split between its feature spaces with
r2_score_split (Dupré la Tour et al., 2022).

All fitting works on kernels (stimulus x stimulus), see features.representations.
A feature space ("band") is one kernel, or a list of kernels (e.g. all layers of a model)
that are combined with equal weight per layer.
"""

import numpy as np
import torch
from himalaya.backend import get_backend, set_backend
from himalaya.kernel_ridge import KernelRidgeCV, MultipleKernelRidgeCV
from himalaya.scoring import r2_score, r2_score_split

from . import config


def use_gpu_if_available():
    set_backend("torch_cuda" if torch.cuda.is_available() else "numpy", on_error="warn")
    print("himalaya backend:", get_backend().name)


def fold_kernels(K, train, test):
    """Training and test kernels for one fold, as if the features were centred with the
    training-set mean (computed from K directly; test images never influence the centring).

    The kernel is also scaled to an average eigenvalue of 1, so one grid of ridge
    penalties (alphas) suits every layer.
    """
    m = K[:, train].mean(axis=1)
    Kc = K - m[:, None] - m[None, :] + K[np.ix_(train, train)].mean()
    Kc /= np.trace(Kc[np.ix_(train, train)]) / train.sum()
    return Kc[np.ix_(train, train)].astype(np.float32), Kc[np.ix_(test, train)].astype(np.float32)


def _band_fold(band, train, test):
    # one feature space for one fold: the mean of its layers' fold kernels. Each fold kernel has an
    # average eigenvalue of 1, so every layer weighs the same, whatever its number of features.
    layers = band if isinstance(band, (list, tuple)) else [band]
    pairs = [fold_kernels(K, train, test) for K in layers]
    return sum(p[0] for p in pairs) / len(pairs), sum(p[1] for p in pairs) / len(pairs)


def _fit(kernels, Y_train, train, test):
    # fit on the training images; returns the fitted model and the test kernel(s) to predict with
    pairs = [_band_fold(band, train, test) for band in kernels]
    if len(kernels) == 1:
        model = KernelRidgeCV(kernel="precomputed", alphas=config.RIDGE_ALPHAS, cv=config.RIDGE_INNER_CV)
        model.fit(pairs[0][0], Y_train)
        return model, pairs[0][1]
    model = MultipleKernelRidgeCV(
        kernels="precomputed", solver="random_search", cv=config.RIDGE_INNER_CV, random_state=0,
        solver_params=dict(n_iter=config.BANDED_N_ITER, alphas=np.asarray(config.RIDGE_ALPHAS), progress_bar=False))
    model.fit(np.stack([p[0] for p in pairs]), Y_train)
    return model, np.stack([p[1] for p in pairs])


def _folds(Y, folds):
    # training/test split per fold; responses centred with the training mean (no intercept)
    for f in np.unique(folds):
        test = folds == f
        train = ~test
        mean = Y[train].mean(axis=0)
        yield train, test, (Y[train] - mean).astype(np.float32), (Y[test] - mean).astype(np.float32)


def cv_r2(kernels, Y, folds):
    """Held-out R^2 of every vertex, averaged over the stimulus folds.

    kernels: list of feature spaces; each is a [n_stim, n_stim] kernel or a list of kernels
             (1 feature space = ridge regression, 2+ = banded ridge regression)
    Y:       responses [n_stim, n_vertices]
    folds:   fold number of every stimulus (data.shared_images: LAION-fMRI's 5-fold split)
    Hyperparameters (ridge penalty, band weights) are chosen by inner cross-validation
    on the training images only.
    """
    backend = get_backend()
    n_folds = len(np.unique(folds))
    r2 = np.zeros(Y.shape[1])
    for train, test, Y_train, Y_test in _folds(Y, folds):
        model, K_test = _fit(kernels, Y_train, train, test)
        r2 += backend.to_numpy(r2_score(backend.asarray(Y_test), model.predict(K_test))) / n_folds
    return r2


def cv_r2_split(kernels, Y, folds):
    """Held-out R^2 of the joint banded ridge model, split into one part per feature space.

    The joint prediction is the sum of one prediction per feature space. r2_score_split
    (product measure; Dupré la Tour et al., 2022) divides the joint R^2 between the feature
    spaces so that the parts add up to the joint R^2. Correlated feature spaces share credit
    according to how the banded model weighted them.

    r2_score_split needs zero-mean responses, so within each test fold the responses and each
    feature space's prediction are mean-centred first.
    returns [n_kernels, n_vertices], averaged over folds
    """
    backend = get_backend()
    n_folds = len(np.unique(folds))
    split = np.zeros((len(kernels), Y.shape[1]))
    for train, test, Y_train, Y_test in _folds(Y, folds):
        model, K_test = _fit(kernels, Y_train, train, test)
        prediction = model.predict(K_test, split=True)                       # [n_kernels, n_test, n_vertices]
        prediction = prediction - prediction.mean(axis=1, keepdims=True)
        Y_test = backend.asarray(Y_test - Y_test.mean(axis=0))
        split += backend.to_numpy(r2_score_split(Y_test, prediction)) / n_folds
    return split


def split_vs_healthy(healthy, lesioned, Y, folds):
    """Fit the lesioned model alone (for its R^2) and the joint banded model of healthy + lesioned
    features, whose R^2 is split between the two feature spaces.

    healthy, lesioned: one feature space each (a kernel, or a list of layer kernels)
    returns the lesioned model's R^2 [n_vertices] and {"healthy features", "lesioned features": [n_vertices]}
    """
    r2_lesioned = cv_r2([lesioned], Y, folds)
    split = cv_r2_split([healthy, lesioned], Y, folds)
    return r2_lesioned, {"healthy features": split[0], "lesioned features": split[1]}


def roi_means(values, roi_masks):
    """Mean of a per-vertex value within each region -> [n_regions]."""
    return np.array([values[mask].mean() for mask in roi_masks.values()])
