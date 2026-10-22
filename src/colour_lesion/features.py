"""Layer representations of a model: RDMs (for RSA) and linear kernels (for encoding models)."""

import numpy as np
import torch
from scipy.spatial.distance import pdist
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor

from . import config


def representations(model, dataset, device, layers=config.LAYERS, batch_size=128):
    """Run the stimuli through the model once and return two dicts, {layer: RDM} and {layer: kernel}.

    RDM: correlation distance between the activation patterns of every pair of stimuli.
    Kernel: linear kernel K = X X^T of the activations [n_stim, n_stim]; encoding models are
    fitted on kernels, so the size of a layer (up to 65k features) does not matter.

    The feature extractor shares its modules with `model`, so a lesion hook
    on `model` (see model.lesion) also applies here.
    """
    extractor = create_feature_extractor(model, return_nodes=layers)
    acts = {layer: [] for layer in layers}
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size):
            out = extractor(batch.to(device))
            for layer in layers:
                acts[layer].append(out[layer].flatten(start_dim=1).cpu().numpy())

    rdms, kernels = {}, {}
    for layer in layers:
        X = np.concatenate(acts.pop(layer))   # pop: free memory as we go
        rdms[layer] = pdist(X, metric="correlation")
        X = X.astype(np.float64)               # float64: the kernel is centred later (fold_kernels)
        kernels[layer] = X @ X.T
    return rdms, kernels
