"""AlexNet: loading, finding colour-selective conv1 channels, and lesioning them."""

from contextlib import contextmanager

import numpy as np
import torch
import torchvision

from . import config


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_alexnet(device):
    """Pretrained AlexNet (Krizhevsky et al., 2012) plus the transforms it expects."""
    weights = torchvision.models.AlexNet_Weights.DEFAULT
    model = torchvision.models.alexnet(weights=weights).to(device)
    model.eval()
    return model, weights.transforms()


def conv1(model):
    return model.features[0]


def _mean_conv1_response(model, batch, device, chunk=100):
    # mean conv1 activation after ReLU, per channel, over images, height and width.
    conv1_relu = model.features[:2]
    total = torch.zeros(conv1(model).out_channels, device=device)
    with torch.no_grad():
        for start in range(0, len(batch), chunk):
            acts = conv1_relu(batch[start:start + chunk].to(device))   # (n, 64, 55, 55)
            total += acts.sum(dim=[0, 2, 3])
    return total / (len(batch) * acts.shape[2] * acts.shape[3])


def colour_selectivity(model, colour_batch, grey_batch, device):
    """Selectivity score per conv1 channel = mean colour activation - mean greyscale activation.

    A high positive score means the channel relies heavily on colour.
    """
    return _mean_conv1_response(model, colour_batch, device) - _mean_conv1_response(model, grey_batch, device)


def top_channels(scores, n=config.N_LESION):
    """Indices of the n most colour-selective channels."""
    return torch.argsort(scores, descending=True)[:n].cpu()


def random_channel_sets(n_sets=config.N_RANDOM_LESIONS, n=config.N_LESION,
                        n_channels=config.N_CONV1_CHANNELS, seed=config.RANDOM_SEED):
    """Control lesions: n_sets sets of n random conv1 channels (fixed seed)."""
    rng = np.random.default_rng(seed)
    return [torch.tensor(rng.choice(n_channels, size=n, replace=False)) for _ in range(n_sets)]


@contextmanager
def lesion(model, channels):
    """Inside this block, the given conv1 channels output zero.

    The hook is always removed afterwards, so the model is healthy again:

        with lesion(model, channels):
            ...  # lesioned model
    """
    channels = torch.as_tensor(channels)

    def zero_channels(module, input, output):
        output = output.clone()
        output[:, channels, :, :] = 0.0
        return output

    handle = conv1(model).register_forward_hook(zero_channels)
    try:
        yield model
    finally:
        handle.remove()
