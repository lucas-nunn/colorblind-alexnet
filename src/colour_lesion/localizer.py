"""Colour localizer: Imagenette images (10 ImageNet classes; Howard, 2019), the same images in colour
and in greyscale. ImageNet is what AlexNet was trained on (Deng et al., 2009)."""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import Imagenette

from . import config


def is_greyscale(img, tol=2.0):
    """True if an image has (almost) no colour. A few ImageNet images are black and white."""
    if img.mode in ("L", "LA", "1", "I", "F"):
        return True
    arr = np.asarray(img.convert("RGB")).astype(np.int16)
    spread = arr.max(axis=-1) - arr.min(axis=-1)   # 0 for a grey pixel
    return spread.mean() < tol


def sample_colour_images(files, n=config.N_LOCALIZER, seed=config.LOCALIZER_SEED):
    """n random colour images from a list of image files (fixed seed).
    Returns their names (file stems) and a uint8 array (n, 224, 224, 3)."""
    files = sorted(files)
    rng = np.random.default_rng(seed)
    ids, imgs = [], []
    for i in rng.permutation(len(files)):
        img = Image.open(files[i])
        if is_greyscale(img):
            continue
        ids.append(files[i].stem)
        imgs.append(np.array(img.convert("RGB").resize((config.IMG_SIZE, config.IMG_SIZE))))
        if len(ids) == n:
            break
    assert len(ids) == n, f"only {len(ids)} usable colour images, wanted {n}"
    return ids, np.stack(imgs)


def load_localizer(root=None, n=config.N_LOCALIZER, seed=config.LOCALIZER_SEED):
    """n colour images from the Imagenette training set (torchvision downloads it once, ~340 MB)."""
    colab = "google.colab" in sys.modules
    root = Path(root or (config.COLAB_LOCALIZER_DIR if colab else config.LOCAL_LOCALIZER_DIR))
    root.mkdir(parents=True, exist_ok=True)
    Imagenette(root, split="train", size="320px", download=True)   # skips the download if present
    return sample_colour_images(list((root / "imagenette2-320" / "train").glob("*/*.JPEG")), n, seed)


# convert to grayscale (function from Ale!)
def to_gray(arr):
    lum = (
        0.299 * arr[:, :, :, 0]
        + 0.587 * arr[:, :, :, 1]
        + 0.114 * arr[:, :, :, 2]
    )
    lum = np.clip(np.round(lum), 0, 255).astype(np.uint8)
    return np.repeat(lum[..., np.newaxis], 3, axis=-1)


# standard AlexNet preprocessing (ToTensor + Normalize); images are already 224x224
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
])


def colour_and_grey_batches(img_array):
    """The same images twice: in colour and in greyscale. Shape and edges are identical,
    so the only difference between the two batches is colour."""
    grey_array = to_gray(img_array)
    colour_batch = torch.stack([preprocess(img) for img in img_array])
    grey_batch = torch.stack([preprocess(img) for img in grey_array])
    return colour_batch, grey_batch, grey_array
