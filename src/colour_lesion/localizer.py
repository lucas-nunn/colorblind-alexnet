"""COCO colour localizer: the same images in colour and in greyscale (Lin et al., 2014)."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from . import config


def is_greyscale(img, tol=2.0):
    """True if an image has (almost) no colour. Some COCO images are black and white."""
    if img.mode in ("L", "LA", "1", "I", "F"):
        return True
    arr = np.asarray(img.convert("RGB")).astype(np.int16)
    spread = arr.max(axis=-1) - arr.min(axis=-1)   # 0 for a grey pixel
    return spread.mean() < tol


def load_coco_localizer(coco_dir, n=config.N_LOCALIZER, seed=config.LOCALIZER_SEED):
    """Pick n random colour images from a folder of COCO jpgs (val2017 or unlabeled2017).

    Returns the COCO ids and a uint8 array (n, 224, 224, 3).
    LAION-fMRI's NSD images (NSD special1000) are not in COCO val2017, and unlabeled2017 has no NSD images.
    """
    files = sorted(Path(coco_dir).glob("*.jpg"))
    assert files, f"no COCO images found in {coco_dir}"

    rng = np.random.default_rng(seed)
    ids, imgs = [], []
    for i in rng.permutation(len(files)):
        img = Image.open(files[i])
        if is_greyscale(img):
            continue
        ids.append(int(files[i].stem))
        imgs.append(np.array(img.convert("RGB").resize((config.IMG_SIZE, config.IMG_SIZE))))
        if len(ids) == n:
            break

    assert len(ids) == n, f"only {len(ids)} usable COCO images, wanted {n}"
    return ids, np.stack(imgs)


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
