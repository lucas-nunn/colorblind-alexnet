"""Experiment settings. Change things here, not inside the analysis code.
Dataset-specific settings (folders, subjects, region masks) live in course.py and laion.py."""

from pathlib import Path

# our regions (each dataset maps them to its own masks)
ROI_NAMES = ["EVC", "hV4", "LOC"]
ENCODING_SUBJECTS = None   # None = all subjects; encoding models are slow, e.g. ["subj01"] for a quick test

# --- colour localizer: Imagenette (10 ImageNet classes; torchvision downloads it, ~340 MB) ---
LOCAL_LOCALIZER_DIR = Path.home() / ".cache/neuro-ai-intensive/imagenette"   # internal disk: small files waste space on exFAT
COLAB_LOCALIZER_DIR = Path("/content/imagenette")   # temporary, downloaded again per session
N_LOCALIZER = 1000      # number of colour localizer images
LOCALIZER_SEED = 42     # fixed seed -> the same images every run
IMG_SIZE = 224          # AlexNet input size

# --- lesion ---
N_LESION = 10           # number of conv1 channels we silence
N_CONV1_CHANNELS = 64   # conv1 of AlexNet has 64 channels
N_RANDOM_LESIONS = 5    # repetitions of the random-lesion control
RANDOM_SEED = 0

# --- model RDMs: AlexNet layers we compare with the brain ---
LAYERS = ["features.2", "features.5", "features.7", "features.9", "features.12",
          "classifier.2", "classifier.5"]

# --- cross-validation over images: 5 folds ---
N_FOLDS = 5
CV_SEED = 0

# --- encoding models (himalaya kernel ridge) ---
RIDGE_ALPHAS = [10.0 ** k for k in range(-3, 7)]   # ridge penalties to try: 0.001 ... 1e6
RIDGE_INNER_CV = 5      # inner folds (training images only) to choose the penalty / band weights
BANDED_N_ITER = 20      # random-search samples of band weights for banded ridge

# standard ImageNet normalisation used by AlexNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
