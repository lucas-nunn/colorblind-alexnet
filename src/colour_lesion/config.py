"""Experiment settings. Change things here, not inside the analysis code."""

# --- colour localizer (COCO unlabeled2017 images) ---
N_LOCALIZER = 1000      # number of colour localizer images
LOCALIZER_SEED = 42     # fixed seed -> the same COCO images every run
IMG_SIZE = 224          # AlexNet input size

# --- lesion ---
N_LESION = 10           # number of conv1 channels we silence
N_CONV1_CHANNELS = 64   # conv1 of AlexNet has 64 channels
N_RANDOM_LESIONS = 5    # repetitions of the random-lesion control
RANDOM_SEED = 0

# --- model RDMs: AlexNet layers we compare with the brain ---
LAYERS = ["features.2", "features.5", "features.7", "features.9", "features.12",
          "classifier.2", "classifier.5"]

# --- reweighted RSA: cross-validation over subjects (leave one out) x stimuli (folds) ---
CV_FOLDS = 5
CV_SEED = 0

# --- encoding models (himalaya kernel ridge) ---
RIDGE_ALPHAS = [10.0 ** k for k in range(-3, 7)]   # ridge penalties to try: 0.001 ... 1e6
RIDGE_INNER_CV = 5      # inner folds (training images only) to choose the penalty / band weights
BANDED_N_ITER = 20      # random-search samples of band weights for banded ridge

# --- brain RDMs ---
SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8]
ENCODING_SUBJECTS = SUBJECTS   # encoding models are slow; set e.g. [1] for a quick test

# our regions, and the Algonauts masks that make up each region
ROI_PARTS = {
    "EVC": ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d"],
    "hV4": ["hV4"],
    "LOC": ["lateral"],
}
ROI_NAMES = list(ROI_PARTS)

# standard ImageNet normalisation used by AlexNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
