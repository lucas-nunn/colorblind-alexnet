"""Experiment settings. Change things here, not inside the analysis code."""

from pathlib import Path

# --- where things live (see data.init) ---
LOCAL_DATA_DIR = Path("/media/chuddy/Extreme SSD/data")                        # LAION-fMRI downloads
LOCAL_WORK_DIR = Path("/media/chuddy/Extreme SSD/data/neuro-ai-intensive")      # cache/ + coco/
COLAB_DATA_DIR = Path("/content/laion-fmri")                                    # big, temporary
COLAB_WORK_DIR = Path("/content/drive/MyDrive/SHARE/NeuroAI Project/data")      # small, persistent

# --- brain data: LAION-fMRI (5 subjects, 7T, single-trial GLMsingle betas) ---
SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
ENCODING_SUBJECTS = SUBJECTS   # encoding models are slow; set e.g. ["sub-01"] for a quick test

# our regions -> LAION-fMRI ROI names
ROIS = {
    "EVC": "EVC",        # V1-V3 (retinotopy)
    "hV4": "hV4",        # retinotopy
    "LOC": "lobjects",   # lateral object-selective cortex (localizer)
}
ROI_NAMES = list(ROIS)
# voxels we keep: stimulus-responsive visual cortex plus our regions
VISUAL_ROIS = ["laiongeneral", *ROIS.values()]

# --- colour localizer (COCO images) ---
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

# --- cross-validation: LAION-fMRI's predefined 5-fold split of the shared images ---
SPLITS = [f"random_{k}" for k in range(5)]

# --- encoding models (himalaya kernel ridge) ---
RIDGE_ALPHAS = [10.0 ** k for k in range(-3, 7)]   # ridge penalties to try: 0.001 ... 1e6
RIDGE_INNER_CV = 5      # inner folds (training images only) to choose the penalty / band weights
BANDED_N_ITER = 20      # random-search samples of band weights for banded ridge

# standard ImageNet normalisation used by AlexNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
