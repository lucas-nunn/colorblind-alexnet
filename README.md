# Colour-lesion model

Lucas Nunn, Max Zimmermann, S. Charlotte Masak. Seminar *Artificial Neural Networks in Cognitive Neuroscience*, WiSe26/27.

We silence AlexNet's most colour-selective conv1 channels and test where the fit of its layer RDMs to healthy fMRI data drops (EVC, hV4, LOC). The analysis lives in [notebooks/lesion_model.ipynb](notebooks/lesion_model.ipynb); the code is in the `colour_lesion` package.

## Data

The Algonauts 2023 data (NSD, 8 subjects) and COCO images, in the same layout on Drive and on the SSD:

```
Drive  MyDrive/SHARE/NeuroAI Project/data/
           algonauts_2023_tutorial_data/subj01..08/   (shortcut)
           COCO/val2017/
SSD    /media/chuddy/Extreme SSD/data/
           algonauts/train/subj01..08/
           neuro-ai-intensive/coco/unlabeled2017/ (or val2017/)
```

The stimuli are the 872 training images that all 8 subjects saw; they are found from the image file names (`train-0001_nsd-00013.png`). Set `DATA_ROOT` in the notebook (or `NAI_DATA_ROOT`) to use another data folder; `NAI_COCO` overrides the COCO folder. If the data sits in a folder someone shared with you, add it to "My Drive" as a shortcut and set `DATA_ROOT`.

## Run on Colab

1. Put `notebooks/lesion_model.ipynb` in the Drive folder that holds the data, and open it in Colab.
2. Use a GPU runtime and run all. The first cell installs this package from GitHub.

## Run locally

```bash
uv sync                    # creates .venv with the package and dev tools
```

Then open `notebooks/lesion_model.ipynb` in VS Code with the `.venv` kernel.

## Editing the notebook

`notebooks/lesion_model.py` is a [Jupytext](https://jupytext.readthedocs.io) twin of the `.ipynb` (readable diffs). After editing one, sync the other: `uv run jupytext --sync notebooks/lesion_model.ipynb`.
