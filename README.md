# Colour-lesion model

Lucas Nunn, Max Zimmermann, S. Charlotte Masak. Seminar *Artificial Neural Networks in Cognitive Neuroscience*, WiSe26/27.

We silence AlexNet's most colour-selective conv1 channels and test where the fit of its layer RDMs to healthy fMRI data drops (EVC, hV4, LOC). The analysis lives in [notebooks/lesion_model.ipynb](notebooks/lesion_model.ipynb); the code is in the `colour_lesion` package.

## Data folder

The same layout on Drive and on the SSD:

```
<data root>/
  stimuli/              872 NSD images         (course folder UvA_encodingtutorial)
  fmri_data/subj01..08/ fMRI + roi_masks       (course folder UvA_encodingtutorial)
  coco/unlabeled2017/   COCO unlabeled2017     (colour localizer; unzip unlabeled2017.zip here)
```

The notebook finds the data root like this:

1. `DATA_ROOT` in the notebook's setup cell, if you fill it in
2. the environment variable `NAI_DATA_ROOT`
3. on Colab: the one Drive folder that contains `stimuli/` and `fmri_data/`. A folder someone shared with you must be added to "My Drive" as a shortcut first.
4. locally: `/media/chuddy/Extreme SSD/data/neuro-ai-intensive`

## Run on Colab

1. Put `notebooks/lesion_model.ipynb` in the Drive folder that holds the data, and open it in Colab.
2. Once only: add a GitHub token with read access to this repo as a Colab secret named `GH_TOKEN` (key icon in the left sidebar).
3. Run all. The first cell installs this package from GitHub.

## Run locally

```bash
uv sync                    # creates .venv with the package and dev tools
```

Then open `notebooks/lesion_model.ipynb` in VS Code with the `.venv` kernel.

## Editing the notebook

`notebooks/lesion_model.py` is a [Jupytext](https://jupytext.readthedocs.io) twin of the `.ipynb` (readable diffs). After editing one, sync the other: `uv run jupytext --sync notebooks/lesion_model.ipynb`.
