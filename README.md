# Colour-lesion model

Lucas Nunn, Max Zimmermann, S. Charlotte Masak. Seminar *Artificial Neural Networks in Cognitive Neuroscience*, WiSe26/27.

We silence AlexNet's most colour-selective conv1 channels and test where its fit to healthy fMRI data drops (EVC, hV4, LOC), with RSA and voxelwise encoding models. The analysis lives in [notebooks/lesion_model.ipynb](notebooks/lesion_model.ipynb); the code is in the `colour_lesion` package.

## Data

* **Brain:** [LAION-fMRI](https://laion-fmri.hebartlab.com) (5 subjects, 7T, single-trial betas), through its `laion_fmri` package. We use the 1,121 images all subjects saw and the dataset's own 5-fold split.
* **Stimulus images:** downloaded by `laion_fmri` on first use. This asks you once to accept the LAION-fMRI Data Use Agreement (name, institutional email, purpose).
* **Colour localizer:** COCO images (`val2017` or `unlabeled2017`) in `<work>/COCO/` or `<work>/coco/`.

Only small files are kept: per subject, the responses of its ~14k visual voxels to the shared images (one `.npz`, ~60 MB) in `<work>/cache/`. Building a cache downloads that subject's betas (~35 GB, one session at a time) and takes a few minutes per subject plus the download.

| | LAION-fMRI downloads (large) | work folder: cache + COCO (small) |
|---|---|---|
| Colab | `/content/laion-fmri` (temporary) | `MyDrive/SHARE/NeuroAI Project/data` |
| Local | `/media/chuddy/Extreme SSD/data` | `/media/chuddy/Extreme SSD/data/neuro-ai-intensive` |

## Run on Colab

1. Put `notebooks/lesion_model.ipynb` on Drive and open it in Colab with a GPU runtime.
2. Run all. The first cell installs this package from GitHub.

With the caches already in the Drive work folder, Colab never downloads betas.

## Run locally

```bash
uv sync                                                      # .venv with the package
uv run python -m colour_lesion.data --delete-raw             # optional: build all caches up front
```

Then open the notebook in VS Code with the `.venv` kernel. `--delete-raw` removes each session's betas after use, so the peak disk use stays at about one session.

## Editing the notebook

`notebooks/lesion_model.py` is a [Jupytext](https://jupytext.readthedocs.io) twin of the `.ipynb` (readable diffs). After editing one, sync the other: `uv run jupytext --sync notebooks/lesion_model.ipynb`.
