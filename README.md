# Colour-lesion model

Lucas Nunn, Max Zimmermann, S. Charlotte Masak. Seminar *Artificial Neural Networks in Cognitive Neuroscience*, WiSe26/27.

We silence AlexNet's most colour-selective conv1 channels and test where its fit to healthy fMRI data drops (EVC, hV4, LOC), with RSA and voxelwise encoding models. The analysis lives in [notebooks/lesion_model.ipynb](notebooks/lesion_model.ipynb); the code is in the `colour_lesion` package.

## Data

* **Brain (default):** the course data, `UvA_encodingtutorial`: the Algonauts 2023 version of NSD, 8 subjects, the 872 images all of them saw.
* **Brain (optional, local only):** [LAION-fMRI](https://laion-fmri.hebartlab.com) via its `laion_fmri` package (`DATASET = "laion"` in the notebook).
* **Colour localizer:** [Imagenette](https://github.com/fastai/imagenette) (10 ImageNet classes), downloaded by torchvision on first use (~340 MB).

## Run on Colab

1. Add the shared course folder **UvA_encodingtutorial** to your own Drive: right-click it → *Organize* → *Add shortcut* → *My Drive*. The notebook expects `/content/drive/MyDrive/UvA_encodingtutorial`.
2. Open `notebooks/lesion_model.ipynb` in Colab with a GPU runtime.
3. Run all. The first cell installs this package from GitHub.

## Run locally

```bash
uv sync                 # course data only
uv sync --extra laion   # plus LAION-fMRI
```

Then open the notebook in VS Code with the `.venv` kernel. Locally the course data is read from a copy in the same layout (`course.LOCAL_DIR`). For LAION-fMRI, `uv run python -m colour_lesion.laion --delete-raw` builds the per-subject caches up front (~35 GB download per subject, one session at a time).

## Editing the notebook

`notebooks/lesion_model.py` is a [Jupytext](https://jupytext.readthedocs.io) twin of the `.ipynb` (readable diffs). After editing one, sync the other: `uv run jupytext --sync notebooks/lesion_model.ipynb`.
