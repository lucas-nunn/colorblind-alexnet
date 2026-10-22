# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Lesion Model
# **Authors:** Lucas Nunn, Max Zimmermann, S. Charlotte Masak · **Seminar:** Artificial Neural Networks in Cognitive Neuroscience · WiSe26/27
#
# We silence AlexNet's most colour-selective conv1 channels (Krizhevsky et al., 2012) and test where its fit to healthy fMRI data suffers. Clinical analogy: **cerebral achromatopsia**, colour blindness after (rare, bilateral) V4 damage, with object recognition intact (Bartolomeo et al., 2014). Colour is decodable from V1–V4 but not from lateral occipital cortex, and only V4 codes perceptual colour space (Brouwer & Heeger, 2009).
#
# **Hypotheses**
# * **H1:** the colour lesion lowers the model–brain fit.
# * **H2:** the drop is larger in **hV4** than in **LOC** (EVC: exploratory).
# * **Control:** the colour lesion matters more than silencing the same number of **random** channels.
#
# **Steps**
# 0. Setup
# 1. Colour localizer
# 2. Lesion: colour channels and random controls
# 3. Brain data (LAION-fMRI)
# 4. Model representations
# 5. RSA: reweighted fit and variance partitioning
# 6. Encoding models: banded ridge regression
# 7. Cortical maps

# %% [markdown]
# ## 0. Setup
# On Colab use a GPU runtime. Locally: `uv sync`.

# %%
import subprocess
import sys

REPO = "https://github.com/lucas-nunn/colorblind-alexnet.git"

if "google.colab" in sys.modules:
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    subprocess.run(pip + [f"git+{REPO}"], check=True)                                     # dependencies
    subprocess.run(pip + ["--force-reinstall", "--no-deps", f"git+{REPO}"], check=True)   # always the latest commit
    for name in [m for m in sys.modules if m.startswith("colour_lesion")]:
        del sys.modules[name]   # forget a version imported earlier in this session

# %% [markdown]
# Folders (empty = defaults, see `config.py`): `DATA_DIR` holds the LAION-fMRI downloads (large; on Colab the temporary `/content/laion-fmri`), `WORK_DIR` the small brain-data cache and the COCO images (on Colab `MyDrive/SHARE/NeuroAI Project/data`). `QUICK` fits the encoding models for one subject only.

# %%
import matplotlib.pyplot as plt
import numpy as np

from colour_lesion import config, data, encoding, features, localizer, maps, model, plots, rsa

DATA_DIR = ""  # @param {type: "string"}
WORK_DIR = ""  # @param {type: "string"}
QUICK = False  # @param {type: "boolean"}

dirs = data.init(DATA_DIR or None, WORK_DIR or None)
device = model.get_device()
print("Device:", device)

# %% [markdown]
# ## 1. Colour localizer
# 1000 colour images from COCO (Lin et al., 2014), each shown in colour and in greyscale: same shapes, only colour differs.

# %%
localizer_ids, img_array = localizer.load_coco_localizer(dirs.coco)
colour_batch, grey_batch, grey_array = localizer.colour_and_grey_batches(img_array)

image_index = 51  # @param {type: "slider", min: 0, max: 999, step: 1}
plots.colour_vs_grey(img_array[image_index], grey_array[image_index],
                     title=f"#{image_index} (COCO {localizer_ids[image_index]})")
plt.show()

# %% [markdown]
# ## 2. Lesion
# Selectivity per conv1 channel = mean activation (after ReLU) to colour − to greyscale images. We silence the 10 most colour-selective channels, and as a control 5 sets of 10 random channels.

# %%
alexnet, alexnet_transforms = model.load_alexnet(device)

selectivity_scores = model.colour_selectivity(alexnet, colour_batch, grey_batch, device)
colour_channels = model.top_channels(selectivity_scores, n=config.N_LESION)
random_channel_sets = model.random_channel_sets()

print("Colour channels:", colour_channels.tolist())
for rep, channels in enumerate(random_channel_sets):
    print(f"Random set {rep + 1}:", sorted(channels.tolist()))

# %% [markdown]
# ## 3. Brain data
# LAION-fMRI: 5 subjects, 7T, single-trial GLMsingle betas (Prince et al., 2022), loaded with the `laion_fmri` package. We use the **1,121 images all subjects saw** (the "shared" pool) and the dataset's predefined 5-fold split of them. Per session, each voxel's betas are z-scored (as in NSD; Allen et al., 2022) and the repetitions of each image are averaged. The first run downloads the betas and builds a small cache per subject (a few minutes per subject); later runs read the cache.
#
# | Region | LAION-fMRI ROI |
# |---|---|
# | **EVC** | `EVC` (V1–V3, retinotopic mapping) |
# | **hV4** | `hV4` (retinotopic mapping) |
# | **LOC** | `lobjects` (lateral object-selective cortex, functional localizer) |

# %%
names, folds, brain = data.load_brain(dirs)
print(f"{len(names)} images seen by all subjects")
for s, d in brain.items():
    print(f"{s}: {d.betas.shape[1]} visual voxels,", {r: int(m.sum()) for r, m in d.rois.items()},
          f"{d.n_reps.mean():.1f} repetitions per image")

brain_rdms = rsa.brain_rdms(brain)
plots.rdm(brain_rdms[0, config.ROI_NAMES.index("hV4")], title=f"hV4, {config.SUBJECTS[0]}")
plt.show()

# %% [markdown]
# ## 4. Model representations
# One pass over the images per model gives each layer's RDM (for RSA) and kernel (for the encoding models). `model.lesion(...)` zeroes the channels only inside the `with` block. The first use of the images asks you to accept the LAION-fMRI Data Use Agreement.

# %%
images = data.StimulusImages(names, transform=alexnet_transforms)

healthy_rdms, healthy_kernels = features.representations(alexnet, images, device)
with model.lesion(alexnet, colour_channels):
    lesioned_rdms, lesioned_kernels = features.representations(alexnet, images, device)

random_rdms, random_kernels = [], []
for channels in random_channel_sets:
    with model.lesion(alexnet, channels):
        rdms, kernels = features.representations(alexnet, images, device)
    random_rdms.append(rdms)
    random_kernels.append(kernels)

for layer in ["features.2", "classifier.5"]:
    plots.rdm(healthy_rdms[layer], title=f"healthy AlexNet, {layer}")
    plots.rdm(lesioned_rdms[layer], title=f"colour-lesioned AlexNet, {layer}")
plt.show()

# %% [markdown]
# ## 5. RSA
# ### 5a. Reweighted RSA
# Each model predicts a brain RDM as a non-negative weighted sum of its layer RDMs (Khaligh-Razavi & Kriegeskorte, 2014). Weights are cross-validated over subjects (leave one out) × images (the 5 folds); score = Pearson r on held-out subject and images (Storrs et al., 2021).

# %%
healthy_scores, healthy_weights = rsa.reweighted_fit(healthy_rdms, brain_rdms, folds)
lesioned_scores, lesioned_weights = rsa.reweighted_fit(lesioned_rdms, brain_rdms, folds)
random_scores = np.mean([rsa.reweighted_fit(r, brain_rdms, folds)[0] for r in random_rdms], axis=0)

plots.score_bars({"healthy": healthy_scores, "colour lesion": lesioned_scores,
                  f"random lesion (mean of {len(random_rdms)})": random_scores},
                 title="RSA: AlexNet–fMRI fit (mean ± SEM across subjects)")
plt.show()

# %% [markdown]
# ### 5b. Variance partitioning
# Fit healthy, lesioned and both together (same cross-validation; r² = squared held-out r) and split the variance (Groen et al., 2018; Hebart et al., 2018): **unique healthy** = r²(both) − r²(lesioned), i.e. what the lesion removes; **shared** = r²(healthy) + r²(lesioned) − r²(both); **unique lesioned** = r²(both) − r²(healthy).

# %%
colour_partition = rsa.variance_partition(healthy_rdms, lesioned_rdms, brain_rdms, folds)
random_partitions = [rsa.variance_partition(healthy_rdms, r, brain_rdms, folds) for r in random_rdms]
random_partition = {part: np.mean([p[part] for p in random_partitions], axis=0) for part in colour_partition}

rsa_partition = {"colour": colour_partition, "random": random_partition}
plots.partition_bars(rsa_partition, title="RSA: variance partitioning (mean across subjects)")
plt.show()
plots.partition_table(rsa_partition)

# %% [markdown]
# ## 6. Encoding models
# Ridge regression per voxel (~14k visual voxels per subject) on all 7 AlexNet layers, equally weighted, one feature space per model; fitted on kernels with himalaya (Dupré la Tour et al., 2022). Same 5 image folds as the RSA; penalties chosen on training images only.
#
# For each lesion, a **banded ridge** joint model (Nunez-Elizalde et al., 2019) of healthy + lesioned features; `r2_score_split` divides its held-out R² into a **healthy** and a **lesioned** part. More healthy credit after the colour lesion than after random lesions = colour matters. Credit between near-identical feature spaces is partly arbitrary, so compare colour with random, not with 0.

# %%
encoding.use_gpu_if_available()
subjects = config.SUBJECTS[:1] if QUICK else config.ENCODING_SUBJECTS
n_subjs, n_regions = len(subjects), len(config.ROI_NAMES)

healthy_band = list(healthy_kernels.values())   # one feature space = the 7 layer kernels
lesioned_band = list(lesioned_kernels.values())
random_bands = [list(kernels.values()) for kernels in random_kernels]

enc_scores = {name: np.zeros((n_subjs, n_regions)) for name in ["healthy", "colour lesion", "random lesion"]}
enc_partition = {lesion: {part: np.zeros((n_subjs, n_regions)) for part in ["healthy features", "lesioned features"]}
                 for lesion in ["colour", "random"]}
healthy_part = {"colour": {}, "random": {}}   # per subject, per voxel (for the maps)

for s, subj in enumerate(subjects):
    Y, roi_masks = brain[subj].betas, brain[subj].rois
    enc_scores["healthy"][s] = encoding.roi_means(encoding.cv_r2([healthy_band], Y, folds), roi_masks)

    r2_lesioned, parts = encoding.split_vs_healthy(healthy_band, lesioned_band, Y, folds)
    enc_scores["colour lesion"][s] = encoding.roi_means(r2_lesioned, roi_masks)
    for part, values in parts.items():
        enc_partition["colour"][part][s] = encoding.roi_means(values, roi_masks)
    healthy_part["colour"][subj] = parts["healthy features"]

    healthy_part["random"][subj] = np.zeros(Y.shape[1])
    for band in random_bands:
        r2_lesioned, parts = encoding.split_vs_healthy(healthy_band, band, Y, folds)
        enc_scores["random lesion"][s] += encoding.roi_means(r2_lesioned, roi_masks) / len(random_bands)
        for part, values in parts.items():
            enc_partition["random"][part][s] += encoding.roi_means(values, roi_masks) / len(random_bands)
        healthy_part["random"][subj] += parts["healthy features"] / len(random_bands)
    print(f"{subj} done")

# %%
plots.score_bars(enc_scores, title="Encoding: mean R² per region (mean ± SEM across subjects)",
                 ylabel="Held-out R²\n(mean over voxels)")
plots.partition_bars(enc_partition, title="Encoding: joint R² split between healthy and lesioned features",
                     ylabel="Held-out R² of the joint model\n(mean over voxels)")
plt.show()
plots.partition_table(enc_partition)

# %% [markdown]
# ## 7. Cortical maps
# The healthy part of every voxel's joint R², projected onto fsaverage with `laion_fmri` (via each subject's FreeSurfer recon) and drawn as flatmaps with nilearn (Abraham et al., 2014). Group maps show vertices covered by at least half of the subjects; colour − random has its own scale (red = colour lesion removes more).

# %%
projected = {subj: maps.to_fsaverage({"colour": healthy_part["colour"][subj], "random": healthy_part["random"][subj],
                                       **{r: brain[subj].rois[r].astype(float) for r in config.ROI_NAMES}},
                                      subj, brain[subj].voxels)
             for subj in subjects}   # one projection per subject
outlines = maps.roi_outlines(projected)
surface = {lesion: {subj: projected[subj][lesion] for subj in subjects} for lesion in ["colour", "random"]}
colour_minus_random = {subj: {h: surface["colour"][subj][h] - surface["random"][subj][h] for h in maps.HEMIS}
                       for subj in subjects}

maps.group_maps({"colour lesion": maps.group_average(list(surface["colour"].values())),
                 f"random lesion (mean of {len(random_bands)})": maps.group_average(list(surface["random"].values()))},
                outlines, title="Healthy part of the joint R² (mean across subjects)", label="R²")
maps.group_maps({"colour − random": maps.group_average(list(colour_minus_random.values()))},
                outlines, title="Colour − random lesion (mean across subjects)", label="Δ R²")
maps.subject_maps(colour_minus_random, outlines, title="Colour − random lesion, per subject", label="Δ R²")
plt.show()

# %% [markdown]
# ---
# ## References
# - Abraham, A., Pedregosa, F., Eickenberg, M., Gervais, P., Mueller, A., Kossaifi, J., Gramfort, A., Thirion, B., & Varoquaux, G. (2014). Machine learning for neuroimaging with scikit-learn. *Frontiers in Neuroinformatics, 8*, 14. https://doi.org/10.3389/fninf.2014.00014
# - Allen, E. J., St-Yves, G., Wu, Y., Breedlove, J. L., Prince, J. S., Dowdle, L. T., … Kay, K. (2022). A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. *Nature Neuroscience, 25*(1), 116–126. https://doi.org/10.1038/s41593-021-00962-x
# - Bartolomeo, P., Bachoud-Lévi, A. C., & Thiebaut de Schotten, M. (2014). The anatomy of cerebral achromatopsia: a reappraisal and comparison of two case reports. *Cortex, 56,* 138–144. https://doi.org/10.1016/j.cortex.2013.01.013
# - Brouwer, G. J., & Heeger, D. J. (2009). Decoding and reconstructing color from responses in human visual cortex. *Journal of Neuroscience, 29*(44), 13992–14003. https://doi.org/10.1523/JNEUROSCI.3577-09.2009
# - Dupré la Tour, T., Eickenberg, M., Nunez-Elizalde, A. O., & Gallant, J. L. (2022). Feature-space selection with banded ridge regression. *NeuroImage, 264*, 119728. https://doi.org/10.1016/j.neuroimage.2022.119728
# - Groen, I. I. A., Greene, M. R., Baldassano, C., Fei-Fei, L., Beck, D. M., & Baker, C. I. (2018). Distinct contributions of functional and deep neural network features to representational similarity of scenes in human brain and behavior. *eLife, 7*, e32962. https://doi.org/10.7554/eLife.32962
# - Hebart, M. N., Bankson, B. B., Harel, A., Baker, C. I., & Cichy, R. M. (2018). The representational dynamics of task and object processing in humans. *eLife, 7*, e32816. https://doi.org/10.7554/eLife.32816
# - Khaligh-Razavi, S.-M., & Kriegeskorte, N. (2014). Deep supervised, but not unsupervised, models may explain IT cortical representation. *PLoS Computational Biology, 10*(11), e1003915. https://doi.org/10.1371/journal.pcbi.1003915
# - Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. *Advances in Neural Information Processing Systems, 25*.
# - LAION-fMRI dataset and `laion_fmri` package, ViCCo-Group. https://laion-fmri.hebartlab.com
# - Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., & Zitnick, C. L. (2014). Microsoft COCO: Common objects in context. *ECCV 2014*, 740–755. https://doi.org/10.1007/978-3-319-10602-1_48
# - Nunez-Elizalde, A. O., Huth, A. G., & Gallant, J. L. (2019). Voxelwise encoding models with non-spherical multivariate normal priors. *NeuroImage, 197*, 482–492. https://doi.org/10.1016/j.neuroimage.2019.04.012
# - Prince, J. S., Charest, I., Kurzawski, J. W., Pyles, J. A., Tarr, M. J., & Kay, K. N. (2022). Improving the accuracy of single-trial fMRI response estimates using GLMsingle. *eLife, 11*, e77599. https://doi.org/10.7554/eLife.77599
# - Storrs, K. R., Kietzmann, T. C., Walther, A., Mehrer, J., & Kriegeskorte, N. (2021). Diverse deep neural networks all predict human inferior temporal cortex well, after training and fitting. *Journal of Cognitive Neuroscience, 33*(10), 2044–2064. https://doi.org/10.1162/jocn_a_01755
