"""Figures. Each function draws one figure and returns it."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import squareform

from . import config


def colour_vs_grey(colour_img, grey_img, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, img, name in zip(axes, [colour_img, grey_img], ["Colour", "Greyscale"]):
        ax.imshow(img)
        ax.set_title(f"{name} {title}")
        ax.axis("off")
    return fig


def dataset_samples(dataset, n_imgs=5, mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD):
    """Show the first n images exactly as the model sees them (after the transforms)."""
    fig, axes = plt.subplots(1, n_imgs, figsize=(n_imgs * 1.5, 1.5))
    for i, ax in enumerate(np.atleast_1d(axes)):
        img = dataset[i].permute(1, 2, 0).numpy() * std + mean
        ax.imshow(np.clip(img, 0, 1))
        ax.set_xticks([])
        ax.set_yticks([])
    return fig


def rdm(rdm_vec, title=None, colormap="viridis"):
    fig, ax = plt.subplots()
    im = ax.imshow(squareform(rdm_vec), cmap=colormap)
    fig.colorbar(im)
    ax.set_title(title)
    ax.set_xlabel("Stim")
    ax.set_ylabel("Stim")
    return fig


def _grouped_bars(scores, roi_names, ylabel, title):
    # scores: {bar name: [n_subjs, n_regions]} -> one group per region, bars of mean ± SEM across subjects
    names = list(scores)
    x = np.arange(len(roi_names))
    bar_w = 0.8 / len(names)
    offsets = (np.arange(len(names)) - (len(names) - 1) / 2) * bar_w   # center bars

    fig, ax = plt.subplots(figsize=(7, 4))
    for k, name in enumerate(names):
        mean = np.nanmean(scores[name], axis=0)
        sem = np.nanstd(scores[name], axis=0, ddof=1) / np.sqrt(len(scores[name]))
        ax.bar(x + offsets[k], mean, width=bar_w, label=name)
        ax.errorbar(x + offsets[k], mean, yerr=sem, fmt="none", capsize=3, color="black", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, roi_names)
    ax.set_xlabel("Brain region")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")   # legend outside, so it does not cover bars
    fig.tight_layout()
    return fig


def score_bars(scores, roi_names=config.ROI_NAMES, title="", ylabel="Cross-validated Pearson r\n(reweighted RSA)"):
    """scores: {model name: [n_subjs, n_regions]}"""
    return _grouped_bars(scores, roi_names, ylabel, title)


# fixed colour per variance part (validated categorical palette: blue, orange, aqua);
# RSA: set-theoretic parts; encoding: R^2 split between the two feature spaces (same colour = same side)
PART_COLOURS = {"shared": "#2a78d6", "unique healthy": "#eb6834", "unique lesioned": "#1baf7a",
                "healthy features": "#eb6834", "lesioned features": "#1baf7a"}
HEALTHY_PARTS = {"unique healthy", "healthy features"}   # the part that gets the error bar


def partition_bars(partitions, roi_names=config.ROI_NAMES, title="", ylabel="Variance explained\n(cross-validated r²)"):
    """Stacked bars of variance parts, one bar per lesion within each region.

    partitions: {lesion name: {part name: [n_subjs, n_regions]}}
                (rsa.variance_partition, or the encoding R^2 split)
    Bars show the mean across subjects; the error bar (SEM) is on the healthy part.
    """
    lesions = list(partitions)
    n = len(lesions)
    bar_w = 0.8 / n
    offsets = (np.arange(n) - (n - 1) / 2) * bar_w

    fig, ax = plt.subplots(figsize=(7, 4))
    for k, lesion in enumerate(lesions):
        x = np.arange(len(roi_names)) + offsets[k]
        pos = np.zeros(len(roi_names))   # positive parts stack upwards,
        neg = np.zeros(len(roi_names))   # small negative parts downwards
        for part, values in partitions[lesion].items():
            colour = PART_COLOURS[part]
            mean = values.mean(axis=0)
            bottom = np.where(mean >= 0, pos, neg)
            ax.bar(x, mean, bottom=bottom, width=bar_w * 0.92, color=colour, edgecolor="white", linewidth=1.5,
                   label=part if k == 0 else None)
            if part in HEALTHY_PARTS:
                sem = values.std(axis=0, ddof=1) / np.sqrt(len(values))
                ax.errorbar(x, bottom + mean, yerr=sem, fmt="none", capsize=3, color="black", linewidth=1)
            pos += np.clip(mean, 0, None)
            neg += np.clip(mean, None, 0)
        for xi in x:
            ax.text(xi, -0.01, lesion, ha="center", va="top", fontsize=8, rotation=0,
                    transform=ax.get_xaxis_transform())

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(roi_names)), roi_names)
    ax.tick_params(axis="x", length=0, pad=16)   # room for the lesion names under each bar
    ax.set_xlabel("Brain region")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    return fig


def partition_table(partitions, roi_names=config.ROI_NAMES):
    """Print {lesion: {part: [n_subjs, n_regions]}} as mean ± SEM across subjects."""
    print(f"{'':8s}{'part':18s}" + "".join(f"{r:>18s}" for r in roi_names))
    for lesion, partition in partitions.items():
        for part, values in partition.items():
            mean, sem = values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(len(values))
            print(f"{lesion:8s}{part:18s}" + "".join(f"{m:+10.4f} ± {e:.4f}" for m, e in zip(mean, sem)))
