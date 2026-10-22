"""Cortical flatmaps on the fsaverage surface with pycortex (Gao et al., 2015).

The Algonauts vertices are a subset of fsaverage; roi_masks/{lh,rh}.all-vertices_fsaverage_space.npy
says which fsaverage vertices they are, in the same order as the fMRI columns.
"""

import cortex
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from . import brain, config

SUBJECT = "fsaverage"
OUTLINE_STYLES = ["solid", "dashed", "dotted"]   # one per region, in config.ROI_NAMES order
N_HEMI = 163842   # fsaverage vertices per hemisphere


def ensure_fsaverage():
    """Download pycortex's fsaverage surfaces once (needed on Colab; ~50 MB)."""
    if SUBJECT not in cortex.db.subjects:
        cortex.utils.download_subject(subject_id=SUBJECT)


def to_fsaverage(values, data, subj):
    """Per-vertex values (left then right hemisphere, as in brain.visual_fmri) -> both fsaverage
    hemispheres [2 * N_HEMI]; vertices outside the subject's Algonauts selection are NaN."""
    subj_dir = data.subject_dir(subj)
    full = np.full(2 * N_HEMI, np.nan)
    n_left = 0
    for h, offset in [("l", 0), ("r", N_HEMI)]:
        selected = np.load(subj_dir / "roi_masks" / f"{h}h.all-vertices_fsaverage_space.npy").astype(bool)
        n = selected.sum()
        full[offset:offset + N_HEMI][selected] = values[n_left:n_left + n]
        n_left += n
    assert n_left == len(values), "values do not match this subject's vertices"
    return full


def group_average(maps, min_subjects=None):
    """Mean over subjects at every fsaverage vertex. Each subject covers slightly different
    vertices; vertices covered by fewer than min_subjects (default: half) are NaN."""
    maps = np.asarray(maps)
    min_subjects = len(maps) // 2 if min_subjects is None else min_subjects
    covered = np.sum(~np.isnan(maps), axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(maps, axis=0)
    mean[covered < min_subjects] = np.nan
    return mean


def roi_outlines(data, subjects=config.SUBJECTS, roi_parts=config.ROI_PARTS):
    """{region: boolean mask over fsaverage} - vertices in the region for at least half of the subjects."""
    outlines = {}
    for roi_name, parts in roi_parts.items():
        votes = np.zeros(2 * N_HEMI)
        for subj in subjects:
            subj_dir = data.subject_dir(subj)
            votes += np.concatenate([
                np.any([brain.get_roi_mask(p, hemi, subj_dir, space="fsaverage") for p in parts], axis=0)
                for hemi in ["left", "right"]])
        outlines[roi_name] = votes >= len(subjects) / 2
    return outlines


def flatmap(values, ax, vmax, cmap="RdBu_r", outlines=None, title=""):
    """Draw one flatmap (symmetric colour scale -vmax..vmax) on a matplotlib Axes, with region outlines."""
    vertex = cortex.Vertex(values, SUBJECT, vmin=-vmax, vmax=vmax, cmap=cmap)
    cortex.quickflat.make_figure(vertex, fig=ax, with_curvature=True, with_rois=False, with_labels=False,
                                 with_colorbar=False, with_sulci=False)
    for k, (roi_name, mask) in enumerate((outlines or {}).items()):
        img, extents = cortex.quickflat.utils.make_flatmap_image(cortex.Vertex(mask.astype(float), SUBJECT))
        ax.contour(np.nan_to_num(img), levels=[0.5], extent=extents, colors="black",
                   linestyles=OUTLINE_STYLES[k % 3], linewidths=1.0,
                   origin="upper")   # same orientation as pycortex's imshow
    _zoom_to_data(ax, values)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def _zoom_to_data(ax, values, margin=0.05):
    # crop the flatmap to the pixels that show data (the visual cortex), plus a small margin;
    # computed from pycortex's own flatmap image, so it uses exactly the same layout
    img, (left, right, bottom, top) = cortex.quickflat.utils.make_flatmap_image(cortex.Vertex(values, SUBJECT))
    rows, cols = np.nonzero(~np.isnan(img))
    height, width = img.shape[:2]
    x = left + (cols + 0.5) / width * (right - left)
    y = top - (rows + 0.5) / height * (top - bottom)   # image row 0 is the top
    pad = margin * max(np.ptp(x), np.ptp(y))
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_ylim(y.min() - pad, y.max() + pad)


def _colorbar_and_legend(fig, axes, vmax, cmap, label, outlines):
    mappable = plt.cm.ScalarMappable(norm=plt.Normalize(-vmax, vmax), cmap=cmap)
    fig.colorbar(mappable, ax=axes, shrink=0.6, label=label)
    handles = [Line2D([], [], color="black", linestyle=OUTLINE_STYLES[k % 3], label=name)
               for k, name in enumerate(outlines or {})]
    if handles:
        fig.legend(handles=handles, loc="lower left", frameon=False, title="Regions")


def group_maps(group, outlines, title, label, cmap="RdBu_r"):
    """One flatmap per map in `group` ({panel title: fsaverage values}), side by side, same colour scale."""
    vmax = np.nanpercentile(np.abs(np.concatenate(list(group.values()))), 99)
    fig, axes = plt.subplots(1, len(group), figsize=(5.5 * len(group), 4.5))
    for ax, (panel, values) in zip(np.atleast_1d(axes), group.items()):
        flatmap(values, ax, vmax, cmap, outlines, panel)
    _colorbar_and_legend(fig, axes, vmax, cmap, label, outlines)
    fig.suptitle(title)
    return fig


def subject_grid(per_subject, outlines, title, label, cmap="RdBu_r"):
    """Small flatmaps, one per subject ({subject: fsaverage values}), same colour scale."""
    vmax = np.nanpercentile(np.abs(np.concatenate(list(per_subject.values()))), 99)
    n_cols = min(4, len(per_subject))
    n_rows = int(np.ceil(len(per_subject) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.75 * n_rows), squeeze=False)
    for ax in axes.flat[len(per_subject):]:
        ax.axis("off")
    for ax, (subj, values) in zip(axes.flat, per_subject.items()):
        flatmap(values, ax, vmax, cmap, outlines, f"subject {subj}")
    _colorbar_and_legend(fig, axes, vmax, cmap, label, outlines)
    fig.suptitle(title)
    return fig
