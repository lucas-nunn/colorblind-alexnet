"""Cortical flatmaps on the fsaverage surface with pycortex (Gao et al., 2015).

The datasets put their values on fsaverage (course.to_fsaverage / laion.to_fsaverage) as
{"L", "R": per-vertex values, 163,842 per hemisphere}.
"""

import warnings

import cortex
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from . import config

SUBJECT = "fsaverage"
HEMIS = ["L", "R"]
OUTLINE_STYLES = ["solid", "dashed", "dotted"]   # one per region, in config.ROI_NAMES order


def ensure_fsaverage():
    """Download pycortex's fsaverage surfaces once (needed on Colab; ~50 MB)."""
    if SUBJECT not in cortex.db.subjects:
        cortex.utils.download_subject(subject_id=SUBJECT)


def _both(m):
    return np.concatenate([m["L"], m["R"]])   # pycortex wants both hemispheres in one array


def group_average(maps, min_subjects=None):
    """Mean over subjects per vertex; vertices covered by fewer than min_subjects (default: half) are NaN."""
    out = {}
    for h in HEMIS:
        stack = np.array([m[h] for m in maps])
        need = len(stack) / 2 if min_subjects is None else min_subjects
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN vertices
            mean = np.nanmean(stack, axis=0)
        mean[np.sum(~np.isnan(stack), axis=0) < need] = np.nan
        out[h] = mean
    return out


def roi_outlines(projected):
    """{region: bool mask over both hemispheres} from projected ROI masks ({subject: {region: {"L", "R"}}}):
    vertices in the region for at least half of the subjects."""
    outlines = {}
    for r in config.ROI_NAMES:
        avg = group_average([{h: np.nan_to_num(p[r][h]) for h in HEMIS} for p in projected.values()],
                            min_subjects=0)
        outlines[r] = _both(avg) >= 0.5
    return outlines


def flatmap(values, ax, vmax, cmap="RdBu_r", outlines=None, title=""):
    """Draw one flatmap ({"L", "R"} values, symmetric colour scale) on a matplotlib Axes, with region outlines."""
    values = _both(values)
    vertex = cortex.Vertex(values, SUBJECT, vmin=-vmax, vmax=vmax, cmap=cmap)
    cortex.quickflat.make_figure(vertex, fig=ax, with_curvature=True, with_rois=False, with_labels=False,
                                 with_colorbar=False, with_sulci=False)
    for k, mask in enumerate((outlines or {}).values()):
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


def _vmax(maps):
    return np.nanpercentile(np.abs(np.concatenate([_both(m) for m in maps])), 99)


def group_maps(group, outlines, title, label, cmap="RdBu_r"):
    """One flatmap per map in `group` ({panel title: {"L", "R"}}), side by side, same colour scale."""
    ensure_fsaverage()
    vmax = _vmax(group.values())
    fig, axes = plt.subplots(1, len(group), figsize=(5.5 * len(group), 4.5))
    for ax, (panel, values) in zip(np.atleast_1d(axes), group.items()):
        flatmap(values, ax, vmax, cmap, outlines, panel)
    _colorbar_and_legend(fig, axes, vmax, cmap, label, outlines)
    fig.suptitle(title)
    return fig


def subject_maps(per_subject, outlines, title, label, cmap="RdBu_r"):
    """Small flatmaps, one per subject ({subject: {"L", "R"}}), same colour scale."""
    ensure_fsaverage()
    vmax = _vmax(per_subject.values())
    n_cols = min(4, len(per_subject))
    n_rows = int(np.ceil(len(per_subject) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.75 * n_rows), squeeze=False)
    for ax in axes.flat[len(per_subject):]:
        ax.axis("off")
    for ax, (subj, values) in zip(axes.flat, per_subject.items()):
        flatmap(values, ax, vmax, cmap, outlines, subj)
    _colorbar_and_legend(fig, axes, vmax, cmap, label, outlines)
    fig.suptitle(title)
    return fig
