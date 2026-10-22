"""Cortical flatmaps on fsaverage, drawn with nilearn (Abraham et al., 2014). The datasets put their
values on fsaverage (course.to_fsaverage / laion.to_fsaverage); the mesh resolution follows the data."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from nilearn import datasets, plotting

from . import config

HEMIS = {"L": "left", "R": "right"}
DENSITIES = {10242: "fsaverage5", 40962: "fsaverage6", 163842: "fsaverage"}   # vertices per hemisphere
OUTLINE_COLOURS = ["black", "dimgray", "darkgray"]   # one per region, in config.ROI_NAMES order


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
    """{region: {"L", "R": 0/1}} from projected ROI masks ({subject: {region: {"L", "R"}}}):
    vertices in the region for at least half of the subjects."""
    outlines = {}
    for r in config.ROI_NAMES:
        avg = group_average([{h: np.nan_to_num(p[r][h]) for h in HEMIS} for p in projected.values()],
                            min_subjects=0)
        outlines[r] = {h: (avg[h] >= 0.5).astype(int) for h in HEMIS}
    return outlines


def _flatmaps(rows, outlines, vmax, label, title, cmap="RdBu_r"):
    # rows: {row title: {"L", "R": values}} -> one row of (left, right) flatmaps each
    density = DENSITIES[len(next(iter(rows.values()))["L"])]
    mesh = datasets.load_fsaverage(density)["flat"]
    sulc = datasets.load_fsaverage_data(mesh=density, mesh_type="flat", data_type="sulcal")
    fig, axes = plt.subplots(len(rows), 2, figsize=(9, 4 * len(rows)), squeeze=False,
                             subplot_kw={"projection": "3d"})
    for (row_title, values), row_axes in zip(rows.items(), axes):
        for ax, (h, hemi) in zip(row_axes, HEMIS.items()):
            plotting.plot_surf_stat_map(surf_mesh=mesh, stat_map=values[h], hemi=hemi, bg_map=sulc, view="dorsal",
                                        axes=ax, figure=fig, colorbar=False, cmap=cmap, vmin=-vmax, vmax=vmax)
            for k, (r, outline) in enumerate(outlines.items()):
                if outline[h].any():
                    plotting.plot_surf_contours(surf_mesh=mesh, roi_map=outline[h], hemi=hemi, axes=ax, figure=fig,
                                                levels=[1], colors=[OUTLINE_COLOURS[k % 3]], labels=[r])
            ax.set_title(f"{row_title} ({hemi})", fontsize=10)
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-vmax, vmax), cmap=cmap), ax=axes, shrink=0.6, label=label)
    fig.suptitle(title)
    return fig


def _vmax(maps):
    return np.nanpercentile(np.abs(np.concatenate([m[h] for m in maps for h in HEMIS])), 99)


def group_maps(group, outlines, title, label):
    """One row of flatmaps per map in `group` ({title: {"L", "R": values}}), same colour scale."""
    return _flatmaps(group, outlines, _vmax(group.values()), label, title)


def subject_maps(per_subject, outlines, title, label):
    """One row of flatmaps per subject ({subject: {"L", "R": values}}), same colour scale."""
    return _flatmaps(per_subject, outlines, _vmax(per_subject.values()), label, title)
