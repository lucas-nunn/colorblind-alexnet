"""Cortical flatmaps on fsaverage5: laion_fmri projects voxel values onto the surface (through each
subject's FreeSurfer recon), nilearn draws the flatmaps."""

import warnings

import laion_fmri as lf
import matplotlib.pyplot as plt
import numpy as np
from laion_fmri.download import download
from nilearn import datasets, plotting

from . import config

DENSITY = "fsaverage5"
HEMIS = {"L": "left", "R": "right"}
OUTLINE_COLOURS = ["black", "dimgray", "darkgray"]   # one per region, in config.ROI_NAMES order
MIN_COVER = 0.1   # see to_fsaverage


def _ensure_freesurfer(sub, subject):
    # the projection needs the subject's white surfaces and sphere.reg from its FreeSurfer recon
    needed = [f"surf/{h}h.{x}" for h in "lr" for x in ("white", "sphere.reg")]
    if not (sub.has_freesurfer() and all((sub.get_freesurfer_dir() / f).exists() for f in needed)):
        download(subject, ses="01", desc="SingletrialBetas", suffix="trials", include_freesurfer=True)


def to_fsaverage(values, subject, voxels):
    """Per-voxel values -> fsaverage5. values: {name: [n_voxels]} (columns of data.SubjectData).

    One projection per subject for all maps (it takes ~10 s whatever their number). Each vertex gets
    the coverage-weighted average of the analysed voxels around it; vertices with < MIN_COVER of
    their weight on analysed voxels are NaN. Returns {name: {"L", "R": [n_vertices]}}.
    """
    sub = lf.load_subject(subject)
    _ensure_freesurfer(sub, subject)
    full = np.zeros((len(values) + 1, sub.get_n_voxels()))
    full[:-1, voxels] = np.stack(list(values.values()))
    full[-1, voxels] = 1.0                                   # last row: coverage
    surf = sub.volume_to_surface(full, target="fsaverage", fsaverage_density=DENSITY)
    out = {name: {} for name in values}
    for h in HEMIS:
        weight = surf[h][-1]
        ok = weight > MIN_COVER
        for k, name in enumerate(values):
            out[name][h] = np.where(ok, surf[h][k] / np.where(ok, weight, 1.0), np.nan)
    return out


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
    mesh = datasets.load_fsaverage(DENSITY)["flat"]
    sulc = datasets.load_fsaverage_data(mesh=DENSITY, mesh_type="flat", data_type="sulcal")
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
