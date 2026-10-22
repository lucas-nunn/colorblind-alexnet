"""Brain RDMs from the Algonauts version of NSD (course dataset)."""

import numpy as np
from scipy.spatial.distance import pdist

from . import config

# which Algonauts mask file each ROI lives in
ROI_CLASSES = {
    "prf-visualrois": ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"],
    "floc-bodies": ["EBA", "FBA-1", "FBA-2", "mTL-bodies"],
    "floc-faces": ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"],
    "floc-places": ["OPA", "PPA", "RSC"],
    "floc-words": ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"],
    "streams": ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"],
}


def get_roi_mask(roi, hemisphere, subj_dir, space="challenge"):
    """Boolean mask of one ROI in one hemisphere.

    space="challenge": over the Algonauts vertices (the columns of the fMRI data)
    space="fsaverage": over all fsaverage vertices (for cortical maps)
    """
    roi_class = next((c for c, rois in ROI_CLASSES.items() if roi in rois), None)
    if roi_class is None:
        raise ValueError(f"Invalid ROI name: {roi}")

    mask_dir = subj_dir / "roi_masks"
    roi_class_map = np.load(mask_dir / f"{hemisphere[0]}h.{roi_class}_{space}_space.npy")
    roi_map = np.load(mask_dir / f"mapping_{roi_class}.npy", allow_pickle=True).item()

    # roi_map is {number: roi name}; find the number of our ROI
    roi_number = list(roi_map.keys())[list(roi_map.values()).index(roi)]
    return roi_class_map == roi_number


def visual_fmri(data, subj, roi_parts=config.ROI_PARTS):
    """All visual-cortex vertices of one subject, both hemispheres (left, then right).

    Returns the responses [n_stim, n_vertices] and {region: boolean mask over those vertices}.
    """
    subj_dir = data.subject_dir(subj)
    fmri = np.concatenate([np.load(subj_dir / "lh_training_fmri.npy"),
                           np.load(subj_dir / "rh_training_fmri.npy")], axis=1)
    roi_masks = {}
    for roi_name, parts in roi_parts.items():
        # a vertex belongs to the region if it belongs to any of its parts
        lh_mask = np.any([get_roi_mask(p, "left", subj_dir) for p in parts], axis=0)
        rh_mask = np.any([get_roi_mask(p, "right", subj_dir) for p in parts], axis=0)
        roi_masks[roi_name] = np.concatenate([lh_mask, rh_mask])
    return fmri, roi_masks


def brain_rdms(data, subjects=config.SUBJECTS, roi_parts=config.ROI_PARTS):
    """Correlation-distance RDM for every subject and region.

    Returns an array [n_subjs, n_regions, RDM_len] and a dict (subj, region) -> n_vertices.
    """
    rdms, n_vertices = [], {}
    for subj in subjects:
        fmri, roi_masks = visual_fmri(data, subj, roi_parts)
        rdms.append([pdist(fmri[:, mask], metric="correlation") for mask in roi_masks.values()])
        for roi_name, mask in roi_masks.items():
            n_vertices[(subj, roi_name)] = int(mask.sum())

    return np.array(rdms), n_vertices
