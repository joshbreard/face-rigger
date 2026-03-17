"""Face landmark estimation from 3D vertex geometry and region masking.

Accepts an ICP-aligned trimesh.Trimesh head mesh (metres, Y-up, centred at origin).
Returns landmark 3-D positions, named keypoints, and a per-vertex region mask (0-9).

Uses purely geometric heuristics (front-face percentile, anatomical proportions)
so it works from any thread — no GPU, no pyrender, no OpenGL context required.

Returns None on any failure so callers fall back to classical methods.
"""
import logging
from typing import Optional

import numpy as np
import trimesh
from scipy.spatial import KDTree

log = logging.getLogger(__name__)

# ── Named keypoints (MediaPipe 478-landmark mesh indices) ─────────────────────
KEYPOINT_INDICES: dict[str, int] = {
    "nose_tip":        4,
    "right_eye_outer": 33,
    "left_eye_outer":  263,
    "mouth_right":     61,
    "mouth_left":      291,
}

# ── Region label definitions ──────────────────────────────────────────────────
# 0=left_eye  1=right_eye  2=nose  3=mouth  4=left_brow  5=right_brow
# 6=forehead  7=left_cheek  8=right_cheek  9=jaw
REGION_NAMES: list[str] = [
    "left_eye", "right_eye", "nose", "mouth",
    "left_brow", "right_brow", "forehead",
    "left_cheek", "right_cheek", "jaw",
]

# MediaPipe landmark indices that seed each region's spatial extent.
_REGION_SEED_LM: list[list[int]] = [
    [33, 160, 159, 133, 153, 144, 246, 7, 163, 173],    # 0 left_eye
    [263, 387, 386, 362, 380, 373, 466, 249, 390, 380],  # 1 right_eye
    [4, 5, 195, 197, 6, 168, 98, 327],                   # 2 nose
    [13, 14, 312, 82, 61, 291, 17, 87, 78, 308],         # 3 mouth
    [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],          # 4 left_brow
    [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],  # 5 right_brow
    [10, 151, 9, 8, 107, 336, 54, 284],                  # 6 forehead
    [116, 123, 147, 213, 192, 207, 187, 50, 36, 118],    # 7 left_cheek
    [345, 352, 376, 433, 411, 427, 280, 266, 347, 330],  # 8 right_cheek
    [152, 148, 176, 149, 150, 136, 32, 262, 369, 395],   # 9 jaw
]

# Shepard boundary-blend radius (metres).
REGION_RADIUS_M: float = 0.065  # 65 mm

# Per-region expansion radii (metres).  Subtracted from each region's min
# seed-distance before argmin, biasing assignment toward larger regions.
# Eye/mouth/jaw need bigger radii so enough target verts land in those regions.
_REGION_EXPANSION_M: list[float] = [
    0.065,  # 0 left_eye
    0.065,  # 1 right_eye
    0.040,  # 2 nose
    0.055,  # 3 mouth
    0.040,  # 4 left_brow
    0.040,  # 5 right_brow
    0.060,  # 6 forehead
    0.055,  # 7 left_cheek  (mouthSmileLeft lives here)
    0.055,  # 8 right_cheek (mouthSmileRight lives here)
    0.070,  # 9 jaw
]


def detect_landmarks(
    mesh: trimesh.Trimesh,
    image_size: int = 512,
) -> Optional[dict]:
    """Estimate face landmarks from 3D vertex geometry and compute a region mask.

    This is a thin wrapper around :func:`detect_landmarks_from_vertices` kept
    for backward compatibility.  The *image_size* parameter is accepted but
    ignored — no rendering is performed.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        ICP-aligned head mesh (metres, Y-up, centred at origin).
    image_size : int
        Ignored (kept for API compatibility).

    Returns
    -------
    dict with keys
        'landmark_3d'           : dict[int, ndarray(3,)]  estimated 3D positions
        'keypoints_3d'          : dict[str, ndarray(3,)]  named keypoints
        'keypoint_vert_indices' : dict[str, int]           vertex indices
        'face_region_mask'      : ndarray(N,) int  labels 0-9
    or None if detection fails.
    """
    return detect_landmarks_from_vertices(mesh)


def detect_landmarks_from_vertices(
    mesh: trimesh.Trimesh,
) -> Optional[dict]:
    """Estimate face landmarks directly from 3D mesh geometry without rendering.

    Uses mesh centroid and vertex spatial distribution to estimate nose_tip,
    eye, mouth, and brow positions based on anatomical proportions of a
    front-facing head mesh (Y-up, centred at origin).

    Returns the same dict structure as detect_landmarks(), or None on failure.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if len(verts) < 50:
        log.warning("detect_landmarks_from_vertices: too few vertices (%d)", len(verts))
        return None

    # Estimate face front: vertices with highest Z (closest to camera)
    z_vals = verts[:, 2]
    z_threshold = np.percentile(z_vals, 80)
    front_mask = z_vals >= z_threshold
    front_verts = verts[front_mask]

    if len(front_verts) < 20:
        log.warning("detect_landmarks_from_vertices: too few front-facing vertices")
        return None

    # Centroid of front-facing vertices
    centroid = front_verts.mean(axis=0)
    y_min_front = float(front_verts[:, 1].min())
    y_max_front = float(front_verts[:, 1].max())
    y_range = y_max_front - y_min_front
    x_min_front = float(front_verts[:, 0].min())
    x_max_front = float(front_verts[:, 0].max())
    x_range = x_max_front - x_min_front

    if y_range < 1e-9 or x_range < 1e-9:
        log.warning("detect_landmarks_from_vertices: degenerate front vertex range")
        return None

    # Anatomical proportions (fractions of face height from bottom of front verts)
    # Nose tip: ~40% up from chin, at center X, max Z
    nose_y = y_min_front + 0.40 * y_range
    nose_region = front_verts[
        (np.abs(front_verts[:, 0] - centroid[0]) < 0.15 * x_range)
        & (np.abs(front_verts[:, 1] - nose_y) < 0.10 * y_range)
    ]
    if len(nose_region) > 0:
        nose_tip_idx = np.argmax(nose_region[:, 2])
        nose_tip = nose_region[nose_tip_idx].copy()
    else:
        nose_tip = np.array([centroid[0], nose_y, z_vals.max()])

    # Eye level: ~65% up from chin
    eye_y = y_min_front + 0.65 * y_range
    eye_x_offset = 0.20 * x_range

    # Mouth: ~22% up from chin
    mouth_y = y_min_front + 0.22 * y_range
    mouth_x_offset = 0.15 * x_range

    # Brow: ~75% up from chin
    brow_y = y_min_front + 0.75 * y_range
    brow_x_offset = 0.18 * x_range

    def _find_nearest_vert(target: np.ndarray) -> int:
        dists = np.linalg.norm(verts - target, axis=1)
        return int(np.argmin(dists))

    # Build keypoints
    keypoints_3d: dict[str, np.ndarray] = {}
    keypoint_vert_indices: dict[str, int] = {}

    kp_targets = {
        "nose_tip": nose_tip,
        "right_eye_outer": np.array([centroid[0] + eye_x_offset, eye_y, centroid[2]]),
        "left_eye_outer": np.array([centroid[0] - eye_x_offset, eye_y, centroid[2]]),
        "mouth_right": np.array([centroid[0] + mouth_x_offset, mouth_y, centroid[2]]),
        "mouth_left": np.array([centroid[0] - mouth_x_offset, mouth_y, centroid[2]]),
    }

    for name, target in kp_targets.items():
        vi = _find_nearest_vert(target)
        keypoints_3d[name] = verts[vi].copy()
        keypoint_vert_indices[name] = vi

    # Build approximate landmark_3d for the seed landmarks used by region mask
    # Map MediaPipe landmark indices to estimated 3D positions using proportions
    landmark_3d: dict[int, np.ndarray] = {}

    # We need landmarks for region seeds. Estimate positions for each.
    _lm_estimates = {
        # nose
        4: nose_tip,
        5: np.array([centroid[0], nose_y - 0.03 * y_range, nose_tip[2] - 0.002]),
        195: np.array([centroid[0], nose_y + 0.05 * y_range, nose_tip[2] - 0.001]),
        197: np.array([centroid[0], nose_y + 0.03 * y_range, nose_tip[2]]),
        6: np.array([centroid[0], nose_y - 0.01 * y_range, nose_tip[2]]),
        168: np.array([centroid[0], nose_y + 0.10 * y_range, nose_tip[2] - 0.003]),
        98: np.array([centroid[0] + 0.10 * x_range, nose_y - 0.05 * y_range, centroid[2]]),
        327: np.array([centroid[0] - 0.10 * x_range, nose_y - 0.05 * y_range, centroid[2]]),
        # left eye region
        33: np.array([centroid[0] - eye_x_offset, eye_y, centroid[2]]),
        160: np.array([centroid[0] - eye_x_offset * 0.8, eye_y + 0.02 * y_range, centroid[2]]),
        159: np.array([centroid[0] - eye_x_offset * 0.6, eye_y + 0.02 * y_range, centroid[2]]),
        133: np.array([centroid[0] - eye_x_offset * 0.3, eye_y, centroid[2]]),
        153: np.array([centroid[0] - eye_x_offset * 0.7, eye_y - 0.02 * y_range, centroid[2]]),
        144: np.array([centroid[0] - eye_x_offset * 0.8, eye_y - 0.02 * y_range, centroid[2]]),
        246: np.array([centroid[0] - eye_x_offset * 1.1, eye_y, centroid[2]]),
        7: np.array([centroid[0] - eye_x_offset * 0.5, eye_y - 0.03 * y_range, centroid[2]]),
        163: np.array([centroid[0] - eye_x_offset * 0.6, eye_y - 0.01 * y_range, centroid[2]]),
        173: np.array([centroid[0] - eye_x_offset * 0.9, eye_y - 0.01 * y_range, centroid[2]]),
        # right eye region
        263: np.array([centroid[0] + eye_x_offset, eye_y, centroid[2]]),
        387: np.array([centroid[0] + eye_x_offset * 0.8, eye_y + 0.02 * y_range, centroid[2]]),
        386: np.array([centroid[0] + eye_x_offset * 0.6, eye_y + 0.02 * y_range, centroid[2]]),
        362: np.array([centroid[0] + eye_x_offset * 0.3, eye_y, centroid[2]]),
        380: np.array([centroid[0] + eye_x_offset * 0.7, eye_y - 0.02 * y_range, centroid[2]]),
        373: np.array([centroid[0] + eye_x_offset * 0.8, eye_y - 0.02 * y_range, centroid[2]]),
        466: np.array([centroid[0] + eye_x_offset * 1.1, eye_y, centroid[2]]),
        249: np.array([centroid[0] + eye_x_offset * 0.5, eye_y - 0.03 * y_range, centroid[2]]),
        390: np.array([centroid[0] + eye_x_offset * 0.6, eye_y - 0.01 * y_range, centroid[2]]),
        # mouth region
        13: np.array([centroid[0], mouth_y + 0.02 * y_range, centroid[2]]),
        14: np.array([centroid[0], mouth_y - 0.02 * y_range, centroid[2]]),
        312: np.array([centroid[0] - 0.05 * x_range, mouth_y, centroid[2]]),
        82: np.array([centroid[0] + 0.05 * x_range, mouth_y, centroid[2]]),
        61: np.array([centroid[0] + mouth_x_offset, mouth_y, centroid[2]]),
        291: np.array([centroid[0] - mouth_x_offset, mouth_y, centroid[2]]),
        17: np.array([centroid[0], mouth_y - 0.05 * y_range, centroid[2]]),
        87: np.array([centroid[0] + 0.08 * x_range, mouth_y - 0.03 * y_range, centroid[2]]),
        78: np.array([centroid[0] + 0.10 * x_range, mouth_y, centroid[2]]),
        308: np.array([centroid[0] - 0.10 * x_range, mouth_y, centroid[2]]),
        # left brow
        70: np.array([centroid[0] - brow_x_offset * 0.6, brow_y, centroid[2]]),
        63: np.array([centroid[0] - brow_x_offset * 0.8, brow_y, centroid[2]]),
        105: np.array([centroid[0] - brow_x_offset * 1.0, brow_y, centroid[2]]),
        66: np.array([centroid[0] - brow_x_offset * 0.4, brow_y, centroid[2]]),
        107: np.array([centroid[0] - brow_x_offset * 0.2, brow_y, centroid[2]]),
        55: np.array([centroid[0] - brow_x_offset * 1.1, brow_y - 0.01 * y_range, centroid[2]]),
        65: np.array([centroid[0] - brow_x_offset * 0.7, brow_y + 0.01 * y_range, centroid[2]]),
        52: np.array([centroid[0] - brow_x_offset * 0.5, brow_y + 0.01 * y_range, centroid[2]]),
        53: np.array([centroid[0] - brow_x_offset * 0.3, brow_y + 0.01 * y_range, centroid[2]]),
        46: np.array([centroid[0] - brow_x_offset * 0.9, brow_y - 0.01 * y_range, centroid[2]]),
        # right brow
        300: np.array([centroid[0] + brow_x_offset * 0.6, brow_y, centroid[2]]),
        293: np.array([centroid[0] + brow_x_offset * 0.8, brow_y, centroid[2]]),
        334: np.array([centroid[0] + brow_x_offset * 1.0, brow_y, centroid[2]]),
        296: np.array([centroid[0] + brow_x_offset * 0.4, brow_y, centroid[2]]),
        336: np.array([centroid[0] + brow_x_offset * 0.2, brow_y, centroid[2]]),
        285: np.array([centroid[0] + brow_x_offset * 1.1, brow_y - 0.01 * y_range, centroid[2]]),
        295: np.array([centroid[0] + brow_x_offset * 0.7, brow_y + 0.01 * y_range, centroid[2]]),
        282: np.array([centroid[0] + brow_x_offset * 0.5, brow_y + 0.01 * y_range, centroid[2]]),
        283: np.array([centroid[0] + brow_x_offset * 0.3, brow_y + 0.01 * y_range, centroid[2]]),
        276: np.array([centroid[0] + brow_x_offset * 0.9, brow_y - 0.01 * y_range, centroid[2]]),
        # forehead
        10: np.array([centroid[0], y_min_front + 0.90 * y_range, centroid[2]]),
        151: np.array([centroid[0], y_min_front + 0.85 * y_range, centroid[2]]),
        9: np.array([centroid[0], y_min_front + 0.80 * y_range, centroid[2]]),
        8: np.array([centroid[0], y_min_front + 0.78 * y_range, centroid[2]]),
        54: np.array([centroid[0] - 0.15 * x_range, y_min_front + 0.85 * y_range, centroid[2]]),
        284: np.array([centroid[0] + 0.15 * x_range, y_min_front + 0.85 * y_range, centroid[2]]),
        # left cheek
        116: np.array([centroid[0] - 0.30 * x_range, nose_y, centroid[2]]),
        123: np.array([centroid[0] - 0.32 * x_range, nose_y - 0.05 * y_range, centroid[2]]),
        147: np.array([centroid[0] - 0.28 * x_range, nose_y - 0.10 * y_range, centroid[2]]),
        213: np.array([centroid[0] - 0.25 * x_range, nose_y + 0.05 * y_range, centroid[2]]),
        192: np.array([centroid[0] - 0.30 * x_range, nose_y + 0.03 * y_range, centroid[2]]),
        207: np.array([centroid[0] - 0.28 * x_range, mouth_y, centroid[2]]),
        187: np.array([centroid[0] - 0.22 * x_range, mouth_y + 0.05 * y_range, centroid[2]]),
        50: np.array([centroid[0] - 0.25 * x_range, nose_y + 0.08 * y_range, centroid[2]]),
        36: np.array([centroid[0] - 0.33 * x_range, nose_y - 0.03 * y_range, centroid[2]]),
        118: np.array([centroid[0] - 0.30 * x_range, nose_y - 0.08 * y_range, centroid[2]]),
        # right cheek
        345: np.array([centroid[0] + 0.30 * x_range, nose_y, centroid[2]]),
        352: np.array([centroid[0] + 0.32 * x_range, nose_y - 0.05 * y_range, centroid[2]]),
        376: np.array([centroid[0] + 0.28 * x_range, nose_y - 0.10 * y_range, centroid[2]]),
        433: np.array([centroid[0] + 0.25 * x_range, nose_y + 0.05 * y_range, centroid[2]]),
        411: np.array([centroid[0] + 0.30 * x_range, nose_y + 0.03 * y_range, centroid[2]]),
        427: np.array([centroid[0] + 0.28 * x_range, mouth_y, centroid[2]]),
        280: np.array([centroid[0] + 0.22 * x_range, mouth_y + 0.05 * y_range, centroid[2]]),
        266: np.array([centroid[0] + 0.25 * x_range, nose_y + 0.08 * y_range, centroid[2]]),
        347: np.array([centroid[0] + 0.33 * x_range, nose_y - 0.03 * y_range, centroid[2]]),
        330: np.array([centroid[0] + 0.30 * x_range, nose_y - 0.08 * y_range, centroid[2]]),
        # jaw
        152: np.array([centroid[0], y_min_front + 0.02 * y_range, centroid[2]]),
        148: np.array([centroid[0] - 0.05 * x_range, y_min_front + 0.05 * y_range, centroid[2]]),
        176: np.array([centroid[0] - 0.10 * x_range, y_min_front + 0.08 * y_range, centroid[2]]),
        149: np.array([centroid[0] - 0.03 * x_range, y_min_front + 0.04 * y_range, centroid[2]]),
        150: np.array([centroid[0] - 0.07 * x_range, y_min_front + 0.06 * y_range, centroid[2]]),
        136: np.array([centroid[0] - 0.15 * x_range, y_min_front + 0.12 * y_range, centroid[2]]),
        32: np.array([centroid[0] - 0.20 * x_range, y_min_front + 0.15 * y_range, centroid[2]]),
        262: np.array([centroid[0] + 0.05 * x_range, y_min_front + 0.05 * y_range, centroid[2]]),
        369: np.array([centroid[0] + 0.15 * x_range, y_min_front + 0.12 * y_range, centroid[2]]),
        395: np.array([centroid[0] + 0.20 * x_range, y_min_front + 0.15 * y_range, centroid[2]]),
    }

    # Snap each estimated landmark to nearest actual mesh vertex
    tree = KDTree(verts)
    for lm_idx, est_pos in _lm_estimates.items():
        _, vi = tree.query(est_pos)
        landmark_3d[lm_idx] = verts[int(vi)].copy()

    face_region_mask = _compute_region_mask(verts, landmark_3d)

    counts = {r: int((face_region_mask == i).sum()) for i, r in enumerate(REGION_NAMES)}
    log.info(
        "Landmarks (3D geometry): 5 keypoints estimated. Region counts: %s",
        counts,
    )

    return {
        "landmark_3d": landmark_3d,
        "keypoints_3d": keypoints_3d,
        "keypoint_vert_indices": keypoint_vert_indices,
        "face_region_mask": face_region_mask,
    }


# ---------------------------------------------------------------------------
# Internal: region mask computation
# ---------------------------------------------------------------------------

def _compute_region_mask(
    verts: np.ndarray,
    landmark_3d: dict[int, np.ndarray],
) -> np.ndarray:
    """Assign each vertex to the anatomically nearest region (0-9).

    Uses minimum Euclidean distance to 3-D seed points per region as a fast
    approximation of geodesic distance.  Falls back to region 9 (jaw) when
    no seed landmarks are available for a region.
    """
    N = len(verts)
    n_regions = len(_REGION_SEED_LM)
    min_dists = np.full((N, n_regions), np.inf, dtype=np.float64)

    for r_idx, lm_indices in enumerate(_REGION_SEED_LM):
        seeds = [landmark_3d[li] for li in lm_indices if li in landmark_3d]
        if not seeds:
            continue
        for s in seeds:
            d = np.linalg.norm(verts - s, axis=1)
            np.minimum(min_dists[:, r_idx], d, out=min_dists[:, r_idx])

    # Subtract per-region expansion to bias assignment toward larger regions.
    # A vertex 0.060 m from eye seeds beats a vertex 0.058 m from cheek seeds
    # when eye expansion (0.065) > cheek expansion (0.055): -0.005 < 0.003.
    expansion = np.array(_REGION_EXPANSION_M, dtype=np.float64)
    adjusted = min_dists - expansion[np.newaxis, :]  # (N, 10)

    mask = np.argmin(adjusted, axis=1).astype(np.int32)
    # Fallback for vertices where all regions had no seeds
    all_inf = np.all(min_dists == np.inf, axis=1)
    mask[all_inf] = 9  # assign to jaw
    return mask
