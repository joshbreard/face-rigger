"""Mediapipe facial landmark detection and face region masking.

Accepts an ICP-aligned trimesh.Trimesh head mesh (metres, Y-up, centred at origin).
Returns landmark 3-D positions, named keypoints, and a per-vertex region mask (0-9).

Returns None on any failure so callers fall back to classical methods.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
from scipy.spatial import KDTree

log = logging.getLogger(__name__)

MODEL_PATH = Path("assets/face_landmarker.task")

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
    """Detect MediaPipe face landmarks on *mesh* and compute a region mask.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        ICP-aligned head mesh (metres, Y-up, centred at origin).
    image_size : int
        Frontal render resolution.

    Returns
    -------
    dict with keys
        'landmark_3d'           : dict[int, ndarray(3,)]  pixel→nearest vert pos
        'keypoints_3d'          : dict[str, ndarray(3,)]  named keypoints
        'keypoint_vert_indices' : dict[str, int]           vertex indices
        'face_region_mask'      : ndarray(N,) int  labels 0-9
    or None if detection fails.
    """
    pixels = _render_and_detect(mesh, image_size)
    if pixels is None:
        return None

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    x_arr, y_arr = verts[:, 0], verts[:, 1]

    x_min_raw = float(x_arr.min()); x_range_raw = float(x_arr.max() - x_min_raw)
    y_min_raw = float(y_arr.min()); y_range_raw = float(y_arr.max() - y_min_raw)
    PAD = 0.05
    x_min = x_min_raw - x_range_raw * PAD
    x_range = x_range_raw * (1.0 + 2.0 * PAD)
    y_min = y_min_raw - y_range_raw * PAD
    y_range = y_range_raw * (1.0 + 2.0 * PAD)

    xy_tree = KDTree(verts[:, :2])

    def _pix_to_vert(px: float, py_pix: float) -> int:
        wx = px / (image_size - 1) * x_range + x_min
        wy = (1.0 - py_pix / (image_size - 1)) * y_range + y_min
        _, vi = xy_tree.query([wx, wy])
        return int(vi)

    landmark_3d: dict[int, np.ndarray] = {}
    for lm_idx in range(len(pixels)):
        vi = _pix_to_vert(*pixels[lm_idx])
        landmark_3d[lm_idx] = verts[vi].copy()

    keypoints_3d: dict[str, np.ndarray] = {}
    keypoint_vert_indices: dict[str, int] = {}
    for name, lm_idx in KEYPOINT_INDICES.items():
        vi = _pix_to_vert(*pixels[lm_idx])
        keypoints_3d[name] = verts[vi].copy()
        keypoint_vert_indices[name] = vi

    face_region_mask = _compute_region_mask(verts, landmark_3d)

    counts = {r: int((face_region_mask == i).sum()) for i, r in enumerate(REGION_NAMES)}
    log.info("Landmarks: %d detected, 5 keypoints. Region counts: %s", len(landmark_3d), counts)

    return {
        "landmark_3d": landmark_3d,
        "keypoints_3d": keypoints_3d,
        "keypoint_vert_indices": keypoint_vert_indices,
        "face_region_mask": face_region_mask,
    }


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
        "Landmarks (3D vertex fallback): 5 keypoints estimated from geometry. Region counts: %s",
        counts,
    )

    return {
        "landmark_3d": landmark_3d,
        "keypoints_3d": keypoints_3d,
        "keypoint_vert_indices": keypoint_vert_indices,
        "face_region_mask": face_region_mask,
    }


# ---------------------------------------------------------------------------
# Internal: software rasterizer + MediaPipe runner
# ---------------------------------------------------------------------------

def _render_pyrender(
    mesh: trimesh.Trimesh,
    image_size: int,
) -> Optional[np.ndarray]:
    """Render *mesh* with pyrender using directional lighting.

    Returns an (H, W, 3) uint8 RGB image, or None if pyrender is unavailable.
    """
    try:
        import pyrender
    except ImportError:
        log.info("pyrender not installed; skipping lit render.")
        return None

    # Ensure minimum 512x512
    image_size = max(image_size, 512)

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if len(verts) < 3:
        return None

    # Build pyrender mesh with material
    # Try to use the GLB's original texture/material
    material = None
    has_texture = (
        hasattr(mesh.visual, "material")
        and mesh.visual.material is not None
    )
    if has_texture:
        try:
            vis = mesh.visual
            if hasattr(vis, "to_color"):
                color_vis = vis.to_color()
                vertex_colors = np.asarray(color_vis.vertex_colors, dtype=np.uint8)
                pr_mesh = pyrender.Mesh.from_trimesh(
                    mesh, smooth=True,
                )
            else:
                pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        except Exception:
            has_texture = False

    if not has_texture:
        # Flat skin-tone fallback (#D4956A)
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.831, 0.584, 0.416, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.8,
        )
        pr_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)

    scene = pyrender.Scene(
        bg_color=[0.0, 0.0, 0.0, 0.0],
        ambient_light=[0.15, 0.15, 0.15],
    )
    scene.add(pr_mesh)

    # Camera: positioned in front of face center, pointing at nose_tip
    centroid = verts.mean(axis=0)
    z_max = float(verts[:, 2].max())
    nose_estimate = np.array([centroid[0], centroid[1], z_max])

    # Place camera in front of the face along +Z
    y_range = float(verts[:, 1].max() - verts[:, 1].min())
    cam_distance = max(y_range * 2.0, 0.3)  # ensure face fills frame
    cam_pos = np.array([nose_estimate[0], nose_estimate[1], nose_estimate[2] + cam_distance])

    # Build camera-to-world matrix (looking at nose_estimate from cam_pos)
    forward = nose_estimate - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-9:
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
    right = right / right_norm
    up = np.cross(right, forward)

    cam_pose = np.eye(4)
    cam_pose[:3, 0] = right
    cam_pose[:3, 1] = up
    cam_pose[:3, 2] = -forward  # OpenGL convention: camera looks along -Z
    cam_pose[:3, 3] = cam_pos

    # Perspective camera
    fov = 2.0 * np.arctan(y_range * 0.7 / cam_distance)
    fov = np.clip(fov, 0.2, 1.2)  # reasonable range
    camera = pyrender.PerspectiveCamera(yfov=fov, aspectRatio=1.0)
    scene.add(camera, pose=cam_pose)

    # Key light: directional from slightly above (ring light effect)
    key_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
    key_light_pose = np.eye(4)
    # Light aimed at face from slightly above and in front
    key_dir = np.array([0.0, -0.3, -1.0])
    key_dir = key_dir / np.linalg.norm(key_dir)
    key_right = np.cross(key_dir, world_up)
    key_right_norm = np.linalg.norm(key_right)
    if key_right_norm > 1e-9:
        key_right = key_right / key_right_norm
    else:
        key_right = np.array([1.0, 0.0, 0.0])
    key_up = np.cross(key_right, key_dir)
    key_light_pose[:3, 0] = key_right
    key_light_pose[:3, 1] = key_up
    key_light_pose[:3, 2] = -key_dir
    key_light_pose[:3, 3] = cam_pos
    scene.add(key_light, pose=key_light_pose)

    # Fill light: from opposite side at 30% intensity
    fill_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.2)
    fill_light_pose = np.eye(4)
    fill_dir = np.array([0.0, 0.1, -1.0])
    fill_dir = fill_dir / np.linalg.norm(fill_dir)
    fill_right = np.cross(fill_dir, world_up)
    fill_right_norm = np.linalg.norm(fill_right)
    if fill_right_norm > 1e-9:
        fill_right = fill_right / fill_right_norm
    else:
        fill_right = np.array([1.0, 0.0, 0.0])
    fill_up = np.cross(fill_right, fill_dir)
    fill_light_pose[:3, 0] = fill_right
    fill_light_pose[:3, 1] = fill_up
    fill_light_pose[:3, 2] = -fill_dir
    fill_light_pose[:3, 3] = cam_pos + np.array([0.0, -y_range * 0.3, 0.0])
    scene.add(fill_light, pose=fill_light_pose)

    # Render offscreen
    try:
        renderer = pyrender.OffscreenRenderer(
            viewport_width=image_size,
            viewport_height=image_size,
        )
        color, _ = renderer.render(scene)
        renderer.delete()
    except Exception as exc:
        log.warning("pyrender offscreen render failed: %s", exc)
        return None

    return color


def _detect_on_image(
    img_rgb: np.ndarray,
    image_size: int,
) -> Optional[np.ndarray]:
    """Run MediaPipe FaceLandmarker on an RGB image.

    Returns (478, 2) float64 pixel coords or None if no face detected.
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        return None

    if not MODEL_PATH.exists():
        return None

    # Convert to uint8 if needed
    if img_rgb.dtype != np.uint8:
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)

    # Ensure 3-channel
    if img_rgb.ndim == 2:
        img_rgb = np.stack([img_rgb, img_rgb, img_rgb], axis=-1)
    elif img_rgb.shape[2] == 4:
        img_rgb = img_rgb[:, :, :3]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img_rgb,
    )

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
    )
    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    lms = result.face_landmarks[0]
    h, w = img_rgb.shape[:2]
    pixels = np.array(
        [[lm.x * w, lm.y * h] for lm in lms],
        dtype=np.float64,
    )
    return pixels


def _render_and_detect(mesh: trimesh.Trimesh, image_size: int) -> Optional[np.ndarray]:
    """Render *mesh* to a frontal image and run FaceLandmarker.

    Tries pyrender (lit with directional lighting) first, then falls back to
    the CPU depth-shaded rasterizer.

    Returns (478, 2) float64 pixel coords or None.
    """
    try:
        import mediapipe as mp  # noqa: F401
        from mediapipe.tasks import python as mp_python  # noqa: F401
        from mediapipe.tasks.python import vision as mp_vision  # noqa: F401
    except ImportError:
        log.info("mediapipe not installed; landmark detection unavailable.")
        return None

    if not MODEL_PATH.exists():
        log.warning(
            "FaceLandmarker model missing at '%s'. "
            "Download: curl -L 'https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task' -o %s",
            MODEL_PATH, MODEL_PATH,
        )
        return None

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    x, y = verts[:, 0], verts[:, 1]
    x_range_raw = float(x.max() - x.min())
    y_range_raw = float(y.max() - y.min())
    if x_range_raw < 1e-9 or y_range_raw < 1e-9:
        log.warning("Landmark detection: degenerate XY range — skipping.")
        return None

    # ── Attempt 1: pyrender with directional lighting ──────────────────────
    pyrender_img = _render_pyrender(mesh, image_size)
    if pyrender_img is not None:
        pixels = _detect_on_image(pyrender_img, image_size)
        if pixels is not None:
            log.info("Landmark detection succeeded with pyrender-lit image.")
            return pixels
        log.info("MediaPipe failed on pyrender image; falling back to CPU rasterizer.")

    # ── Attempt 2: CPU depth-shaded rasterizer (original path) ─────────────
    PAD = 0.05
    x_min = x.min() - x_range_raw * PAD; x_range = x_range_raw * (1.0 + 2 * PAD)
    y_min = y.min() - y_range_raw * PAD; y_range = y_range_raw * (1.0 + 2 * PAD)
    px = np.clip(((x - x_min) / x_range * (image_size - 1)).astype(np.int32), 0, image_size - 1)
    py = np.clip(((1.0 - (y - y_min) / y_range) * (image_size - 1)).astype(np.int32), 0, image_size - 1)

    img = _rasterize(mesh, verts, np.asarray(mesh.faces, dtype=np.int32), px, py, image_size)
    img_rgb = np.stack([img, img, img], axis=-1)

    pixels = _detect_on_image(img_rgb, image_size)
    if pixels is None:
        log.warning("No face detected in rendered vertex image.")
    return pixels


def _rasterize(
    mesh: trimesh.Trimesh,
    verts: np.ndarray,
    faces: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    image_size: int,
) -> np.ndarray:
    """Painter's-algorithm CPU rasterizer: front-facing triangles, depth-shaded."""
    img = np.zeros((image_size, image_size), dtype=np.uint8)
    normals = mesh.face_normals
    front = normals[:, 2] > 0.0
    z_face = verts[faces, 2].mean(axis=1)
    front_idx = np.where(front)[0]
    if len(front_idx) == 0:
        return img

    order = front_idx[np.argsort(z_face[front_idx])]
    sf = faces[order]
    sz = z_face[order]
    z_lo, z_hi = float(sz[0]), float(sz[-1])
    z_rng = max(z_hi - z_lo, 1e-9)
    N_LV = 20
    lv_arr = np.floor((sz - z_lo) / z_rng * (N_LV - 1)).astype(np.int32).clip(0, N_LV - 1)
    pts2d = np.stack([px, py], axis=1).astype(np.int32)

    try:
        import cv2
        for lv in range(N_LV):
            sel = lv_arr == lv
            if not sel.any():
                continue
            color = int(80 + lv / (N_LV - 1) * 140)
            tris = pts2d[sf[sel]]
            cv2.fillPoly(img, list(tris.reshape(-1, 3, 1, 2)), color=color)
        return img
    except ImportError:
        pass

    try:
        from PIL import Image as PILImage, ImageDraw
        pil = PILImage.fromarray(img, mode="L")
        draw = ImageDraw.Draw(pil)
        for lv in range(N_LV):
            sel = lv_arr == lv
            if not sel.any():
                continue
            color = int(80 + lv / (N_LV - 1) * 140)
            for f in sf[sel]:
                tri = [(int(pts2d[f[i], 0]), int(pts2d[f[i], 1])) for i in range(3)]
                draw.polygon(tri, fill=color)
        return np.array(pil)
    except ImportError:
        pass

    # Last-resort dot fallback
    log.warning("No cv2/PIL available; using large-dot landmark render fallback.")
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            if dx * dx + dy * dy <= 16:
                img[
                    np.clip(py + dy, 0, image_size - 1),
                    np.clip(px + dx, 0, image_size - 1),
                ] = 200
    return img


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
