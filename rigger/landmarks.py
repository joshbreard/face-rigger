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


# ---------------------------------------------------------------------------
# Internal: software rasterizer + MediaPipe runner
# ---------------------------------------------------------------------------

def _render_and_detect(mesh: trimesh.Trimesh, image_size: int) -> Optional[np.ndarray]:
    """Render *mesh* to a frontal depth-shaded image and run FaceLandmarker.

    Returns (478, 2) float64 pixel coords or None.
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
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

    PAD = 0.05
    x_min = x.min() - x_range_raw * PAD; x_range = x_range_raw * (1.0 + 2 * PAD)
    y_min = y.min() - y_range_raw * PAD; y_range = y_range_raw * (1.0 + 2 * PAD)
    px = np.clip(((x - x_min) / x_range * (image_size - 1)).astype(np.int32), 0, image_size - 1)
    py = np.clip(((1.0 - (y - y_min) / y_range) * (image_size - 1)).astype(np.int32), 0, image_size - 1)

    img = _rasterize(mesh, verts, np.asarray(mesh.faces, dtype=np.int32), px, py, image_size)
    img_rgb = np.stack([img, img, img], axis=-1)

    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
    )
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    try:
        with mp_vision.FaceLandmarker.create_from_options(opts) as lmkr:
            result = lmkr.detect(mp_img)
    except Exception as exc:
        log.warning("FaceLandmarker exception: %s", exc)
        return None

    if not result.face_landmarks:
        log.warning("No face detected in rendered vertex image.")
        return None

    lms = result.face_landmarks[0]
    if len(lms) < 478:
        log.warning("Expected 478 landmarks, got %d.", len(lms))
        return None

    return np.array([[lm.x * image_size, lm.y * image_size] for lm in lms], dtype=np.float64)


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
