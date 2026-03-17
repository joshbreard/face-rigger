"""ARKit morph-target transfer using Claire's blendshape skin (bs_skin.npz).

Transfer strategy
-----------------
Primary: barycentric surface projection onto a convex hull built from Claire's
frontalMask vertices (requires the ``rtree`` package).  For each Meshy vertex,
find the nearest triangle on the hull, compute barycentric coordinates, and
interpolate the displacement from the 3 corner vertices.

Fallback: IDW k=4 (inverse-distance-weighted average of the 4 nearest Claire
vertices).  Activates automatically if rtree is absent or hull build fails.

Data note
---------
bs_skin.npz blendshape arrays store **delta/displacement vectors** in cm —
NOT absolute vertex positions.  Scale factor 0.01 converts to metres; no
subtraction of neutral is needed.
"""

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import trimesh
import trimesh.convex
import trimesh.proximity
import trimesh.triangles
from scipy.linalg import lu_factor, lu_solve
from scipy.spatial import KDTree

log = logging.getLogger(__name__)

BS_SKIN_PATH = Path("assets/bs_skin.npz")

# Canonical ARKit 52 blendshape names (order matches Apple's ARFaceAnchor spec)
ARKIT_BLENDSHAPES: list[str] = [
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
    "tongueOut",
]

assert len(ARKIT_BLENDSHAPES) == 52, "Must have exactly 52 blendshapes."

# ---------------------------------------------------------------------------
# Smile-shape local rescale thresholds.
# Applied only to mouthSmileLeft / mouthSmileRight in the landmark pipeline.
# ---------------------------------------------------------------------------

# Minimum reference amplitude used when Claire's local p95 displacement is
# near-zero (i.e. her canonical smile barely moves those verts).  We floor
# p95_claire to this value so the ratio is still well-defined.
SMILE_LOCAL_P95_MIN: float = 0.002  # metres (2 mm)

# Target mean local displacement for smile shapes after all rescaling.  If the
# resulting mean is still below this, a secondary boost is applied.
SMILE_LOCAL_MEAN_TARGET: float = 0.003  # metres (3 mm)

# Clamp range for the secondary mean-boost scale factor.
SMILE_MEAN_BOOST_MIN: float = 0.5
SMILE_MEAN_BOOST_MAX: float = 3.0

# ---------------------------------------------------------------------------
# Module-level Claire data — populated by _load_bs_skin() at import time.
# ---------------------------------------------------------------------------

# (V, 3) Claire neutral vertices scaled to metres, then centred at origin.
# Exposed so rigger/aligner.py can use the same point cloud for ICP.
claire_neutral_m: np.ndarray | None = None

# {arkit_name: (V, 3)} blendshape displacement vectors in metres (direct deltas).
_claire_deltas_m: dict[str, np.ndarray] | None = None

# Convex hull of frontal-face vertices — used for barycentric transfer.
# None if build failed (rtree absent) → IDW fallback activates.
_claire_hull: trimesh.Trimesh | None = None

# KDTree on the frontal subset of claire_neutral_m (for hull corner resolution).
_claire_frontal_tree: KDTree | None = None

# Maps frontal-subset local indices → full claire_neutral_m indices.
_claire_frontal_indices: np.ndarray | None = None


def _load_bs_skin() -> None:
    """Load Claire's blendshape skin from *BS_SKIN_PATH*. Called at import time."""
    global claire_neutral_m, _claire_deltas_m
    global _claire_hull, _claire_frontal_tree, _claire_frontal_indices

    if not BS_SKIN_PATH.exists():
        log.error(
            "STARTUP ERROR: '%s' not found. "
            "Place assets/bs_skin.npz in the project root. "
            "POST /rig will fail until the file is present.",
            BS_SKIN_PATH,
        )
        return

    data = np.load(BS_SKIN_PATH, allow_pickle=True)

    # Neutral vertices (cm) → scale to metres, then centre at origin.
    if "neutral" not in data.files:
        raise KeyError(
            f"'neutral' key not found in '{BS_SKIN_PATH}'. "
            f"Keys present: {list(data.files)}"
        )
    neutral_m = data["neutral"].astype(np.float64) * 0.01  # cm → m
    centroid = neutral_m.mean(axis=0)
    claire_neutral_m = neutral_m - centroid

    # Blendshape arrays store delta vectors (cm) — scale to metres directly.
    # Do NOT subtract neutral_m: the arrays are already displacements.
    deltas: dict[str, np.ndarray] = {}
    missing = []
    for name in ARKIT_BLENDSHAPES:
        if name not in data.files:
            missing.append(name)
            continue
        deltas[name] = data[name].astype(np.float64) * 0.01  # cm → m (delta)

    if missing:
        log.warning(
            "%d blendshape key(s) not found in '%s' — will be zero: %s",
            len(missing),
            BS_SKIN_PATH,
            missing,
        )

    _claire_deltas_m = deltas

    # Log jawOpen stats to sanity-check the data interpretation.
    if "jawOpen" in deltas:
        jaw_mags = np.linalg.norm(deltas["jawOpen"], axis=1)
        log.info(
            "jawOpen delta check: mean=%.5fm  max=%.5fm  nonzero=%d / %d",
            jaw_mags.mean(), jaw_mags.max(),
            (jaw_mags > 1e-9).sum(), len(jaw_mags),
        )

    log.info(
        "Loaded Claire blendshape skin: %d vertices, %d / 52 blendshapes (delta cm→m, centred).",
        len(neutral_m),
        len(deltas),
    )

    # ── Build frontal convex hull for barycentric transfer ──────────────────
    if "frontalMask" not in data.files:
        log.warning("'frontalMask' not in bs_skin.npz — barycentric transfer unavailable; IDW k=4 will be used.")
        return

    frontal_indices = data["frontalMask"].astype(np.int64)
    frontal_verts = claire_neutral_m[frontal_indices]  # already centred/scaled

    log.info(
        "frontalMask: %d frontal-face vertices selected from %d total.",
        len(frontal_indices), len(claire_neutral_m),
    )

    try:
        import rtree  # noqa: F401 — presence check only; trimesh.proximity needs it
        hull_src = trimesh.Trimesh(vertices=frontal_verts, process=False)
        _claire_hull = trimesh.convex.convex_hull(hull_src)
        _claire_frontal_tree = KDTree(frontal_verts)
        _claire_frontal_indices = frontal_indices
        log.info(
            "Built Claire frontal convex hull: %d vertices, %d triangles. "
            "Barycentric transfer enabled.",
            len(_claire_hull.vertices),
            len(_claire_hull.faces),
        )
    except ImportError:
        log.warning(
            "rtree package not installed — barycentric transfer unavailable. "
            "Run `pip install rtree` to enable it. IDW k=4 fallback will be used."
        )
    except Exception as exc:
        log.warning(
            "Frontal hull build failed (%s) — IDW k=4 fallback will be used.", exc
        )


_load_bs_skin()


# ---------------------------------------------------------------------------
# Transfer helpers
# ---------------------------------------------------------------------------

def _precompute_bary_mapping(
    target_verts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find nearest hull triangle for each target vertex; resolve Claire indices.

    Returns
    -------
    closest_pts : (N, 3)  — projection of each target vertex onto hull surface
    triangles   : (N, 3, 3) — 3D positions of the 3 corner vertices per triangle
    tri_claire_indices : (N, 3) — index into full claire_neutral_m for each corner
    """
    closest_pts, _, tri_ids = trimesh.proximity.closest_point(_claire_hull, target_verts)
    triangles = _claire_hull.vertices[_claire_hull.faces[tri_ids]]  # (N, 3, 3)

    # Resolve hull corner positions → frontal local index → full claire index
    flat = triangles.reshape(-1, 3)                          # (N*3, 3)
    _, flat_local = _claire_frontal_tree.query(flat)         # (N*3,)
    flat_global = _claire_frontal_indices[flat_local]        # (N*3,)
    tri_claire_indices = flat_global.reshape(len(target_verts), 3)  # (N, 3)

    return closest_pts, triangles, tri_claire_indices


def _transfer_barycentric_with_mapping(
    claire_disp: np.ndarray,
    mapping: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Interpolate *claire_disp* using precomputed barycentric mapping."""
    closest_pts, triangles, tri_claire_indices = mapping
    bary = trimesh.triangles.points_to_barycentric(triangles, closest_pts)  # (N, 3)
    bary = np.clip(bary, 0.0, None)
    row_sums = bary.sum(axis=1, keepdims=True)
    bary /= np.where(row_sums > 1e-12, row_sums, 1.0)
    corner_disp = claire_disp[tri_claire_indices]            # (N, 3, 3)
    return (corner_disp * bary[:, :, np.newaxis]).sum(axis=1)  # (N, 3)


def _transfer_idw(
    target_verts: np.ndarray,
    claire_disp: np.ndarray,
    tree: KDTree,
    k: int = 4,
) -> np.ndarray:
    """Inverse-distance-weighted transfer; fallback when hull unavailable."""
    k = min(k, len(claire_neutral_m))
    dists, idxs = tree.query(target_verts, k=k)  # (N, k)
    w = 1.0 / (dists + 1e-12)
    w /= w.sum(axis=1, keepdims=True)
    return (claire_disp[idxs] * w[:, :, np.newaxis]).sum(axis=1)  # (N, 3)


# Mediapipe landmark index -> canonical Claire anchor position (metres, centred).
# Covers 8 anatomical control points across the face.
_LANDMARK_TO_CLAIRE_POS: dict[int, tuple[float, float, float]] = {
    152: (0.0,    -0.050,  0.020),   # jaw tip
    33:  (-0.030,  0.020,  0.040),   # left eye centre
    263: ( 0.030,  0.020,  0.040),   # right eye centre
    4:   ( 0.0,   -0.005,  0.055),   # nose tip
    61:  (-0.025, -0.025,  0.040),   # left mouth corner
    291: ( 0.025, -0.025,  0.040),   # right mouth corner
    70:  (-0.030,  0.040,  0.030),   # left brow peak
    300: ( 0.030,  0.040,  0.030),   # right brow peak
}


def _detect_landmarks(mesh: trimesh.Trimesh, image_size: int = 512) -> np.ndarray | None:
    """Project *mesh* to a frontal 2D image with filled triangles and run mediapipe FaceLandmarker.

    Parameters
    ----------
    mesh : trimesh.Trimesh in ICP-aligned metres, centred at origin.

    Returns
    -------
    (478, 2) pixel coords of the detected landmarks, or None on failure.
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        return None

    MODEL_PATH = Path("assets/face_landmarker.task")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model file: {MODEL_PATH}. "
            "Run: curl -L 'https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task'"
            " -o assets/face_landmarker.task"
        )

    mesh_verts = np.asarray(mesh.vertices, dtype=np.float64)
    mesh_faces = np.asarray(mesh.faces, dtype=np.int32)

    x = mesh_verts[:, 0]
    y = mesh_verts[:, 1]
    x_min_raw = float(x.min())
    x_range_raw = float(x.max() - x_min_raw)
    y_min_raw = float(y.min())
    y_range_raw = float(y.max() - y_min_raw)

    if x_range_raw < 1e-9 or y_range_raw < 1e-9:
        log.warning("Landmark detection: degenerate vertex XY range — skipping.")
        return None

    # 5% padding on each side so the face doesn't butt against the image border.
    PAD = 0.05
    x_min = x_min_raw - x_range_raw * PAD
    x_range = x_range_raw * (1.0 + 2.0 * PAD)
    y_min = y_min_raw - y_range_raw * PAD
    y_range = y_range_raw * (1.0 + 2.0 * PAD)

    # Map world XY -> pixel coords (flip Y so +Y = up in world = up in image).
    px = np.clip(((x - x_min) / x_range * (image_size - 1)).astype(np.int32), 0, image_size - 1)
    py = np.clip(((1.0 - (y - y_min) / y_range) * (image_size - 1)).astype(np.int32), 0, image_size - 1)

    # ── Render filled triangles with depth shading ────────────────────────────
    # Front-facing triangles (normal_z > 0) are depth-sorted and shaded by Z
    # to produce a recognisable face silhouette that MediaPipe's detector can latch onto.
    img = np.zeros((image_size, image_size), dtype=np.uint8)
    pts_2d = np.stack([px, py], axis=1).astype(np.int32)

    face_normals = mesh.face_normals          # (F, 3) unit normals
    front_mask = face_normals[:, 2] > 0.0    # only camera-facing triangles

    z_verts = mesh_verts[:, 2]
    face_z = z_verts[mesh_faces].mean(axis=1)  # mean Z per face

    front_idx = np.where(front_mask)[0]
    rendered = False
    if len(front_idx) > 0:
        # Back-to-front order (painter's algorithm).
        order = front_idx[np.argsort(face_z[front_idx])]
        sorted_faces = mesh_faces[order]   # (F', 3)
        sorted_z     = face_z[order]
        z_lo = float(sorted_z[0])
        z_hi = float(sorted_z[-1])
        z_rng = max(z_hi - z_lo, 1e-9)

        # Quantise depth into 20 grey levels (80–220) for batch rendering.
        N_LEVELS = 20
        level_arr = np.floor(
            (sorted_z - z_lo) / z_rng * (N_LEVELS - 1)
        ).astype(np.int32).clip(0, N_LEVELS - 1)

        try:
            import cv2
            for lv in range(N_LEVELS):
                sel = level_arr == lv
                if not sel.any():
                    continue
                color = int(80 + lv / (N_LEVELS - 1) * 140)
                tris = pts_2d[sorted_faces[sel]]  # (M, 3, 2)
                # cv2.fillPoly expects a list of (N_pts, 1, 2) int32 arrays.
                polys = list(tris.reshape(len(tris), 3, 1, 2))
                cv2.fillPoly(img, polys, color=color)
            rendered = True
        except ImportError:
            pass

        if not rendered:
            try:
                from PIL import Image as PILImage, ImageDraw
                pil_img = PILImage.fromarray(img, mode="L")
                draw = ImageDraw.Draw(pil_img)
                for lv in range(N_LEVELS):
                    sel = level_arr == lv
                    if not sel.any():
                        continue
                    color = int(80 + lv / (N_LEVELS - 1) * 140)
                    for f in sorted_faces[sel]:
                        tri = [
                            (int(pts_2d[f[0], 0]), int(pts_2d[f[0], 1])),
                            (int(pts_2d[f[1], 0]), int(pts_2d[f[1], 1])),
                            (int(pts_2d[f[2], 0]), int(pts_2d[f[2], 1])),
                        ]
                        draw.polygon(tri, fill=color)
                img = np.array(pil_img)
                rendered = True
            except ImportError:
                pass

    if not rendered:
        # Last-resort fallback: large dots (radius 4 px).
        log.warning(
            "Landmark render: neither cv2 nor PIL available; using large dot fallback "
            "(face detection may fail)."
        )
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                if dx * dx + dy * dy <= 16:
                    img[
                        np.clip(py + dy, 0, image_size - 1),
                        np.clip(px + dx, 0, image_size - 1),
                    ] = 200

    img_rgb = np.stack([img, img, img], axis=-1)  # Tasks API needs HxWx3 SRGB

    BaseOptions = mp_python.BaseOptions
    FaceLandmarker = mp_vision.FaceLandmarker
    FaceLandmarkerOptions = mp_vision.FaceLandmarkerOptions
    VisionRunningMode = mp_vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
    )
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    with FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        log.warning("Landmark detection: no face detected in projected vertex image.")
        return None

    lms = result.face_landmarks[0]
    if len(lms) < 478:
        log.warning("Landmark detection: expected 478 landmarks, got %d.", len(lms))
        return None

    # Landmark .x / .y are normalised [0, 1]; multiply by image_size for pixels.
    coords = np.array([[lm.x * image_size, lm.y * image_size] for lm in lms],
                      dtype=np.float64)  # (478, 2)
    return coords


def _build_landmark_anchors(
    meshy_verts: np.ndarray,
    landmark_pixels: np.ndarray,
    image_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Map detected landmark pixels to Meshy vertex indices and Claire anchor indices.

    Parameters
    ----------
    meshy_verts : (N, 3) Meshy head vertices (metres, centred).
    landmark_pixels : (478, 2) pixel coords from _detect_landmarks.

    Returns
    -------
    meshy_anchor_indices : (K,) int  — index into meshy_verts per control point
    claire_anchor_indices : (K,) int — index into claire_neutral_m per control point
    """
    x = meshy_verts[:, 0]
    y = meshy_verts[:, 1]
    x_min_raw = float(x.min())
    x_range_raw = float(x.max() - x_min_raw)
    y_min_raw = float(y.min())
    y_range_raw = float(y.max() - y_min_raw)

    # Must match the 5% padding used in _detect_landmarks.
    PAD = 0.05
    x_min = x_min_raw - x_range_raw * PAD
    x_range = x_range_raw * (1.0 + 2.0 * PAD)
    y_min = y_min_raw - y_range_raw * PAD
    y_range = y_range_raw * (1.0 + 2.0 * PAD)

    # KDTree on Meshy XY projection for fast nearest-vertex lookup.
    meshy_xy_tree = KDTree(meshy_verts[:, :2])

    meshy_anchor_indices: list[int] = []
    claire_anchor_indices: list[int] = []

    for lm_idx, claire_pos in _LANDMARK_TO_CLAIRE_POS.items():
        px, py_pix = landmark_pixels[lm_idx]

        # Unnormalise pixel -> world XY (inverse of _detect_landmarks mapping).
        world_x = px / (image_size - 1) * x_range + x_min
        world_y = (1.0 - py_pix / (image_size - 1)) * y_range + y_min

        _, nearest_meshy = meshy_xy_tree.query([world_x, world_y])
        meshy_anchor_indices.append(int(nearest_meshy))

        # Nearest Claire frontal vertex to the canonical anatomical position.
        _, local_idx = _claire_frontal_tree.query(np.array(claire_pos))
        claire_anchor_indices.append(int(_claire_frontal_indices[local_idx]))

    return (
        np.array(meshy_anchor_indices, dtype=np.int64),
        np.array(claire_anchor_indices, dtype=np.int64),
    )


def _transfer_landmark_anchored(
    target_verts: np.ndarray,
    claire_disp: np.ndarray,
    meshy_anchor_indices: np.ndarray,
    claire_anchor_indices: np.ndarray,
    idw_tree: KDTree,
) -> np.ndarray:
    """RBF (thin-plate spline) displacement transfer anchored at landmarks.

    Uses phi(r) = r^2 * log(r + 1e-10).  Weights are solved once via LU
    factorisation of Phi; the same factorisation is reused for all 3 spatial
    components (x, y, z).  Vertices further than 0.08 m from any anchor are
    blended towards an IDW result.

    Parameters
    ----------
    target_verts : (N, 3)
    claire_disp  : (V_claire, 3)
    meshy_anchor_indices  : (K,)
    claire_anchor_indices : (K,)
    idw_tree     : KDTree on all claire_neutral_m

    Returns
    -------
    (N, 3) displacement array.
    """
    meshy_anchors = target_verts[meshy_anchor_indices]  # (K, 3)
    K = len(meshy_anchors)

    # Build K×K RBF matrix.
    diff_aa = meshy_anchors[:, np.newaxis, :] - meshy_anchors[np.newaxis, :, :]  # (K,K,3)
    r_aa = np.linalg.norm(diff_aa, axis=2)  # (K, K)
    Phi = r_aa ** 2 * np.log(r_aa + 1e-10)  # thin-plate spline kernel

    rhs = claire_disp[claire_anchor_indices]  # (K, 3)

    # Factorise once, solve for x / y / z independently.
    lu, piv = lu_factor(Phi)
    w = lu_solve((lu, piv), rhs)  # (K, 3)

    # Evaluate at every target vertex.
    diff_ta = target_verts[:, np.newaxis, :] - meshy_anchors[np.newaxis, :, :]  # (N,K,3)
    r_ta = np.linalg.norm(diff_ta, axis=2)  # (N, K)
    D = r_ta ** 2 * np.log(r_ta + 1e-10)   # (N, K)
    rbf_disp = D @ w  # (N, 3)

    # IDW fallback for vertices far from any anchor.
    idw_disp = _transfer_idw(target_verts, claire_disp, idw_tree, k=4)

    anchor_tree = KDTree(meshy_anchors)
    d_anchor, _ = anchor_tree.query(target_verts)  # (N,)

    # blend_weight = 1 (pure RBF) when d <= 0.08; decays exponentially beyond.
    blend_weight = np.where(d_anchor > 0.08, np.exp(-d_anchor / 0.04), 1.0)
    blend_weight = blend_weight[:, np.newaxis]  # (N, 1) for broadcasting

    return blend_weight * rbf_disp + (1.0 - blend_weight) * idw_disp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transfer_morph_targets(
    target_mesh: trimesh.Trimesh,
    alignment_meta: dict | None = None,
) -> tuple[trimesh.Trimesh, dict[str, np.ndarray]]:
    """Transfer ARKit morph targets from Claire's blendshape skin to *target_mesh*.

    Parameters
    ----------
    target_mesh:
        Aligned Meshy head mesh (metres, centred) from ``rigger.aligner.align_icp``.
    alignment_meta:
        Optional dict returned by align_icp (bbox_ratio, fitness, rmse).

    Returns
    -------
    (rigged_mesh, blendshapes)
        rigged_mesh : trimesh.Trimesh
        blendshapes : dict[str, ndarray] — shape (N_target, 3) per blendshape
    """
    if claire_neutral_m is None or _claire_deltas_m is None:
        raise RuntimeError(
            f"Claire blendshape data not loaded. "
            f"Ensure '{BS_SKIN_PATH}' exists and is valid."
        )

    # NOTE: kaolin_deformer is not called here.
    # Deforming target_mesh vertices before transfer causes delta vectors computed
    # in ICP-aligned space to be applied against the original undeformed GLB vertices,
    # producing inconsistent displacements and visible mesh tears.  The deformer
    # module (rigger/kaolin_deformer.py) is retained for future use once the
    # pipeline is extended to propagate the deformed-mesh reference frame through
    # to the GLB patch step.

    target_verts = np.array(target_mesh.vertices, dtype=np.float64)

    log.info(
        "Transfer: target mesh has %d vertices.  Centroid=[%.4f, %.4f, %.4f]",
        len(target_verts), *target_verts.mean(axis=0),
    )

    # Full IDW tree (always built — used as landmark-RBF fallback and IDW path).
    idw_tree_full = KDTree(claire_neutral_m)

    # ── Try landmark detection ───────────────────────────────────────────────
    _landmark_anchors = None
    if _claire_frontal_tree is not None and _claire_frontal_indices is not None:
        try:
            import mediapipe  # noqa: F401 — presence check
            import mediapipe.tasks  # noqa: F401 — ensure Tasks API is available
            lm_pixels = _detect_landmarks(target_mesh)
            if lm_pixels is not None:
                meshy_idx, claire_idx = _build_landmark_anchors(target_verts, lm_pixels)
                _landmark_anchors = (meshy_idx, claire_idx)
                log.info("LANDMARK anchors built: %d control points.", len(meshy_idx))
            else:
                log.warning("Landmark detection failed; falling back to barycentric/IDW.")
        except ImportError:
            log.info("mediapipe not installed; using barycentric/IDW transfer.")

    # ── Prepare barycentric / IDW structures (only when landmark unavailable) ─
    use_barycentric = _claire_hull is not None
    mapping = None
    idw_tree = None  # used only in pure-IDW path

    if _landmark_anchors is None:
        if use_barycentric:
            log.info(
                "Transfer method: BARYCENTRIC (frontal hull %d tris, %d verts).",
                len(_claire_hull.faces), len(_claire_hull.vertices),
            )
            log.info("Precomputing barycentric mapping for %d target vertices...", len(target_verts))
            mapping = _precompute_bary_mapping(target_verts)
        else:
            log.info(
                "Transfer method: IDW k=4 (barycentric hull unavailable). "
                "Reusing full KD-tree on %d Claire vertices.",
                len(claire_neutral_m),
            )
            idw_tree = idw_tree_full

    # Reference Claire jawOpen magnitude for scale mismatch detection.
    jaw_claire_disp = _claire_deltas_m.get("jawOpen", np.zeros_like(claire_neutral_m))
    jaw_claire_mags = np.linalg.norm(jaw_claire_disp, axis=1)
    jaw_claire_mean = float(jaw_claire_mags[jaw_claire_mags > 1e-9].mean()) if (jaw_claire_mags > 1e-9).any() else 0.0
    log.info("Claire jawOpen reference: mean=%.6fm (%.2fmm)", jaw_claire_mean, jaw_claire_mean * 1000)

    # ── Anatomical masking: per-mesh Y thresholds ───────────────────────────
    # All coordinates are in ICP-aligned space (metres, Y-up).
    _y_verts = target_verts[:, 1]
    _z_verts = target_verts[:, 2]
    _x_verts = target_verts[:, 0]
    _face_height = float(_y_verts.max() - _y_verts.min())
    _x_range_mesh = float(_x_verts.max() - _x_verts.min())

    # Y-distribution diagnostic: shows where the face zones actually sit.
    _y_pcts = np.percentile(_y_verts, [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95])
    log.info(
        "Target Y percentiles (ICP-aligned, metres): "
        "p5=%.4f  p10=%.4f  p20=%.4f  p30=%.4f  p40=%.4f  p50=%.4f  "
        "p60=%.4f  p70=%.4f  p80=%.4f  p90=%.4f  p95=%.4f",
        *_y_pcts,
    )
    log.info(
        "Target mesh extent: Y=[%.4f, %.4f]  Z=[%.4f, %.4f]  X=[%.4f, %.4f]  "
        "face_height=%.4fm",
        float(_y_verts.min()), float(_y_verts.max()),
        float(_z_verts.min()), float(_z_verts.max()),
        float(_x_verts.min()), float(_x_verts.max()),
        _face_height,
    )

    # Nose tip: frontmost (max Z) vertex near the facial centre (|X| < 15 % of width).
    _center_mask = np.abs(_x_verts) < _x_range_mesh * 0.15
    _n_center_candidates = int(_center_mask.sum())
    if _n_center_candidates >= 1:
        _center_indices = np.where(_center_mask)[0]
        _nose_local = int(np.argmax(_z_verts[_center_mask]))
        _nose_tip_idx = int(_center_indices[_nose_local])
        _nose_y = float(_y_verts[_nose_tip_idx])
        log.info(
            "Nose tip vertex: idx=%d  pos=(X=%.4f, Y=%.4f, Z=%.4f)  "
            "candidates_in_center_band=%d  |X|_threshold=%.4fm",
            _nose_tip_idx,
            float(_x_verts[_nose_tip_idx]),
            float(_y_verts[_nose_tip_idx]),
            float(_z_verts[_nose_tip_idx]),
            _n_center_candidates,
            _x_range_mesh * 0.15,
        )
    else:
        _nose_y = float(np.percentile(_y_verts, 40))
        log.warning("Nose tip: no center-band vertices found — fallback nose_y=%.4fm", _nose_y)

    # Jaw-region upper cutoff.
    #
    # We tried deriving this from Claire's jawOpen displacement field but that
    # approach fails: jawOpen displaces vertices all the way up to the cheeks,
    # so "max Y of significant jaw verts" lands at cheek level — far above the
    # mouth.  The correct boundary is simpler: the nose tip Y IS the cutoff.
    # Anything at or above the nose base must not move with jaw blendshapes.
    # We subtract 5 mm (nose_y - 0.005) so the cutoff sits just below the nose
    # tip, fully suppressing the nose and all geometry above it.
    #
    # Sanity check from this run's data:
    #   nose_y ≈ -0.0147 m  (nose tip, correctly found as max-Z centre vertex)
    #   Old value (nose_y + 0.010) ≈ -0.0047 m  ← too high, nose still moved
    #   New value (nose_y - 0.005) ≈ -0.0197 m  ← below nose base → correct
    _jaw_cutoff_y = _nose_y - 0.005

    # Lower lip / upper-mouth boundary.
    # Estimate chin as the 5th-percentile Y (lowest real face geometry, not outliers).
    _chin_y = float(np.percentile(_y_verts, 5))
    _nose_to_chin = abs(_nose_y - _chin_y)
    # Upper lip sits roughly 20 % of the nose-to-chin distance below the nose.
    _lower_lip_y = _nose_y - 0.20 * _nose_to_chin

    # Nose base Y: slightly below nose tip (≈ 5 % of face height lower).
    _nose_base_y = _nose_y - 0.05 * _face_height

    log.info(
        "Anatomical mask thresholds — "
        "jaw_cutoff_y=%.4fm (nose_y=%.4fm - 5mm)  "
        "lower_lip_y=%.4fm  nose_base_y=%.4fm  "
        "chin_y(p5)=%.4fm  nose_to_chin=%.4fm  face_height=%.4fm",
        _jaw_cutoff_y, _nose_y, _lower_lip_y, _nose_base_y,
        _chin_y, _nose_to_chin, _face_height,
    )

    # Blendshape → mask group membership.
    _MASK_LOWER_JAW: frozenset[str] = frozenset({
        "jawOpen", "jawForward", "jawLeft", "jawRight",
        "mouthClose", "mouthFunnel", "mouthPucker",
        "mouthLeft", "mouthRight",
        "mouthLowerDownLeft", "mouthLowerDownRight",
        "mouthRollLower", "mouthShrugLower",
        "mouthPressLeft", "mouthPressRight",
    })
    _MASK_UPPER_MOUTH: frozenset[str] = frozenset({
        "mouthShrugUpper", "mouthUpperUpLeft", "mouthUpperUpRight", "mouthRollUpper",
    })
    _MASK_NOSE_SNEER: frozenset[str] = frozenset({
        "noseSneerLeft", "noseSneerRight",
    })

    # Boolean vertex masks (computed once, reused across blendshapes).
    _above_jaw_cutoff = _y_verts > _jaw_cutoff_y   # at/above cutoff → frozen for jaw shapes
    _below_lip_mask   = _y_verts < _lower_lip_y    # vertices below lower lip
    _below_nose_base  = _y_verts < _nose_base_y    # vertices below nose base

    # Soft falloff weights for jawOpen only: linear ramp in the 15 mm zone
    # just below jaw_cutoff_y.  Outside the blend zone the weight is either
    # 0 (above cutoff, fully suppressed) or 1 (below cutoff-15mm, full motion).
    # Shape: (N,) float32, values in [0, 1].
    _JAW_BLEND_MM   = 0.015                        # 15 mm blend zone
    _jaw_blend_low  = _jaw_cutoff_y - _JAW_BLEND_MM
    _jaw_open_weights = np.clip(
        (_jaw_cutoff_y - _y_verts) / _JAW_BLEND_MM,
        0.0, 1.0,
    ).astype(np.float32)                           # 0 at cutoff, 1 at cutoff-15mm

    log.info(
        "Vertex mask counts — above_jaw_cutoff (Y>%.4f): %d/%d  "
        "in_jaw_blend_zone (%.4f<Y<=%.4f): %d/%d  "
        "below_lower_lip (Y<%.4f): %d/%d  below_nose_base (Y<%.4f): %d/%d",
        _jaw_cutoff_y, int(_above_jaw_cutoff.sum()), len(target_verts),
        _jaw_blend_low, _jaw_cutoff_y,
        int(((_y_verts > _jaw_blend_low) & (_y_verts <= _jaw_cutoff_y)).sum()), len(target_verts),
        _lower_lip_y,  int(_below_lip_mask.sum()),   len(target_verts),
        _nose_base_y,  int(_below_nose_base.sum()),  len(target_verts),
    )

    target_morph_targets: dict[str, np.ndarray] = {}
    jaw_transferred_mean: float | None = None

    for name in ARKIT_BLENDSHAPES:
        claire_disp = _claire_deltas_m.get(name, np.zeros_like(claire_neutral_m))

        if _landmark_anchors is not None:
            target_disp = _transfer_landmark_anchored(
                target_verts, claire_disp, *_landmark_anchors, idw_tree_full
            )
        elif use_barycentric:
            target_disp = _transfer_barycentric_with_mapping(claire_disp, mapping)
        else:
            target_disp = _transfer_idw(target_verts, claire_disp, idw_tree)

        mags = np.linalg.norm(target_disp, axis=1)
        log.info(
            "%-30s  min=%.5fm  max=%.5fm  mean=%.5fm  nonzero=%d/%d",
            name, mags.min(), mags.max(), mags.mean(),
            int((mags > 1e-9).sum()), len(target_verts),
        )
        if mags.mean() < 0.001:
            log.warning("NEAR-ZERO: %s mean=%.6fm — check scale/alignment.", name, mags.mean())

        if name == "jawOpen":
            nonzero = mags[mags > 1e-9]
            jaw_transferred_mean = float(nonzero.mean()) if len(nonzero) else 0.0

        # ── Post-transfer anatomical masking ────────────────────────────────
        # "active" = displacement magnitude > 0.1 mm (meaningful motion)
        _active = mags > 1e-4
        if name in _MASK_LOWER_JAW:
            n_masked        = int(_above_jaw_cutoff.sum())
            n_active_masked = int((_above_jaw_cutoff & _active).sum())
            target_disp = target_disp.copy()
            if name == "jawOpen":
                # Soft falloff in the 15 mm blend zone below the cutoff to
                # avoid a visible seam at the philtrum/upper-lip boundary.
                # Vertices above the cutoff are still hard-zeroed.
                target_disp *= _jaw_open_weights[:, np.newaxis]
                n_blended = int(((_jaw_open_weights > 0.0) & (_jaw_open_weights < 1.0)).sum())
                n_blended_active = int(
                    ((_jaw_open_weights > 0.0) & (_jaw_open_weights < 1.0) & _active).sum()
                )
                log.info(
                    "Mask [jawOpen]         %-30s  "
                    "zeroed %d/%d verts above jaw_cutoff_y=%.4fm  "
                    "blended %d/%d verts in blend zone (%.4f–%.4fm)  "
                    "(%d zeroed, %d blended had |disp|>0.1mm)",
                    name, n_masked, len(target_verts), _jaw_cutoff_y,
                    n_blended, len(target_verts), _jaw_blend_low, _jaw_cutoff_y,
                    n_active_masked, n_blended_active,
                )
            else:
                if n_masked > 0:
                    target_disp[_above_jaw_cutoff] = 0.0
                log.info(
                    "Mask [jaw/lower-mouth] %-30s  "
                    "zeroed %d/%d verts above jaw_cutoff_y=%.4fm  "
                    "(%d had |disp|>0.1mm — these were suppressed)",
                    name, n_masked, len(target_verts), _jaw_cutoff_y, n_active_masked,
                )
        elif name in _MASK_UPPER_MOUTH:
            n_masked        = int(_below_lip_mask.sum())
            n_active_masked = int((_below_lip_mask & _active).sum())
            if n_masked > 0:
                target_disp = target_disp.copy()
                target_disp[_below_lip_mask] = 0.0
            log.info(
                "Mask [upper-mouth]    %-30s  "
                "zeroed %d/%d verts below lower_lip_y=%.4fm  "
                "(%d had |disp|>0.1mm — these were suppressed)",
                name, n_masked, len(target_verts), _lower_lip_y, n_active_masked,
            )
        elif name in _MASK_NOSE_SNEER:
            n_masked        = int(_below_nose_base.sum())
            n_active_masked = int((_below_nose_base & _active).sum())
            if n_masked > 0:
                target_disp = target_disp.copy()
                target_disp[_below_nose_base] = 0.0
            log.info(
                "Mask [nose-sneer]     %-30s  "
                "zeroed %d/%d verts below nose_base_y=%.4fm  "
                "(%d had |disp|>0.1mm — these were suppressed)",
                name, n_masked, len(target_verts), _nose_base_y, n_active_masked,
            )

        target_morph_targets[name] = target_disp

    # ── Scale mismatch detection and correction ─────────────────────────────
    jaw_ratio: float | None = None
    if jaw_claire_mean > 1e-9 and jaw_transferred_mean is not None and jaw_transferred_mean > 1e-9:
        jaw_ratio = jaw_claire_mean / jaw_transferred_mean
        log.info(
            "Scale check: Claire jawOpen mean=%.6fm, transferred=%.6fm, ratio=%.3f",
            jaw_claire_mean, jaw_transferred_mean, jaw_ratio,
        )
        if jaw_ratio > 2.0 or jaw_ratio < 0.5:
            log.warning(
                "SCALE MISMATCH (ratio=%.3f) — rescaling all 52 targets by %.4f.",
                jaw_ratio, jaw_ratio,
            )
            target_morph_targets = {k: v * jaw_ratio for k, v in target_morph_targets.items()}

    # ── Post-RBF local rescaling for eye and smile shapes ───────────────────────
    # For blendshapes whose transfer is known to under/over-shoot in a localised
    # anatomical region, rebalance by matching the Claire source 95th-percentile
    # local displacement magnitude.
    _RESCALE_EYE_PREFIXES = ("eyeBlink", "eyeLook", "eyeWide", "eyeSquint")
    _RESCALE_SMILE_SHAPES = frozenset({"mouthSmileLeft", "mouthSmileRight"})
    _RESCALE_RADIUS_M = 0.012  # 12 mm neighbourhood radius

    # Landmark index order within _LANDMARK_TO_CLAIRE_POS (dict-insertion order, Python 3.7+).
    _lm_key_list = list(_LANDMARK_TO_CLAIRE_POS.keys())
    # [152, 33, 263, 4, 61, 291, 70, 300]

    for _rname in ARKIT_BLENDSHAPES:
        if _rname not in target_morph_targets:
            continue

        _is_eye = any(_rname.startswith(p) for p in _RESCALE_EYE_PREFIXES)
        _is_smile = _rname in _RESCALE_SMILE_SHAPES
        if not (_is_eye or _is_smile):
            continue

        _is_left = _rname.endswith("Left")
        if _is_eye:
            _lm_idx = 33 if _is_left else 263   # left/right eye centre
        else:
            _lm_idx = 61 if _is_left else 291   # left/right mouth corner

        # Anchor position in Claire space (canonical).
        _anchor_claire = np.array(_LANDMARK_TO_CLAIRE_POS[_lm_idx])

        # Anchor position in target space: use detected landmark vertex if available,
        # otherwise fall back to the same canonical position (meshes are ICP-aligned).
        if _landmark_anchors is not None:
            _k = _lm_key_list.index(_lm_idx)
            _anchor_target = target_verts[_landmark_anchors[0][_k]]
        else:
            _anchor_target = _anchor_claire.copy()

        # Local vertex masks (within 12 mm of anchor).
        _dists_t = np.linalg.norm(target_verts - _anchor_target, axis=1)
        _local_t  = _dists_t <= _RESCALE_RADIUS_M
        _dists_c  = np.linalg.norm(claire_neutral_m - _anchor_claire, axis=1)
        _local_c  = _dists_c <= _RESCALE_RADIUS_M

        _n_local_t = int(_local_t.sum())
        _n_local_c = int(_local_c.sum())

        if _n_local_t < 3 or _n_local_c < 3:
            log.warning(
                "LocalRescale [%s]: too few local verts (target=%d, claire=%d) within %.0fmm — skipping.",
                _rname, _n_local_t, _n_local_c, _RESCALE_RADIUS_M * 1000,
            )
            continue

        # Claire source 95th-percentile local displacement magnitude.
        _claire_disp_r = _claire_deltas_m.get(_rname, np.zeros_like(claire_neutral_m))
        _claire_local_mags = np.linalg.norm(_claire_disp_r[_local_c], axis=1)
        _claire_p95 = float(np.percentile(_claire_local_mags, 95))

        # Transferred 95th-percentile local displacement magnitude.
        _xfer_disp = target_morph_targets[_rname]
        _xfer_local_mags = np.linalg.norm(_xfer_disp[_local_t], axis=1)
        _xfer_p95 = float(np.percentile(_xfer_local_mags, 95))

        # Determine the reference p95 amplitude.  For smile shapes whose
        # Claire p95 is near-zero we clamp up to SMILE_LOCAL_P95_MIN so
        # the ratio is still meaningful (and typically > 1, scaling the
        # transferred smile up toward a physiologically plausible amplitude).
        _p95_ref = _claire_p95
        if _claire_p95 < 1e-6 or _xfer_p95 < 1e-6:
            if _is_smile and _xfer_p95 > 1e-6:
                _p95_ref = SMILE_LOCAL_P95_MIN
                log.info(
                    "LocalRescale [%s]: p95_claire near-zero (claire=%.4fmm, xfer=%.4fmm) — "
                    "smile fallback: using ref=%.4fmm.",
                    _rname, _claire_p95 * 1000, _xfer_p95 * 1000, _p95_ref * 1000,
                )
            else:
                log.info(
                    "LocalRescale [%s]: p95 near-zero (claire=%.4fmm, xfer=%.4fmm) — skipping.",
                    _rname, _claire_p95 * 1000, _xfer_p95 * 1000,
                )
                continue

        _ratio = _p95_ref / _xfer_p95
        if 0.5 <= _ratio <= 2.0:
            log.info(
                "LocalRescale [%s]: ratio=%.3f within [0.5, 2.0] — no rescale needed "
                "(claire_p95=%.3fmm, xfer_p95=%.3fmm, local_t=%d, local_c=%d).",
                _rname, _ratio, _claire_p95 * 1000, _xfer_p95 * 1000, _n_local_t, _n_local_c,
            )
            continue

        _ratio_clamped = float(np.clip(_ratio, 0.0, 10.0))
        log.info(
            "LocalRescale [%s]: p95_claire=%.4fmm, p95_target=%.4fmm, "
            "using %s with ref=%.4fmm and ratio=%.3f (clamped=%.3f)  "
            "local_t=%d  local_c=%d  APPLYING.",
            _rname, _claire_p95 * 1000, _xfer_p95 * 1000,
            "smile fallback" if (_is_smile and _p95_ref > _claire_p95 + 1e-7) else "local rescale",
            _p95_ref * 1000, _ratio, _ratio_clamped, _n_local_t, _n_local_c,
        )
        target_morph_targets[_rname] = _xfer_disp * _ratio_clamped

    # ── Smile mean-boost safeguard ───────────────────────────────────────────
    # If, after local rescaling, a smile shape's local mean displacement is
    # still below SMILE_LOCAL_MEAN_TARGET, scale its local vertices up toward
    # that target — but only when the global jaw ratio is within a sane range
    # so we don't try to compensate for a fundamentally broken transfer.
    _jaw_ratio_ok = jaw_ratio is not None and 0.3 <= jaw_ratio <= 2.0
    for _sname, _slm_idx in [("mouthSmileLeft", 61), ("mouthSmileRight", 291)]:
        if _sname not in target_morph_targets:
            continue
        if not _jaw_ratio_ok:
            log.info(
                "LocalRescale [%s]: skipping mean boost — jaw_ratio=%s outside [0.3, 2.0].",
                _sname, f"{jaw_ratio:.3f}" if jaw_ratio is not None else "N/A",
            )
            continue

        # Recompute anchor / local mask (same logic as local rescale loop above).
        _anchor_s = np.array(_LANDMARK_TO_CLAIRE_POS[_slm_idx])
        if _landmark_anchors is not None:
            _k_s = _lm_key_list.index(_slm_idx)
            _anchor_s_t = target_verts[_landmark_anchors[0][_k_s]]
        else:
            _anchor_s_t = _anchor_s.copy()

        _dists_s = np.linalg.norm(target_verts - _anchor_s_t, axis=1)
        _local_s = _dists_s <= _RESCALE_RADIUS_M

        if int(_local_s.sum()) < 3:
            continue

        _sdisp = target_morph_targets[_sname]
        _local_mean = float(np.linalg.norm(_sdisp[_local_s], axis=1).mean())

        if _local_mean < SMILE_LOCAL_MEAN_TARGET:
            _boost = float(np.clip(
                SMILE_LOCAL_MEAN_TARGET / max(_local_mean, 1e-6),
                SMILE_MEAN_BOOST_MIN, SMILE_MEAN_BOOST_MAX,
            ))
            log.info(
                "LocalRescale [%s]: boosting local mean from %.4fmm to %.4fmm (scale=%.3f).",
                _sname, _local_mean * 1000, _local_mean * _boost * 1000, _boost,
            )
            _boosted = _sdisp.copy()
            _boosted[_local_s] *= _boost
            target_morph_targets[_sname] = _boosted
        else:
            log.info(
                "LocalRescale [%s]: local mean=%.4fmm ≥ %.1fmm target — no boost needed.",
                _sname, _local_mean * 1000, SMILE_LOCAL_MEAN_TARGET * 1000,
            )

    # ── Shape-specific displacement validation (post-rescaling) ─────────────────
    for _vname, _thresh_m, _label in [
        ("mouthSmileLeft",  0.003, "smile"),
        ("mouthSmileRight", 0.003, "smile"),
        ("eyeBlinkLeft",    0.001, "eyeBlink"),
        ("eyeBlinkRight",   0.001, "eyeBlink"),
    ]:
        _vdisp = target_morph_targets.get(_vname)
        if _vdisp is None:
            continue
        _vmean = float(np.linalg.norm(_vdisp, axis=1).mean())
        if _vmean < _thresh_m:
            log.warning(
                "VALIDATION [%s] mean=%.4fmm < %.1fmm threshold — "
                "transfer may be insufficient for this %s shape.",
                _vname, _vmean * 1000, _thresh_m * 1000, _label,
            )
        else:
            log.info(
                "Validation [%s]: mean=%.4fmm ✓ (threshold %.1fmm).",
                _vname, _vmean * 1000, _thresh_m * 1000,
            )

    # ── Upper-face anchoring: remove global translation from lower-face shapes ──
    # These blendshapes should deform only the jaw / mouth / lower cheek region.
    # Any displacement appearing on the upper face (eyes, brows, forehead) is
    # an artefact — either a rigid-body offset baked into the source data, or
    # spatial-interpolation spread.  We cancel it by subtracting the mean
    # displacement of the top-40 % of the aligned head (Y > 60th percentile)
    # from the whole array, which removes the rigid component while preserving
    # the relative jaw-to-skull motion.
    _LOWER_FACE_BLENDSHAPES: frozenset[str] = frozenset({
        "jawOpen", "jawForward", "jawLeft", "jawRight",
        "mouthClose",
        "mouthDimpleLeft", "mouthDimpleRight",
        "mouthFrownLeft", "mouthFrownRight",
        "mouthFunnel",
        "mouthLeft", "mouthRight",
        "mouthLowerDownLeft", "mouthLowerDownRight",
        "mouthPressLeft", "mouthPressRight",
        "mouthPucker",
        "mouthRollLower", "mouthRollUpper",
        "mouthShrugLower", "mouthShrugUpper",
        "mouthSmileLeft", "mouthSmileRight",
        "mouthStretchLeft", "mouthStretchRight",
        "mouthUpperUpLeft", "mouthUpperUpRight",
        "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
        "noseSneerLeft", "noseSneerRight",
        "tongueOut",
    })

    y_arr = target_verts[:, 1]
    upper_thresh = float(y_arr.min()) + 0.60 * float(y_arr.max() - y_arr.min())
    upper_mask = y_arr > upper_thresh
    n_upper = int(upper_mask.sum())

    if n_upper > 5:
        anchored_count = 0
        for name in _LOWER_FACE_BLENDSHAPES:
            disp = target_morph_targets.get(name)
            if disp is None:
                continue
            mean_upper = disp[upper_mask].mean(axis=0)   # (3,)
            correction_mag = float(np.linalg.norm(mean_upper))
            if correction_mag > 5e-5:   # skip if already < 0.05 mm
                target_morph_targets[name] = disp - mean_upper
                anchored_count += 1
                log.debug(
                    "upper-face anchor %s: removed offset [%.5f %.5f %.5f]m (%.3fmm)",
                    name, *mean_upper, correction_mag * 1000,
                )
        if anchored_count:
            log.info(
                "Upper-face anchoring applied to %d/%d lower-face blendshapes "
                "(upper_thresh Y=%.4f, n_upper=%d).",
                anchored_count, len(_LOWER_FACE_BLENDSHAPES), upper_thresh, n_upper,
            )

    method_name = (
        "landmark-anchored RBF" if _landmark_anchors is not None
        else ("barycentric" if use_barycentric else "IDW k=4")
    )
    log.info("All 52 morph targets transferred via %s.", method_name)

    rigged = trimesh.Trimesh(
        vertices=target_verts.copy(),
        faces=np.array(target_mesh.faces),
        process=False,
    )
    return rigged, target_morph_targets


# ===========================================================================
# Partition-of-Unity RBF (POU-RBF) morph target transfer
# ===========================================================================

# Canonical anatomical anchor positions for Claire (metres, centred at origin).
# Used to build Claire's per-vertex region mask without running MediaPipe on her.
_CLAIRE_REGION_ANCHORS: list[tuple[float, float, float]] = [
    (-0.030,  0.020,  0.040),  # 0 left_eye
    ( 0.030,  0.020,  0.040),  # 1 right_eye
    ( 0.000, -0.005,  0.055),  # 2 nose
    ( 0.000, -0.030,  0.040),  # 3 mouth
    (-0.030,  0.040,  0.030),  # 4 left_brow
    ( 0.030,  0.040,  0.030),  # 5 right_brow
    ( 0.000,  0.070,  0.020),  # 6 forehead
    (-0.045, -0.008,  0.025),  # 7 left_cheek
    ( 0.045, -0.008,  0.025),  # 8 right_cheek
    ( 0.000, -0.060,  0.015),  # 9 jaw
]

# Region label → blendshape names that primarily affect that region.
REGION_BLENDSHAPES: dict[int, list[str]] = {
    0: [  # left_eye
        "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft",
        "eyeLookOutLeft", "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft",
    ],
    1: [  # right_eye
        "eyeBlinkRight", "eyeLookDownRight", "eyeLookInRight",
        "eyeLookOutRight", "eyeLookUpRight", "eyeSquintRight", "eyeWideRight",
    ],
    2: [  # nose
        "noseSneerLeft", "noseSneerRight",
    ],
    3: [  # mouth (central)
        "mouthClose", "mouthFunnel", "mouthPucker",
        "mouthLeft", "mouthRight",
        "mouthRollLower", "mouthRollUpper",
        "mouthShrugLower", "mouthShrugUpper",
        "tongueOut",
    ],
    4: [  # left_brow
        "browDownLeft", "browOuterUpLeft",
    ],
    5: [  # right_brow
        "browDownRight", "browOuterUpRight",
    ],
    6: [  # forehead
        "browInnerUp",
    ],
    7: [  # left_cheek
        "cheekSquintLeft", "cheekPuff",
        "mouthSmileLeft", "mouthDimpleLeft",
        "mouthFrownLeft", "mouthStretchLeft",
        "mouthPressLeft", "mouthLowerDownLeft", "mouthUpperUpLeft",
    ],
    8: [  # right_cheek
        "cheekSquintRight",
        "mouthSmileRight", "mouthDimpleRight",
        "mouthFrownRight", "mouthStretchRight",
        "mouthPressRight", "mouthLowerDownRight", "mouthUpperUpRight",
    ],
    9: [  # jaw
        "jawOpen", "jawForward", "jawLeft", "jawRight",
    ],
}

# Inverse map: blendshape name → list of region indices it primarily affects.
_BLENDSHAPE_REGIONS: dict[str, list[int]] = {}
for _r_idx, _bs_list in REGION_BLENDSHAPES.items():
    for _bs_name in _bs_list:
        _BLENDSHAPE_REGIONS.setdefault(_bs_name, []).append(_r_idx)
# Any blendshape not in REGION_BLENDSHAPES gets global treatment (all regions).
for _bs_name in ARKIT_BLENDSHAPES:
    if _bs_name not in _BLENDSHAPE_REGIONS:
        _BLENDSHAPE_REGIONS[_bs_name] = list(range(10))

# Shepard boundary-blend radius (metres).
_POU_REGION_RADIUS_M: float = 0.065  # 65 mm

# Per-region expansion biases for Claire's mask (mirrors landmarks._REGION_EXPANSION_M).
_CLAIRE_REGION_EXPANSION_M: np.ndarray = np.array(
    [0.065, 0.065, 0.040, 0.055, 0.040, 0.040, 0.060, 0.055, 0.055, 0.070],
    dtype=np.float64,
)

# Cached Claire per-vertex region mask — computed lazily on first POU-RBF call.
_claire_region_mask_cache: np.ndarray | None = None


def _get_claire_region_mask() -> np.ndarray:
    """Return (lazily compute) Claire's per-vertex region mask (0-9)."""
    global _claire_region_mask_cache
    if _claire_region_mask_cache is not None:
        return _claire_region_mask_cache
    if claire_neutral_m is None:
        raise RuntimeError("Claire data not loaded.")
    anchors = np.array(_CLAIRE_REGION_ANCHORS, dtype=np.float64)   # (10, 3)
    dists = np.linalg.norm(
        claire_neutral_m[:, np.newaxis, :] - anchors[np.newaxis, :, :],
        axis=2,
    )  # (V_claire, 10)
    adjusted = dists - _CLAIRE_REGION_EXPANSION_M[np.newaxis, :]
    _claire_region_mask_cache = np.argmin(adjusted, axis=1).astype(np.int32)
    log.info(
        "Claire region mask: %s",
        {i: int((_claire_region_mask_cache == i).sum()) for i in range(10)},
    )
    return _claire_region_mask_cache


_RBF_MAX_EVAL_VERTS: int = 3000
"""Max target vertices to evaluate the RBF on directly.  Meshes above this
threshold are decimated for the RBF solve, then results are up-sampled to
the full vertex set via nearest-neighbour lookup."""

_RBF_BATCH_TIMEOUT_S: float = 10.0
"""Per-batch wall-clock timeout.  If a batch of blendshapes exceeds this,
timed-out shapes fall back to IDW and a warning is logged."""


def _pou_rbf_single_worker(
    args: tuple,
) -> tuple[str, np.ndarray]:
    """ProcessPoolExecutor worker: transfer one blendshape via POU-RBF.

    Accepts a tuple so it works with executor.map.  All numpy arrays are passed
    explicitly; module-level globals are not used (workers are separate processes).

    When the target mesh exceeds *_RBF_MAX_EVAL_VERTS*, the RBF is evaluated on
    a decimated proxy and the resulting displacements are up-sampled to the full
    vertex set via nearest-neighbour lookup.
    """
    (name, claire_disp, regions, target_verts, face_region_mask,
     claire_neutral, claire_mask, shepard_w,
     decimated_verts, deci_to_full_idx) = args

    import logging as _logging
    from scipy.interpolate import RBFInterpolator
    from scipy.spatial import KDTree

    _log = _logging.getLogger(__name__)
    t0 = time.time()

    N = len(target_verts)
    full_tree = KDTree(claire_neutral)
    use_decimated = decimated_verts is not None

    def _idw(pts: np.ndarray) -> np.ndarray:
        k = min(4, len(claire_neutral))
        dists, idxs = full_tree.query(pts, k=k)
        w = 1.0 / (dists + 1e-12)
        w /= w.sum(axis=1, keepdims=True)
        return (claire_disp[idxs] * w[:, :, np.newaxis]).sum(axis=1)

    def _idw_full() -> tuple[str, np.ndarray]:
        """Full-mesh IDW fallback."""
        return name, _idw(target_verts)

    # Global IDW for blendshapes spanning all regions
    if len(regions) >= 10:
        return _idw_full()

    target_disp = np.zeros((N, 3), dtype=np.float64)

    for r_idx in regions:
        # --- timeout check at region granularity ---
        if (time.time() - t0) > _RBF_BATCH_TIMEOUT_S:
            _log.warning(
                "POU-RBF timeout (%.0fs) on '%s' — falling back to IDW.",
                _RBF_BATCH_TIMEOUT_S, name,
            )
            return _idw_full()

        tgt_r_idx = np.where(face_region_mask == r_idx)[0]
        if len(tgt_r_idx) == 0:
            continue

        claire_r_idx = np.where(claire_mask == r_idx)[0]

        if len(claire_r_idx) < 4:
            r_disp = _idw(target_verts[tgt_r_idx])
            target_disp[tgt_r_idx] += shepard_w[tgt_r_idx, r_idx, np.newaxis] * r_disp
            continue

        cp_pos = claire_neutral[claire_r_idx]
        cp_disp = claire_disp[claire_r_idx]

        # ── Decide whether to evaluate on decimated proxy ──────────────
        if use_decimated:
            deci_r_idx = np.where(face_region_mask[deci_to_full_idx] == r_idx)[0]
            if len(deci_r_idx) == 0:
                # Decimation dropped all verts in this region — IDW fallback
                r_disp = _idw(target_verts[tgt_r_idx])
                target_disp[tgt_r_idx] += shepard_w[tgt_r_idx, r_idx, np.newaxis] * r_disp
                continue
            eval_pos = decimated_verts[deci_r_idx]
        else:
            eval_pos = target_verts[tgt_r_idx]

        try:
            rbf = RBFInterpolator(
                cp_pos, cp_disp,
                kernel="thin_plate_spline",
                degree=1,
                smoothing=1e-3,
            )
            r_disp_eval = rbf(eval_pos)
        except Exception:
            r_disp = _idw(target_verts[tgt_r_idx])
            target_disp[tgt_r_idx] += shepard_w[tgt_r_idx, r_idx, np.newaxis] * r_disp
            continue

        # ── Up-sample from decimated proxy to full mesh via NN ─────────
        if use_decimated:
            deci_tree = KDTree(eval_pos)
            _, nn_idx = deci_tree.query(target_verts[tgt_r_idx], k=1)
            r_disp = r_disp_eval[nn_idx]
        else:
            r_disp = r_disp_eval

        target_disp[tgt_r_idx] += shepard_w[tgt_r_idx, r_idx, np.newaxis] * r_disp

    return name, target_disp


def transfer_morph_targets_pou_rbf(
    aligned_head: trimesh.Trimesh,
    alignment_meta: dict | None,
    face_region_mask: np.ndarray,
) -> tuple[trimesh.Trimesh, dict[str, np.ndarray]]:
    """Partition-of-Unity RBF morph target transfer — no cross-region contamination.

    For each blendshape, a local thin-plate-spline RBF is fitted using only
    Claire's vertices in the blendshape's anatomical region.  The RBF is
    evaluated only on the target vertices in that same region.  Vertices
    outside the region receive zero displacement (hard localization), with
    smooth Shepard-weighted blending at region boundaries.

    Parameters
    ----------
    aligned_head : trimesh.Trimesh
        ICP-aligned Meshy head (metres, centred at origin).
    alignment_meta : dict | None
        From align_icp — not used by POU-RBF but kept for signature parity.
    face_region_mask : np.ndarray shape (N,) int
        Per-vertex region label 0-9 from rigger.landmarks.detect_landmarks.

    Returns
    -------
    (rigged_mesh, blendshapes) — identical format to transfer_morph_targets.
    """
    if claire_neutral_m is None or _claire_deltas_m is None:
        raise RuntimeError(f"Claire blendshape data not loaded from '{BS_SKIN_PATH}'.")

    try:
        from scipy.interpolate import RBFInterpolator
    except ImportError as exc:
        raise ImportError("scipy.interpolate.RBFInterpolator required for POU-RBF.") from exc

    target_verts = np.asarray(aligned_head.vertices, dtype=np.float64)
    N = len(target_verts)
    V_claire = len(claire_neutral_m)
    claire_mask = _get_claire_region_mask()

    # Full IDW tree for fallbacks
    full_tree = KDTree(claire_neutral_m)

    # Precompute Shepard blending weights: (N, 10)
    shepard_w = _compute_pou_shepard_weights(target_verts, face_region_mask)

    target_morph_targets: dict[str, np.ndarray] = {}

    # ── Subsample target vertices for RBF evaluation if above threshold ──
    decimated_verts: np.ndarray | None = None
    deci_to_full_idx: np.ndarray | None = None
    if N > _RBF_MAX_EVAL_VERTS:
        # Uniform stride through vertex indices to get an evenly-spaced subset.
        stride = max(1, N // _RBF_MAX_EVAL_VERTS)
        deci_to_full_idx = np.arange(0, N, stride, dtype=np.intp)
        decimated_verts = target_verts[deci_to_full_idx]
        log.info(
            "POU-RBF: subsampled %d → %d vertices for RBF evaluation (stride=%d).",
            N, len(decimated_verts), stride,
        )

    # Build per-blendshape args for parallel workers.
    claire_neutral_snap = claire_neutral_m  # local ref for closure safety
    worker_args = [
        (
            name,
            _claire_deltas_m.get(name, np.zeros((V_claire, 3), dtype=np.float64)),
            _BLENDSHAPE_REGIONS.get(name, list(range(10))),
            target_verts,
            face_region_mask,
            claire_neutral_snap,
            claire_mask,
            shepard_w,
            decimated_verts,
            deci_to_full_idx,
        )
        for name in ARKIT_BLENDSHAPES
    ]

    _BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4))
    n_batches = (len(worker_args) + _BATCH_SIZE - 1) // _BATCH_SIZE
    log.info(
        "POU-RBF: dispatching 52 blendshapes in %d batches of %d (BATCH_SIZE=%d).",
        n_batches, _BATCH_SIZE, _BATCH_SIZE,
    )

    def _idw_fallback(bs_name: str) -> tuple[str, np.ndarray]:
        """Compute IDW k=4 transfer for a single blendshape (timeout fallback)."""
        c_disp = _claire_deltas_m.get(
            bs_name, np.zeros((V_claire, 3), dtype=np.float64),
        )
        k = min(4, len(claire_neutral_m))
        dists, idxs = full_tree.query(target_verts, k=k)
        w = 1.0 / (dists + 1e-12)
        w /= w.sum(axis=1, keepdims=True)
        disp = (c_disp[idxs] * w[:, :, np.newaxis]).sum(axis=1)
        return bs_name, disp

    from concurrent.futures import TimeoutError as FuturesTimeoutError

    results: list[tuple[str, np.ndarray]] = []
    with ProcessPoolExecutor(max_workers=_BATCH_SIZE) as executor:
        for batch_idx, batch_start in enumerate(
            range(0, len(worker_args), _BATCH_SIZE)
        ):
            batch = worker_args[batch_start : batch_start + _BATCH_SIZE]
            batch_names = [a[0] for a in batch]
            batch_num = batch_idx + 1

            t_batch_start = time.time()
            log.info(
                "POU-RBF batch %d/%d start: %s",
                batch_num, n_batches, batch_names,
            )

            futures = {
                executor.submit(_pou_rbf_single_worker, args): args[0]
                for args in batch
            }

            batch_results: list[tuple[str, np.ndarray]] = []
            timed_out_names: list[str] = []

            for fut in futures:
                bs_name = futures[fut]
                remaining = _RBF_BATCH_TIMEOUT_S - (time.time() - t_batch_start)
                try:
                    result = fut.result(timeout=max(0.0, remaining))
                    batch_results.append(result)
                except FuturesTimeoutError:
                    timed_out_names.append(bs_name)
                    fut.cancel()

            t_batch_elapsed = time.time() - t_batch_start

            # IDW fallback for any blendshapes that timed out.
            if timed_out_names:
                log.warning(
                    "POU-RBF batch %d/%d: %d blendshape(s) timed out after %.1fs, "
                    "falling back to IDW: %s",
                    batch_num, n_batches, len(timed_out_names),
                    t_batch_elapsed, timed_out_names,
                )
                for bs_name in timed_out_names:
                    batch_results.append(_idw_fallback(bs_name))

            log.info(
                "POU-RBF batch %d/%d done: %.2fs (%d ok, %d idw-fallback)",
                batch_num, n_batches, t_batch_elapsed,
                len(batch_results) - len(timed_out_names), len(timed_out_names),
            )

            results.extend(batch_results)
            if batch_start + _BATCH_SIZE < len(worker_args):
                time.sleep(0.1)

    # Reassemble in canonical order and log per-shape stats.
    _CHECK_SHAPES = {"eyeBlinkLeft", "eyeBlinkRight", "mouthSmileLeft", "mouthSmileRight"}
    for name, target_disp in results:
        target_morph_targets[name] = target_disp
        mags = np.linalg.norm(target_disp, axis=1)
        nonzero = int((mags > 1e-9).sum())
        log.info(
            "POU-RBF %-30s  mean=%.5fm  max=%.5fm  nonzero=%d/%d",
            name, mags.mean(), mags.max(), nonzero, N,
        )
        if name in _CHECK_SHAPES:
            print(
                f"[REGION-CHECK] {name}: nonzero={nonzero}/{N}",
                flush=True,
            )

    # Upper-face anchoring: remove rigid offset from lower-face blendshapes
    # (same correction as the classic pipeline)
    _LOWER_FACE: frozenset[str] = frozenset({
        "jawOpen", "jawForward", "jawLeft", "jawRight",
        "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
        "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
        "mouthLeft", "mouthRight", "mouthLowerDownLeft", "mouthLowerDownRight",
        "mouthPressLeft", "mouthPressRight", "mouthPucker",
        "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
        "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
        "mouthUpperUpLeft", "mouthUpperUpRight",
        "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
        "noseSneerLeft", "noseSneerRight", "tongueOut",
    })
    y_arr = target_verts[:, 1]
    upper_thresh = float(y_arr.min()) + 0.60 * float(y_arr.max() - y_arr.min())
    upper_mask = y_arr > upper_thresh
    n_upper = int(upper_mask.sum())
    if n_upper > 5:
        anchored = 0
        for _name in _LOWER_FACE:
            disp = target_morph_targets.get(_name)
            if disp is None:
                continue
            mean_upper = disp[upper_mask].mean(axis=0)
            if float(np.linalg.norm(mean_upper)) > 5e-5:
                target_morph_targets[_name] = disp - mean_upper
                anchored += 1
        if anchored:
            log.info("POU-RBF: upper-face anchoring applied to %d shapes.", anchored)

    log.info("All 52 morph targets transferred via POU-RBF.")
    rigged = trimesh.Trimesh(
        vertices=target_verts.copy(),
        faces=np.array(aligned_head.faces),
        process=False,
    )
    return rigged, target_morph_targets


def _compute_pou_shepard_weights(
    target_verts: np.ndarray,
    face_region_mask: np.ndarray,
) -> np.ndarray:
    """Compute (N, 10) Shepard blend weights for smooth region boundaries.

    For each vertex, the weight for its own region is 1.0.
    For adjacent regions, weight = max(0, R - dist(v, region_centroid)).
    Normalised per-vertex so weights sum to 1.

    When a vertex is deep inside a region, w[own]=1, all others=0 → hard assign.
    Near a boundary, neighbouring region weights become non-zero → smooth blend.
    """
    N = len(target_verts)
    n_regions = 10
    R = _POU_REGION_RADIUS_M

    # Compute region centroids from target vertices
    centroids = np.zeros((n_regions, 3), dtype=np.float64)
    for r in range(n_regions):
        idx = np.where(face_region_mask == r)[0]
        if len(idx) > 0:
            centroids[r] = target_verts[idx].mean(axis=0)
        # else centroid stays at origin — will get near-zero weight from distance

    # Shepard: w_i(x) = max(0, R - dist(x, c_i)) / (R * max(dist(x, c_i), 1e-6))
    raw_w = np.zeros((N, n_regions), dtype=np.float64)
    for r in range(n_regions):
        d = np.linalg.norm(target_verts - centroids[r], axis=1)  # (N,)
        raw_w[:, r] = np.maximum(0.0, R - d) / (R * np.maximum(d, 1e-6))

    # Vertices in region r: force their own weight to at least 1.0 before normalising
    for r in range(n_regions):
        own = face_region_mask == r
        raw_w[own, r] = np.maximum(raw_w[own, r], 1.0)

    # Normalise rows to sum = 1
    row_sum = raw_w.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > 1e-12, row_sum, 1.0)
    return raw_w / row_sum
