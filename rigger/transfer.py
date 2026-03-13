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
from pathlib import Path

import numpy as np
import trimesh
import trimesh.convex
import trimesh.proximity
import trimesh.triangles
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

    # Print all keys so the actual structure is visible in the server log.
    print("=== bs_skin.npz structure ===", flush=True)
    for key in data.files:
        val = data[key]
        shape = val.shape if hasattr(val, "shape") else "?"
        dtype = val.dtype if hasattr(val, "dtype") else "?"
        print(f"  {key!r:30s}  shape={shape}  dtype={dtype}", flush=True)
    print("=============================", flush=True)

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


# Run at import so keys are printed during server startup.
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

    target_verts = np.array(target_mesh.vertices, dtype=np.float64)

    log.info(
        "Transfer: target mesh has %d vertices.  Centroid=[%.4f, %.4f, %.4f]",
        len(target_verts), *target_verts.mean(axis=0),
    )

    use_barycentric = _claire_hull is not None
    mapping = None
    idw_tree = None

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
            "Building KD-tree on %d Claire vertices...",
            len(claire_neutral_m),
        )
        idw_tree = KDTree(claire_neutral_m)

    # Reference Claire jawOpen magnitude for scale mismatch detection.
    jaw_claire_disp = _claire_deltas_m.get("jawOpen", np.zeros_like(claire_neutral_m))
    jaw_claire_mags = np.linalg.norm(jaw_claire_disp, axis=1)
    jaw_claire_mean = float(jaw_claire_mags[jaw_claire_mags > 1e-9].mean()) if (jaw_claire_mags > 1e-9).any() else 0.0
    log.info("Claire jawOpen reference: mean=%.6fm (%.2fmm)", jaw_claire_mean, jaw_claire_mean * 1000)

    target_morph_targets: dict[str, np.ndarray] = {}
    jaw_transferred_mean: float | None = None

    for name in ARKIT_BLENDSHAPES:
        claire_disp = _claire_deltas_m.get(name, np.zeros_like(claire_neutral_m))

        if use_barycentric:
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

        target_morph_targets[name] = target_disp

    # ── Scale mismatch detection and correction ─────────────────────────────
    if jaw_claire_mean > 1e-9 and jaw_transferred_mean is not None and jaw_transferred_mean > 1e-9:
        ratio = jaw_claire_mean / jaw_transferred_mean
        log.info(
            "Scale check: Claire jawOpen mean=%.6fm, transferred=%.6fm, ratio=%.3f",
            jaw_claire_mean, jaw_transferred_mean, ratio,
        )
        if ratio > 2.0 or ratio < 0.5:
            log.warning(
                "SCALE MISMATCH (ratio=%.3f) — rescaling all 52 targets by %.4f.",
                ratio, ratio,
            )
            target_morph_targets = {k: v * ratio for k, v in target_morph_targets.items()}

    log.info(
        "All 52 morph targets transferred via %s.",
        "barycentric" if use_barycentric else "IDW k=4",
    )

    rigged = trimesh.Trimesh(
        vertices=target_verts.copy(),
        faces=np.array(target_mesh.faces),
        process=False,
    )
    return rigged, target_morph_targets
