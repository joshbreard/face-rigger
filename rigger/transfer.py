"""ARKit morph-target transfer using Claire's blendshape skin (bs_skin.npz).

For each of the 52 ARKit blendshapes, the per-vertex displacement from
Claire's neutral pose is transferred to the aligned Meshy head via nearest-
neighbour lookup: for each target vertex, find the nearest vertex in Claire's
(centred) neutral mesh and copy its displacement.

Claire's mesh is in centimetres; Meshy GLBs are in metres. A 0.01 scale
factor is applied to Claire's positions (and displacement vectors) so both
meshes share the same coordinate space during the spatial lookup.
"""

import logging
from pathlib import Path

import numpy as np
import trimesh
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

# {arkit_name: (V, 3)} blendshape displacement vectors in metres.
_claire_deltas_m: dict[str, np.ndarray] | None = None


def _load_bs_skin() -> None:
    """Load Claire's blendshape skin from *BS_SKIN_PATH*. Called at import time."""
    global claire_neutral_m, _claire_deltas_m

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

    # Centre neutral so it matches the coordinate frame that aligner.py
    # produces for the aligned Meshy head.
    claire_neutral_m = neutral_m - centroid

    # Each ARKit blendshape is stored as absolute vertex positions (cm).
    # Displacement = posed_positions_m − neutral_m.
    deltas: dict[str, np.ndarray] = {}
    missing = []
    for name in ARKIT_BLENDSHAPES:
        if name not in data.files:
            missing.append(name)
            continue
        posed_m = data[name].astype(np.float64) * 0.01  # cm → m
        deltas[name] = posed_m - neutral_m               # displacement in m

    if missing:
        log.warning(
            "%d blendshape key(s) not found in '%s' — will be zero: %s",
            len(missing),
            BS_SKIN_PATH,
            missing,
        )

    _claire_deltas_m = deltas

    log.info(
        "Loaded Claire blendshape skin: %d vertices, %d / 52 blendshapes (scaled cm→m, centred).",
        len(neutral_m),
        len(deltas),
    )


# Run at import so keys are printed during server startup.
_load_bs_skin()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transfer_morph_targets(
    target_mesh: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, dict[str, np.ndarray]]:
    """Transfer ARKit morph targets from Claire's blendshape skin to *target_mesh*.

    Parameters
    ----------
    target_mesh:
        The aligned Meshy head mesh (vertices must be in the same coordinate
        frame as *claire_neutral_m*, i.e. metres, centred at the origin —
        produced by ``rigger.aligner.align_icp``).

    Returns
    -------
    (rigged_mesh, blendshapes)
        rigged_mesh  : trimesh.Trimesh — geometry-only copy of *target_mesh*.
        blendshapes  : dict[str, np.ndarray] — 52 displacement arrays,
            shape ``(N_target, 3)``, keyed by ARKit blendshape name.
    """
    if claire_neutral_m is None or _claire_deltas_m is None:
        raise RuntimeError(
            f"Claire blendshape data not loaded. "
            f"Ensure '{BS_SKIN_PATH}' exists and is valid."
        )

    target_verts = np.array(target_mesh.vertices, dtype=np.float64)

    log.info(
        "Building KD-tree on Claire neutral (%d vertices, metres, centred)...",
        len(claire_neutral_m),
    )
    tree = KDTree(claire_neutral_m)

    log.info(
        "Querying nearest Claire vertex for each of %d target vertices...",
        len(target_verts),
    )
    _, nn_indices = tree.query(target_verts)  # shape (N_target,)

    target_morph_targets: dict[str, np.ndarray] = {}

    for name in ARKIT_BLENDSHAPES:
        claire_disp = _claire_deltas_m.get(name, np.zeros_like(claire_neutral_m))  # (V_claire, 3)
        target_disp = claire_disp[nn_indices]    # (N_target, 3)  — direct copy
        target_morph_targets[name] = target_disp
        log.debug(
            "Transferred '%s': %d / %d target verts have non-zero displacement.",
            name,
            (np.linalg.norm(target_disp, axis=1) > 1e-9).sum(),
            len(target_verts),
        )

    log.info("All 52 morph targets transferred.")

    rigged = trimesh.Trimesh(
        vertices=target_verts.copy(),
        faces=np.array(target_mesh.faces),
        process=False,
    )
    return rigged, target_morph_targets
