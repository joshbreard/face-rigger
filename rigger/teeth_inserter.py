"""Load a static teeth GLB asset and align it to a character's mouth."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pygltflib

from rigger.transfer import ARKIT_BLENDSHAPES

log = logging.getLogger(__name__)

TEETH_ASSET_PATH = Path(__file__).resolve().parent.parent / "assets" / "free_teeth_base_mesh.glb"

# Jaw-related blendshapes that get non-zero displacement on lower teeth.
_JAW_LATERAL_SCALE = 0.3


def _read_primitive_vertices(gltf: pygltflib.GLTF2, binary: bytes, prim: pygltflib.Primitive) -> tuple[np.ndarray, np.ndarray]:
    """Return (verts (N,3) float64, faces (F,3) uint32) for a single primitive."""
    pos_acc = gltf.accessors[prim.attributes.POSITION]
    pos_bv = gltf.bufferViews[pos_acc.bufferView]
    pos_offset = (pos_bv.byteOffset or 0) + (pos_acc.byteOffset or 0)
    pos_bytes = binary[pos_offset: pos_offset + pos_acc.count * 3 * 4]
    verts = np.frombuffer(pos_bytes, dtype=np.float32).reshape(-1, 3).astype(np.float64)

    idx_acc = gltf.accessors[prim.indices]
    idx_bv = gltf.bufferViews[idx_acc.bufferView]
    idx_offset = (idx_bv.byteOffset or 0) + (idx_acc.byteOffset or 0)
    comp_size = {5125: 4, 5123: 2, 5121: 1}.get(idx_acc.componentType, 4)
    dtype_map = {5125: np.uint32, 5123: np.uint16, 5121: np.uint8}
    idx_dtype = dtype_map.get(idx_acc.componentType, np.uint32)
    idx_bytes = binary[idx_offset: idx_offset + idx_acc.count * comp_size]
    indices = np.frombuffer(idx_bytes, dtype=idx_dtype).astype(np.uint32)
    faces = indices.reshape(-1, 3)
    return verts, faces


def align_and_embed_teeth(
    mouth_left_3d: "np.ndarray | list | None",
    mouth_right_3d: "np.ndarray | list | None",
    jaw_open_amplitude: float,
    alignment_meta: dict | None,
) -> dict | None:
    """Load, align, and prepare teeth geometry for embedding.

    Returns a dict with upper/lower teeth verts, faces, and blendshape data,
    or None if the asset is missing or inputs are insufficient.
    """
    if not TEETH_ASSET_PATH.exists():
        log.warning("Teeth asset not found at %s — skipping teeth insertion.", TEETH_ASSET_PATH)
        return None

    if mouth_left_3d is None or mouth_right_3d is None:
        log.warning("mouth_left_3d or mouth_right_3d is None — skipping teeth insertion.")
        return None

    mouth_left = np.asarray(mouth_left_3d, dtype=np.float64)
    mouth_right = np.asarray(mouth_right_3d, dtype=np.float64)

    # ── Load GLB primitives ──────────────────────────────────────────────────
    gltf = pygltflib.GLTF2.load(str(TEETH_ASSET_PATH))
    binary = gltf.binary_blob() or b""

    # Collect all mesh primitives across all meshes.
    all_prims: list[tuple[np.ndarray, np.ndarray]] = []
    for mesh in gltf.meshes:
        for prim in mesh.primitives or []:
            if prim.attributes.POSITION is not None and prim.indices is not None:
                verts, faces = _read_primitive_vertices(gltf, binary, prim)
                all_prims.append((verts, faces))

    if len(all_prims) < 2:
        log.warning("Teeth GLB has fewer than 2 primitives (%d) — cannot identify upper/lower teeth.", len(all_prims))
        return None

    # Sort by vertex count descending — two largest are upper and lower teeth.
    all_prims.sort(key=lambda vf: len(vf[0]), reverse=True)

    cand_a_verts, cand_a_faces = all_prims[0]
    cand_b_verts, cand_b_faces = all_prims[1]

    mean_y_a = cand_a_verts[:, 1].mean()
    mean_y_b = cand_b_verts[:, 1].mean()

    if mean_y_a >= mean_y_b:
        upper_verts, upper_faces = cand_a_verts, cand_a_faces
        lower_verts, lower_faces = cand_b_verts, cand_b_faces
    else:
        upper_verts, upper_faces = cand_b_verts, cand_b_faces
        lower_verts, lower_faces = cand_a_verts, cand_a_faces

    log.info(
        "Teeth primitives: upper=%d verts (mean Y=%.4f), lower=%d verts (mean Y=%.4f)",
        len(upper_verts), upper_verts[:, 1].mean(),
        len(lower_verts), lower_verts[:, 1].mean(),
    )

    # Cavity: third primitive if present.
    cavity_verts: np.ndarray | None = None
    cavity_faces: np.ndarray | None = None
    if len(all_prims) >= 3:
        cavity_verts, cavity_faces = all_prims[2]
        log.info("Cavity primitive: %d verts", len(cavity_verts))
    else:
        log.info("No cavity primitive found in teeth GLB.")

    # ── Compute alignment parameters ────────────────────────────────────────
    mouth_center = (mouth_left + mouth_right) / 2.0
    mouth_width = float(np.linalg.norm(mouth_left - mouth_right))
    lip_y = float(mouth_left[1])

    # Combine all teeth verts to get current X extent.
    combined = np.vstack([upper_verts, lower_verts])
    if cavity_verts is not None:
        combined = np.vstack([combined, cavity_verts])

    teeth_x_min, teeth_x_max = combined[:, 0].min(), combined[:, 0].max()
    teeth_x_span = teeth_x_max - teeth_x_min
    if teeth_x_span < 1e-8:
        log.warning("Teeth X span is near zero — cannot scale.")
        return None

    target_width = mouth_width * 1.05
    scale = target_width / teeth_x_span

    # Center of the teeth mesh in its own space.
    teeth_center = combined.mean(axis=0)

    log.info(
        "Teeth alignment: mouth_center=[%.4f,%.4f,%.4f], mouth_width=%.4f, "
        "scale=%.4f, lip_y=%.4f, jaw_amp=%.4f",
        *mouth_center, mouth_width, scale, lip_y, jaw_open_amplitude,
    )

    # ── Apply transform to all teeth groups ─────────────────────────────────
    def transform(verts: np.ndarray) -> np.ndarray:
        v = (verts - teeth_center) * scale
        v[:, 0] += mouth_center[0]
        v[:, 1] += lip_y
        v[:, 2] += mouth_center[2]
        return v

    upper_verts = transform(upper_verts)
    lower_verts = transform(lower_verts)
    if cavity_verts is not None:
        cavity_verts = transform(cavity_verts)

    # ── Inverse ICP: transform teeth from Claire space to original model space ──
    if alignment_meta is not None:
        _scale_factor = float(alignment_meta["scale_factor"])
        _T = np.array(alignment_meta["icp_transformation"], dtype=np.float64)
        _T_inv_rot = np.linalg.inv(_T[:3, :3])
        _inv_scale = 1.0 / _scale_factor if _scale_factor > 1e-8 else 1.0
        src_centre = np.array(alignment_meta["source_centre"], dtype=np.float64)

        def inv_icp(verts: np.ndarray) -> np.ndarray:
            return (verts @ _T_inv_rot.T) * _inv_scale + src_centre

        upper_verts = inv_icp(upper_verts)
        lower_verts = inv_icp(lower_verts)
        if cavity_verts is not None:
            cavity_verts = inv_icp(cavity_verts)

    # ── Generate lower-teeth blendshapes ─────────────────────────────────────
    n_lower = len(lower_verts)
    lower_blendshapes: dict[str, np.ndarray] = {}

    # Compute jaw displacement direction in original model space.
    if alignment_meta is not None:
        # Transform the displacement vectors (not positions) back to original space.
        jaw_down = np.array([0.0, -jaw_open_amplitude, 0.0], dtype=np.float64)
        jaw_down_orig = (jaw_down @ _T_inv_rot.T) * _inv_scale

        jaw_left_dir = np.array([-jaw_open_amplitude * _JAW_LATERAL_SCALE, 0.0, 0.0], dtype=np.float64)
        jaw_left_orig = (jaw_left_dir @ _T_inv_rot.T) * _inv_scale

        jaw_right_dir = np.array([jaw_open_amplitude * _JAW_LATERAL_SCALE, 0.0, 0.0], dtype=np.float64)
        jaw_right_orig = (jaw_right_dir @ _T_inv_rot.T) * _inv_scale

        jaw_fwd_dir = np.array([0.0, 0.0, -jaw_open_amplitude * _JAW_LATERAL_SCALE], dtype=np.float64)
        jaw_fwd_orig = (jaw_fwd_dir @ _T_inv_rot.T) * _inv_scale
    else:
        jaw_down_orig = np.array([0.0, -jaw_open_amplitude, 0.0], dtype=np.float64)
        jaw_left_orig = np.array([-jaw_open_amplitude * _JAW_LATERAL_SCALE, 0.0, 0.0], dtype=np.float64)
        jaw_right_orig = np.array([jaw_open_amplitude * _JAW_LATERAL_SCALE, 0.0, 0.0], dtype=np.float64)
        jaw_fwd_orig = np.array([0.0, 0.0, -jaw_open_amplitude * _JAW_LATERAL_SCALE], dtype=np.float64)

    for name in ARKIT_BLENDSHAPES:
        if name == "jawOpen":
            disp = np.tile(jaw_down_orig, (n_lower, 1))
        elif name == "jawLeft":
            disp = np.tile(jaw_left_orig, (n_lower, 1))
        elif name == "jawRight":
            disp = np.tile(jaw_right_orig, (n_lower, 1))
        elif name == "jawForward":
            disp = np.tile(jaw_fwd_orig, (n_lower, 1))
        else:
            disp = np.zeros((n_lower, 3), dtype=np.float64)
        lower_blendshapes[name] = disp.astype(np.float32)

    log.info(
        "Teeth data prepared: upper=%d verts, lower=%d verts, cavity=%s verts, jaw_amp=%.4f",
        len(upper_verts), n_lower,
        len(cavity_verts) if cavity_verts is not None else "None",
        jaw_open_amplitude,
    )

    return {
        "lower_teeth_verts": lower_verts.astype(np.float32),
        "lower_teeth_faces": lower_faces,
        "lower_teeth_blendshapes": lower_blendshapes,
        "upper_teeth_verts": upper_verts.astype(np.float32),
        "upper_teeth_faces": upper_faces,
        "cavity_verts": cavity_verts.astype(np.float32) if cavity_verts is not None else None,
        "cavity_faces": cavity_faces,
    }
