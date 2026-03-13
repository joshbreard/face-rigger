"""Write a rigged GLB using pygltflib with proper GLTF 2.0 morph-target structure.

GLTF morph-target layout produced
----------------------------------
mesh
  primitive
    attributes.POSITION   -> base vertex positions
    attributes.TEXCOORD_0 -> UV coordinates (copied from original GLB if present)
    attributes.NORMAL     -> normals (copied from original GLB if present)
    indices               -> face indices
    targets[0..51]        -> Attributes(POSITION=<delta accessor>) per blendshape
mesh.extras.targetNames          -> 52 ARKit names
primitive.extras.targetNames     -> 52 ARKit names (for Three.js / Babylon.js compat)
mesh.weights                     -> [0.0, ...] (52 zeros)
material                         -> copied from original GLB

Each morph-target POSITION accessor stores displacement vectors (delta from base).
"""

import logging
import struct
from pathlib import Path

import numpy as np
import pygltflib
import trimesh

from rigger.transfer import ARKIT_BLENDSHAPES

log = logging.getLogger(__name__)

# GLTF component type constants
_FLOAT = pygltflib.FLOAT
_UNSIGNED_INT = pygltflib.UNSIGNED_INT
_UNSIGNED_SHORT = 5123
_UNSIGNED_BYTE = 5121

# Bytes per component for known types
_COMPONENT_BYTES = {
    _FLOAT: 4,
    _UNSIGNED_INT: 4,
    _UNSIGNED_SHORT: 2,
    _UNSIGNED_BYTE: 1,
    5126: 4,  # FLOAT alias
}
_TYPE_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def write_rigged_glb(
    head_verts: np.ndarray,
    head_faces: np.ndarray,
    blendshapes: dict[str, np.ndarray],
    body_mesh: "trimesh.Trimesh | None",
    output_path: Path,
    original_glb_bytes: bytes | None = None,
    original_head_name: str | None = None,
    head_alignment_meta: dict | None = None,
    body_parts: "list[tuple[str, trimesh.Trimesh]] | None" = None,
    head_vert_indices: "np.ndarray | None" = None,
    body_vert_indices: "np.ndarray | None" = None,
    head_uvs: "np.ndarray | None" = None,
) -> None:
    """Assemble and save a GLB with 52 ARKit morph targets on the head mesh.

    Parameters
    ----------
    head_verts : (N, 3) float array — rigged head vertex positions
    head_faces : (F, 3) int array  — triangle face indices
    blendshapes : name -> (N, 3) displacement arrays
    body_mesh : fallback body mesh (used only when body_parts is None/empty)
    output_path : destination .glb path
    original_glb_bytes : raw bytes of the input GLB (for UV/material extraction)
    original_head_name : geometry node name used to locate the head primitive
    body_parts : list of (name, trimesh) per body part; preferred over body_mesh
    """
    builder = _GLBBuilder()

    # ── Try to extract UV/normals/materials from the original GLB ────────────
    orig_uvs: bytes | None = None
    orig_normals: bytes | None = None
    orig_uv_count: int = 0
    orig_normal_count: int = 0
    orig_material_idx: int | None = None
    orig_gltf: pygltflib.GLTF2 | None = None

    if original_glb_bytes is not None:
        try:
            orig_gltf, orig_uvs, orig_normals, orig_uv_count, orig_normal_count, orig_material_idx = \
                _extract_head_attributes(
                    original_glb_bytes, original_head_name, len(head_verts),
                    force_first_primitive=(head_vert_indices is not None),
                )
            if orig_uvs:
                log.info("Extracted UV data from original GLB (%d bytes, %d verts).", len(orig_uvs), orig_uv_count)
            if orig_normals:
                log.info("Extracted normal data from original GLB (%d bytes, %d verts).", len(orig_normals), orig_normal_count)
            if orig_material_idx is not None:
                log.info("Head primitive material index: %d", orig_material_idx)

            # Slice UVs/normals down to just the head vertices when the original
            # GLB is a single merged mesh and head_vert_indices are provided.
            if head_vert_indices is not None:
                if orig_uvs and orig_uv_count != len(head_verts):
                    uv_arr = np.frombuffer(orig_uvs, dtype=np.float32).reshape(-1, 2)
                    uv_arr = uv_arr[head_vert_indices]
                    orig_uvs = uv_arr.astype(np.float32).tobytes()
                    orig_uv_count = len(head_vert_indices)
                    log.info("Sliced head UVs to %d verts using head_vert_indices.", orig_uv_count)
                if orig_normals and orig_normal_count != len(head_verts):
                    norm_arr = np.frombuffer(orig_normals, dtype=np.float32).reshape(-1, 3)
                    norm_arr = norm_arr[head_vert_indices]
                    orig_normals = norm_arr.astype(np.float32).tobytes()
                    orig_normal_count = len(head_vert_indices)
                    log.info("Sliced head normals to %d verts using head_vert_indices.", orig_normal_count)
        except Exception as exc:
            log.warning("Could not extract attributes from original GLB (%s); output will have no UVs/materials.", exc)

    # ── Head base geometry ───────────────────────────────────────────────────
    hv = np.asarray(head_verts, dtype=np.float32)
    hi = np.asarray(head_faces, dtype=np.uint32).flatten()

    a_hv = builder.add_vec3(hv, target=pygltflib.ARRAY_BUFFER, with_bounds=True)
    a_hi = builder.add_scalar_u32(hi, target=pygltflib.ELEMENT_ARRAY_BUFFER)

    # ── Optional UV and normal accessors ─────────────────────────────────────
    a_uv: int | None = None
    a_norm: int | None = None

    if head_uvs is not None and len(head_uvs) == len(hv):
        uv_bytes = np.asarray(head_uvs, dtype=np.float32).tobytes()
        a_uv = builder.add_raw(uv_bytes, len(head_uvs), _FLOAT, "VEC2", pygltflib.ARRAY_BUFFER)
        log.info("Head UVs: using pre-extracted trimesh UV array (%d verts).", len(head_uvs))
    elif orig_uvs and orig_uv_count == len(hv):
        a_uv = builder.add_raw(orig_uvs, orig_uv_count, _FLOAT, "VEC2", pygltflib.ARRAY_BUFFER)
    elif orig_uvs:
        log.warning(
            "UV vertex count mismatch (UV=%d, head=%d); skipping UV copy.",
            orig_uv_count, len(hv),
        )

    if orig_normals and orig_normal_count == len(hv):
        a_norm = builder.add_raw(orig_normals, orig_normal_count, _FLOAT, "VEC3", pygltflib.ARRAY_BUFFER)
    elif orig_normals:
        log.warning(
            "Normal vertex count mismatch (normals=%d, head=%d); skipping normal copy.",
            orig_normal_count, len(hv),
        )

    log.info(
        "Head primitive UV accessor: %s (orig_uv_count=%d, head_verts=%d), normal accessor: %s",
        a_uv, orig_uv_count, len(hv), a_norm,
    )

    # ── 52 morph-target displacement accessors ───────────────────────────────
    morph_target_list: list[pygltflib.Attributes] = []
    for name in ARKIT_BLENDSHAPES:
        delta = np.asarray(blendshapes.get(name, np.zeros((len(hv), 3))), dtype=np.float32)
        if delta.shape != (len(hv), 3):
            log.warning("Blendshape '%s' shape %s != (%d,3); zeroing.", name, delta.shape, len(hv))
            delta = np.zeros((len(hv), 3), dtype=np.float32)
        a_mt = builder.add_vec3(delta, target=pygltflib.ARRAY_BUFFER, with_bounds=True)
        morph_target_list.append(pygltflib.Attributes(POSITION=a_mt))

    # ── Head mesh primitive ──────────────────────────────────────────────────
    prim_attrs = pygltflib.Attributes(POSITION=a_hv)
    if a_uv is not None:
        prim_attrs.TEXCOORD_0 = a_uv
    if a_norm is not None:
        prim_attrs.NORMAL = a_norm

    target_names_list = list(ARKIT_BLENDSHAPES)
    head_prim = pygltflib.Primitive(
        attributes=prim_attrs,
        indices=a_hi,
        targets=morph_target_list,
        material=orig_material_idx,
        extras={"targetNames": target_names_list},
    )
    head_mesh_gltf = pygltflib.Mesh(
        name="head",
        primitives=[head_prim],
        weights=[0.0] * len(ARKIT_BLENDSHAPES),
        extras={"targetNames": target_names_list},
    )

    meshes: list[pygltflib.Mesh] = [head_mesh_gltf]

    head_node = pygltflib.Node(name="head", mesh=0)
    if head_alignment_meta is not None:
        icp_centroid = hv.mean(axis=0)
        log.info(
            "Head centroid after ICP (Claire space): [%.4f, %.4f, %.4f]",
            *icp_centroid,
        )
        src_centre = np.array(head_alignment_meta["source_centre"], dtype=np.float64)
        log.info(
            "Head centroid before ICP (original Meshy space): [%.4f, %.4f, %.4f]",
            *src_centre,
        )
        head_node.matrix = _inverse_icp_matrix(head_alignment_meta)
    nodes: list[pygltflib.Node] = [head_node]

    # ── Body parts (pass-through with UVs/normals/materials) ─────────────────
    _active_body_parts = body_parts if body_parts else (
        [("body", body_mesh)] if body_mesh is not None else []
    )
    if _active_body_parts:
        orig_binary_for_body = orig_gltf.binary_blob() if orig_gltf is not None else None
        for part_name, part_mesh in _active_body_parts:
            bv_arr = np.asarray(part_mesh.vertices, dtype=np.float32)
            bi_arr = np.asarray(part_mesh.faces, dtype=np.uint32).flatten()
            a_bv = builder.add_vec3(bv_arr, target=pygltflib.ARRAY_BUFFER, with_bounds=True)
            a_bi = builder.add_scalar_u32(bi_arr, target=pygltflib.ELEMENT_ARRAY_BUFFER)

            a_buv: int | None = None
            a_bnorm: int | None = None
            part_material_idx: int | None = None

            if orig_gltf is not None and orig_binary_for_body is not None:
                try:
                    buv_bytes, bnorm_bytes, buv_count, bnorm_count, part_material_idx = \
                        _extract_body_part_attributes(
                            orig_gltf, orig_binary_for_body, part_name,
                            len(bv_arr), original_head_name,
                        )

                    # Slice to body vertices when original GLB is a single merged mesh.
                    if body_vert_indices is not None:
                        if buv_bytes and buv_count != len(bv_arr):
                            buv_arr = np.frombuffer(buv_bytes, dtype=np.float32).reshape(-1, 2)
                            buv_arr = buv_arr[body_vert_indices]
                            buv_bytes = buv_arr.astype(np.float32).tobytes()
                            buv_count = len(body_vert_indices)
                            log.info("Body part '%s': sliced UVs to %d verts.", part_name, buv_count)
                        if bnorm_bytes and bnorm_count != len(bv_arr):
                            bnorm_arr = np.frombuffer(bnorm_bytes, dtype=np.float32).reshape(-1, 3)
                            bnorm_arr = bnorm_arr[body_vert_indices]
                            bnorm_bytes = bnorm_arr.astype(np.float32).tobytes()
                            bnorm_count = len(body_vert_indices)
                            log.info("Body part '%s': sliced normals to %d verts.", part_name, bnorm_count)

                    if buv_bytes and buv_count == len(bv_arr):
                        a_buv = builder.add_raw(buv_bytes, buv_count, _FLOAT, "VEC2", pygltflib.ARRAY_BUFFER)
                        log.info("Body part '%s': copied %d UV verts.", part_name, buv_count)
                    elif buv_bytes:
                        log.warning(
                            "Body part '%s' UV count mismatch (UV=%d, verts=%d); skipping UVs.",
                            part_name, buv_count, len(bv_arr),
                        )
                    if bnorm_bytes and bnorm_count == len(bv_arr):
                        a_bnorm = builder.add_raw(bnorm_bytes, bnorm_count, _FLOAT, "VEC3", pygltflib.ARRAY_BUFFER)
                        log.info("Body part '%s': copied %d normals.", part_name, bnorm_count)
                    elif bnorm_bytes:
                        log.warning(
                            "Body part '%s' normal count mismatch (normals=%d, verts=%d); skipping normals.",
                            part_name, bnorm_count, len(bv_arr),
                        )
                except Exception as exc:
                    log.warning("Could not extract body part '%s' attributes: %s", part_name, exc)

            body_prim_attrs = pygltflib.Attributes(POSITION=a_bv)
            if a_buv is not None:
                body_prim_attrs.TEXCOORD_0 = a_buv
            if a_bnorm is not None:
                body_prim_attrs.NORMAL = a_bnorm

            body_prim = pygltflib.Primitive(
                attributes=body_prim_attrs,
                indices=a_bi,
                material=part_material_idx,
            )
            meshes.append(pygltflib.Mesh(name=part_name, primitives=[body_prim]))
            nodes.append(pygltflib.Node(name=part_name, mesh=len(meshes) - 1))

    # ── Assemble GLTF ────────────────────────────────────────────────────────
    gltf = pygltflib.GLTF2(
        asset=pygltflib.Asset(version="2.0", generator="face-rigger"),
        scene=0,
        scenes=[pygltflib.Scene(nodes=list(range(len(nodes))))],
        nodes=nodes,
        meshes=meshes,
        accessors=builder.accessors,
        bufferViews=builder.buffer_views,
        buffers=[pygltflib.Buffer(byteLength=0)],
    )

    # Copy materials/textures/images/samplers from original verbatim.
    if orig_gltf is not None:
        if orig_gltf.materials:
            gltf.materials = orig_gltf.materials
        if orig_gltf.textures:
            gltf.textures = orig_gltf.textures
        if orig_gltf.samplers:
            gltf.samplers = orig_gltf.samplers
        if orig_gltf.images:
            blob_before_images = len(builder.binary_blob())
            # Re-embed images into the builder so their bytes are part of the blob.
            gltf.images = _rebase_images(
                orig_gltf, original_glb_bytes, builder, gltf
            )
            blob_after_images = len(builder.binary_blob())
            log.info(
                "Image rebase: blob grew from %d B to %d B (+%d B for %d image(s)).",
                blob_before_images, blob_after_images,
                blob_after_images - blob_before_images,
                len(gltf.images),
            )
            # bufferViews may have been copied by pygltflib at GLTF2() construction;
            # reassign so the image bufferView entries added above are present.
            gltf.bufferViews = builder.buffer_views

    # Capture binary blob AFTER all chunks (including images) have been added.
    binary_blob = builder.binary_blob()
    gltf.buffers[0].byteLength = len(binary_blob)
    gltf.set_binary_blob(binary_blob)
    # set_binary_blob replaces bufferViews with its own internal list, discarding
    # the image bufferView added by _rebase_images — reassign to restore it.
    gltf.bufferViews = builder.buffer_views

    log.info(
        "Writing GLB -> %s  [accessors=%d  bufferViews=%d  binary=%d B  morphTargets=%d  uvs=%s  normals=%s]",
        output_path,
        len(gltf.accessors),
        len(gltf.bufferViews),
        len(binary_blob),
        len(morph_target_list),
        a_uv is not None,
        a_norm is not None,
    )
    gltf.save(str(output_path))
    log.info("GLB saved.")


# ---------------------------------------------------------------------------
# Attribute extraction helpers
# ---------------------------------------------------------------------------

def _extract_head_attributes(
    glb_bytes: bytes,
    head_name: str | None,
    expected_vert_count: int,
    force_first_primitive: bool = False,
) -> tuple:
    """Extract UV/normal bytes and material index from the original GLB.

    Returns (orig_gltf, uv_bytes, normal_bytes, uv_count, normal_count, material_idx).

    When *force_first_primitive* is True (single merged mesh case) the function
    returns the first primitive in the file regardless of name/count, so the
    caller can slice UVs/normals down to just the head vertices afterward.
    """
    orig_gltf = pygltflib.GLTF2.load_from_bytes(glb_bytes)
    orig_binary = orig_gltf.binary_blob()

    head_prim = _find_head_primitive(
        orig_gltf, head_name, expected_vert_count,
        force_first_primitive=force_first_primitive,
    )
    if head_prim is None:
        log.warning("Could not locate head primitive in original GLB for attribute extraction.")
        return orig_gltf, None, None, 0, 0, None

    uv_bytes: bytes | None = None
    uv_count: int = 0
    normal_bytes: bytes | None = None
    normal_count: int = 0
    material_idx: int | None = head_prim.material

    attrs = head_prim.attributes
    if attrs.TEXCOORD_0 is not None:
        uv_bytes, uv_count = _read_accessor_bytes(orig_gltf, orig_binary, attrs.TEXCOORD_0)
    if attrs.NORMAL is not None:
        normal_bytes, normal_count = _read_accessor_bytes(orig_gltf, orig_binary, attrs.NORMAL)

    return orig_gltf, uv_bytes, normal_bytes, uv_count, normal_count, material_idx


def _extract_body_part_attributes(
    orig_gltf: pygltflib.GLTF2,
    orig_binary: bytes,
    part_name: str,
    expected_vert_count: int,
    head_name: str | None,
) -> tuple:
    """Extract UV/normal bytes and material index for a single body part.

    Finds the GLTF mesh matching *part_name* (excluding the head mesh), then
    reads TEXCOORD_0 and NORMAL from its first primitive.

    Returns (uv_bytes, normal_bytes, uv_count, normal_count, material_idx).
    """
    def _is_head_mesh(mesh_name: str | None) -> bool:
        if not mesh_name or not head_name:
            return False
        return head_name.lower() in mesh_name.lower() or mesh_name.lower() in head_name.lower()

    # Try name match first (case-insensitive substring).
    prim = None
    for mesh in orig_gltf.meshes:
        if _is_head_mesh(mesh.name):
            continue
        if mesh.name and (
            part_name.lower() in mesh.name.lower()
            or mesh.name.lower() in part_name.lower()
        ):
            if mesh.primitives:
                prim = mesh.primitives[0]
                log.info("Body part primitive found by name: mesh='%s' for part='%s'", mesh.name, part_name)
                break

    if prim is None:
        # Fallback: closest vertex count among non-head meshes.
        best_prim = None
        best_diff = float("inf")
        for mesh in orig_gltf.meshes:
            if _is_head_mesh(mesh.name):
                continue
            for p in mesh.primitives or []:
                if p.attributes.POSITION is None:
                    continue
                acc = orig_gltf.accessors[p.attributes.POSITION]
                diff = abs(acc.count - expected_vert_count)
                if diff < best_diff:
                    best_diff = diff
                    best_prim = p
        if best_prim is None:
            log.warning("Could not locate primitive for body part '%s'.", part_name)
            return None, None, 0, 0, None
        prim = best_prim
        log.info(
            "Body part '%s' primitive found by vertex count proximity (expected=%d, diff=%d).",
            part_name, expected_vert_count, best_diff,
        )

    uv_bytes: bytes | None = None
    uv_count: int = 0
    normal_bytes: bytes | None = None
    normal_count: int = 0

    if prim.attributes.TEXCOORD_0 is not None:
        uv_bytes, uv_count = _read_accessor_bytes(orig_gltf, orig_binary, prim.attributes.TEXCOORD_0)
    if prim.attributes.NORMAL is not None:
        normal_bytes, normal_count = _read_accessor_bytes(orig_gltf, orig_binary, prim.attributes.NORMAL)

    return uv_bytes, normal_bytes, uv_count, normal_count, prim.material


def _find_head_primitive(
    gltf: pygltflib.GLTF2,
    head_name: str | None,
    expected_vert_count: int,
    force_first_primitive: bool = False,
) -> "pygltflib.Primitive | None":
    """Locate the head primitive by mesh name, then fall back to vertex count.

    When *force_first_primitive* is True the function returns the first
    primitive it finds without any name or vertex-count matching.  This is
    correct for single-merged-mesh GLBs where the head is a vertex-level slice
    of that one primitive; the caller holds head_vert_indices and will slice
    UVs/normals itself.
    """
    # Single-merged-mesh fast path: skip matching entirely.
    if force_first_primitive:
        for mesh in gltf.meshes:
            for prim in mesh.primitives or []:
                if prim.attributes.POSITION is not None:
                    log.info(
                        "Head primitive: single-merged-mesh mode, using first primitive "
                        "(mesh='%s', verts=%d); UVs/normals will be sliced by head_vert_indices.",
                        mesh.name,
                        gltf.accessors[prim.attributes.POSITION].count,
                    )
                    return prim
        return None

    # Try by name first.
    if head_name is not None:
        for mesh in gltf.meshes:
            if mesh.name and (
                head_name.lower() in mesh.name.lower()
                or mesh.name.lower() in head_name.lower()
            ):
                if mesh.primitives:
                    log.info("Head primitive found by name match: mesh='%s'", mesh.name)
                    return mesh.primitives[0]

    # Fallback: find the primitive whose POSITION accessor vertex count is
    # closest to expected_vert_count.
    best_prim = None
    best_diff = float("inf")
    for mesh in gltf.meshes:
        for prim in mesh.primitives:
            if prim.attributes.POSITION is None:
                continue
            acc = gltf.accessors[prim.attributes.POSITION]
            diff = abs(acc.count - expected_vert_count)
            if diff < best_diff:
                best_diff = diff
                best_prim = prim

    if best_prim is not None:
        found_count = gltf.accessors[best_prim.attributes.POSITION].count
        log.info(
            "Head primitive found by vertex count proximity (expected=%d, found=%d, diff=%d).",
            expected_vert_count,
            found_count,
            best_diff,
        )
        if best_diff > 0:
            log.warning(
                "Head primitive: name match failed for '%s'; vertex-count fallback diff=%d "
                "(expected=%d, found=%d). Material/UV may be wrong.",
                head_name, best_diff, expected_vert_count, found_count,
            )
    return best_prim


def _read_accessor_bytes(
    gltf: pygltflib.GLTF2,
    binary: bytes,
    accessor_idx: int,
) -> tuple[bytes, int]:
    """Read raw bytes for a GLTF accessor; return (raw_bytes, element_count)."""
    acc = gltf.accessors[accessor_idx]
    bv = gltf.bufferViews[acc.bufferView]

    n_components = _TYPE_COMPONENTS.get(acc.type, 1)
    comp_bytes = _COMPONENT_BYTES.get(acc.componentType, 4)
    stride = bv.byteStride or (n_components * comp_bytes)
    byte_offset = (bv.byteOffset or 0) + (acc.byteOffset or 0)

    if bv.byteStride and bv.byteStride != n_components * comp_bytes:
        # Interleaved buffer: de-interleave into packed bytes.
        chunks = []
        element_size = n_components * comp_bytes
        for i in range(acc.count):
            start = byte_offset + i * stride
            chunks.append(binary[start: start + element_size])
        raw = b"".join(chunks)
    else:
        total = acc.count * n_components * comp_bytes
        raw = binary[byte_offset: byte_offset + total]

    return raw, acc.count


def _rebase_images(
    orig_gltf: pygltflib.GLTF2,
    orig_glb_bytes: bytes,
    builder: "_GLBBuilder",
    new_gltf: pygltflib.GLTF2,
) -> list:
    """Copy embedded images into the new binary blob and update their bufferViews."""
    orig_binary = orig_gltf.binary_blob()
    if orig_binary is None:
        return orig_gltf.images or []

    images_out = []
    for img in orig_gltf.images:
        if img.bufferView is None:
            # URI-referenced image — keep as-is.
            images_out.append(img)
            continue
        orig_bv = orig_gltf.bufferViews[img.bufferView]
        start = orig_bv.byteOffset or 0
        raw_img = orig_binary[start: start + orig_bv.byteLength]

        new_bv_idx = builder.add_image_bytes(raw_img)
        new_img = pygltflib.Image(
            mimeType=img.mimeType,
            bufferView=new_bv_idx,
        )
        if img.name:
            new_img.name = img.name
        images_out.append(new_img)
        log.info(
            "Rebased image '%s': %d bytes -> bufferView %d",
            img.name or "<unnamed>", len(raw_img), new_bv_idx,
        )

    return images_out


# ---------------------------------------------------------------------------
# ICP inverse transform helper
# ---------------------------------------------------------------------------

def _inverse_icp_matrix(alignment_meta: dict) -> list[float]:
    """Return the inverse ICP transform as a GLTF column-major flat list.

    The forward ICP pipeline is:
        v_aligned = T @ (scale * (v_orig - src_centre))

    So the inverse (placing aligned head verts back into Meshy world space) is:
        v_orig = (1/scale) * T_inv @ v_aligned + src_centre

    As a 4x4 matrix M = Translate(src_centre) @ Scale(1/scale) @ T_inv.
    GLTF matrices are column-major (M.T flattened).
    """
    src_centre = np.array(alignment_meta["source_centre"], dtype=np.float64)
    scale_factor = float(alignment_meta["scale_factor"])
    T = np.array(alignment_meta["icp_transformation"], dtype=np.float64)

    T_inv = np.linalg.inv(T)
    inv_scale = 1.0 / scale_factor if scale_factor > 1e-8 else 1.0

    scale_mat = np.diag([inv_scale, inv_scale, inv_scale, 1.0])
    translate_mat = np.eye(4)
    translate_mat[:3, 3] = src_centre

    M = translate_mat @ scale_mat @ T_inv
    return M.T.flatten().tolist()  # column-major for GLTF


# ---------------------------------------------------------------------------
# Internal buffer builder
# ---------------------------------------------------------------------------

class _GLBBuilder:
    """Incrementally build the flat binary buffer, buffer views, and accessors."""

    def __init__(self) -> None:
        self._chunks: list[bytes] = []
        self.buffer_views: list[pygltflib.BufferView] = []
        self.accessors: list[pygltflib.Accessor] = []

    def add_vec3(self, data: np.ndarray, *, target: int, with_bounds: bool = False) -> int:
        """Append (N, 3) float32 array; return accessor index."""
        arr = np.asarray(data, dtype=np.float32)
        assert arr.ndim == 2 and arr.shape[1] == 3
        bv_idx = self._add_chunk(arr.tobytes(), target)
        min_v = arr.min(axis=0).tolist() if with_bounds else None
        max_v = arr.max(axis=0).tolist() if with_bounds else None
        return self._add_accessor(bv_idx, len(arr), _FLOAT, "VEC3", min_v=min_v, max_v=max_v)

    def add_scalar_u32(self, data: np.ndarray, *, target: int) -> int:
        """Append flat uint32 array; return accessor index."""
        arr = np.asarray(data, dtype=np.uint32).flatten()
        bv_idx = self._add_chunk(arr.tobytes(), target)
        return self._add_accessor(bv_idx, len(arr), _UNSIGNED_INT, "SCALAR")

    def add_raw(
        self, raw: bytes, count: int, component_type: int, acc_type: str, target: int
    ) -> int:
        """Append pre-encoded bytes verbatim; return accessor index."""
        bv_idx = self._add_chunk(raw, target)
        return self._add_accessor(bv_idx, count, component_type, acc_type)

    def add_image_bytes(self, raw: bytes) -> int:
        """Append raw image bytes (no target); return bufferView index."""
        return self._add_chunk(raw, target=None)

    def binary_blob(self) -> bytes:
        return b"".join(self._chunks)

    def _current_offset(self) -> int:
        return sum(len(c) for c in self._chunks)

    def _add_chunk(self, raw: bytes, target: int | None) -> int:
        offset = self._current_offset()
        padded = raw + b"\x00" * ((-len(raw)) % 4)
        self._chunks.append(padded)
        bv = pygltflib.BufferView(
            buffer=0,
            byteOffset=offset,
            byteLength=len(raw),
            target=target,
        )
        idx = len(self.buffer_views)
        self.buffer_views.append(bv)
        return idx

    def _add_accessor(
        self,
        bv_idx: int,
        count: int,
        component_type: int,
        acc_type: str,
        min_v: list | None = None,
        max_v: list | None = None,
    ) -> int:
        acc = pygltflib.Accessor(
            bufferView=bv_idx,
            byteOffset=0,
            componentType=component_type,
            count=count,
            type=acc_type,
        )
        if min_v is not None:
            acc.min = [float(v) for v in min_v]
        if max_v is not None:
            acc.max = [float(v) for v in max_v]
        idx = len(self.accessors)
        self.accessors.append(acc)
        return idx
