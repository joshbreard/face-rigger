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
) -> None:
    """Assemble and save a GLB with 52 ARKit morph targets on the head mesh.

    Parameters
    ----------
    head_verts : (N, 3) float array — rigged head vertex positions
    head_faces : (F, 3) int array  — triangle face indices
    blendshapes : name -> (N, 3) displacement arrays
    body_mesh : body mesh passed through unchanged; None if whole model is head
    output_path : destination .glb path
    original_glb_bytes : raw bytes of the input GLB (for UV/material extraction)
    original_head_name : geometry node name used to locate the head primitive
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
                _extract_head_attributes(original_glb_bytes, original_head_name, len(head_verts))
            if orig_uvs:
                log.info("Extracted UV data from original GLB (%d bytes, %d verts).", len(orig_uvs), orig_uv_count)
            if orig_normals:
                log.info("Extracted normal data from original GLB (%d bytes, %d verts).", len(orig_normals), orig_normal_count)
            if orig_material_idx is not None:
                log.info("Head primitive material index: %d", orig_material_idx)
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

    if orig_uvs and orig_uv_count == len(hv):
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
    nodes: list[pygltflib.Node] = [pygltflib.Node(name="head", mesh=0)]

    # ── Body mesh (pass-through) ─────────────────────────────────────────────
    if body_mesh is not None:
        bv_arr = np.asarray(body_mesh.vertices, dtype=np.float32)
        bi_arr = np.asarray(body_mesh.faces, dtype=np.uint32).flatten()
        a_bv = builder.add_vec3(bv_arr, target=pygltflib.ARRAY_BUFFER, with_bounds=True)
        a_bi = builder.add_scalar_u32(bi_arr, target=pygltflib.ELEMENT_ARRAY_BUFFER)
        body_prim = pygltflib.Primitive(
            attributes=pygltflib.Attributes(POSITION=a_bv),
            indices=a_bi,
        )
        meshes.append(pygltflib.Mesh(name="body", primitives=[body_prim]))
        nodes.append(pygltflib.Node(name="body", mesh=len(meshes) - 1))

    # ── Assemble GLTF ────────────────────────────────────────────────────────
    binary_blob = builder.binary_blob()

    gltf = pygltflib.GLTF2(
        asset=pygltflib.Asset(version="2.0", generator="face-rigger"),
        scene=0,
        scenes=[pygltflib.Scene(nodes=list(range(len(nodes))))],
        nodes=nodes,
        meshes=meshes,
        accessors=builder.accessors,
        bufferViews=builder.buffer_views,
        buffers=[pygltflib.Buffer(byteLength=len(binary_blob))],
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
            # Re-embed images: shift bufferView indices to account for our new BVs,
            # and update byte offsets to point into our new binary blob.
            gltf.images = _rebase_images(
                orig_gltf, original_glb_bytes, builder, gltf
            )

    gltf.set_binary_blob(binary_blob)

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
) -> tuple:
    """Extract UV/normal bytes and material index from the original GLB.

    Returns (orig_gltf, uv_bytes, normal_bytes, uv_count, normal_count, material_idx).
    """
    orig_gltf = pygltflib.GLTF2.load_from_bytes(glb_bytes)
    orig_binary = orig_gltf.binary_blob()

    head_prim = _find_head_primitive(orig_gltf, head_name, expected_vert_count)
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


def _find_head_primitive(
    gltf: pygltflib.GLTF2,
    head_name: str | None,
    expected_vert_count: int,
) -> "pygltflib.Primitive | None":
    """Locate the head primitive by mesh name, then fall back to vertex count."""
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
        log.info(
            "Head primitive found by vertex count proximity (expected=%d, found=%d, diff=%d).",
            expected_vert_count,
            gltf.accessors[best_prim.attributes.POSITION].count,
            best_diff,
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

        # Append image bytes to builder (no target for image data).
        new_bv_idx = builder._add_chunk(raw_img, target=None)
        # Shift new_bv_idx by count already registered (images added after).
        actual_bv_idx = len(new_gltf.bufferViews) + new_bv_idx
        # Actually builder._add_chunk already appended to builder.buffer_views —
        # the index is absolute within builder.buffer_views.
        new_img = pygltflib.Image(
            mimeType=img.mimeType,
            bufferView=new_bv_idx,
        )
        if img.name:
            new_img.name = img.name
        images_out.append(new_img)

    return images_out


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
