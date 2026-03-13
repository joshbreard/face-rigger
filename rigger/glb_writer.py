"""Write a rigged GLB using pygltflib with proper GLTF 2.0 morph-target structure.

GLTF morph-target layout produced
──────────────────────────────────
mesh
  └─ primitive
       ├─ attributes.POSITION  → accessor for base vertex positions
       ├─ indices               → accessor for face indices
       └─ targets[0..51]        → one Attributes(POSITION=<accessor>) per blendshape
mesh.extras.targetNames         → ["browDownLeft", ..., "tongueOut"]   (52 names)
mesh.weights                    → [0.0, …, 0.0]                        (52 zeros)

Each morph-target POSITION accessor stores displacement vectors (delta from base),
not absolute positions.
"""

import logging
from pathlib import Path

import numpy as np
import pygltflib
import trimesh

from rigger.transfer import ARKIT_BLENDSHAPES

log = logging.getLogger(__name__)


def write_rigged_glb(
    head_verts: np.ndarray,
    head_faces: np.ndarray,
    blendshapes: dict[str, np.ndarray],
    body_mesh: "trimesh.Trimesh | None",
    output_path: Path,
) -> None:
    """Assemble and save a GLB file with 52 ARKit morph targets on the head mesh.

    Parameters
    ----------
    head_verts : (N, 3) array — rigged head vertex positions
    head_faces : (F, 3) array — triangle face indices into head_verts
    blendshapes : name → (N, 3) displacement array for each of the 52 targets
    body_mesh : unmodified body (no morph targets); None if the whole model is head
    output_path : destination .glb path
    """
    builder = _GLBBuilder()

    # ── Head base geometry ──────────────────────────────────────────────────
    hv = np.asarray(head_verts, dtype=np.float32)
    hi = np.asarray(head_faces, dtype=np.uint32).flatten()

    a_hv = builder.add_vec3(hv, target=pygltflib.ARRAY_BUFFER, with_bounds=True)
    a_hi = builder.add_scalar_u32(hi, target=pygltflib.ELEMENT_ARRAY_BUFFER)

    # ── 52 morph-target displacement accessors ──────────────────────────────
    morph_target_list: list[pygltflib.Attributes] = []
    for name in ARKIT_BLENDSHAPES:
        delta = np.asarray(
            blendshapes.get(name, np.zeros((len(hv), 3))), dtype=np.float32
        )
        if delta.shape != (len(hv), 3):
            log.warning(
                "Blendshape '%s' has shape %s, expected (%d, 3); zeroing.",
                name, delta.shape, len(hv),
            )
            delta = np.zeros((len(hv), 3), dtype=np.float32)
        a_mt = builder.add_vec3(delta, target=pygltflib.ARRAY_BUFFER, with_bounds=True)
        morph_target_list.append(pygltflib.Attributes(POSITION=a_mt))

    # ── Head mesh primitive ─────────────────────────────────────────────────
    head_prim = pygltflib.Primitive(
        attributes=pygltflib.Attributes(POSITION=a_hv),
        indices=a_hi,
        targets=morph_target_list,
    )
    head_mesh_gltf = pygltflib.Mesh(
        name="head",
        primitives=[head_prim],
        weights=[0.0] * len(ARKIT_BLENDSHAPES),
        extras={"targetNames": list(ARKIT_BLENDSHAPES)},
    )

    meshes: list[pygltflib.Mesh] = [head_mesh_gltf]
    nodes: list[pygltflib.Node] = [pygltflib.Node(name="head", mesh=0)]

    # ── Body mesh (pass-through, no morph targets) ──────────────────────────
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

    # ── Assemble GLTF2 ──────────────────────────────────────────────────────
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
    gltf.set_binary_blob(binary_blob)

    log.info(
        "Writing GLB → %s  [accessors=%d  bufferViews=%d  binary=%d B  morphTargets=%d]",
        output_path, len(builder.accessors), len(builder.buffer_views),
        len(binary_blob), len(morph_target_list),
    )
    gltf.save(str(output_path))
    log.info("GLB saved.")


# ── Internal buffer builder ─────────────────────────────────────────────────

class _GLBBuilder:
    """Incrementally build the flat binary buffer, buffer views, and accessors."""

    def __init__(self) -> None:
        self._chunks: list[bytes] = []
        self.buffer_views: list[pygltflib.BufferView] = []
        self.accessors: list[pygltflib.Accessor] = []

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def add_vec3(
        self,
        data: np.ndarray,
        *,
        target: int,
        with_bounds: bool = False,
    ) -> int:
        """Append a (N, 3) float32 array; return accessor index."""
        arr = np.asarray(data, dtype=np.float32)
        assert arr.ndim == 2 and arr.shape[1] == 3
        bv_idx = self._add_chunk(arr.tobytes(), target)
        min_v = arr.min(axis=0).tolist() if with_bounds else None
        max_v = arr.max(axis=0).tolist() if with_bounds else None
        return self._add_accessor(
            bv_idx, len(arr),
            pygltflib.FLOAT, "VEC3",
            min_v=min_v, max_v=max_v,
        )

    def add_scalar_u32(self, data: np.ndarray, *, target: int) -> int:
        """Append a flat uint32 array; return accessor index."""
        arr = np.asarray(data, dtype=np.uint32).flatten()
        bv_idx = self._add_chunk(arr.tobytes(), target)
        return self._add_accessor(bv_idx, len(arr), pygltflib.UNSIGNED_INT, "SCALAR")

    def binary_blob(self) -> bytes:
        return b"".join(self._chunks)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _current_offset(self) -> int:
        return sum(len(c) for c in self._chunks)

    def _add_chunk(self, raw: bytes, target: int | None) -> int:
        """Append raw bytes (4-byte padded) and register a BufferView."""
        offset = self._current_offset()
        padded = raw + b"\x00" * ((-len(raw)) % 4)  # pad to 4-byte boundary
        self._chunks.append(padded)

        bv = pygltflib.BufferView(
            buffer=0,
            byteOffset=offset,
            byteLength=len(raw),   # spec: actual data length, not padded
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
