"""Stage 0 mesh enhancer: region-aware remeshing of the face before rigging.

Runs AFTER the user confirms the neck cutoff plane and BEFORE the
separator/RBF pipeline. Produces a remeshed head with:
  - Dense isotropic topology around eyes, mouth, and brows (3 mm target)
  - Moderate density on the rest of the face (6 mm target)
  - Eyeball geometry preserved exactly (no remesh)
  - Pre-cut mouth slit on clean topology
  - Approximate UV transfer via nearest-vertex projection
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import trimesh
from scipy.spatial import KDTree

from rigger.landmarks import detect_landmarks_from_vertices
from rigger.mouth_slit import cut_mouth_slit
from rigger.separator import separate_head_body

log = logging.getLogger("rigger.enhancer")


class EnhancerLandmarkError(Exception):
    """Raised when the enhancer cannot detect face landmarks after all attempts."""
    pass

# ── Constants ─────────────────────────────────────────────────────────────────
_EYEBALL_SEARCH_RADIUS_M = 0.007       # 7 mm
_EYEBALL_SPHERE_TOLERANCE_M = 0.001    # 1 mm
_FACE_RADIUS_FROM_NOSE_M = 0.040       # 40 mm
_DENSE_REGION_RADIUS_M = 0.015         # 15 mm from eye/mouth/brow landmarks
_TARGET_EDGE_DENSE_M = 0.003           # 3 mm
_TARGET_EDGE_STANDARD_M = 0.006        # 6 mm
_REMESH_ITERATIONS = 5


def _decode_gltf_accessor(
    gltf: Any,
    accessor_idx: int,
    binary_blob: bytes,
) -> np.ndarray:
    """Decode a GLTF accessor into a numpy array from the binary blob."""
    acc = gltf.accessors[accessor_idx]
    bv = gltf.bufferViews[acc.bufferView]

    _COMP_DTYPE = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
                   5123: np.uint16, 5125: np.uint32, 5126: np.float32}
    _TYPE_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
                   "MAT2": 4, "MAT3": 9, "MAT4": 16}

    dtype = _COMP_DTYPE[acc.componentType]
    n_comp = _TYPE_COUNT[acc.type]
    item_bytes = np.dtype(dtype).itemsize * n_comp
    stride = bv.byteStride or item_bytes
    start = (bv.byteOffset or 0) + (acc.byteOffset or 0)

    if stride == item_bytes:
        arr = np.frombuffer(binary_blob, dtype=dtype,
                            count=acc.count * n_comp, offset=start)
        return arr.reshape(acc.count, n_comp).copy() if n_comp > 1 else arr.copy()

    # Interleaved / non-tightly-packed
    out = np.empty((acc.count, n_comp), dtype=dtype)
    for i in range(acc.count):
        out[i] = np.frombuffer(binary_blob, dtype=dtype,
                                count=n_comp, offset=start + i * stride)
    return out


class MeshEnhancer:
    """Region-aware head mesh enhancer for face rigging."""

    def enhance(
        self,
        glb_path: str,
        neck_cutoff_y: float,
        neck_cutoff_z_range: tuple[float, float],
    ) -> tuple[str, dict[str, Any]]:
        """Enhance a GLB head mesh and return the path to the enhanced GLB.

        Parameters
        ----------
        glb_path : str
            Path to the input GLB file.
        neck_cutoff_y : float
            Y value of the neck cutoff plane (y_front = y_back = neck_cutoff_y
            for a horizontal cut).
        neck_cutoff_z_range : tuple[float, float]
            (z_min, z_max) of the full mesh, used for the tilted cutting plane.

        Returns
        -------
        tuple[str, dict]
            (path_to_enhanced_glb, stats_dict)
        """
        t_total = time.perf_counter()

        # ── Step 1: Load and isolate head mesh ────────────────────────────────
        glb_p = Path(glb_path)
        head_mesh, body_mesh, scene_meta = separate_head_body(
            glb_p,
            y_back=neck_cutoff_y,
            y_front=neck_cutoff_y,
        )
        original_head_verts = len(head_mesh.vertices)
        log.info(
            "Step 1: head isolated — %d verts, body=%s verts",
            original_head_verts,
            len(body_mesh.vertices) if body_mesh else 0,
        )

        # Keep a copy of the original head for UV transfer
        original_head = head_mesh.copy()

        # ── Step 2: Detect face landmarks on head mesh ────────────────────────
        # Use 3D vertex geometry directly — no render needed.
        # pyrender is incompatible with background threads on macOS (NSWindow
        # must be instantiated on the main thread).
        t2 = time.perf_counter()
        lm_result = detect_landmarks_from_vertices(head_mesh)
        if lm_result is None:
            raise EnhancerLandmarkError(
                "Enhancer Step 2 failed: could not detect face landmarks for "
                "remesh guidance. Check that the head mesh is front-facing and "
                "neck_cutoff_y is set correctly."
            )
        log.info("Step 2: landmarks detected in %.2fs", time.perf_counter() - t2)

        # Extract key region centers
        kp = lm_result["keypoints_3d"]
        lm3d = lm_result["landmark_3d"]

        region_centers = self._extract_region_centers(kp, lm3d)
        log.info(
            "Step 2: region centers — %s",
            {k: [f"{v:.4f}" for v in c] for k, c in region_centers.items()},
        )

        # ── Step 3: Isolate eyeball geometry ──────────────────────────────────
        t3 = time.perf_counter()
        verts = np.asarray(head_mesh.vertices, dtype=np.float64)

        eyeball_mask = np.zeros(len(verts), dtype=bool)
        total_eyeball = 0

        for eye_name in ("left_eye", "right_eye"):
            center = region_centers.get(eye_name)
            if center is None:
                continue
            eye_verts = self._isolate_eyeball(verts, center)
            eyeball_mask |= eye_verts
            total_eyeball += int(eye_verts.sum())

        log.info(
            "Step 3: eyeball isolated — %d verts tagged (%.1fs)",
            total_eyeball,
            time.perf_counter() - t3,
        )

        # ── Step 4: Region-aware remesh using pymeshlab ───────────────────────
        t4 = time.perf_counter()
        nose_tip = region_centers.get("nose_tip")
        if nose_tip is None:
            log.warning("Step 4: no nose_tip landmark — skipping remesh")
            return glb_path, {"original_head_verts": original_head_verts,
                              "enhanced_head_verts": original_head_verts,
                              "eye_region_density": 0.0, "mouth_region_density": 0.0,
                              "eyeball_verts_preserved": total_eyeball, "remesh_duration_ms": 0}

        enhanced_head, remesh_stats = self._region_remesh(
            head_mesh, eyeball_mask, region_centers, nose_tip,
        )
        log.info(
            "Step 4: remesh complete — %d -> %d verts, "
            "eye_density=%.1fmm mouth_density=%.1fmm (%.1fs)",
            original_head_verts,
            len(enhanced_head.vertices),
            remesh_stats["eye_region_density"] * 1000,
            remesh_stats["mouth_region_density"] * 1000,
            time.perf_counter() - t4,
        )

        # ── Step 5: Pre-cut mouth slit ────────────────────────────────────────
        t5 = time.perf_counter()
        enhanced_head = cut_mouth_slit(enhanced_head)
        log.info(
            "Step 5: mouth slit cut — %d verts after slit (%.1fs)",
            len(enhanced_head.vertices),
            time.perf_counter() - t5,
        )

        # ── Step 6: UV texture projection ─────────────────────────────────────
        t6 = time.perf_counter()
        self._project_uvs(original_head, enhanced_head)
        log.info("Step 6: UV projection complete (%.1fs)", time.perf_counter() - t6)

        # ── Step 7: Reconstruct GLB ───────────────────────────────────────────
        t7 = time.perf_counter()
        output_path = self._reconstruct_glb(
            glb_p, enhanced_head, body_mesh, scene_meta,
        )
        log.info("Step 7: GLB reconstructed at %s (%.1fs)", output_path, time.perf_counter() - t7)

        total_s = time.perf_counter() - t_total
        log.info("Stage 0 total: %.1fs", total_s)

        stats = {
            "original_head_verts": original_head_verts,
            "enhanced_head_verts": len(enhanced_head.vertices),
            "eye_region_density": remesh_stats["eye_region_density"] * 1000,   # mm
            "mouth_region_density": remesh_stats["mouth_region_density"] * 1000,  # mm
            "eyeball_verts_preserved": remesh_stats["eyeball_verts_preserved"],
            "remesh_duration_ms": int((time.perf_counter() - t4) * 1000),
        }

        return str(output_path), stats

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_region_centers(
        keypoints: dict[str, np.ndarray],
        landmark_3d: dict[int, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Extract key region center positions from landmarks."""
        centers: dict[str, np.ndarray] = {}

        # Direct keypoints
        if "nose_tip" in keypoints:
            centers["nose_tip"] = keypoints["nose_tip"]

        # Eye centers from MediaPipe landmarks
        # Left eye: landmarks 33 (outer), 133 (inner), 159 (top), 145 (bottom)
        left_eye_lms = [33, 133, 159, 145, 160, 144, 153, 246]
        left_eye_pts = [landmark_3d[i] for i in left_eye_lms if i in landmark_3d]
        if left_eye_pts:
            centers["left_eye"] = np.mean(left_eye_pts, axis=0)

        # Right eye: landmarks 263 (outer), 362 (inner), 386 (top), 374 (bottom)
        right_eye_lms = [263, 362, 386, 374, 387, 373, 380, 466]
        right_eye_pts = [landmark_3d[i] for i in right_eye_lms if i in landmark_3d]
        if right_eye_pts:
            centers["right_eye"] = np.mean(right_eye_pts, axis=0)

        # Mouth center from landmarks
        mouth_lms = [13, 14, 61, 291, 78, 308]
        mouth_pts = [landmark_3d[i] for i in mouth_lms if i in landmark_3d]
        if mouth_pts:
            centers["mouth_center"] = np.mean(mouth_pts, axis=0)

        # Brow centers
        left_brow_lms = [70, 63, 105, 66, 107]
        left_brow_pts = [landmark_3d[i] for i in left_brow_lms if i in landmark_3d]
        if left_brow_pts:
            centers["left_brow"] = np.mean(left_brow_pts, axis=0)

        right_brow_lms = [300, 293, 334, 296, 336]
        right_brow_pts = [landmark_3d[i] for i in right_brow_lms if i in landmark_3d]
        if right_brow_pts:
            centers["right_brow"] = np.mean(right_brow_pts, axis=0)

        return centers

    @staticmethod
    def _isolate_eyeball(
        verts: np.ndarray,
        eye_center: np.ndarray,
    ) -> np.ndarray:
        """Identify eyeball vertices near *eye_center* by sphere fitting.

        Returns a boolean mask over all vertices.
        """
        dists = np.linalg.norm(verts - eye_center, axis=1)
        nearby_mask = dists <= _EYEBALL_SEARCH_RADIUS_M
        nearby_idx = np.where(nearby_mask)[0]

        if len(nearby_idx) < 8:
            return np.zeros(len(verts), dtype=bool)

        nearby_pts = verts[nearby_idx]

        # Least-squares sphere fit: minimize || |p - c|^2 - r^2 ||^2
        # Linearise: 2*cx*x + 2*cy*y + 2*cz*z + (r^2 - cx^2 - cy^2 - cz^2) = x^2 + y^2 + z^2
        A = np.column_stack([
            2 * nearby_pts,
            np.ones(len(nearby_pts)),
        ])
        b = np.sum(nearby_pts ** 2, axis=1)

        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        sphere_center = result[:3]
        sphere_r = np.sqrt(result[3] + np.sum(sphere_center ** 2))

        # Mark verts within tolerance of the sphere surface
        dist_to_center = np.linalg.norm(verts - sphere_center, axis=1)
        on_sphere = np.abs(dist_to_center - sphere_r) <= _EYEBALL_SPHERE_TOLERANCE_M
        # Intersect with the nearby region to avoid false positives far from the eye
        eyeball_mask = on_sphere & nearby_mask

        return eyeball_mask

    @staticmethod
    def _region_remesh(
        head_mesh: trimesh.Trimesh,
        eyeball_mask: np.ndarray,
        region_centers: dict[str, np.ndarray],
        nose_tip: np.ndarray,
    ) -> tuple[trimesh.Trimesh, dict[str, Any]]:
        """Apply region-aware isotropic remeshing using pymeshlab.

        Remeshes only the face region (excluding eyeball verts and verts
        far from nose_tip). Eyeball verts are merged back unchanged.
        """
        import pymeshlab

        verts = np.asarray(head_mesh.vertices, dtype=np.float64)
        faces = np.asarray(head_mesh.faces, dtype=np.int32)

        # Define face region: not eyeball, within 40mm of nose_tip
        dist_from_nose = np.linalg.norm(verts - nose_tip, axis=1)
        face_region = (~eyeball_mask) & (dist_from_nose <= _FACE_RADIUS_FROM_NOSE_M)

        # Determine per-face region membership: a face is "face region" if
        # ALL its verts are in the face region
        face_in_region = np.all(face_region[faces], axis=1)
        face_outside = ~face_in_region

        # Dense sub-regions: within 15mm of eye/mouth/brow landmarks
        dense_landmarks = []
        for key in ("left_eye", "right_eye", "mouth_center", "left_brow", "right_brow"):
            if key in region_centers:
                dense_landmarks.append(region_centers[key])

        # ── Remesh face region in two passes: dense regions, then standard ────
        # Extract face-region sub-mesh
        region_face_indices = np.where(face_in_region)[0]
        if len(region_face_indices) < 10:
            log.warning("_region_remesh: too few face-region faces (%d); skipping", len(region_face_indices))
            return head_mesh, {
                "eye_region_density": 0.0,
                "mouth_region_density": 0.0,
                "eyeball_verts_preserved": int(eyeball_mask.sum()),
                "remesh_duration_ms": 0,
            }

        region_faces = faces[region_face_indices]
        region_vert_idx = np.unique(region_faces.flatten())
        old_to_new = {int(old): new for new, old in enumerate(region_vert_idx)}
        region_verts = verts[region_vert_idx]
        region_faces_remapped = np.vectorize(old_to_new.get)(region_faces).astype(np.int32)

        # Determine which region verts are in the dense sub-region
        dense_mask_region = np.zeros(len(region_verts), dtype=bool)
        for lm in dense_landmarks:
            d = np.linalg.norm(region_verts - lm, axis=1)
            dense_mask_region |= (d <= _DENSE_REGION_RADIUS_M)

        # Compute blended target edge length per-face: use face centroid distance
        # to dense landmarks to decide target
        face_centroids = region_verts[region_faces_remapped].mean(axis=1)
        min_dist_to_dense = np.full(len(face_centroids), np.inf)
        for lm in dense_landmarks:
            d = np.linalg.norm(face_centroids - lm, axis=1)
            np.minimum(min_dist_to_dense, d, out=min_dist_to_dense)

        # Weighted average target: 3mm where dense, 6mm where not
        t = np.clip(min_dist_to_dense / _DENSE_REGION_RADIUS_M, 0.0, 1.0)
        target_edge_blend = _TARGET_EDGE_DENSE_M * (1 - t) + _TARGET_EDGE_STANDARD_M * t
        avg_target = float(target_edge_blend.mean())

        # pymeshlab remesh with the blended average target
        ms = pymeshlab.MeshSet()
        m = pymeshlab.Mesh(
            vertex_matrix=region_verts.astype(np.float64),
            face_matrix=region_faces_remapped.astype(np.int32),
        )
        ms.add_mesh(m)

        # pymeshlab PureValue expects millimetres; mesh coords are in metres
        avg_target_mm = avg_target * 1000.0
        ms.meshing_isotropic_explicit_remeshing(
            targetlen=pymeshlab.PureValue(avg_target_mm),
            iterations=_REMESH_ITERATIONS,
            adaptive=True,
            featuredeg=30.0,
            checksurfdist=True,
            maxsurfdist=pymeshlab.PureValue(avg_target_mm * 0.5),
        )

        remeshed = ms.current_mesh()
        remeshed_verts = remeshed.vertex_matrix()
        remeshed_faces = remeshed.face_matrix()

        # ── Merge back: outside-region faces + eyeball verts + remeshed face ──
        # Collect outside-region sub-mesh
        outside_face_indices = np.where(face_outside)[0]
        if len(outside_face_indices) > 0:
            outside_faces_raw = faces[outside_face_indices]
            outside_vert_idx = np.unique(outside_faces_raw.flatten())
            outside_old_to_new = {int(old): new for new, old in enumerate(outside_vert_idx)}
            outside_verts = verts[outside_vert_idx]
            outside_faces_remapped = np.vectorize(outside_old_to_new.get)(outside_faces_raw).astype(np.int32)

            # Offset remeshed faces by outside vert count
            offset = len(outside_verts)
            merged_verts = np.vstack([outside_verts, remeshed_verts])
            merged_faces = np.vstack([outside_faces_remapped, remeshed_faces + offset])
        else:
            merged_verts = remeshed_verts
            merged_faces = remeshed_faces

        enhanced = trimesh.Trimesh(
            vertices=merged_verts,
            faces=merged_faces,
            process=False,
        )

        # Transfer UVs from the original head mesh via nearest-vertex lookup.
        # Remeshing destroys UV coordinates, so we recover them here.
        orig_vis = head_mesh.visual
        if hasattr(orig_vis, "uv") and orig_vis.uv is not None:
            orig_uv = np.asarray(orig_vis.uv, dtype=np.float32)
            if len(orig_uv) == len(verts):
                tree = KDTree(verts)
                _, nearest_idx = tree.query(merged_verts)
                new_uv = orig_uv[nearest_idx]
                enhanced.visual = trimesh.visual.TextureVisuals(
                    uv=new_uv,
                    material=orig_vis.material if hasattr(orig_vis, "material") else None,
                )

        # ── Compute density stats ────────────────────────────────────────────
        eye_density = _compute_region_edge_density(
            enhanced, region_centers, ("left_eye", "right_eye"), _DENSE_REGION_RADIUS_M,
        )
        mouth_density = _compute_region_edge_density(
            enhanced, region_centers, ("mouth_center",), _DENSE_REGION_RADIUS_M,
        )

        stats = {
            "eye_region_density": eye_density,
            "mouth_region_density": mouth_density,
            "eyeball_verts_preserved": int(eyeball_mask.sum()),
            "remesh_duration_ms": 0,  # filled by caller
        }
        return enhanced, stats

    @staticmethod
    def _project_uvs(
        original_mesh: trimesh.Trimesh,
        enhanced_mesh: trimesh.Trimesh,
    ) -> None:
        """Project UVs from original mesh onto enhanced mesh via nearest-vertex lookup."""
        orig_vis = original_mesh.visual
        if not hasattr(orig_vis, "uv") or orig_vis.uv is None:
            log.info("_project_uvs: original mesh has no UVs; skipping")
            return

        orig_uv = np.asarray(orig_vis.uv, dtype=np.float32)
        orig_verts = np.asarray(original_mesh.vertices, dtype=np.float64)

        tree = KDTree(orig_verts)
        new_verts = np.asarray(enhanced_mesh.vertices, dtype=np.float64)
        _, nearest_idx = tree.query(new_verts)

        new_uv = orig_uv[nearest_idx]

        enhanced_mesh.visual = trimesh.visual.TextureVisuals(
            uv=new_uv,
            material=orig_vis.material if hasattr(orig_vis, "material") else None,
        )

    @staticmethod
    def _reconstruct_glb(
        original_glb_path: Path,
        enhanced_head: trimesh.Trimesh,
        body_mesh: trimesh.Trimesh | None,
        scene_meta: dict[str, Any],
    ) -> Path:
        """Replace the head mesh in the original GLB with the enhanced head.

        Preserves body mesh, materials, textures, and all other nodes.
        """
        import pygltflib

        original_glb_bytes = original_glb_path.read_bytes()
        orig_gltf = pygltflib.GLTF2.load_from_bytes(original_glb_bytes)
        orig_binary: bytes = orig_gltf.binary_blob() or b""

        head_name = scene_meta.get("original_head_name")

        # Find the head primitive in the original GLTF
        head_mesh_idx: int | None = None
        head_prim_idx: int | None = None
        for mi, mesh in enumerate(orig_gltf.meshes):
            if head_name and mesh.name and (
                head_name.lower() in mesh.name.lower()
                or mesh.name.lower() in head_name.lower()
            ):
                head_mesh_idx = mi
                head_prim_idx = 0
                break

        # Fallback: first mesh
        if head_mesh_idx is None:
            head_mesh_idx = 0
            head_prim_idx = 0

        # Build new vertex/face data for the head
        hv = np.asarray(enhanced_head.vertices, dtype=np.float32)
        hf = np.asarray(enhanced_head.faces, dtype=np.uint32).flatten()

        # Build new binary data: original blob + head verts + head indices + head UVs
        new_chunks: list[bytes] = []
        current_offset = len(orig_binary)

        def _add_chunk(data: bytes) -> tuple[int, int]:
            nonlocal current_offset
            offset = current_offset
            padded = data + b"\x00" * ((-len(data)) % 4)
            new_chunks.append(padded)
            current_offset += len(padded)
            return offset, len(data)

        # Head vertex positions
        hv_bytes = hv.tobytes()
        hv_offset, hv_len = _add_chunk(hv_bytes)

        # Head face indices
        hf_bytes = hf.tobytes()
        hf_offset, hf_len = _add_chunk(hf_bytes)

        # Head UVs (if present)
        has_uv = False
        huv_offset, huv_len = 0, 0
        if hasattr(enhanced_head.visual, "uv") and enhanced_head.visual.uv is not None:
            huv = np.asarray(enhanced_head.visual.uv, dtype=np.float32)
            if len(huv) == len(hv):
                huv_bytes = huv.tobytes()
                huv_offset, huv_len = _add_chunk(huv_bytes)
                has_uv = True

        # Create new buffer views and accessors for the head primitive
        # Vertex positions
        pos_bv_idx = len(orig_gltf.bufferViews)
        orig_gltf.bufferViews.append(pygltflib.BufferView(
            buffer=0, byteOffset=hv_offset, byteLength=hv_len,
            target=pygltflib.ARRAY_BUFFER,
        ))
        pos_acc_idx = len(orig_gltf.accessors)
        orig_gltf.accessors.append(pygltflib.Accessor(
            bufferView=pos_bv_idx, byteOffset=0,
            componentType=pygltflib.FLOAT, count=len(hv), type="VEC3",
            min=hv.min(axis=0).tolist(), max=hv.max(axis=0).tolist(),
        ))

        # Face indices
        idx_bv_idx = len(orig_gltf.bufferViews)
        orig_gltf.bufferViews.append(pygltflib.BufferView(
            buffer=0, byteOffset=hf_offset, byteLength=hf_len,
            target=pygltflib.ELEMENT_ARRAY_BUFFER,
        ))
        idx_acc_idx = len(orig_gltf.accessors)
        orig_gltf.accessors.append(pygltflib.Accessor(
            bufferView=idx_bv_idx, byteOffset=0,
            componentType=pygltflib.UNSIGNED_INT, count=len(hf), type="SCALAR",
        ))

        # UV accessor
        uv_acc_idx = None
        if has_uv:
            uv_bv_idx = len(orig_gltf.bufferViews)
            orig_gltf.bufferViews.append(pygltflib.BufferView(
                buffer=0, byteOffset=huv_offset, byteLength=huv_len,
                target=pygltflib.ARRAY_BUFFER,
            ))
            uv_acc_idx = len(orig_gltf.accessors)
            orig_gltf.accessors.append(pygltflib.Accessor(
                bufferView=uv_bv_idx, byteOffset=0,
                componentType=pygltflib.FLOAT, count=len(hv), type="VEC2",
            ))

        # Patch the head primitive to point to new data
        prim = orig_gltf.meshes[head_mesh_idx].primitives[head_prim_idx]
        orig_material_idx = prim.material  # save before patching

        prim.attributes.POSITION = pos_acc_idx
        prim.indices = idx_acc_idx
        if uv_acc_idx is not None:
            prim.attributes.TEXCOORD_0 = uv_acc_idx
        else:
            # No UVs — clear stale TEXCOORD_0 to avoid accessor count mismatch
            # which causes viewers to drop the material entirely
            prim.attributes.TEXCOORD_0 = None
        # Clear normals — trimesh will recompute on load
        prim.attributes.NORMAL = None
        # Clear any existing morph targets (will be re-added by the rig pipeline)
        prim.targets = None
        orig_gltf.meshes[head_mesh_idx].weights = None

        # ── Preserve material from the original head primitive ────────────
        if orig_material_idx is not None:
            if has_uv:
                # UVs present — keep original material with texture references
                prim.material = orig_material_idx
            else:
                # UVs lost — texture sampling won't work.  Create a fallback
                # material that preserves baseColorFactor (skin tone) and PBR
                # properties but drops the baseColorTexture reference.
                orig_mat = orig_gltf.materials[orig_material_idx]
                pbr = orig_mat.pbrMetallicRoughness
                if pbr and pbr.baseColorTexture is not None:
                    fallback_pbr = pygltflib.PbrMetallicRoughness(
                        baseColorFactor=pbr.baseColorFactor or [1.0, 1.0, 1.0, 1.0],
                        metallicFactor=pbr.metallicFactor if pbr.metallicFactor is not None else 0.0,
                        roughnessFactor=pbr.roughnessFactor if pbr.roughnessFactor is not None else 1.0,
                    )
                    fallback_mat = pygltflib.Material(
                        name=(orig_mat.name or "head") + "_fallback",
                        pbrMetallicRoughness=fallback_pbr,
                        doubleSided=orig_mat.doubleSided,
                    )
                    fallback_idx = len(orig_gltf.materials)
                    orig_gltf.materials.append(fallback_mat)
                    prim.material = fallback_idx
                else:
                    # No texture on the original material — safe to use as-is
                    prim.material = orig_material_idx

        # Update buffer size
        new_binary = orig_binary + b"".join(new_chunks)
        orig_gltf.buffers[0].byteLength = len(new_binary)

        # set_binary_blob can reset GLTF JSON arrays — save and restore them
        saved_bv = orig_gltf.bufferViews
        saved_acc = orig_gltf.accessors
        saved_materials = orig_gltf.materials
        saved_images = orig_gltf.images
        saved_textures = orig_gltf.textures
        saved_samplers = orig_gltf.samplers
        saved_meshes = orig_gltf.meshes
        orig_gltf.set_binary_blob(new_binary)
        orig_gltf.bufferViews = saved_bv
        orig_gltf.accessors = saved_acc
        orig_gltf.materials = saved_materials
        orig_gltf.images = saved_images
        orig_gltf.textures = saved_textures
        orig_gltf.samplers = saved_samplers
        orig_gltf.meshes = saved_meshes

        # Write to temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
        output_path = Path(tmp.name)
        tmp.close()
        orig_gltf.save(str(output_path))

        return output_path


def _compute_region_edge_density(
    mesh: trimesh.Trimesh,
    region_centers: dict[str, np.ndarray],
    region_keys: tuple[str, ...],
    radius: float,
) -> float:
    """Compute average edge length in the vicinity of given region centers."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    edges = mesh.edges_unique
    if len(edges) == 0:
        return 0.0

    # Find edges near any of the given region centers
    edge_midpoints = (verts[edges[:, 0]] + verts[edges[:, 1]]) / 2.0
    in_region = np.zeros(len(edges), dtype=bool)
    for key in region_keys:
        center = region_centers.get(key)
        if center is None:
            continue
        d = np.linalg.norm(edge_midpoints - center, axis=1)
        in_region |= (d <= radius)

    if not in_region.any():
        return 0.0

    region_edges = edges[in_region]
    edge_lengths = np.linalg.norm(
        verts[region_edges[:, 0]] - verts[region_edges[:, 1]], axis=1,
    )
    return float(edge_lengths.mean())
