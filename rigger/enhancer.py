"""Stage 0 mesh enhancer: region-aware remeshing of the face before rigging.

Runs AFTER the user confirms the neck cutoff plane and BEFORE the
separator/RBF pipeline. Produces a remeshed head with:
  - Dense isotropic topology around eyes, mouth, nose, and brows (3 mm target)
  - Moderate density on the rest of the face (5.5 mm target)
  - Eyeball geometry preserved exactly (no remesh)
  - Pre-cut mouth slit on clean topology
  - UV transfer via barycentric interpolation (seam-safe)
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
_NOSE_REGION_RADIUS_M = 0.025          # 25 mm — covers tip + nostrils + bridge base
_TARGET_EDGE_DENSE_M = 0.003           # 3 mm
_TARGET_EDGE_STANDARD_M = 0.0055       # 5.5 mm
_REMESH_ITERATIONS = 5
_BOUNDARY_SNAP_TOL_M = 0.005           # 5 mm — vertices within this of neck_cutoff_y are snapped


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

        # ── Step 5b: Snap neck boundary verts to original positions ──────────
        # After remeshing + mouth slit, boundary verts near the neck cut
        # plane may have shifted.  Snap them back to their pre-remesh
        # positions so the seam matches the body mesh exactly.
        t5b = time.perf_counter()
        enhanced_verts = np.asarray(enhanced_head.vertices, dtype=np.float64)
        original_verts = np.asarray(original_head.vertices, dtype=np.float64)

        orig_boundary_mask = (
            np.abs(original_verts[:, 1] - neck_cutoff_y) < _BOUNDARY_SNAP_TOL_M
        )
        enh_boundary_mask = (
            np.abs(enhanced_verts[:, 1] - neck_cutoff_y) < _BOUNDARY_SNAP_TOL_M
        )

        n_snapped = 0
        if enh_boundary_mask.any() and orig_boundary_mask.any():
            orig_boundary_pts = original_verts[orig_boundary_mask]
            tree = KDTree(orig_boundary_pts)
            enh_boundary_idx = np.where(enh_boundary_mask)[0]
            _, nearest = tree.query(enhanced_verts[enh_boundary_idx])
            enhanced_verts[enh_boundary_idx] = orig_boundary_pts[nearest]
            enhanced_head.vertices = enhanced_verts
            n_snapped = len(enh_boundary_idx)

        log.info(
            "Step 5b: snapped %d neck boundary verts (tol=%.1fmm, %.1fs)",
            n_snapped, _BOUNDARY_SNAP_TOL_M * 1000,
            time.perf_counter() - t5b,
        )

        # ── Step 6: Reconstruct GLB ───────────────────────────────────────────
        t6 = time.perf_counter()
        output_path = self._reconstruct_glb(
            glb_p, enhanced_head, body_mesh, scene_meta,
        )
        out_glb_path = str(output_path)
        log.info("Step 6: GLB reconstructed at %s (%.1fs)", output_path, time.perf_counter() - t6)

        # ── Step 7: UV texture projection ─────────────────────────────────────
        t7 = time.perf_counter()
        enhanced_verts_np = np.asarray(enhanced_head.vertices, dtype=np.float32)
        self._project_uvs(out_glb_path, str(glb_p), enhanced_verts_np)
        log.info("Step 7: UV projection complete (%.1fs)", time.perf_counter() - t7)

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
        """Two-pass region-aware isotropic remeshing using pymeshlab.

        Pass 1: coarse remesh of the full face region at the standard target.
        Pass 2: fine remesh of dense sub-regions (eyes, mouth, nose, brows)
        extracted from the coarse result, refined at the dense target.

        Eyeball verts and geometry outside the face region are preserved.
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

        # Dense sub-regions: (center, radius) tuples
        dense_regions = []
        for key in ("left_eye", "right_eye", "mouth_center", "left_brow", "right_brow"):
            if key in region_centers:
                dense_regions.append((region_centers[key], _DENSE_REGION_RADIUS_M))
        if "nose_tip" in region_centers:
            dense_regions.append((region_centers["nose_tip"], _NOSE_REGION_RADIUS_M))

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

        # ── Pass 1: coarse remesh of entire face region at standard target ──
        ms = pymeshlab.MeshSet()
        m = pymeshlab.Mesh(
            vertex_matrix=region_verts.astype(np.float64),
            face_matrix=region_faces_remapped.astype(np.int32),
        )
        ms.add_mesh(m)

        coarse_mm = _TARGET_EDGE_STANDARD_M * 1000.0
        ms.meshing_isotropic_explicit_remeshing(
            targetlen=pymeshlab.PureValue(coarse_mm),
            iterations=_REMESH_ITERATIONS,
            adaptive=True,
            featuredeg=30.0,
            checksurfdist=True,
            maxsurfdist=pymeshlab.PureValue(coarse_mm * 0.5),
        )

        coarse_result = ms.current_mesh()
        coarse_verts = coarse_result.vertex_matrix()
        coarse_faces = coarse_result.face_matrix()

        # ── Pass 2: fine remesh of dense sub-regions ─────────────────────
        # Identify coarse faces whose centroids fall within a dense region
        coarse_centroids = coarse_verts[coarse_faces].mean(axis=1)
        dense_face_mask = np.zeros(len(coarse_faces), dtype=bool)
        for lm, radius in dense_regions:
            d = np.linalg.norm(coarse_centroids - lm, axis=1)
            dense_face_mask |= (d <= radius)

        if dense_face_mask.any() and dense_face_mask.sum() >= 4:
            # Extract dense sub-mesh from coarse result
            dense_sub_faces = coarse_faces[dense_face_mask]
            dense_vert_idx = np.unique(dense_sub_faces.flatten())
            dense_map = {int(old): new for new, old in enumerate(dense_vert_idx)}
            dense_sub_verts = coarse_verts[dense_vert_idx]
            dense_sub_faces_r = np.vectorize(dense_map.get)(dense_sub_faces).astype(np.int32)

            ms2 = pymeshlab.MeshSet()
            m2 = pymeshlab.Mesh(
                vertex_matrix=dense_sub_verts.astype(np.float64),
                face_matrix=dense_sub_faces_r.astype(np.int32),
            )
            ms2.add_mesh(m2)

            fine_mm = _TARGET_EDGE_DENSE_M * 1000.0
            ms2.meshing_isotropic_explicit_remeshing(
                targetlen=pymeshlab.PureValue(fine_mm),
                iterations=_REMESH_ITERATIONS,
                adaptive=True,
                featuredeg=30.0,
                checksurfdist=True,
                maxsurfdist=pymeshlab.PureValue(fine_mm * 0.5),
            )

            fine_result = ms2.current_mesh()
            fine_verts = fine_result.vertex_matrix()
            fine_faces = fine_result.face_matrix()

            # Combine: non-dense coarse faces + fine dense faces
            std_sub_faces = coarse_faces[~dense_face_mask]
            std_vert_idx = np.unique(std_sub_faces.flatten())
            std_map = {int(old): new for new, old in enumerate(std_vert_idx)}
            std_sub_verts = coarse_verts[std_vert_idx]
            std_sub_faces_r = np.vectorize(std_map.get)(std_sub_faces).astype(np.int32)

            offset = len(std_sub_verts)
            remeshed_verts = np.vstack([std_sub_verts, fine_verts])
            remeshed_faces = np.vstack([std_sub_faces_r, fine_faces + offset])
        else:
            remeshed_verts = coarse_verts
            remeshed_faces = coarse_faces

        # ── Merge back: outside-region faces + remeshed face region ──────
        outside_face_indices = np.where(face_outside)[0]
        if len(outside_face_indices) > 0:
            outside_faces_raw = faces[outside_face_indices]
            outside_vert_idx = np.unique(outside_faces_raw.flatten())
            outside_old_to_new = {int(old): new for new, old in enumerate(outside_vert_idx)}
            outside_verts = verts[outside_vert_idx]
            outside_faces_remapped = np.vectorize(outside_old_to_new.get)(outside_faces_raw).astype(np.int32)

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

        # Transfer UVs from the original head mesh via barycentric interpolation.
        # Remeshing destroys UV coordinates, so we recover them here.
        # Using closest-point + barycentric weights avoids seam tears that
        # nearest-neighbor (KDTree) causes when remeshed vertices near UV seams
        # get matched to source vertices on the wrong side.
        orig_vis = head_mesh.visual
        if hasattr(orig_vis, "uv") and orig_vis.uv is not None:
            orig_uv = np.asarray(orig_vis.uv, dtype=np.float32)
            if len(orig_uv) == len(verts):
                from trimesh.proximity import closest_point as _closest_point
                closest_pts, _dists, tri_ids = _closest_point(head_mesh, merged_verts)
                src_faces_arr = np.asarray(head_mesh.faces)
                bary = trimesh.triangles.points_to_barycentric(
                    head_mesh.triangles[tri_ids], closest_pts,
                )
                v0 = src_faces_arr[tri_ids, 0]
                v1 = src_faces_arr[tri_ids, 1]
                v2 = src_faces_arr[tri_ids, 2]
                new_uv = (bary[:, 0:1] * orig_uv[v0] +
                          bary[:, 1:2] * orig_uv[v1] +
                          bary[:, 2:3] * orig_uv[v2])
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

    def _project_uvs(self, out_gltf_path, source_glb_path, enhanced_vertices):
        """Transfer UVs from source GLB to the enhanced mesh.

        Uses per-face centroid proximity so that all three vertices of
        each enhanced face are UV-mapped via the *same* source triangle,
        preventing within-face UV seam crossings.  Vertices at UV seam
        boundaries are then duplicated so the glTF index buffer can
        represent the discontinuity cleanly.
        """
        import pygltflib, copy
        import numpy as np
        from trimesh.proximity import closest_point as _closest_point

        # --- 1. Read source UVs, vertices, and face indices ---
        src = pygltflib.GLTF2.load(source_glb_path)
        src_blob = src.binary_blob()

        src_prim = None
        for mesh in src.meshes:
            for prim in mesh.primitives:
                if prim.attributes.TEXCOORD_0 is not None:
                    src_prim = prim
                    break
            if src_prim:
                break

        if src_prim is None:
            log.warning("_project_uvs: source has no TEXCOORD_0, skipping")
            return

        src_verts = _decode_gltf_accessor(
            src, src_prim.attributes.POSITION, src_blob,
        ).astype(np.float32)
        src_uvs = _decode_gltf_accessor(
            src, src_prim.attributes.TEXCOORD_0, src_blob,
        ).astype(np.float32)

        if src_prim.indices is None:
            log.warning("_project_uvs: source primitive has no indices, skipping")
            return
        src_faces = _decode_gltf_accessor(
            src, src_prim.indices, src_blob,
        ).reshape(-1, 3).astype(np.int32)

        # --- 2. Build source trimesh for proximity queries ---
        src_mesh = trimesh.Trimesh(
            vertices=src_verts, faces=src_faces, process=False,
        )

        # --- 3. Load output GLB to get enhanced face topology ---
        out = pygltflib.GLTF2.load(out_gltf_path)
        out_blob_raw = out.binary_blob() or b""
        enh_prim = out.meshes[0].primitives[0]
        enh_faces = _decode_gltf_accessor(
            out, enh_prim.indices, out_blob_raw,
        ).reshape(-1, 3).astype(np.int32)

        # --- 4. Per-face barycentric UV transfer ---
        # For each enhanced face, find the closest source triangle to
        # the face *centroid*.  All three vertices of the face are then
        # UV-interpolated via that single source triangle so the result
        # never straddles a UV seam within one face.
        face_verts = enhanced_vertices[enh_faces]       # (F, 3, 3)
        face_centroids = face_verts.mean(axis=1)        # (F, 3)

        _, _, face_tri_ids = _closest_point(src_mesh, face_centroids)

        all_face_verts = face_verts.reshape(-1, 3)      # (F*3, 3)
        all_tri_ids = np.repeat(face_tri_ids, 3)        # (F*3,)

        bary = trimesh.triangles.points_to_barycentric(
            src_mesh.triangles[all_tri_ids], all_face_verts,
        )
        # Clamp negative bary coords (vertex outside its face's source
        # triangle) and renormalise so weights sum to 1.
        bary = np.clip(bary, 0.0, None)
        nan_mask = np.isnan(bary).any(axis=1)
        if nan_mask.any():
            bary[nan_mask] = 1.0 / 3.0
        bary_sum = bary.sum(axis=1, keepdims=True)
        bary = bary / np.where(bary_sum > 0, bary_sum, 1.0)

        v_idx = src_faces[all_tri_ids]                  # (F*3, 3)
        per_face_uvs = (
            bary[:, 0:1] * src_uvs[v_idx[:, 0]] +
            bary[:, 1:2] * src_uvs[v_idx[:, 1]] +
            bary[:, 2:3] * src_uvs[v_idx[:, 2]]
        ).astype(np.float32)                            # (F*3, 2)

        # --- 5. Vertex deduplication with UV-seam splitting ---
        # Key each face-vertex by (original_vertex_index, quantised_uv).
        # Interior vertices merge (same position, same UV); vertices on
        # UV seams are duplicated (same position, different quantised UV).
        all_orig_idx = enh_faces.flatten()              # (F*3,)
        _UV_TOL = 0.005
        quv = np.round(per_face_uvs / _UV_TOL).astype(np.int32)
        keys = np.column_stack([all_orig_idx.reshape(-1, 1), quv])

        _, unique_pos, inverse = np.unique(
            keys, axis=0, return_index=True, return_inverse=True,
        )

        new_positions = enhanced_vertices[
            all_orig_idx[unique_pos]
        ].astype(np.float32)
        new_uvs = per_face_uvs[unique_pos]
        new_indices = inverse.astype(np.uint32)         # (F*3,) flat

        log.info(
            "_project_uvs: %d -> %d verts after UV seam split (%d faces)",
            len(enhanced_vertices), len(new_positions), len(enh_faces),
        )

        # --- 6. Write new POSITION / INDEX / UV into output GLB ---
        out_blob = bytearray(out_blob_raw)

        def _append(gltf, blob, arr, type_str,
                    component_type=5126, target=34962):
            pad = (4 - len(blob) % 4) % 4
            blob += b"\x00" * pad
            raw = arr.tobytes()
            bv = pygltflib.BufferView(
                buffer=0, byteOffset=len(blob),
                byteLength=len(raw), target=target,
            )
            gltf.bufferViews.append(bv)
            kw = dict(bufferView=len(gltf.bufferViews) - 1,
                      byteOffset=0, componentType=component_type,
                      count=len(arr), type=type_str)
            if type_str == "VEC3" and arr.ndim == 2:
                kw["min"] = arr.min(axis=0).tolist()
                kw["max"] = arr.max(axis=0).tolist()
            gltf.accessors.append(pygltflib.Accessor(**kw))
            blob += raw
            return len(gltf.accessors) - 1, blob

        pos_acc, out_blob = _append(
            out, out_blob, new_positions, "VEC3",
            component_type=5126, target=34962)
        idx_acc, out_blob = _append(
            out, out_blob, new_indices, "SCALAR",
            component_type=5125, target=34963)
        uv_acc, out_blob = _append(
            out, out_blob, new_uvs, "VEC2",
            component_type=5126, target=34962)

        enh_prim.attributes.POSITION = pos_acc
        enh_prim.indices = idx_acc
        enh_prim.attributes.TEXCOORD_0 = uv_acc

        # --- 7. Copy materials + texture images from source ---
        out.materials = copy.deepcopy(src.materials)
        out.textures  = copy.deepcopy(src.textures)
        out.images    = copy.deepcopy(src.images)
        out.samplers  = copy.deepcopy(src.samplers)

        for img in out.images:
            if img.bufferView is not None:
                sbv = src.bufferViews[img.bufferView]
                chunk = src_blob[sbv.byteOffset:sbv.byteOffset + sbv.byteLength]
                pad = (4 - len(out_blob) % 4) % 4
                out_blob += b"\x00" * pad
                new_bv = copy.deepcopy(sbv)
                new_bv.byteOffset = len(out_blob)
                new_bv.buffer = 0
                out.bufferViews.append(new_bv)
                img.bufferView = len(out.bufferViews) - 1
                out_blob += chunk

        enh_prim.material = 0
        out.buffers[0].byteLength = len(out_blob)

        # set_binary_blob can reset GLTF JSON arrays — save and restore
        saved_bv = out.bufferViews
        saved_acc = out.accessors
        saved_materials = out.materials
        saved_images = out.images
        saved_textures = out.textures
        saved_samplers = out.samplers
        saved_meshes = out.meshes
        out.set_binary_blob(bytes(out_blob))
        out.bufferViews = saved_bv
        out.accessors = saved_acc
        out.materials = saved_materials
        out.images = saved_images
        out.textures = saved_textures
        out.samplers = saved_samplers
        out.meshes = saved_meshes

        out.save(out_gltf_path)

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
