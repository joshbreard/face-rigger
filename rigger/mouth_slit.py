"""Mouth slit: split the closed-mouth seam into distinct upper/lower lip loops.

Algorithm
---------
1. Find candidate vertices inside a configurable lip bounding box
   (Y-range relative to mesh height, Z-range restricted to the frontal face).
2. Among candidates, identify *seam vertices* — those whose adjacent face
   centroids span both above AND below the vertex's own Y position, meaning
   the vertex sits on the boundary between the upper and lower lip surfaces.
3. Duplicate each seam vertex.  Original indices (lower copy) keep their slot;
   upper copies are appended after all existing vertices.
4. Rewire faces: if a face centroid is *above* a seam vertex's Y, remap that
   vertex slot to the upper copy.  Lower / coplanar faces keep the original.

This creates topologically distinct upper and lower lip vertex loops while
leaving the visual geometry unchanged.  Because new vertices are appended at
the end, indices [0 .. N-1] are stable — blendshape displacement arrays for
those indices remain valid for downstream patching via patch_glb_add_morph_targets.
"""

import logging

import numpy as np
import trimesh

log = logging.getLogger(__name__)

# Lip region defaults — fractions of the head bounding-box height from the bottom.
_LIP_Y_LO_FRAC: float = 0.08   # just above the chin
_LIP_Y_HI_FRAC: float = 0.25   # below the nose
# Minimum Z fraction from the back of the mesh; restricts to front-facing geo.
_LIP_Z_LO_FRAC: float = 0.35


def cut_mouth_slit(
    mesh: trimesh.Trimesh,
    *,
    lip_y_lo_frac: float = _LIP_Y_LO_FRAC,
    lip_y_hi_frac: float = _LIP_Y_HI_FRAC,
    lip_z_lo_frac: float = _LIP_Z_LO_FRAC,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Split the mouth seam into distinct upper and lower lip vertex loops.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Head mesh (output of separate_head_body).
    lip_y_lo_frac, lip_y_hi_frac : float
        Y-range for the lip region as fractions from the bottom of the mesh
        bounding box.  Defaults: [0.08, 0.25].
    lip_z_lo_frac : float
        Frontal-face Z threshold as a fraction from the back of the mesh.
        Only vertices with Z >= z_min + lip_z_lo_frac * z_range are
        considered seam candidates.  Default: 0.35.

    Returns
    -------
    (trimesh.Trimesh, np.ndarray)
        A tuple of (new_mesh, seam_orig_indices) where new_mesh has seam
        vertices duplicated so upper and lower lip triangles reference
        independent vertex loops, and seam_orig_indices is a 1-D int32
        array of length N_seam containing the original (pre-split) vertex
        indices that were duplicated as upper-lip copies.  The upper-lip
        copies occupy indices [N_orig .. N_orig + N_seam - 1] in the
        returned mesh.  If no seam vertices were found, seam_orig_indices
        is empty.
    """
    verts = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int32)

    y_min = float(verts[:, 1].min())
    y_max = float(verts[:, 1].max())
    z_min = float(verts[:, 2].min())
    z_max = float(verts[:, 2].max())
    y_range = y_max - y_min
    z_range = z_max - z_min

    lip_y_lo = y_min + lip_y_lo_frac * y_range
    lip_y_hi = y_min + lip_y_hi_frac * y_range
    lip_z_lo = z_min + lip_z_lo_frac * z_range

    log.info(
        "cut_mouth_slit: mesh bbox Y=[%.4f, %.4f] Z=[%.4f, %.4f]; "
        "lip region Y=[%.4f, %.4f] Z>=[%.4f].",
        y_min, y_max, z_min, z_max, lip_y_lo, lip_y_hi, lip_z_lo,
    )

    # ── Candidate vertices: inside the lip bounding box ───────────────────────
    in_lip = (
        (verts[:, 1] >= lip_y_lo)
        & (verts[:, 1] <= lip_y_hi)
        & (verts[:, 2] >= lip_z_lo)
    )
    candidate_indices = np.where(in_lip)[0]

    if len(candidate_indices) == 0:
        log.warning(
            "cut_mouth_slit: no vertices found in lip region "
            "Y=[%.4f, %.4f] Z>=[%.4f]; mesh returned unchanged.",
            lip_y_lo, lip_y_hi, lip_z_lo,
        )
        return mesh, np.array([], dtype=np.int32)

    log.info(
        "cut_mouth_slit: %d candidate lip vertices in lip region.",
        len(candidate_indices),
    )

    # ── Per-face centroid Y ───────────────────────────────────────────────────
    face_centroid_y = verts[faces, 1].mean(axis=1)  # (F,)

    # ── Build candidate-vertex → adjacent-face mapping ────────────────────────
    vert_to_faces: dict[int, list[int]] = {int(vi): [] for vi in candidate_indices}
    for fi in range(len(faces)):
        for vi in faces[fi]:
            vi_int = int(vi)
            if vi_int in vert_to_faces:
                vert_to_faces[vi_int].append(fi)

    # ── Seam vertices: candidates with adjacent faces on both sides of their Y ─
    seam_vertices: list[int] = []
    for vi in candidate_indices:
        adj = vert_to_faces[int(vi)]
        if not adj:
            continue
        vy = float(verts[vi, 1])
        cy = face_centroid_y[adj]
        if np.any(cy > vy) and np.any(cy < vy):
            seam_vertices.append(int(vi))

    if not seam_vertices:
        log.warning(
            "cut_mouth_slit: no seam vertices identified "
            "(no candidate has adjacent faces on both sides of its Y); "
            "mesh returned unchanged.",
        )
        return mesh, np.array([], dtype=np.int32)

    n_seam = len(seam_vertices)
    log.info("cut_mouth_slit: %d seam vertices will be split.", n_seam)

    # ── Duplicate seam vertices — upper copies appended after all originals ───
    seam_arr = np.array(seam_vertices, dtype=np.int32)
    # old vertex index → new index of its upper copy
    old_to_upper: dict[int, int] = {
        int(vi): len(verts) + i for i, vi in enumerate(seam_arr)
    }
    new_verts = np.vstack([verts, verts[seam_arr]])

    # ── Rewire faces: upper-side faces use the upper vertex copies ────────────
    new_faces = faces.copy()
    seam_set = set(seam_vertices)

    for fi in range(len(faces)):
        fcy = face_centroid_y[fi]
        for j in range(3):
            vi = int(faces[fi, j])
            if vi in seam_set and fcy > float(verts[vi, 1]):
                new_faces[fi, j] = old_to_upper[vi]

    # ── Copy visual data; extend UV array for the new seam vertices ───────────
    new_visual = None
    try:
        vis = mesh.visual
        if hasattr(vis, "uv") and vis.uv is not None:
            old_uv = np.asarray(vis.uv, dtype=np.float32)
            new_uv = np.vstack([old_uv, old_uv[seam_arr]])
            new_visual = trimesh.visual.TextureVisuals(
                uv=new_uv,
                material=vis.material if hasattr(vis, "material") else None,
            )
        else:
            new_visual = vis.copy()
    except Exception as exc:
        log.debug("cut_mouth_slit: could not copy visual data: %s", exc)

    result = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
    if new_visual is not None:
        result.visual = new_visual

    log.info(
        "cut_mouth_slit: done — %d → %d vertices (%d seam vertices split into upper/lower copies).",
        len(verts), len(new_verts), n_seam,
    )
    return result, seam_arr
