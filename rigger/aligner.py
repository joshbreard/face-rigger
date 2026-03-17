"""ICP-based alignment of the Meshy head to Claire's neutral mesh.

Uses Open3D's point-to-point ICP (two-pass: coarse then fine) to find the
rigid transform that best aligns the Meshy head point cloud to Claire's neutral
vertex cloud, then applies that transform to the Meshy head mesh so subsequent
morph-target transfer is in a consistent coordinate frame.
"""

import logging

import numpy as np
import open3d as o3d
import trimesh

from rigger.transfer import claire_neutral_m

log = logging.getLogger(__name__)

# Two-pass ICP hyper-parameters
_ICP_COARSE_DIST = 0.15   # metres — coarse pass casts a wide net
_ICP_FINE_DIST = 0.05     # metres — fine pass refines from coarse result
_ICP_COARSE_ITER = 100
_ICP_FINE_ITER = 200
_RELATIVE_FITNESS = 1e-6
_RELATIVE_RMSE = 1e-6
_ICP_FITNESS_WARN = 0.1   # warn if fitness drops below this after fine pass


def align_icp(
    source_mesh: trimesh.Trimesh,
    landmarks: dict | None = None,
) -> tuple[trimesh.Trimesh, dict]:
    """Align *source_mesh* (Meshy head) to Claire's neutral.

    If *landmarks* (from rigger.landmarks.detect_landmarks) are provided and
    contain at least 3 keypoints, delegates to align_landmark_nicp() which
    uses Procrustes pre-alignment + landmark-weighted ICP.  Falls back to
    the classic two-pass rigid ICP when landmarks are absent or insufficient.

    Returns
    -------
    (aligned_source, alignment_meta)
        aligned_source : trimesh.Trimesh with updated vertex positions
        alignment_meta : dict with keys fitness, rmse, bbox_ratio, scale_factor
    """
    # Try landmark-guided NICP first
    if landmarks is not None:
        kp = landmarks.get("keypoints_3d", {})
        if len(kp) >= 3:
            try:
                log.info(
                    "align_icp: delegating to align_landmark_nicp (%d keypoints).",
                    len(kp),
                )
                return align_landmark_nicp(
                    source_mesh,
                    reference_mesh=None,
                    landmarks_src=landmarks,
                )
            except Exception as exc:
                log.warning(
                    "align_landmark_nicp failed (%s) — falling back to rigid ICP.", exc
                )

    if claire_neutral_m is None:
        raise RuntimeError(
            "Claire neutral mesh not loaded. "
            "Ensure 'assets/bs_skin.npz' exists before calling align_icp()."
        )

    # ── Diagnostics: pre-alignment state ────────────────────────────────────
    src_verts_orig = np.array(source_mesh.vertices, dtype=np.float64)
    src_centroid = src_verts_orig.mean(axis=0)
    target_centroid = claire_neutral_m.mean(axis=0)
    log.info(
        "Pre-ICP source:  %d verts  centroid=[%.4f, %.4f, %.4f]",
        len(src_verts_orig), *src_centroid,
    )
    log.info(
        "Pre-ICP target:  %d verts  centroid=[%.4f, %.4f, %.4f]",
        len(claire_neutral_m), *target_centroid,
    )

    source_pcd = _mesh_to_pcd(source_mesh)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(claire_neutral_m)

    # ── Pre-alignment: centre source at origin ───────────────────────────────
    source_centre = source_pcd.get_center()
    source_pcd.translate(-source_centre)

    # ── Pre-alignment: scale source to match Claire's bbox diagonal ──────────
    source_scale = _bbox_diagonal(source_pcd)
    target_scale = _bbox_diagonal(target_pcd)
    if source_scale > 1e-8 and target_scale > 1e-8:
        scale_factor = target_scale / source_scale
        source_pcd.scale(scale_factor, center=(0, 0, 0))
        log.info(
            "Pre-scale: source_diag=%.4f -> target_diag=%.4f  factor=%.4f",
            source_scale, target_scale, scale_factor,
        )
    else:
        scale_factor = 1.0

    # ── Pass 1: coarse ICP ───────────────────────────────────────────────────
    log.info(
        "ICP coarse pass: max_corr=%.3fm, max_iter=%d",
        _ICP_COARSE_DIST, _ICP_COARSE_ITER,
    )
    coarse = o3d.pipelines.registration.registration_icp(
        source=source_pcd,
        target=target_pcd,
        max_correspondence_distance=_ICP_COARSE_DIST,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=_ICP_COARSE_ITER,
            relative_fitness=_RELATIVE_FITNESS,
            relative_rmse=_RELATIVE_RMSE,
        ),
    )
    log.info("ICP coarse: fitness=%.6f  RMSE=%.6f", coarse.fitness, coarse.inlier_rmse)

    # ── Pass 2: fine ICP initialised from coarse result ──────────────────────
    log.info(
        "ICP fine pass:   max_corr=%.3fm, max_iter=%d",
        _ICP_FINE_DIST, _ICP_FINE_ITER,
    )
    fine = o3d.pipelines.registration.registration_icp(
        source=source_pcd,
        target=target_pcd,
        max_correspondence_distance=_ICP_FINE_DIST,
        init=coarse.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=_ICP_FINE_ITER,
            relative_fitness=_RELATIVE_FITNESS,
            relative_rmse=_RELATIVE_RMSE,
        ),
    )
    log.info("ICP fine:   fitness=%.6f  RMSE=%.6f", fine.fitness, fine.inlier_rmse)

    if fine.fitness < _ICP_FITNESS_WARN:
        log.warning(
            "ICP fitness LOW (%.4f < %.2f) — alignment may be poor. "
            "Check that the head mesh is frontal and scale is reasonable.",
            fine.fitness, _ICP_FITNESS_WARN,
        )

    # ── Apply transform to mesh vertices ────────────────────────────────────
    aligned_verts = src_verts_orig.copy()
    aligned_verts -= source_centre          # centre
    aligned_verts *= scale_factor           # scale
    T = np.array(fine.transformation)
    ones = np.ones((len(aligned_verts), 1))
    aligned_verts = (T @ np.hstack([aligned_verts, ones]).T).T[:, :3]

    aligned_source = trimesh.Trimesh(
        vertices=aligned_verts,
        faces=np.array(source_mesh.faces),
        process=False,
    )

    # ── Post-alignment diagnostics ───────────────────────────────────────────
    aligned_pcd = _mesh_to_pcd(aligned_source)
    aligned_diag = _bbox_diagonal(aligned_pcd)
    claire_diag = target_scale
    bbox_ratio = aligned_diag / claire_diag if claire_diag > 1e-8 else 0.0

    log.info(
        "Post-ICP aligned: centroid=[%.4f, %.4f, %.4f]  bbox_diag=%.4f  ratio_vs_claire=%.3f",
        *aligned_verts.mean(axis=0), aligned_diag, bbox_ratio,
    )

    alignment_meta = {
        "fitness": float(fine.fitness),
        "rmse": float(fine.inlier_rmse),
        "bbox_ratio": float(bbox_ratio),
        "scale_factor": float(scale_factor),
        "source_centre": source_centre.tolist(),
        "icp_transformation": fine.transformation.tolist(),
    }
    log.info("Alignment complete: %s", alignment_meta)
    return aligned_source, alignment_meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def align_landmark_nicp(
    head_mesh: trimesh.Trimesh,
    reference_mesh: trimesh.Trimesh | None,
    landmarks_src: dict,
    landmarks_tgt: dict | None = None,
    lambda_weight: float = 50.0,
) -> tuple[trimesh.Trimesh, dict]:
    """Procrustes pre-alignment + landmark-weighted ICP.

    1. Rigid Procrustes from 5 named keypoints (eyes, nose, mouth corners).
    2. ICP where landmark point-pairs are duplicated ``lambda_weight`` times
       so they are weighted that many times more than geometric correspondences.

    Parameters
    ----------
    head_mesh : trimesh.Trimesh
        Raw (pre-alignment) Meshy head mesh.
    reference_mesh : trimesh.Trimesh | None
        Reference mesh (ignored; uses Claire from transfer module).
    landmarks_src : dict
        Output of rigger.landmarks.detect_landmarks() on *head_mesh*.
    landmarks_tgt : dict | None
        Canonical target keypoints.  Defaults to Claire's known positions.
    lambda_weight : float
        How many times to duplicate landmark pairs for weighting.

    Returns
    -------
    (aligned_mesh, alignment_meta) — same format as align_icp().
    """
    if claire_neutral_m is None:
        raise RuntimeError("Claire data not loaded.")

    src_kp = landmarks_src.get("keypoints_3d", {})
    tgt_kp: dict[str, np.ndarray] = landmarks_tgt or _CLAIRE_KEYPOINTS_M
    common_keys = [k for k in src_kp if k in tgt_kp]

    if len(common_keys) < 3:
        raise ValueError(
            f"align_landmark_nicp: need ≥3 matching keypoints, got {common_keys}."
        )

    src_pts = np.array([src_kp[k] for k in common_keys], dtype=np.float64)
    tgt_pts = np.array([tgt_kp[k] for k in common_keys], dtype=np.float64)

    # ── Step 1: Centre + scale (same pre-alignment as align_icp) ────────────
    src_verts_orig = np.asarray(head_mesh.vertices, dtype=np.float64)
    source_centre = src_verts_orig.mean(axis=0)
    src_verts_c = src_verts_orig - source_centre
    src_pts_c = src_pts - source_centre

    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src_verts_c)
    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(claire_neutral_m)

    source_scale = _bbox_diagonal(src_pcd)
    target_scale = _bbox_diagonal(tgt_pcd)
    scale_factor = (
        target_scale / source_scale
        if source_scale > 1e-8 and target_scale > 1e-8
        else 1.0
    )
    src_verts_scaled = src_verts_c * scale_factor
    src_pts_scaled = src_pts_c * scale_factor

    # ── Step 2: Procrustes rotation from keypoints ───────────────────────────
    R_proc, t_proc = _procrustes(src_pts_scaled, tgt_pts)
    T_proc = np.eye(4)
    T_proc[:3, :3] = R_proc
    T_proc[:3, 3] = t_proc
    log.info(
        "Procrustes (%d keypoints): scale=%.4f  det(R)=%.4f  t_norm=%.4f",
        len(common_keys), scale_factor,
        float(np.linalg.det(R_proc)), float(np.linalg.norm(t_proc)),
    )

    # Apply Procrustes to all vertices
    ones = np.ones((len(src_verts_scaled), 1))
    proc_verts = (T_proc @ np.hstack([src_verts_scaled, ones]).T).T[:, :3]
    # Apply to keypoint positions (for landmark duplicate injection)
    ones_kp = np.ones((len(src_pts_scaled), 1))
    proc_pts = (T_proc @ np.hstack([src_pts_scaled, ones_kp]).T).T[:, :3]

    # ── Step 3: Landmark-weighted ICP ───────────────────────────────────────
    # Inject landmark pairs as lambda_weight integer duplicates so ICP
    # treats them as high-confidence correspondences.
    lam = max(1, int(lambda_weight))
    aug_src = np.vstack([proc_verts, np.tile(proc_pts, (lam, 1))])
    aug_tgt = np.vstack([claire_neutral_m, np.tile(tgt_pts, (lam, 1))])

    aug_src_pcd = o3d.geometry.PointCloud()
    aug_src_pcd.points = o3d.utility.Vector3dVector(aug_src)
    aug_tgt_pcd = o3d.geometry.PointCloud()
    aug_tgt_pcd.points = o3d.utility.Vector3dVector(aug_tgt)

    coarse = o3d.pipelines.registration.registration_icp(
        source=aug_src_pcd,
        target=aug_tgt_pcd,
        max_correspondence_distance=_ICP_COARSE_DIST,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=_ICP_COARSE_ITER,
            relative_fitness=_RELATIVE_FITNESS,
            relative_rmse=_RELATIVE_RMSE,
        ),
    )
    fine = o3d.pipelines.registration.registration_icp(
        source=aug_src_pcd,
        target=aug_tgt_pcd,
        max_correspondence_distance=_ICP_FINE_DIST,
        init=coarse.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=_ICP_FINE_ITER,
            relative_fitness=_RELATIVE_FITNESS,
            relative_rmse=_RELATIVE_RMSE,
        ),
    )
    log.info("Landmark-NICP fine: fitness=%.6f  RMSE=%.6f", fine.fitness, fine.inlier_rmse)

    # Compose full transform T_fine ∘ T_proc
    T_full = fine.transformation @ T_proc

    # Apply to original (pre-centre, pre-scale) vertices
    aligned_verts = src_verts_orig.copy()
    aligned_verts -= source_centre
    aligned_verts *= scale_factor
    ones_full = np.ones((len(aligned_verts), 1))
    aligned_verts = (T_full @ np.hstack([aligned_verts, ones_full]).T).T[:, :3]

    aligned_mesh = trimesh.Trimesh(
        vertices=aligned_verts,
        faces=np.array(head_mesh.faces),
        process=False,
    )

    src_pcd2 = _mesh_to_pcd(aligned_mesh)
    aligned_diag = _bbox_diagonal(src_pcd2)
    bbox_ratio = aligned_diag / target_scale if target_scale > 1e-8 else 0.0

    alignment_meta = {
        "fitness": float(fine.fitness),
        "rmse": float(fine.inlier_rmse),
        "bbox_ratio": float(bbox_ratio),
        "scale_factor": float(scale_factor),
        "source_centre": source_centre.tolist(),
        "icp_transformation": T_full.tolist(),
        "method": "landmark_nicp",
    }
    log.info("Landmark-NICP complete: %s", alignment_meta)
    return aligned_mesh, alignment_meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _procrustes(src_pts: np.ndarray, tgt_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optimal rigid rotation + translation mapping src_pts → tgt_pts (no scaling)."""
    src_c = src_pts.mean(axis=0)
    tgt_c = tgt_pts.mean(axis=0)
    H = (src_pts - src_c).T @ (tgt_pts - tgt_c)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:   # fix reflection
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = tgt_c - R @ src_c
    return R, t


# Canonical Claire keypoint positions (metres, centred at origin).
# Keys match rigger.landmarks.KEYPOINT_INDICES names.
_CLAIRE_KEYPOINTS_M: dict[str, np.ndarray] = {
    "nose_tip":        np.array([ 0.000, -0.005,  0.055]),
    "right_eye_outer": np.array([-0.030,  0.020,  0.040]),
    "left_eye_outer":  np.array([ 0.030,  0.020,  0.040]),
    "mouth_right":     np.array([-0.025, -0.025,  0.040]),
    "mouth_left":      np.array([ 0.025, -0.025,  0.040]),
}


def _mesh_to_pcd(mesh: trimesh.Trimesh) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(mesh.vertices, dtype=np.float64))
    return pcd


def _bbox_diagonal(pcd: o3d.geometry.PointCloud) -> float:
    aabb = pcd.get_axis_aligned_bounding_box()
    return float(np.linalg.norm(np.array(aabb.get_extent())))
