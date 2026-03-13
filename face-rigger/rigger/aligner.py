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
) -> tuple[trimesh.Trimesh, dict]:
    """Align *source_mesh* (Meshy head) to Claire's neutral using two-pass ICP.

    Returns
    -------
    (aligned_source, alignment_meta)
        aligned_source : trimesh.Trimesh with updated vertex positions
        alignment_meta : dict with keys fitness, rmse, bbox_ratio, scale_factor
    """
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
    }
    log.info("Alignment complete: %s", alignment_meta)
    return aligned_source, alignment_meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mesh_to_pcd(mesh: trimesh.Trimesh) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(mesh.vertices, dtype=np.float64))
    return pcd


def _bbox_diagonal(pcd: o3d.geometry.PointCloud) -> float:
    aabb = pcd.get_axis_aligned_bounding_box()
    return float(np.linalg.norm(np.array(aabb.get_extent())))
