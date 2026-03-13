"""ICP-based alignment of the Meshy head to Claire's neutral mesh.

Uses Open3D's point-to-point ICP to find the rigid transform that best aligns
the Meshy head point cloud to Claire's neutral vertex cloud (loaded from
assets/bs_skin.npz via rigger.transfer), then applies that transform to the
Meshy head mesh so subsequent morph-target transfer is in a consistent
coordinate frame.
"""

import logging
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from rigger.transfer import claire_neutral_m

log = logging.getLogger(__name__)

# ICP hyper-parameters
_MAX_CORRESPONDENCE_DIST = 0.05   # metres
_MAX_ITERATIONS = 200
_RELATIVE_FITNESS = 1e-6
_RELATIVE_RMSE = 1e-6


def align_icp(
    source_mesh: trimesh.Trimesh,
) -> trimesh.Trimesh:
    """Align *source_mesh* (Meshy head) to Claire's neutral using ICP.

    Claire's neutral point cloud (metres, centred) is imported from
    ``rigger.transfer.claire_neutral_m`` — the same array used later for
    blendshape transfer, ensuring both steps share an identical coordinate
    frame.

    Parameters
    ----------
    source_mesh:
        The Meshy head mesh to align.

    Returns
    -------
    aligned_source : trimesh.Trimesh
        *source_mesh* after applying the ICP rigid transform.
    """
    if claire_neutral_m is None:
        raise RuntimeError(
            "Claire neutral mesh not loaded. "
            "Ensure 'assets/bs_skin.npz' exists before calling align_icp()."
        )

    log.info(
        "Using Claire neutral as ICP target (%d vertices, metres, centred).",
        len(claire_neutral_m),
    )

    source_pcd = _mesh_to_pcd(source_mesh)

    # Build target cloud directly from the centred Claire neutral.
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(claire_neutral_m)

    # Coarse pre-alignment: centre source so ICP converges even when the
    # models come from wildly different coordinate systems.
    # The target (claire_neutral_m) is already centred at the origin.
    source_centre = source_pcd.get_center()
    source_pcd.translate(-source_centre)

    # Scale normalisation: rescale source to match Claire's bounding-box diagonal.
    source_scale = _bbox_diagonal(source_pcd)
    target_scale = _bbox_diagonal(target_pcd)
    if source_scale > 1e-8 and target_scale > 1e-8:
        scale_factor = target_scale / source_scale
        source_pcd.scale(scale_factor, center=(0, 0, 0))
        log.info(
            "Pre-scaled source by %.4f (source diag=%.4f → target diag=%.4f).",
            scale_factor,
            source_scale,
            target_scale,
        )
    else:
        scale_factor = 1.0

    log.info(
        "Running ICP registration (max_corr=%.4f, max_iter=%d)...",
        _MAX_CORRESPONDENCE_DIST,
        _MAX_ITERATIONS,
    )
    result = o3d.pipelines.registration.registration_icp(
        source=source_pcd,
        target=target_pcd,
        max_correspondence_distance=_MAX_CORRESPONDENCE_DIST,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=_MAX_ITERATIONS,
            relative_fitness=_RELATIVE_FITNESS,
            relative_rmse=_RELATIVE_RMSE,
        ),
    )
    log.info(
        "ICP done — fitness=%.6f, RMSE=%.6f.",
        result.fitness,
        result.inlier_rmse,
    )

    # Apply the same pre-processing (centre + scale) then ICP transform to the
    # source *mesh* (not just the point cloud) so vertex positions are updated.
    aligned_verts = np.array(source_mesh.vertices, dtype=np.float64)

    # 1. Centre
    aligned_verts -= source_centre
    # 2. Scale
    aligned_verts *= scale_factor
    # 3. ICP rigid transform (4×4 homogeneous)
    T = np.array(result.transformation)
    ones = np.ones((len(aligned_verts), 1))
    aligned_verts_h = np.hstack([aligned_verts, ones])  # (N, 4)
    aligned_verts = (T @ aligned_verts_h.T).T[:, :3]

    aligned_source = trimesh.Trimesh(
        vertices=aligned_verts,
        faces=np.array(source_mesh.faces),
        process=False,
    )

    log.info("Alignment complete.")
    return aligned_source


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mesh_to_pcd(mesh: trimesh.Trimesh) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(mesh.vertices, dtype=np.float64))
    return pcd


def _bbox_diagonal(pcd: o3d.geometry.PointCloud) -> float:
    aabb = pcd.get_axis_aligned_bounding_box()
    extent = np.array(aabb.get_extent())
    return float(np.linalg.norm(extent))
