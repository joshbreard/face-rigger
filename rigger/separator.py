"""Head / body separation logic.

Strategy
--------
1. Load the GLB with trimesh and inspect every mesh node name.
2. If any node name contains "head" or "face" (case-insensitive), treat that
   mesh as the head and concatenate everything else as the body.
3. Fallback: merge all meshes into one, compute the full bounding-box, and
   keep only vertices whose Y coordinate is in the top 22 % of that range as
   the "head" sub-mesh; the rest becomes the body.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import trimesh.scene

log = logging.getLogger(__name__)

HEAD_KEYWORDS = ("head", "face")
HEAD_Y_FRACTION = 0.22  # top fraction of bounding box treated as head


def separate_head_body(
    glb_path: Path,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh | None, dict[str, Any]]:
    """Load *glb_path* and split it into a head mesh and a body mesh.

    Returns
    -------
    head_mesh : trimesh.Trimesh
    body_mesh : trimesh.Trimesh | None
        None when the entire model is the head (no leftover geometry).
    scene_meta : dict
        Preserved metadata (transform graph, extras, etc.) for round-tripping.
    """
    scene_or_mesh = trimesh.load(str(glb_path), force="scene", process=False)

    if isinstance(scene_or_mesh, trimesh.Trimesh):
        log.info("GLB contained a single mesh; treating entire mesh as head.")
        return scene_or_mesh, None, {}

    scene: trimesh.scene.Scene = scene_or_mesh
    scene_meta: dict[str, Any] = {"graph": scene.graph, "metadata": scene.metadata}

    named_head_meshes: list[trimesh.Trimesh] = []
    named_body_meshes: list[trimesh.Trimesh] = []

    geometry_names = list(scene.geometry.keys())
    log.info("Scene contains %d geometry nodes: %s", len(geometry_names), geometry_names)

    for name, geom in scene.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            log.debug("Skipping non-Trimesh geometry '%s'.", name)
            continue
        if any(kw in name.lower() for kw in HEAD_KEYWORDS):
            log.info("Name-match head: '%s'", name)
            named_head_meshes.append(geom)
        else:
            named_body_meshes.append(geom)

    if named_head_meshes:
        head_mesh = _concat(named_head_meshes)
        body_mesh = _concat(named_body_meshes) if named_body_meshes else None
        log.info(
            "Name-based split: head vertices=%d, body vertices=%s",
            len(head_mesh.vertices),
            len(body_mesh.vertices) if body_mesh else 0,
        )
        return head_mesh, body_mesh, scene_meta

    # Fallback: bounding-box Y split
    log.info(
        "No head/face node names found; falling back to top-%.0f%% bounding-box split.",
        HEAD_Y_FRACTION * 100,
    )
    all_meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
    full_mesh = _concat(all_meshes)
    head_mesh, body_mesh = _split_by_y(full_mesh)
    log.info(
        "Bounding-box split: head vertices=%d, body vertices=%d",
        len(head_mesh.vertices),
        len(body_mesh.vertices) if body_mesh else 0,
    )
    return head_mesh, body_mesh, scene_meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _concat(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)


def _split_by_y(
    mesh: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh | None]:
    """Split *mesh* into top-22% (head) and the rest (body) by Y coordinate."""
    verts = np.array(mesh.vertices)
    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    y_threshold = y_max - HEAD_Y_FRACTION * (y_max - y_min)

    log.debug("Y range [%.4f, %.4f]; head threshold Y >= %.4f", y_min, y_max, y_threshold)

    # Find faces where ALL three vertices are above the threshold → head
    face_verts_y = verts[mesh.faces, 1]  # (F, 3)
    head_face_mask = face_verts_y.min(axis=1) >= y_threshold
    body_face_mask = ~head_face_mask

    head_mesh = _submesh(mesh, head_face_mask)
    body_mesh = _submesh(mesh, body_face_mask) if body_face_mask.any() else None
    return head_mesh, body_mesh


def _submesh(mesh: trimesh.Trimesh, face_mask: np.ndarray) -> trimesh.Trimesh:
    """Extract faces selected by boolean *face_mask* into a new Trimesh."""
    selected_faces = mesh.faces[face_mask]
    used_vertex_indices = np.unique(selected_faces)
    index_map = np.zeros(len(mesh.vertices), dtype=np.int64)
    index_map[used_vertex_indices] = np.arange(len(used_vertex_indices))
    new_verts = mesh.vertices[used_vertex_indices]
    new_faces = index_map[selected_faces]
    return trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
