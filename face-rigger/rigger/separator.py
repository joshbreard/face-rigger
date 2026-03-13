"""Head / body separation logic.

Strategy
--------
1. Load the GLB with trimesh and inspect every mesh node.
2. If any node name contains a HEAD_KEYWORD (case-insensitive), treat that
   mesh as the head; everything else is the body.
3. Fallback: merge all meshes, compute the full bounding box, and pick the
   largest mesh (by vertex count) whose centroid Y >= the top-50% threshold.
   If no single mesh qualifies, keep all geometry above that threshold.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import trimesh.scene

log = logging.getLogger(__name__)

# Meshy exports use names like "Wolf3D_Head", "Head", "Face", "Hair", etc.
HEAD_KEYWORDS = ("wolf3d", "head", "face", "hair", "skull", "neck")
HEAD_Y_FRACTION = 0.50  # top fraction of full bounding box treated as head


def separate_head_body(
    glb_path: Path,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh | None, dict[str, Any]]:
    """Load *glb_path* and split it into a head mesh and a body mesh.

    Returns
    -------
    head_mesh : trimesh.Trimesh
    body_mesh : trimesh.Trimesh | None
    scene_meta : dict
        Includes ``original_head_name`` — the geometry node name of the
        selected head mesh (used by glb_writer to re-extract UV/materials).
    """
    scene_or_mesh = trimesh.load(str(glb_path), force="scene", process=False)

    if isinstance(scene_or_mesh, trimesh.Trimesh):
        log.info("GLB contained a single mesh (%d verts); treating as head.", len(scene_or_mesh.vertices))
        return scene_or_mesh, None, {"original_head_name": None}

    scene: trimesh.scene.Scene = scene_or_mesh
    scene_meta: dict[str, Any] = {"graph": scene.graph, "metadata": scene.metadata}

    trimeshes: dict[str, trimesh.Trimesh] = {}
    log.info("Scene geometry nodes:")
    for name, geom in scene.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            log.info("  %-30s  (skipped — not a Trimesh)", name)
            continue
        verts = np.array(geom.vertices)
        y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
        y_centroid = verts[:, 1].mean()
        log.info(
            "  %-30s  verts=%6d  Y=[%.4f, %.4f]  centroid_Y=%.4f",
            name, len(verts), y_min, y_max, y_centroid,
        )
        trimeshes[name] = geom

    if not trimeshes:
        raise ValueError("GLB contains no Trimesh geometry nodes.")

    # ── Strategy 1: keyword match ────────────────────────────────────────────
    named_head: list[trimesh.Trimesh] = []
    named_body: list[trimesh.Trimesh] = []
    head_name: str | None = None

    for name, geom in trimeshes.items():
        if any(kw in name.lower() for kw in HEAD_KEYWORDS):
            log.info("Keyword match -> HEAD: '%s'", name)
            named_head.append(geom)
            head_name = name
        else:
            named_body.append(geom)

    if named_head:
        head_mesh = _concat(named_head)
        body_mesh = _concat(named_body) if named_body else None
        log.info(
            "Name-based split: head=%d verts, body=%s verts, head_name='%s'",
            len(head_mesh.vertices),
            len(body_mesh.vertices) if body_mesh else 0,
            head_name,
        )
        scene_meta["original_head_name"] = head_name
        return head_mesh, body_mesh, scene_meta

    # ── Strategy 2: Y-centroid threshold fallback ────────────────────────────
    log.info(
        "No keyword matches; falling back to top-%.0f%% bounding-box Y split.",
        HEAD_Y_FRACTION * 100,
    )
    all_names = list(trimeshes.keys())
    all_meshes = list(trimeshes.values())
    full_mesh = _concat(all_meshes)
    all_verts = np.array(full_mesh.vertices)
    y_min_full, y_max_full = all_verts[:, 1].min(), all_verts[:, 1].max()
    y_threshold = y_max_full - HEAD_Y_FRACTION * (y_max_full - y_min_full)
    log.info(
        "Full Y range [%.4f, %.4f]; head threshold Y >= %.4f",
        y_min_full, y_max_full, y_threshold,
    )

    head_candidates: list[tuple[str, trimesh.Trimesh]] = []
    body_candidates: list[tuple[str, trimesh.Trimesh]] = []
    for name, geom in zip(all_names, all_meshes):
        centroid_y = np.array(geom.vertices)[:, 1].mean()
        if centroid_y >= y_threshold:
            log.info("  Y-centroid %.4f >= threshold -> HEAD: '%s'", centroid_y, name)
            head_candidates.append((name, geom))
        else:
            body_candidates.append((name, geom))

    if head_candidates:
        # Largest head candidate by vertex count drives the head_name.
        head_candidates.sort(key=lambda x: len(x[1].vertices), reverse=True)
        head_name = head_candidates[0][0]
        head_mesh = _concat([g for _, g in head_candidates])
        body_mesh = _concat([g for _, g in body_candidates]) if body_candidates else None
    else:
        log.warning("No mesh centroid above Y threshold; using full mesh as head.")
        head_mesh = full_mesh
        body_mesh = None
        head_name = all_names[0] if all_names else None

    log.info(
        "Y-fallback split: head=%d verts, body=%s verts, head_name='%s'",
        len(head_mesh.vertices),
        len(body_mesh.vertices) if body_mesh else 0,
        head_name,
    )
    scene_meta["original_head_name"] = head_name
    return head_mesh, body_mesh, scene_meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _concat(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)
