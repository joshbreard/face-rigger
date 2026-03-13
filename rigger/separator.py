"""Head / body separation logic.

Strategy
--------
1. Load the GLB with trimesh and inspect every mesh node.
2. If any node name contains a HEAD_KEYWORD (case-insensitive), treat that
   mesh as the head; everything else is the body.
3. Gemini Vision fallback: render 3 views of the model, ask Gemini to identify
   the fraction from the top where the head ends, and use that as the Y cutoff.
4. Last resort: if GEMINI_API_KEY is not set, use HEAD_Y_FRACTION (0.20) as
   a hardcoded fallback to pick the top 20% of the bounding box as head.
"""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import trimesh.scene

log = logging.getLogger(__name__)

# Meshy exports use names like "Wolf3D_Head", "Head", "Face", "Hair", etc.
HEAD_KEYWORDS = ("wolf3d", "head", "face", "hair", "skull", "neck")
HEAD_Y_FRACTION = 0.20  # top fraction used when Gemini is unavailable


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
        return scene_or_mesh, None, {"original_head_name": None, "body_parts": []}

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
    named_body_parts: list[tuple[str, trimesh.Trimesh]] = []
    head_name: str | None = None

    for name, geom in trimeshes.items():
        if any(kw in name.lower() for kw in HEAD_KEYWORDS):
            log.info("Keyword match -> HEAD: '%s'", name)
            named_head.append(geom)
            head_name = name
        else:
            named_body_parts.append((name, geom))

    if named_head:
        head_mesh = _concat(named_head)
        body_mesh = _concat([g for _, g in named_body_parts]) if named_body_parts else None
        log.info(
            "Name-based split: head=%d verts, body=%s verts, head_name='%s'",
            len(head_mesh.vertices),
            len(body_mesh.vertices) if body_mesh else 0,
            head_name,
        )
        scene_meta["original_head_name"] = head_name
        scene_meta["body_parts"] = named_body_parts
        return head_mesh, body_mesh, scene_meta

    # ── Strategy 2: Gemini Vision head detection ─────────────────────────────
    log.info("No keyword matches; using Gemini Vision to determine head cutoff.")
    cutoff_fraction = _gemini_head_cutoff(_render_views(glb_path))
    log.info("Gemini head cutoff fraction: %.3f", cutoff_fraction)

    all_verts_arr = np.array(_concat(list(trimeshes.values())).vertices)
    y_min_full = all_verts_arr[:, 1].min()
    y_max_full = all_verts_arr[:, 1].max()
    total_height = y_max_full - y_min_full
    y_cutoff = y_max_full - cutoff_fraction * total_height
    log.info(
        "Full Y range [%.4f, %.4f]; head cutoff Y >= %.4f",
        y_min_full, y_max_full, y_cutoff,
    )

    # ── Single-mesh GLB: vertex-level slice ──────────────────────────────────
    if len(trimeshes) == 1:
        single_name, single_mesh = list(trimeshes.items())[0]
        verts = np.array(single_mesh.vertices)
        faces = np.array(single_mesh.faces)

        head_mask = verts[:, 1] >= y_cutoff
        body_mask = ~head_mask

        head_vert_indices = np.where(head_mask)[0]
        body_vert_indices = np.where(body_mask)[0]

        if len(head_vert_indices) == 0:
            log.warning("Vertex slice produced empty head; using full mesh as head.")
            scene_meta["original_head_name"] = single_name
            scene_meta["body_parts"] = []
            return single_mesh, None, scene_meta

        head_old_to_new = {int(old): new for new, old in enumerate(head_vert_indices)}
        head_face_mask = np.all(np.isin(faces, head_vert_indices), axis=1)
        head_faces_orig = faces[head_face_mask]
        head_faces_remapped = np.vectorize(head_old_to_new.get)(head_faces_orig)
        head_mesh = trimesh.Trimesh(
            vertices=verts[head_vert_indices],
            faces=head_faces_remapped,
            process=False,
        )

        body_mesh = None
        if len(body_vert_indices) > 0:
            body_old_to_new = {int(old): new for new, old in enumerate(body_vert_indices)}
            body_face_mask = np.all(np.isin(faces, body_vert_indices), axis=1)
            body_faces_orig = faces[body_face_mask]
            if len(body_faces_orig) > 0:
                body_faces_remapped = np.vectorize(body_old_to_new.get)(body_faces_orig)
                body_mesh = trimesh.Trimesh(
                    vertices=verts[body_vert_indices],
                    faces=body_faces_remapped,
                    process=False,
                )

        log.info(
            "Vertex-slice split: head=%d verts, body=%d verts, cutoff_Y=%.4f",
            len(head_mesh.vertices),
            len(body_mesh.vertices) if body_mesh else 0,
            y_cutoff,
        )
        scene_meta["original_head_name"] = single_name
        scene_meta["body_parts"] = [(single_name + "_body", body_mesh)] if body_mesh else []
        return head_mesh, body_mesh, scene_meta

    # ── Multi-mesh GLB: per-mesh centroid loop ────────────────────────────────
    head_candidates: list[tuple[str, trimesh.Trimesh]] = []
    body_candidates: list[tuple[str, trimesh.Trimesh]] = []
    for name, geom in trimeshes.items():
        centroid_y = np.array(geom.vertices)[:, 1].mean()
        if centroid_y >= y_cutoff:
            log.info("  Y-centroid %.4f >= cutoff -> HEAD: '%s'", centroid_y, name)
            head_candidates.append((name, geom))
        else:
            body_candidates.append((name, geom))

    if head_candidates:
        # Largest head candidate by vertex count drives the head_name.
        head_candidates.sort(key=lambda x: len(x[1].vertices), reverse=True)
        head_name = head_candidates[0][0]
        head_mesh = _concat([g for _, g in head_candidates])
        body_mesh = _concat([g for _, g in body_candidates]) if body_candidates else None
        scene_meta["body_parts"] = body_candidates
    else:
        log.warning("No mesh centroid above Y cutoff; using full mesh as head.")
        full_mesh = _concat(list(trimeshes.values()))
        head_mesh = full_mesh
        body_mesh = None
        all_names = list(trimeshes.keys())
        head_name = all_names[0] if all_names else None
        scene_meta["body_parts"] = []

    log.info(
        "Gemini-fallback split: head=%d verts, body=%s verts, head_name='%s'",
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


def _render_views(glb_path: Path) -> list[bytes]:
    """Render front, side, and 45-degree views of the GLB; return PNG bytes."""
    import math

    try:
        scene = trimesh.load(str(glb_path), force="scene", process=False)
    except Exception as exc:
        log.warning("_render_views: failed to load GLB: %s", exc)
        return []

    # Camera transforms: columns are [right, up, -forward, position] in homogeneous coords.
    # trimesh uses OpenGL convention: camera looks down -Z.
    def _look_from(eye, target=(0, 0, 0), up=(0, 1, 0)):
        """Build a 4x4 camera-to-world transform."""
        eye = np.array(eye, dtype=float)
        target = np.array(target, dtype=float)
        up = np.array(up, dtype=float)
        fwd = target - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        true_up = np.cross(right, fwd)
        mat = np.eye(4)
        mat[:3, 0] = right
        mat[:3, 1] = true_up
        mat[:3, 2] = -fwd   # camera -Z points toward scene
        mat[:3, 3] = eye
        return mat

    # Compute a reasonable camera distance from the scene bounds.
    try:
        bounds = scene.bounds  # (2, 3)
        center = bounds.mean(axis=0)
        extent = (bounds[1] - bounds[0]).max()
        dist = extent * 2.0
    except Exception:
        center = np.zeros(3)
        dist = 2.0

    camera_transforms = [
        _look_from(center + np.array([0, 0, dist]), center),           # front (+Z)
        _look_from(center + np.array([dist, 0, 0]), center),           # side (+X)
        _look_from(center + np.array([dist * math.cos(math.radians(45)),
                                       0,
                                       dist * math.sin(math.radians(45))]), center),  # 45°
    ]

    images: list[bytes] = []
    for i, transform in enumerate(camera_transforms):
        try:
            scene.camera_transform = transform
            png = scene.save_image(resolution=[512, 512])
            if png:
                images.append(png)
        except Exception as exc:
            log.warning("_render_views: view %d failed: %s", i, exc)

    return images


def _gemini_head_cutoff(view_images: list[bytes]) -> float:
    """Ask Gemini Vision where the head ends; returns fraction from top [0.05, 0.40]."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        log.info("GEMINI_API_KEY not set; using default head cutoff fraction %.2f", HEAD_Y_FRACTION)
        return HEAD_Y_FRACTION

    if not view_images:
        log.warning("_gemini_head_cutoff: no view images available; using default %.2f", HEAD_Y_FRACTION)
        return HEAD_Y_FRACTION

    try:
        import google.generativeai as genai
    except ImportError:
        log.warning("google-generativeai not installed; using default head cutoff fraction %.2f", HEAD_Y_FRACTION)
        return HEAD_Y_FRACTION

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        parts: list[Any] = []
        for png_bytes in view_images:
            parts.append({
                "inline_data": {
                    "mime_type": "image/png",
                    "data": png_bytes,
                }
            })
        parts.append(
            "These images show a 3D humanoid character from different angles. "
            "At what fraction from the TOP of the character (0.0 = very top of head, "
            "1.0 = bottom of feet) does the head end and the neck/torso begin? "
            "Respond with ONLY a single decimal number between 0.05 and 0.40. No text."
        )

        response = model.generate_content(parts)
        raw = response.text.strip()
        value = float(raw)
        clamped = max(0.05, min(0.40, value))
        log.info("Gemini returned head cutoff %.4f (clamped to %.4f)", value, clamped)
        return clamped

    except Exception as exc:
        log.warning("_gemini_head_cutoff: API call failed (%s); using default %.2f", exc, HEAD_Y_FRACTION)
        return HEAD_Y_FRACTION
