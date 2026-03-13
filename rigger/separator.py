"""Head / body separation logic.

Strategy
--------
1. Load the GLB with trimesh and inspect every mesh node.
2. If any node name contains a HEAD_KEYWORD (case-insensitive), treat that
   mesh as the head; everything else is the body.
3. Gemini Vision fallback: render 2 views of the model, ask Gemini to identify
   the fraction from the top where the head ends, and use that as the Y cutoff.
4. Last resort: if GEMINI_API_KEY is not set, use HEAD_Y_FRACTION (0.20) as
   a hardcoded fallback to pick the top 20% of the bounding box as head.
"""

import logging
import os
import re
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
            if head_name is None:
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

        head_uvs = _extract_gltf_head_uvs(glb_path, head_name)
        if head_uvs is None:
            log.info("Head mesh has no GLTF UV data; falling back to GLTF re-extraction in writer.")
        scene_meta["head_uvs"] = head_uvs

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
        scene_meta["head_vert_indices"] = head_vert_indices
        scene_meta["body_vert_indices"] = body_vert_indices
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

    head_uvs_g = _extract_gltf_head_uvs(glb_path, head_name)
    if head_uvs_g is None:
        log.info("Head mesh has no GLTF UV data; falling back to GLTF re-extraction in writer.")
    scene_meta["head_uvs"] = head_uvs_g

    return head_mesh, body_mesh, scene_meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_gltf_head_uvs(glb_path: Path, head_name: str | None) -> np.ndarray | None:
    """Extract TEXCOORD_0 UV data for the head mesh directly from the GLTF binary.

    Returns (M, 2) float32 array or None if not found.
    """
    try:
        import pygltflib
        orig_gltf = pygltflib.GLTF2.load(str(glb_path))
        orig_binary = orig_gltf.binary_blob()
        if orig_binary is None:
            return None

        prim = None
        if head_name is not None:
            for mesh in orig_gltf.meshes:
                if mesh.name and (
                    head_name.lower() in mesh.name.lower()
                    or mesh.name.lower() in head_name.lower()
                ):
                    if mesh.primitives:
                        prim = mesh.primitives[0]
                        break

        if prim is None or prim.attributes.TEXCOORD_0 is None:
            return None

        acc = orig_gltf.accessors[prim.attributes.TEXCOORD_0]
        bv = orig_gltf.bufferViews[acc.bufferView]
        byte_offset = (bv.byteOffset or 0) + (acc.byteOffset or 0)
        raw = orig_binary[byte_offset: byte_offset + acc.count * 8]  # 2 floats * 4 bytes
        uvs = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()
        log.info("Extracted %d UV verts from GLTF binary for head '%s'.", len(uvs), head_name)
        return uvs
    except Exception as exc:
        log.warning("_extract_gltf_head_uvs failed: %s", exc)
        return None


def _concat(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)


def _render_views(glb_path: Path) -> list[bytes]:
    """Render front and side silhouette views using matplotlib (headless, no display required)."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        scene_or_mesh = trimesh.load(str(glb_path), force="scene", process=False)
        if isinstance(scene_or_mesh, trimesh.Trimesh):
            verts = np.array(scene_or_mesh.vertices)
        else:
            meshes = [g for g in scene_or_mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                log.warning("_render_views: no Trimesh geometry found in GLB.")
                return []
            verts = np.concatenate([np.array(m.vertices) for m in meshes], axis=0)
    except Exception as exc:
        log.warning("_render_views: failed to load GLB: %s", exc)
        return []

    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]

    y_min, y_max = float(y.min()), float(y.max())
    y_range = y_max - y_min

    # Reference fractions from the top (high Y) where lines will be drawn.
    ref_fractions = [0.10, 0.15, 0.20, 0.25, 0.30]

    views = [
        ("front (X-Y)", "X", "Y", x, y),
        ("side (Z-Y)",  "Z", "Y", z, y),
    ]

    images: list[bytes] = []

    for label, xlabel, ylabel, hx, hy in views:
        try:
            fig, ax = plt.subplots(figsize=(4, 6), dpi=128)

            # Build a 2-D histogram and display it as a filled silhouette image.
            h, xedges, yedges = np.histogram2d(hx, hy, bins=128)
            # Transpose so Y is the vertical axis (imshow row 0 = low Y by default).
            # We flip vertically so the top of the image corresponds to high Y (head).
            h_img = h.T[::-1, :]
            # Mask empty bins so background stays white.
            h_masked = np.ma.masked_where(h_img == 0, h_img)
            ax.imshow(
                h_masked,
                aspect="auto",
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                origin="lower",
                cmap="Blues",
                interpolation="nearest",
            )

            # Draw horizontal reference lines at 10 %, 15 %, 20 %, 25 %, 30 % from the top.
            x_left = float(hx.min())
            x_right = float(hx.max())
            for frac in ref_fractions:
                y_line = y_max - frac * y_range
                ax.axhline(y=y_line, color="red", linewidth=1.0, linestyle="--", alpha=0.85)
                ax.text(
                    x_right, y_line,
                    f" {int(frac * 100)}%",
                    va="center", ha="left",
                    fontsize=7, color="red",
                    clip_on=False,
                )

            ax.set_xlim(x_left, x_right)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(label)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            images.append(buf.getvalue())
        except Exception as exc:
            log.warning("_render_views: %s view failed: %s", label, exc)

    log.info("_render_views: produced %d view(s) for Gemini.", len(images))
    return images


def _gemini_head_cutoff(view_images: list[bytes]) -> float:
    """Ask Gemini Vision where the head ends; returns fraction from top [0.10, 0.30]."""
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
            "These images show 2D projections (front and side views) of a single merged 3D humanoid "
            "character mesh. The Y axis is the vertical axis (top = head, bottom = feet). "
            "Horizontal reference lines are drawn at 10%, 15%, 20%, 25%, and 30% from the top. "
            "At what fraction from the TOP does the head end and the neck/torso begin? "
            "Reply with ONLY a single decimal number."
        )

        response = model.generate_content(parts)
        raw = response.text.strip()
        match = re.search(r"[-+]?\d*\.?\d+", raw)
        if not match:
            log.warning("_gemini_head_cutoff: could not parse number from response %r; using default.", raw)
            return HEAD_Y_FRACTION
        value = float(match.group())
        clamped = max(0.10, min(0.30, value))
        log.info("Gemini returned head cutoff %.4f (clamped to %.4f)", value, clamped)
        return clamped

    except Exception as exc:
        log.warning("_gemini_head_cutoff: API call failed (%s); using default %.2f", exc, HEAD_Y_FRACTION)
        return HEAD_Y_FRACTION
