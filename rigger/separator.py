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
HEAD_Y_FRACTION = 0.28  # top fraction used when Gemini is unavailable


def separate_head_body(
    glb_path: Path,
    y_back: float | None = None,
    y_front: float | None = None,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh | None, dict[str, Any]]:
    """Load *glb_path* and split it into a head mesh and a body mesh.

    Parameters
    ----------
    glb_path : Path
    y_back : float | None
        If provided together with *y_front*, skip geometric/Gemini detection
        and use a tilted cutting plane instead.  *y_back* is the cut Y at the
        back of the model (z = z_min of the full mesh).
    y_front : float | None
        Cut Y at the front of the model (z = z_max of the full mesh).

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

    # ── Strategy 2: geometry-based jaw/neck detection (or user plane) ─────────
    all_verts_arr = np.array(_concat(list(trimeshes.values())).vertices)
    y_min_full = float(all_verts_arr[:, 1].min())
    y_max_full = float(all_verts_arr[:, 1].max())
    z_min_full = float(all_verts_arr[:, 2].min())
    z_max_full = float(all_verts_arr[:, 2].max())
    total_height = y_max_full - y_min_full

    use_plane = y_back is not None and y_front is not None
    y_cutoff: float | None = None

    if use_plane:
        log.info(
            "Using provided diagonal plane: y_back=%.4f  y_front=%.4f  z=[%.4f, %.4f]",
            y_back, y_front, z_min_full, z_max_full,
        )
    else:
        y_cutoff = _find_jaw_cutoff_geometric(all_verts_arr)

        if y_cutoff is not None:
            cutoff_fraction = (y_max_full - y_cutoff) / total_height
            log.info(
                "Geometric jaw cutoff: Y=%.4f  fraction=%.3f from top",
                y_cutoff, cutoff_fraction,
            )
        else:
            # ── Strategy 3: Gemini Vision head detection ──────────────────────
            log.info("Geometric detection inconclusive; using Gemini Vision to determine jaw cutoff.")
            cutoff_fraction = _gemini_head_cutoff(_render_views(glb_path))
            log.info("Gemini jaw cutoff fraction: %.3f", cutoff_fraction)
            y_cutoff = y_max_full - cutoff_fraction * total_height

        log.info(
            "Full Y range [%.4f, %.4f]; jaw/head cutoff Y >= %.4f",
            y_min_full, y_max_full, y_cutoff,
        )

    # ── Single-mesh GLB: vertex-level slice ──────────────────────────────────
    if len(trimeshes) == 1:
        single_name, single_mesh = list(trimeshes.items())[0]
        verts = np.array(single_mesh.vertices)
        faces = np.array(single_mesh.faces)

        if use_plane:
            head_mask = _plane_classify_verts(verts, y_back, y_front, z_min_full, z_max_full)
        else:
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

        # Verify trimesh did not reorder vertices (process=False should guarantee this).
        sliced_verts = verts[head_vert_indices]
        max_diff = float(np.abs(np.array(head_mesh.vertices) - sliced_verts).max()) if len(head_vert_indices) > 0 else 0.0
        log.info(
            "Head vertex order check: head_mesh.vertices[0]=%s  verts[head_vert_indices][0]=%s  max_abs_diff=%.6e",
            head_mesh.vertices[0].tolist() if len(head_mesh.vertices) > 0 else "N/A",
            sliced_verts[0].tolist() if len(sliced_verts) > 0 else "N/A",
            max_diff,
        )
        if max_diff > 1e-6:
            log.warning("Vertex reordering detected! UVs indexed by head_vert_indices will be misaligned.")

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

        if use_plane:
            log.info(
                "Vertex-slice split (plane): head=%d verts, body=%d verts, y_back=%.4f y_front=%.4f",
                len(head_mesh.vertices),
                len(body_mesh.vertices) if body_mesh else 0,
                y_back, y_front,
            )
        else:
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
        geom_verts = np.array(geom.vertices)
        if use_plane:
            centroid = geom_verts.mean(axis=0).reshape(1, 3)
            is_head = bool(_plane_classify_verts(centroid, y_back, y_front, z_min_full, z_max_full)[0])
            log.info("  Plane-classified -> %s: '%s'", "HEAD" if is_head else "BODY", name)
        else:
            centroid_y = float(geom_verts[:, 1].mean())
            is_head = centroid_y >= y_cutoff
            log.info("  Y-centroid %.4f >= cutoff -> %s: '%s'", centroid_y, "HEAD" if is_head else "BODY", name)
        if is_head:
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
        stride = bv.byteStride or 8  # default: 2 floats packed = 8 bytes
        if stride != 8:
            # Interleaved buffer: de-interleave manually.
            chunks = []
            for i in range(acc.count):
                start = byte_offset + i * stride
                chunks.append(orig_binary[start: start + 8])
            raw = b"".join(chunks)
            log.info(
                "Head UV de-interleave: stride=%d, count=%d for '%s'.",
                stride, acc.count, head_name,
            )
        else:
            raw = orig_binary[byte_offset: byte_offset + acc.count * 8]
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


def _plane_classify_verts(
    verts: np.ndarray,
    y_back: float,
    y_front: float,
    z_min: float,
    z_max: float,
) -> np.ndarray:
    """Return boolean mask — True for vertices on the 'head' side of the tilted plane.

    The cutting plane passes through P1=(0, y_back, z_min) and P2=(0, y_front, z_max).
    The plane normal [0, dz, -dy] points toward the head (upward) side.
    """
    P1 = np.array([0.0, y_back, z_min])
    dz = z_max - z_min
    dy = y_front - y_back
    normal = np.array([0.0, dz, -dy])
    norm_len = float(np.linalg.norm(normal))
    if norm_len < 1e-9:
        # Degenerate (perfectly horizontal); fall back to horizontal cut at y_back
        return verts[:, 1] >= y_back
    normal /= norm_len
    return (np.dot(verts - P1, normal) >= 0)


def _find_jaw_cutoff_geometric(verts: np.ndarray) -> float | None:
    """Find the neck cutoff using cross-sectional area, filtering out arm geometry.

    Strategy
    --------
    1. Measure head X width from the top 15 % of the model (skull — no arms).
    2. Compute the T-pose arm X threshold (3.5 × head width).  This is used
       to *exclude* arm-containing slices, not as a hard zone boundary.
    3. Search from 40 % below the top down to 8 % below the top.  At each
       slice, skip it if x_span exceeds the arm threshold (arms present).
       For remaining slices compute cross-sectional area (x_span × z_span).
    4. Smooth the area profile and find the minimum — that is the narrowest
       point of the neck cylinder, used directly as the cutoff Y.

    Using a fixed 40 %-from-top lower bound (rather than tpose_y) ensures
    we search through the actual neck even when the arm/shoulder geometry
    is detected close to the head (e.g. muscular or clothed characters).

    Returns an absolute Y cutoff value, or None if detection is unreliable.
    """
    from scipy.ndimage import uniform_filter1d

    y_min = float(verts[:, 1].min())
    y_max = float(verts[:, 1].max())
    total_height = y_max - y_min
    if total_height < 1e-6:
        return None

    # ── Step 1: head X width (top 15 % = skull only) ─────────────────────────
    skull_mask = verts[:, 1] >= y_max - 0.15 * total_height
    if skull_mask.sum() < 20:
        log.warning("_find_jaw_cutoff_geometric: too few skull-top vertices.")
        return None
    head_x_width      = float(verts[skull_mask, 0].max() - verts[skull_mask, 0].min())
    # Use 3.5× head width so broad shoulders don't trigger the arm filter too
    # early; full T-pose arm span is typically 4–5× head width.
    tpose_x_threshold = head_x_width * 3.5

    log.info(
        "_find_jaw_cutoff_geometric: head_x_width=%.4f  tpose_threshold=%.4f",
        head_x_width, tpose_x_threshold,
    )

    # ── Step 2: scan the neck zone, skipping arm-containing slices ───────────
    # Fixed bounds: 8 % from top (just below skull base) → 40 % from top
    # (well below the shoulder/collar), so the actual neck cylinder is always
    # inside this range regardless of where the arms branch off.
    y_high    = y_max - 0.08 * total_height
    y_low     = y_max - 0.40 * total_height
    n_samples = 400
    band_half = 0.015 * total_height
    y_samples = np.linspace(y_low, y_high, n_samples)
    # y_samples[0] = y_low (lower), y_samples[-1] = y_high (closer to head)

    areas = np.full(n_samples, np.nan)
    for i, y_level in enumerate(y_samples):
        mask = (verts[:, 1] >= y_level - band_half) & (verts[:, 1] <= y_level + band_half)
        if mask.sum() < 10:
            continue
        x_span = float(verts[mask, 0].max() - verts[mask, 0].min())
        # Skip slices that include arm geometry
        if x_span > tpose_x_threshold:
            continue
        z_span = float(verts[mask, 2].max() - verts[mask, 2].min())
        areas[i] = x_span * z_span

    valid = np.isfinite(areas)
    if valid.sum() < 10:
        log.warning("_find_jaw_cutoff_geometric: too few valid slices in neck search zone.")
        return None

    fill_val = float(np.nanmax(areas[valid]))
    smoothed = uniform_filter1d(
        np.where(valid, areas, fill_val).astype(float), size=15
    )
    smoothed = np.where(valid, smoothed, np.nan)

    neck_idx  = int(np.nanargmin(smoothed))
    neck_area = float(smoothed[neck_idx])
    neck_y    = float(y_samples[neck_idx])
    fraction  = (y_max - neck_y) / total_height

    log.info(
        "_find_jaw_cutoff_geometric: neck_y=%.4f  neck_area=%.6f  fraction=%.3f from top",
        neck_y, neck_area, fraction,
    )

    if not (0.08 <= fraction <= 0.40):
        log.warning(
            "_find_jaw_cutoff_geometric: fraction %.3f outside [0.08, 0.40]; ignoring.",
            fraction,
        )
        return None

    return neck_y


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
    ref_fractions = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

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
    """Ask Gemini Vision where the head ends; returns fraction from top [0.15, 0.30]."""
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
            "Horizontal reference lines are drawn at 10%, 15%, 20%, 25%, 30%, 35%, and 40% from the top. "
            "I need to find the BOTTOM OF THE JAW / CHIN — the point where the jawline meets the neck. "
            "Everything from the chin/jawline UPWARD will be treated as the head for facial animation. "
            "The jaw bottom is typically the LOWEST point of the chin, just above where the neck begins. "
            "This is usually deeper into the figure than just the head crown — typically 25% to 38% from the top. "
            "Look at both the front view (where you can see face width narrowing at the chin) and the side view "
            "(where you can see the chin profile). "
            "At what fraction from the TOP is the bottom of the jaw/chin? "
            "Reply with ONLY a single decimal number between 0.15 and 0.40."
        )

        response = model.generate_content(parts)
        raw = response.text.strip()
        match = re.search(r"[-+]?\d*\.?\d+", raw)
        if not match:
            log.warning("_gemini_head_cutoff: could not parse number from response %r; using default.", raw)
            return HEAD_Y_FRACTION
        value = float(match.group())
        clamped = max(0.18, min(0.40, value))
        log.info("Gemini returned jaw cutoff %.4f (clamped to %.4f)", value, clamped)
        return clamped

    except Exception as exc:
        log.warning("_gemini_head_cutoff: API call failed (%s); using default %.2f", exc, HEAD_Y_FRACTION)
        return HEAD_Y_FRACTION
