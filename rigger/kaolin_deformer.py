"""Facial landmark-guided Laplacian mesh deformation.

Sits between aligner.py and transfer.py.  Adjusts the ICP-aligned Meshy
head's vertex positions so they better match Claire's facial landmark
structure, improving morph-target transfer for jaw, brows, and lips.

Correspondence strategy
-----------------------
Both meshes are already in Claire's coordinate space after ICP alignment, so
correspondences are built by querying a KDTree on each mesh for the nearest
vertex to each of a set of fixed anatomical 3D positions (jaw, brows, eyes,
nose, lips, cheeks, forehead).  No rendering or mediapipe model required.

Primary path  : scipy cotangent-Laplacian with soft anatomical constraints.
Stretch goal  : Kaolin NRICP when kaolin + GPU are present.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import trimesh
from scipy.spatial import KDTree

log = logging.getLogger(__name__)

# ── Optional Kaolin ───────────────────────────────────────────────────────────
# Optional: pip install kaolin from https://github.com/NVIDIAGameWorks/kaolin
# If not installed, scipy Laplacian fallback is used automatically
# kaolin  (commented out intentionally — install manually if GPU available)
try:
    import kaolin  # noqa: F401
    _KAOLIN_AVAILABLE = True
    log.debug("kaolin_deformer: kaolin imported — NRICP path available.")
except Exception:
    _KAOLIN_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────
_LANDMARK_WEIGHT: float = 10.0

# Anatomical anchor positions in Claire's coordinate space (metres, centred).
# After ICP alignment, aligned_head and template share this coordinate system,
# so querying each mesh for the nearest vertex to these positions gives
# reliable anatomical correspondences without any rendering or face detection.
#
# Positions cover: jaw/chin, eyes (inner/outer/centre), brows (inner/peak/outer),
# nose (tip/base/alae), lips (corners/cupid/chin), cheeks, forehead, nasolabial.
_ANATOMICAL_ANCHORS: list[tuple[float, float, float]] = [
    # Jaw / chin (5)
    ( 0.000, -0.050,  0.020),   # chin tip
    (-0.020, -0.047,  0.026),   # chin left
    ( 0.020, -0.047,  0.026),   # chin right
    (-0.040, -0.040,  0.010),   # jaw angle left
    ( 0.040, -0.040,  0.010),   # jaw angle right
    # Left eye (3)
    (-0.030,  0.020,  0.040),   # left eye centre
    (-0.046,  0.018,  0.028),   # left eye outer corner
    (-0.015,  0.020,  0.044),   # left eye inner corner
    # Right eye (3)
    ( 0.030,  0.020,  0.040),   # right eye centre
    ( 0.046,  0.018,  0.028),   # right eye outer corner
    ( 0.015,  0.020,  0.044),   # right eye inner corner
    # Left brow (3)
    (-0.030,  0.040,  0.030),   # left brow peak
    (-0.048,  0.033,  0.020),   # left brow outer
    (-0.012,  0.038,  0.035),   # left brow inner
    # Right brow (3)
    ( 0.030,  0.040,  0.030),   # right brow peak
    ( 0.048,  0.033,  0.020),   # right brow outer
    ( 0.012,  0.038,  0.035),   # right brow inner
    # Nose (4)
    ( 0.000, -0.005,  0.055),   # nose tip
    ( 0.000, -0.015,  0.048),   # nose base / columella
    (-0.018, -0.010,  0.048),   # left ala
    ( 0.018, -0.010,  0.048),   # right ala
    # Mouth (5)
    (-0.025, -0.025,  0.040),   # left mouth corner
    ( 0.025, -0.025,  0.040),   # right mouth corner
    ( 0.000, -0.018,  0.050),   # cupid's bow / upper lip centre
    ( 0.000, -0.033,  0.044),   # lower lip centre
    ( 0.000, -0.043,  0.036),   # chin-lip groove
    # Cheeks (2)
    (-0.050,  0.000,  0.025),   # left cheek
    ( 0.050,  0.000,  0.025),   # right cheek
    # Forehead (3)
    ( 0.000,  0.068,  0.018),   # forehead centre
    (-0.030,  0.066,  0.010),   # forehead left
    ( 0.030,  0.066,  0.010),   # forehead right
    # Nasolabial / philtrum (3)
    ( 0.000, -0.015,  0.052),   # philtrum
    (-0.030, -0.018,  0.038),   # left nasolabial fold
    ( 0.030, -0.018,  0.038),   # right nasolabial fold
]
# Total: 5 + 3 + 3 + 3 + 3 + 4 + 5 + 2 + 3 + 3 = 34 anchors


# ── Public API ────────────────────────────────────────────────────────────────

def deform_to_template(
    aligned_head: trimesh.Trimesh,
    template_mesh: trimesh.Trimesh,
) -> trimesh.Trimesh:
    """Deform *aligned_head* to better match *template_mesh* facial landmarks.

    Uses Kaolin NRICP when available; otherwise falls back to scipy cotangent-
    Laplacian deformation.  Returns a new trimesh.Trimesh — topology (faces,
    UV indices) is never modified and no vertices are added or removed.

    Parameters
    ----------
    aligned_head:
        ICP-aligned Meshy head mesh (metres, centred at origin).
    template_mesh:
        Claire's neutral mesh (metres, centred) — only vertex positions used.

    Returns
    -------
    trimesh.Trimesh with updated vertex positions and identical faces.
    """
    src_verts = np.array(aligned_head.vertices, dtype=np.float64)
    tgt_verts = np.array(template_mesh.vertices, dtype=np.float64)

    # ── Build correspondences from anatomical 3D anchor positions ─────────────
    src_lm_indices, tgt_lm_positions = _build_anatomical_correspondences(
        src_verts, tgt_verts,
    )
    n_pairs = len(src_lm_indices)
    log.info("kaolin_deformer: %d anatomical anchor pairs built.", n_pairs)

    if n_pairs < 4:
        log.warning("kaolin_deformer: too few pairs (%d); skipping deformation.", n_pairs)
        return aligned_head

    # ── Facial region mask and geometry metrics ───────────────────────────────
    face_mask, face_height = _compute_face_mask(src_verts)

    # ── Deform ───────────────────────────────────────────────────────────────
    faces_arr = np.array(aligned_head.faces)
    if _KAOLIN_AVAILABLE:
        try:
            new_verts = _deform_kaolin(src_verts, faces_arr, src_lm_indices, tgt_lm_positions)
            log.info("kaolin_deformer: Kaolin NRICP path used.")
        except Exception as exc:
            log.warning("kaolin_deformer: Kaolin failed (%s); using Laplacian.", exc)
            new_verts = _deform_laplacian(src_verts, faces_arr, src_lm_indices, tgt_lm_positions)
            log.info("kaolin_deformer: scipy Laplacian fallback (after Kaolin error).")
    else:
        new_verts = _deform_laplacian(src_verts, faces_arr, src_lm_indices, tgt_lm_positions)
        log.info("kaolin_deformer: scipy Laplacian fallback used.")

    # ── Apply face-region mask and per-axis displacement clamp ───────────────
    max_delta = 0.15 * face_height
    delta = new_verts - src_verts
    delta[~face_mask] = 0.0
    delta = np.clip(delta, -max_delta, max_delta)
    final_verts = src_verts + delta

    # Safety net: if deformation produced any non-finite vertex, return original.
    if not np.all(np.isfinite(final_verts)):
        log.warning(
            "kaolin_deformer: deformed vertices contain NaN/Inf; returning original mesh."
        )
        return aligned_head

    result = trimesh.Trimesh(
        vertices=final_verts,
        faces=faces_arr.copy(),
        process=False,
    )
    # Carry visual data (UV, material) forward unchanged.
    if hasattr(aligned_head, "visual") and aligned_head.visual is not None:
        try:
            result.visual = aligned_head.visual.copy()
        except Exception:
            pass

    return result


# ── Anatomical correspondence building ────────────────────────────────────────

def _build_anatomical_correspondences(
    src_verts: np.ndarray,
    tgt_verts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build correspondences by querying each mesh for its nearest vertex to
    each anatomical anchor position in Claire's coordinate space.

    Returns
    -------
    src_indices   : (K,) int indices into src_verts (aligned_head).
    tgt_positions : (K, 3) world-space positions from tgt_verts (Claire).
    """
    src_tree = KDTree(src_verts)
    tgt_tree = KDTree(tgt_verts)

    anchors = np.array(_ANATOMICAL_ANCHORS, dtype=np.float64)  # (K, 3)

    _, src_indices = src_tree.query(anchors)    # (K,)
    _, tgt_indices = tgt_tree.query(anchors)    # (K,)
    tgt_positions = tgt_verts[tgt_indices]      # (K, 3)

    return src_indices.astype(np.int64), tgt_positions


# ── Face region mask ──────────────────────────────────────────────────────────

def _compute_face_mask(verts: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (boolean face-region mask, face_height).

    Facial region: above neck Y cutoff (bottom 5 % of bounding box) AND
    inside a bounding sphere centred on the face centroid with
    radius = 0.6 × face_height.
    """
    y_min = float(verts[:, 1].min())
    y_max = float(verts[:, 1].max())
    face_height = max(y_max - y_min, 1e-6)

    neck_cutoff = y_min + 0.05 * face_height
    centroid = verts.mean(axis=0)
    radius = 0.6 * face_height

    above_neck = verts[:, 1] > neck_cutoff
    in_sphere = np.linalg.norm(verts - centroid, axis=1) < radius
    return (above_neck & in_sphere), face_height


# ── Cotangent Laplacian ───────────────────────────────────────────────────────

def _build_cotangent_laplacian(
    verts: np.ndarray,
    faces: np.ndarray,
) -> scipy.sparse.csr_matrix:
    """Build the cotangent-weighted graph Laplacian L.

    L[i, j] < 0  for neighbouring vertices j ≠ i.
    L[i, i] > 0  equals the sum of positive cotangent weights in row i.
    """
    N = len(verts)
    if len(faces) == 0:
        return scipy.sparse.eye(N, format="csr")

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    def _cot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        dot = (a * b).sum(axis=1)
        cross_norm = np.linalg.norm(np.cross(a, b), axis=1)
        return dot / (cross_norm + 1e-12)

    # Cotangent of the angle at each vertex — gives weight for the opposite edge.
    # Clamp to _COT_MAX to prevent numerical overflow from near-degenerate triangles.
    _COT_MAX = 1e3
    cot0 = np.clip(_cot(v1 - v0, v2 - v0), 0.0, _COT_MAX)  # angle at v0 → edge (v1, v2)
    cot1 = np.clip(_cot(v0 - v1, v2 - v1), 0.0, _COT_MAX)  # angle at v1 → edge (v0, v2)
    cot2 = np.clip(_cot(v0 - v2, v1 - v2), 0.0, _COT_MAX)  # angle at v2 → edge (v0, v1)

    half = 0.5
    row = np.concatenate([
        faces[:, 1], faces[:, 2],
        faces[:, 0], faces[:, 2],
        faces[:, 0], faces[:, 1],
    ])
    col = np.concatenate([
        faces[:, 2], faces[:, 1],
        faces[:, 2], faces[:, 0],
        faces[:, 1], faces[:, 0],
    ])
    dat = np.concatenate([
        -cot0 * half, -cot0 * half,
        -cot1 * half, -cot1 * half,
        -cot2 * half, -cot2 * half,
    ])

    L = scipy.sparse.coo_matrix((dat, (row, col)), shape=(N, N)).tocsr()
    # Diagonal = negative row-sum so that L @ ones = 0 on an interior mesh.
    diag_vals = np.array(-L.sum(axis=1)).flatten()
    L = L + scipy.sparse.diags(diag_vals, format="csr")
    return L


# ── Laplacian deformation (primary deliverable) ───────────────────────────────

def _deform_laplacian(
    src_verts: np.ndarray,
    faces: np.ndarray,
    src_lm_indices: np.ndarray,
    tgt_lm_positions: np.ndarray,
    weight: float = _LANDMARK_WEIGHT,
) -> np.ndarray:
    """Solve a soft-constrained Laplacian system to deform src_verts.

    System per axis:
        (L + Wp) x = (L @ src_verts + Wp @ target)

    where Wp is a diagonal matrix with *weight* at pinned landmark rows and 0
    elsewhere.  Free vertices keep their original Laplacian coordinates (smooth
    shape preservation); pinned vertices are pulled toward their target with
    strength proportional to *weight*.

    Returns (N, 3) deformed vertex positions.
    """
    N = len(src_verts)
    L = _build_cotangent_laplacian(src_verts, faces)

    # Original Laplacian differential coordinates (preserve local curvature).
    delta = L @ src_verts  # (N, 3)

    # Soft-constraint diagonal: add weight at pinned vertex rows.
    pin_diag = np.zeros(N, dtype=np.float64)
    pin_diag[src_lm_indices] = weight
    Wp = scipy.sparse.diags(pin_diag, format="csr")

    # Small regularization (1e-6 * I) ensures the system is non-singular even
    # for isolated / disconnected vertices that would otherwise give all-zero rows.
    reg = 1e-6 * scipy.sparse.eye(N, format="csr")
    A = (L + Wp + reg).tocsr()

    B_pin = np.zeros((N, 3), dtype=np.float64)
    B_pin[src_lm_indices] = tgt_lm_positions * weight
    b = delta + B_pin  # (N, 3)

    # Guard: non-finite RHS (from overflow in L @ src_verts) → bail out early.
    if not np.all(np.isfinite(b)):
        log.warning(
            "kaolin_deformer: Laplacian RHS has non-finite values (degenerate geometry?); "
            "returning original mesh."
        )
        return src_verts.copy()

    new_verts = src_verts.copy()
    for axis in range(3):
        try:
            sol = scipy.sparse.linalg.spsolve(A, b[:, axis])
            if np.all(np.isfinite(sol)):
                new_verts[:, axis] = sol
            else:
                log.warning(
                    "kaolin_deformer: spsolve returned non-finite values for axis %d; "
                    "keeping original.", axis
                )
        except Exception as exc:
            log.warning(
                "kaolin_deformer: spsolve failed for axis %d (%s); keeping original.", axis, exc
            )

    return new_verts


# ── Kaolin NRICP (stretch goal) ───────────────────────────────────────────────

def _deform_kaolin(
    src_verts: np.ndarray,
    faces: np.ndarray,
    src_lm_indices: np.ndarray,
    tgt_lm_positions: np.ndarray,
) -> np.ndarray:
    """Non-rigid ICP deformation via Kaolin (GPU preferred).

    Optional stretch-goal path.  Any exception propagates to the caller,
    which falls back to Laplacian deformation.
    """
    import torch
    import kaolin.ops.mesh as kaolin_mesh  # type: ignore[import]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    verts_t  = torch.from_numpy(src_verts.astype(np.float32)).unsqueeze(0).to(device)
    faces_t  = torch.from_numpy(faces.astype(np.int64)).to(device)
    target_t = torch.from_numpy(tgt_lm_positions.astype(np.float32)).to(device)
    pin_idx  = torch.from_numpy(src_lm_indices.astype(np.int64)).to(device)

    deformed = kaolin_mesh.deform(
        vertices=verts_t,
        faces=faces_t,
        target_vertices=target_t,
        target_vertex_indices=pin_idx,
    )

    return deformed.squeeze(0).detach().cpu().numpy().astype(np.float64)
