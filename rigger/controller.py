"""Multi-attempt rig controller with automatic retries and human-assist fallback."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from rigger.aligner import align_icp
from rigger.glb_writer import patch_glb_add_morph_targets
from rigger.landmarks import detect_landmarks
from rigger.mouth_slit import cut_mouth_slit
from rigger.separator import separate_head_body
from rigger.transfer import (
    claire_neutral_m,
    transfer_morph_targets,
    transfer_morph_targets_pou_rbf,
)
from rigger.validator import score_rig

log = logging.getLogger("face-rigger.controller")

# ── Configuration ──────────────────────────────────────────────────────────
SCORE_OK: float = 0.7
MAX_ATTEMPTS: int = 3

# Attempt strategies: (use_pou_rbf, rbf_config_or_None, description)
_STRATEGIES: list[tuple[bool, dict | None, str]] = [
    (True,  None,                        "POU-RBF + Landmark NICP"),
    (True,  {"rbf_radius_scale": 2.0},   "POU-RBF (wider radius)"),
    (False, None,                        "Classic ICP + TPS (conservative fallback)"),
]


def run_rig_attempt(
    glb_bytes: bytes,
    y_back: Optional[float],
    y_front: Optional[float],
    use_pou_rbf: bool,
    extra_landmarks: Optional[dict] = None,
    rbf_config: Optional[dict] = None,
) -> tuple[bytes, dict, dict]:
    """Run a single rig attempt with the given configuration.

    Returns (rigged_glb_bytes, validation_data, scene_meta).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        input_path = tmp / "input.glb"
        output_path = tmp / "rigged_output.glb"
        input_path.write_bytes(glb_bytes)

        log.info("Separating head and body meshes...")
        head_mesh, body_mesh, scene_meta = separate_head_body(
            input_path, y_back=y_back, y_front=y_front,
        )

        log.info("Cutting mouth slit...")
        head_mesh = cut_mouth_slit(head_mesh)

        log.info("Detecting MediaPipe landmarks...")
        lm_result = None
        try:
            lm_result = detect_landmarks(head_mesh)
            if lm_result is not None:
                log.info("Landmarks detected — using landmark-NICP alignment.")
            else:
                log.info("Landmark detection returned None — falling back to rigid ICP.")
        except Exception as exc:
            log.warning("Landmark detection raised: %s", exc)

        # Merge extra_landmarks into detected keypoints if provided.
        if extra_landmarks and lm_result is not None:
            _merge_extra_landmarks(lm_result, extra_landmarks)
        elif extra_landmarks and lm_result is None:
            # Build a minimal landmark result from user hints alone.
            lm_result = _build_landmark_result_from_hints(extra_landmarks, head_mesh)

        log.info("Aligning head to Claire neutral via ICP...")
        aligned_head, alignment_meta = align_icp(head_mesh, landmarks=lm_result)

        # Re-detect landmarks on aligned mesh for POU-RBF region mask.
        aligned_lm_result = None
        if use_pou_rbf:
            try:
                aligned_lm_result = detect_landmarks(aligned_head)
                if aligned_lm_result is not None:
                    log.info("Aligned-mesh landmarks detected — POU-RBF enabled.")
                else:
                    log.info("Aligned-mesh landmark detection failed — falling back to classic transfer.")
            except Exception as exc:
                log.warning("Aligned landmark detection raised: %s", exc)

        log.info("Transferring 52 ARKit morph targets...")
        if use_pou_rbf and aligned_lm_result is not None:
            _rbf_radius_scale = (rbf_config or {}).get("rbf_radius_scale", 1.0)
            rigged_head, blendshapes = transfer_morph_targets_pou_rbf(
                aligned_head,
                alignment_meta,
                aligned_lm_result["face_region_mask"],
                rbf_radius_scale=_rbf_radius_scale,
            )
            scene_meta["face_region_mask"] = aligned_lm_result["face_region_mask"]
        else:
            rigged_head, blendshapes = transfer_morph_targets(aligned_head, alignment_meta)
            scene_meta["face_region_mask"] = None

        # Inverse ICP transform: blendshapes back to original model space.
        _scale = float(alignment_meta["scale_factor"])
        _T = np.array(alignment_meta["icp_transformation"], dtype=np.float64)
        _T_inv_rot = np.linalg.inv(_T[:3, :3])
        _inv_scale = 1.0 / _scale if _scale > 1e-8 else 1.0
        blendshapes_orig = {
            name: (disp @ _T_inv_rot.T) * _inv_scale
            for name, disp in blendshapes.items()
        }

        log.info("Patching original GLB with morph-target accessors...")
        patch_glb_add_morph_targets(
            original_glb_bytes=glb_bytes,
            blendshapes=blendshapes_orig,
            output_path=output_path,
            original_head_name=scene_meta.get("original_head_name"),
            head_vert_indices=scene_meta.get("head_vert_indices"),
        )

        validation_data = score_rig(blendshapes_orig)
        log.info(
            "Attempt score=%.2f pass=%d/52 critical_failures=%s",
            validation_data["overall_score"],
            validation_data["pass_c"],
            validation_data["critical_failures"],
        )

        rigged_glb_bytes = output_path.read_bytes()
        return rigged_glb_bytes, validation_data, scene_meta


def rig_with_retries(
    glb_bytes: bytes,
    y_back: Optional[float],
    y_front: Optional[float],
    max_attempts: int = MAX_ATTEMPTS,
    extra_landmarks: Optional[dict] = None,
) -> tuple[bytes, dict, int, bool]:
    """Run the rigging pipeline up to max_attempts times.

    Returns (best_glb_bytes, best_validation_data, best_attempt_index, needs_human_guidance).
    """
    best_glb: Optional[bytes] = None
    best_vd: Optional[dict] = None
    best_score: float = -1.0
    best_crit: int = 999
    best_idx: int = 1
    best_meta: Optional[dict] = None

    strategies = _STRATEGIES[:max_attempts]

    for attempt_idx, (use_pou_rbf, rbf_config, desc) in enumerate(strategies, start=1):
        log.info(
            "=== Rig attempt %d/%d: %s ===",
            attempt_idx, max_attempts, desc,
        )

        try:
            glb_out, vd, meta = run_rig_attempt(
                glb_bytes=glb_bytes,
                y_back=y_back,
                y_front=y_front,
                use_pou_rbf=use_pou_rbf,
                extra_landmarks=extra_landmarks,
                rbf_config=rbf_config,
            )
        except Exception as exc:
            log.warning("Attempt %d failed with exception: %s", attempt_idx, exc)
            continue

        score = vd["overall_score"]
        crit = len(vd["critical_failures"])

        # Track best result: highest score, then fewest critical failures, then earliest attempt.
        if (score > best_score) or (score == best_score and crit < best_crit):
            best_glb = glb_out
            best_vd = vd
            best_score = score
            best_crit = crit
            best_idx = attempt_idx
            best_meta = meta

        log.info(
            "Attempt %d result: score=%.2f critical=%d (best so far: attempt %d score=%.2f)",
            attempt_idx, score, crit, best_idx, best_score,
        )

        # Early exit if good enough.
        if score >= SCORE_OK:
            log.info("Score %.2f >= threshold %.2f — accepting attempt %d.", score, SCORE_OK, attempt_idx)
            return best_glb, best_vd, best_idx, False, best_meta

    if best_glb is None:
        raise RuntimeError("All rig attempts failed.")

    needs_human = best_score < SCORE_OK
    if needs_human:
        log.info(
            "All %d attempts below threshold (best=%.2f < %.2f) — needs human guidance.",
            max_attempts, best_score, SCORE_OK,
        )
    return best_glb, best_vd, best_idx, needs_human, best_meta


# ── Landmark merging helpers ──────────────────────────────────────────────

def _merge_extra_landmarks(lm_result: dict, extra: dict) -> None:
    """Override auto-detected keypoints with user-provided coordinates."""
    kp = lm_result.get("keypoints_3d", {})
    for name in ("nose_tip", "left_eye_outer", "right_eye_outer", "mouth_left", "mouth_right"):
        coords = extra.get(name)
        if coords is not None:
            kp[name] = np.array(coords, dtype=np.float64)
            log.info("Merged extra landmark '%s': %s", name, coords)


def _build_landmark_result_from_hints(extra: dict, head_mesh) -> Optional[dict]:
    """Build a minimal landmark dict from user-provided hints alone."""
    kp: dict[str, np.ndarray] = {}
    for name in ("nose_tip", "left_eye_outer", "right_eye_outer", "mouth_left", "mouth_right"):
        coords = extra.get(name)
        if coords is not None:
            kp[name] = np.array(coords, dtype=np.float64)
    if len(kp) < 3:
        return None
    return {"keypoints_3d": kp, "keypoint_vert_indices": {}, "landmark_3d": {}, "face_region_mask": np.array([])}
