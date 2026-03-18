"""Quality scoring for 52-blendshape ARKit face rigs."""
from __future__ import annotations

import numpy as np

# Minimum mean displacement (metres) for each ARKit blendshape to pass.
BLENDSHAPE_THRESHOLDS: dict[str, float] = {
    # Jaw / major mouth
    "jawOpen":    0.0003,
    "jawLeft":    0.0003,
    "jawRight":   0.0003,
    "jawForward": 0.0003,
    # Mouth open/close
    "mouthClose":  0.008,
    "mouthFunnel": 0.008,
    "mouthPucker": 0.008,
    # Mouth corners / smile / frown
    "mouthSmileLeft":   0.0003,
    "mouthSmileRight":  0.0003,
    "mouthFrownLeft":   0.006,
    "mouthFrownRight":  0.006,
    "mouthDimpleLeft":  0.005,
    "mouthDimpleRight": 0.005,
    # Mouth left / right slide
    "mouthLeft":  0.006,
    "mouthRight": 0.006,
    # Lip movements
    "mouthStretchLeft":    0.004,
    "mouthStretchRight":   0.004,
    "mouthRollLower":      0.004,
    "mouthRollUpper":      0.004,
    "mouthShrugLower":     0.004,
    "mouthShrugUpper":     0.004,
    "mouthPressLeft":      0.004,
    "mouthPressRight":     0.004,
    "mouthLowerDownLeft":  0.004,
    "mouthLowerDownRight": 0.004,
    "mouthUpperUpLeft":    0.004,
    "mouthUpperUpRight":   0.004,
    # Eye blinks
    "eyeBlinkLeft":  0.005,
    "eyeBlinkRight": 0.005,
    # Eye look directions
    "eyeLookUpLeft":    0.003,
    "eyeLookUpRight":   0.003,
    "eyeLookDownLeft":  0.003,
    "eyeLookDownRight": 0.003,
    "eyeLookInLeft":    0.003,
    "eyeLookInRight":   0.003,
    "eyeLookOutLeft":   0.003,
    "eyeLookOutRight":  0.003,
    # Eye squint / wide
    "eyeSquintLeft":  0.003,
    "eyeSquintRight": 0.003,
    "eyeWideLeft":    0.003,
    "eyeWideRight":   0.003,
    # Brow
    "browDownLeft":    0.003,
    "browDownRight":   0.003,
    "browInnerUp":     0.003,
    "browOuterUpLeft": 0.003,
    "browOuterUpRight":0.003,
    # Nose
    "noseSneerLeft":  0.002,
    "noseSneerRight": 0.002,
    # Cheek
    "cheekPuff":        0.003,
    "cheekSquintLeft":  0.003,
    "cheekSquintRight": 0.003,
    # Tongue
    "tongueOut": 0.005,
}

# Blendshapes that must pass for the rig to be considered usable at all.
_CRITICAL: frozenset[str] = frozenset({
    "jawOpen",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthClose",
})


def score_blendshape(delta: np.ndarray, threshold: float) -> dict:
    """Score a single blendshape displacement array (N, 3).

    Returns a dict with mean/max displacement, threshold, pass flag, and
    a 0–1 score clipped at 1.0.
    """
    mags = np.linalg.norm(delta, axis=1)
    mean_m = float(mags.mean())
    max_m = float(mags.max())
    score = float(min(mean_m / threshold, 1.0)) if threshold > 0 else 0.0
    return {
        "mean_displacement_m": round(mean_m, 6),
        "mean_displacement_mm": round(mean_m * 1000, 3),
        "max_displacement_m": round(max_m, 6),
        "threshold_m": threshold,
        "pass": mean_m >= threshold,
        "score": round(score, 4),
    }


def score_rig(blendshapes_orig: dict[str, np.ndarray]) -> dict:
    """Score all 52 ARKit blendshapes and return a summary dict."""
    results: dict[str, dict] = {}
    for name, threshold in BLENDSHAPE_THRESHOLDS.items():
        delta = blendshapes_orig.get(name, np.zeros((1, 3)))
        results[name] = score_blendshape(delta, threshold)

    passing = [n for n, r in results.items() if r["pass"]]
    failing = [n for n, r in results.items() if not r["pass"]]
    critical_failures = [n for n in failing if n in _CRITICAL]
    overall_score = float(np.mean([r["score"] for r in results.values()]))

    return {
        "blendshapes": results,
        "overall_score": round(overall_score, 4),
        "pass_c": len(passing),
        "fail_count": len(failing),
        "failing_blendshapes": failing,
        "critical_failures": critical_failures,
        "all_pass": len(failing) == 0,
    }
