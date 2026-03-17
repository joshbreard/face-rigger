import asyncio
import logging
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.background import BackgroundTask

import json

from rigger.aligner import align_icp
from rigger.controller import rig_with_retries, SCORE_OK, MAX_ATTEMPTS
from rigger.glb_writer import patch_glb_add_morph_targets
from rigger.landmarks import detect_landmarks
from rigger.mouth_slit import cut_mouth_slit
from rigger.separator import _find_jaw_cutoff_geometric, separate_head_body
from rigger.transfer import (
    ARKIT_BLENDSHAPES,
    BS_SKIN_PATH,
    _claire_deltas_m,
    claire_neutral_m,
    transfer_morph_targets,
    transfer_morph_targets_pou_rbf,
)
from rigger.validator import score_rig

TMP_DIR = Path("tmp")

rig_semaphore: asyncio.Semaphore  # initialised in startup_event

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("face-rigger")

app = FastAPI(title="face-rigger", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Rig-Id", "X-Rig-Pass", "X-Rig-Score",
        "X-Rig-Failures", "X-Rig-Attempt", "X-Rig-Status",
    ],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup_event() -> None:
    global rig_semaphore
    TMP_DIR.mkdir(exist_ok=True)
    max_concurrent = int(os.getenv("MAX_CONCURRENT_RIGS", "2"))
    rig_semaphore = asyncio.Semaphore(max_concurrent)
    log.info("Rig semaphore initialised (MAX_CONCURRENT_RIGS=%d).", max_concurrent)
    if not BS_SKIN_PATH.exists():
        log.error(
            "STARTUP ERROR: Claire blendshape skin not found at '%s'. "
            "Place assets/bs_skin.npz in the project root. "
            "POST /rig will return HTTP 503 until the file is present.",
            BS_SKIN_PATH,
        )
    elif claire_neutral_m is None:
        log.error(
            "STARTUP ERROR: '%s' was found but failed to load. "
            "Check the key-structure printout above for details.",
            BS_SKIN_PATH,
        )
    else:
        log.info(
            "Claire blendshape skin loaded from '%s' (%d vertices).",
            BS_SKIN_PATH,
            len(claire_neutral_m),
        )


@app.get("/health")
async def health() -> JSONResponse:
    """Report pipeline readiness and Claire data loading status."""
    loaded = claire_neutral_m is not None
    n_bs = len(_claire_deltas_m) if _claire_deltas_m else 0
    return JSONResponse({
        "status": "ok" if loaded else "degraded",
        "bs_skin_loaded": loaded,
        "claire_vertex_count": int(len(claire_neutral_m)) if loaded else None,
        "blendshapes_loaded": n_bs,
        "arkit_blendshapes_expected": 52,
    })


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(content=Path("static/index.html").read_text())


@app.post("/preview")
async def preview(file: UploadFile = File(...)) -> JSONResponse:
    """Save upload to a temp file, run geometric neck detection, return plane params."""
    if not file.filename or not file.filename.lower().endswith(".glb"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a .glb file.")

    glb_bytes = await file.read()
    temp_id = str(uuid.uuid4())
    tmp_path = TMP_DIR / f"{temp_id}.glb"
    tmp_path.write_bytes(glb_bytes)

    # Schedule automatic cleanup after 10 minutes.
    asyncio.get_event_loop().create_task(_delete_after_delay(tmp_path, 600.0))

    try:
        import trimesh as _trimesh
        scene_or_mesh = _trimesh.load(str(tmp_path), force="scene", process=False)
        if isinstance(scene_or_mesh, _trimesh.Trimesh):
            all_verts = np.array(scene_or_mesh.vertices)
        else:
            meshes = [
                g for g in scene_or_mesh.geometry.values()
                if isinstance(g, _trimesh.Trimesh)
            ]
            if not meshes:
                raise ValueError("No Trimesh geometry found in GLB.")
            all_verts = np.concatenate([np.array(m.vertices) for m in meshes], axis=0)

        y_min = float(all_verts[:, 1].min())
        y_max = float(all_verts[:, 1].max())
        z_min = float(all_verts[:, 2].min())
        z_max = float(all_verts[:, 2].max())
        total_height = y_max - y_min

        neck_y = _find_jaw_cutoff_geometric(all_verts)
        if neck_y is None:
            neck_y = y_max - 0.25 * total_height  # fallback: 25 % from top

        tilt = 0.04 * total_height
        y_front = float(neck_y)
        y_back = float(neck_y - tilt)

    except Exception as exc:
        log.exception("Preview detection failed: %s", exc)
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    log.info(
        "Preview: temp_id=%s  y_back=%.4f  y_front=%.4f  z=[%.4f, %.4f]",
        temp_id, y_back, y_front, z_min, z_max,
    )
    return JSONResponse({
        "temp_id": temp_id,
        "y_front": round(y_front, 5),
        "y_back":  round(y_back,  5),
        "y_min":   round(y_min,   5),
        "y_max":   round(y_max,   5),
        "z_min":   round(z_min,   5),
        "z_max":   round(z_max,   5),
    })


@app.get("/preview/{temp_id}")
async def get_preview(temp_id: str) -> FileResponse:
    """Serve the temp GLB for Three.js to load during the adjust step."""
    if not re.match(r"^[0-9a-f\-]+$", temp_id):
        raise HTTPException(status_code=400, detail="Invalid temp_id.")
    tmp_path = TMP_DIR / f"{temp_id}.glb"
    if not tmp_path.exists():
        raise HTTPException(status_code=404, detail="Preview file not found or expired.")
    return FileResponse(path=str(tmp_path), media_type="model/gltf-binary")


@app.get("/validate/{rig_id}")
async def validate_rig(rig_id: str) -> JSONResponse:
    """Return per-blendshape validation data (mean displacement + pass/fail) for a rigged GLB."""
    if not re.match(r"^[0-9a-f\-]+$", rig_id):
        raise HTTPException(status_code=400, detail="Invalid rig_id.")
    path = TMP_DIR / f"{rig_id}.validate.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Validation data not found or expired.")
    return JSONResponse(json.loads(path.read_text()))


@app.get("/rig-status/{rig_id}")
async def rig_status(rig_id: str) -> JSONResponse:
    """Return status and validation summary for a rig."""
    if not re.match(r"^[0-9a-f\-]+$", rig_id):
        raise HTTPException(status_code=400, detail="Invalid rig_id.")
    path = TMP_DIR / f"{rig_id}.validate.json"
    if path.exists():
        data = json.loads(path.read_text())
        return JSONResponse({
            "status": "needs_human" if data.get("needs_human") else "done",
            "rig_id": rig_id,
            "overall_score": data.get("overall_score"),
            "critical_failures": data.get("critical_failures", []),
            "attempt_index": data.get("attempt_index"),
        })
    return JSONResponse({"status": "unknown", "rig_id": rig_id})


@app.get("/rig-preview/{rig_id}")
async def rig_preview(rig_id: str, variant: str = "original") -> FileResponse:
    """Serve the stored GLB for a rig_id.

    variant=original (default) returns the unrigged input GLB (for landmark annotation).
    variant=rigged returns the best rigged output GLB.
    """
    if not re.match(r"^[0-9a-f\-]+$", rig_id):
        raise HTTPException(status_code=400, detail="Invalid rig_id.")
    if variant == "rigged":
        glb_path = TMP_DIR / f"{rig_id}.glb"
        if not glb_path.exists():
            raise HTTPException(status_code=404, detail="Rigged GLB not found or expired.")
        return FileResponse(path=str(glb_path), media_type="model/gltf-binary")
    # Default: original GLB for landmark annotation.
    orig_path = TMP_DIR / f"{rig_id}.orig.glb"
    if orig_path.exists():
        return FileResponse(path=str(orig_path), media_type="model/gltf-binary")
    # Fallback to rigged GLB.
    glb_path = TMP_DIR / f"{rig_id}.glb"
    if not glb_path.exists():
        raise HTTPException(status_code=404, detail="GLB not found or expired.")
    return FileResponse(path=str(glb_path), media_type="model/gltf-binary")


class RigHints(BaseModel):
    nose_tip: Optional[list[float]] = None
    left_eye_outer: Optional[list[float]] = None
    right_eye_outer: Optional[list[float]] = None
    mouth_left: Optional[list[float]] = None
    mouth_right: Optional[list[float]] = None


@app.post("/rig", response_model=None)
async def rig(
    file: Optional[UploadFile] = File(None),
    temp_id: Optional[str] = Form(None),
    y_back: Optional[float] = Form(None),
    y_front: Optional[float] = Form(None),
    use_pou_rbf: bool = Form(True),
):
    # Determine the source GLB bytes.
    if temp_id is not None:
        tmp_file = TMP_DIR / f"{temp_id}.glb"
        if not tmp_file.exists():
            raise HTTPException(status_code=404, detail="Temp file not found — please re-upload.")
        glb_bytes = tmp_file.read_bytes()
    elif file is not None:
        if not file.filename or not file.filename.lower().endswith(".glb"):
            raise HTTPException(status_code=400, detail="Uploaded file must be a .glb file.")
        glb_bytes = await file.read()
    else:
        raise HTTPException(status_code=400, detail="Provide a .glb file or a temp_id.")

    if claire_neutral_m is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Claire blendshape skin not loaded from '{BS_SKIN_PATH}'. "
                "See README for setup instructions."
            ),
        )

    source_name = f"temp:{temp_id}" if temp_id else (file.filename if file else "unknown")
    log.info(
        "Rig request: source=%s  y_back=%s  y_front=%s  use_pou_rbf=%s",
        source_name, y_back, y_front, use_pou_rbf,
    )
    log.info("Read %d bytes.", len(glb_bytes))

    def _run_pipeline() -> tuple:
        best_glb, validation_data, attempt_idx, needs_human = rig_with_retries(
            glb_bytes=glb_bytes,
            y_back=y_back,
            y_front=y_front,
        )

        rig_id = str(uuid.uuid4())

        # Persist the rigged GLB for potential retry.
        glb_path = TMP_DIR / f"{rig_id}.glb"
        glb_path.write_bytes(best_glb)

        # Also persist the original GLB bytes for re-rigging with hints.
        orig_path = TMP_DIR / f"{rig_id}.orig.glb"
        orig_path.write_bytes(glb_bytes)

        # Persist validation JSON.
        validate_path = TMP_DIR / f"{rig_id}.validate.json"
        validate_meta = {
            **validation_data,
            "y_back": y_back,
            "y_front": y_front,
            "attempt_index": attempt_idx,
            "needs_human": needs_human,
        }
        validate_path.write_text(json.dumps(validate_meta))

        return validation_data, rig_id, attempt_idx, needs_human

    async with rig_semaphore:
        try:
            loop = asyncio.get_event_loop()
            validation_data, rig_id, attempt_idx, needs_human = await loop.run_in_executor(
                None, _run_pipeline
            )
        except Exception as exc:
            log.exception("Pipeline failed: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Schedule cleanup of all rig files.
    for suffix in (".glb", ".orig.glb", ".validate.json", ".hints.json"):
        p = TMP_DIR / f"{rig_id}{suffix}"
        asyncio.get_event_loop().create_task(_delete_after_delay(p, 600.0))

    # Clean up the temp preview file now that rigging is done.
    if temp_id is not None:
        _delete_file(TMP_DIR / f"{temp_id}.glb")

    all_pass = validation_data["all_pass"]
    overall_score = validation_data["overall_score"]
    failures_header = ",".join(validation_data["failing_blendshapes"])

    expose = "X-Rig-Id, X-Rig-Pass, X-Rig-Score, X-Rig-Failures, X-Rig-Attempt, X-Rig-Status"

    if needs_human:
        log.info("Rig needs human guidance: rig_id=%s  score=%.2f", rig_id, overall_score)
        return JSONResponse(
            content={
                "rig_id": rig_id,
                "status": "needs_human",
                "overall_score": overall_score,
                "critical_failures": validation_data["critical_failures"],
                "failing_blendshapes": validation_data["failing_blendshapes"],
                "attempts": attempt_idx,
            },
            headers={
                "X-Rig-Id": rig_id,
                "X-Rig-Status": "needs_human",
                "X-Rig-Score": f"{overall_score:.2f}",
                "X-Rig-Attempt": str(attempt_idx),
                "Access-Control-Expose-Headers": expose,
            },
        )

    glb_path = TMP_DIR / f"{rig_id}.glb"
    log.info("Returning rigged GLB: %s  rig_id=%s  attempt=%d", glb_path, rig_id, attempt_idx)
    return FileResponse(
        path=str(glb_path),
        media_type="model/gltf-binary",
        filename="rigged_output.glb",
        headers={
            "X-Rig-Id": rig_id,
            "X-Rig-Pass": str(all_pass).lower(),
            "X-Rig-Score": f"{overall_score:.2f}",
            "X-Rig-Failures": failures_header,
            "X-Rig-Attempt": str(attempt_idx),
            "X-Rig-Status": "done",
            "Access-Control-Expose-Headers": expose,
        },
    )


@app.post("/rig-hints/{rig_id}")
async def rig_hints(rig_id: str, hints: RigHints) -> JSONResponse:
    """Save user-provided landmark hints for a given rig_id."""
    if not re.match(r"^[0-9a-f\-]+$", rig_id):
        raise HTTPException(status_code=400, detail="Invalid rig_id.")
    validate_path = TMP_DIR / f"{rig_id}.validate.json"
    if not validate_path.exists():
        raise HTTPException(status_code=404, detail="Rig not found or expired.")

    hints_path = TMP_DIR / f"{rig_id}.hints.json"
    hints_data = hints.model_dump(exclude_none=True)
    hints_path.write_text(json.dumps(hints_data))
    log.info("Saved hints for rig_id=%s: %s", rig_id, list(hints_data.keys()))
    return JSONResponse({"status": "ok", "rig_id": rig_id, "landmarks_provided": list(hints_data.keys())})


@app.post("/rig-retry/{rig_id}", response_model=None)
async def rig_retry(rig_id: str):
    """Re-run rigging with user-provided landmark hints."""
    if not re.match(r"^[0-9a-f\-]+$", rig_id):
        raise HTTPException(status_code=400, detail="Invalid rig_id.")

    orig_path = TMP_DIR / f"{rig_id}.orig.glb"
    if not orig_path.exists():
        raise HTTPException(status_code=404, detail="Original GLB not found or expired.")

    hints_path = TMP_DIR / f"{rig_id}.hints.json"
    if not hints_path.exists():
        raise HTTPException(status_code=400, detail="No hints provided. POST /rig-hints/{rig_id} first.")

    validate_path = TMP_DIR / f"{rig_id}.validate.json"
    if not validate_path.exists():
        raise HTTPException(status_code=404, detail="Rig metadata not found or expired.")

    glb_bytes = orig_path.read_bytes()
    hints_data = json.loads(hints_path.read_text())
    validate_meta = json.loads(validate_path.read_text())
    y_back = validate_meta.get("y_back")
    y_front = validate_meta.get("y_front")

    log.info("Rig retry: rig_id=%s  hints=%s", rig_id, list(hints_data.keys()))

    if claire_neutral_m is None:
        raise HTTPException(status_code=503, detail="Claire blendshape skin not loaded.")

    def _run_retry() -> tuple:
        best_glb, validation_data, attempt_idx, needs_human = rig_with_retries(
            glb_bytes=glb_bytes,
            y_back=y_back,
            y_front=y_front,
            extra_landmarks=hints_data,
        )

        # Update the stored files for this rig_id.
        glb_out_path = TMP_DIR / f"{rig_id}.glb"
        glb_out_path.write_bytes(best_glb)

        new_meta = {
            **validation_data,
            "y_back": y_back,
            "y_front": y_front,
            "attempt_index": attempt_idx,
            "needs_human": needs_human,
        }
        validate_path.write_text(json.dumps(new_meta))
        log.info("Retry complete: rig_id=%s score=%.2f needs_human=%s", rig_id, validation_data["overall_score"], needs_human)

        return validation_data, attempt_idx, needs_human

    async with rig_semaphore:
        try:
            loop = asyncio.get_event_loop()
            validation_data, attempt_idx, needs_human = await loop.run_in_executor(
                None, _run_retry
            )
        except Exception as exc:
            log.exception("Retry pipeline failed: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    all_pass = validation_data["all_pass"]
    overall_score = validation_data["overall_score"]
    failures_header = ",".join(validation_data["failing_blendshapes"])
    expose = "X-Rig-Id, X-Rig-Pass, X-Rig-Score, X-Rig-Failures, X-Rig-Attempt, X-Rig-Status"

    if needs_human:
        return JSONResponse(
            content={
                "rig_id": rig_id,
                "status": "needs_human",
                "overall_score": overall_score,
                "critical_failures": validation_data["critical_failures"],
                "failing_blendshapes": validation_data["failing_blendshapes"],
                "attempts": attempt_idx,
            },
            headers={
                "X-Rig-Id": rig_id,
                "X-Rig-Status": "needs_human",
                "X-Rig-Score": f"{overall_score:.2f}",
                "X-Rig-Attempt": str(attempt_idx),
                "Access-Control-Expose-Headers": expose,
            },
        )

    glb_path = TMP_DIR / f"{rig_id}.glb"
    return FileResponse(
        path=str(glb_path),
        media_type="model/gltf-binary",
        filename="rigged_output.glb",
        headers={
            "X-Rig-Id": rig_id,
            "X-Rig-Pass": str(all_pass).lower(),
            "X-Rig-Score": f"{overall_score:.2f}",
            "X-Rig-Failures": failures_header,
            "X-Rig-Attempt": str(attempt_idx),
            "X-Rig-Status": "done",
            "Access-Control-Expose-Headers": expose,
        },
    )


def _delete_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


async def _delete_after_delay(path: Path, delay: float = 600.0) -> None:
    await asyncio.sleep(delay)
    _delete_file(path)
