import logging
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from rigger.aligner import align_icp
from rigger.glb_writer import write_rigged_glb
from rigger.separator import separate_head_body
from rigger.transfer import (
    ARKIT_BLENDSHAPES,
    BS_SKIN_PATH,
    _claire_deltas_m,
    claire_neutral_m,
    transfer_morph_targets,
)

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
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Key blendshapes that must have mean displacement > 1mm to pass validation.
_KEY_BLENDSHAPES = ["jawOpen", "eyeBlinkLeft", "eyeBlinkRight", "mouthSmileLeft", "mouthSmileRight"]


@app.on_event("startup")
async def startup_event() -> None:
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


@app.post("/rig")
async def rig(file: UploadFile = File(...)) -> FileResponse:
    if not file.filename or not file.filename.lower().endswith(".glb"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a .glb file.")

    if claire_neutral_m is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Claire blendshape skin not loaded from '{BS_SKIN_PATH}'. "
                "See README for setup instructions."
            ),
        )

    log.info("Received file: %s (%s)", file.filename, file.content_type)
    glb_bytes = await file.read()
    log.info("Read %d bytes from upload.", len(glb_bytes))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        input_path = tmp / "input.glb"
        output_path = tmp / "rigged_output.glb"
        input_path.write_bytes(glb_bytes)

        try:
            log.info("Separating head and body meshes...")
            head_mesh, body_mesh, scene_meta = separate_head_body(input_path)

            log.info("Aligning head to Claire neutral via ICP...")
            aligned_head, alignment_meta = align_icp(head_mesh)

            log.info("Transferring 52 ARKit morph targets...")
            rigged_head, blendshapes = transfer_morph_targets(aligned_head, alignment_meta)

            log.info("Writing rigged GLB with pygltflib morph-target accessors...")
            write_rigged_glb(
                head_verts=np.array(rigged_head.vertices),
                head_faces=np.array(rigged_head.faces),
                blendshapes=blendshapes,
                body_mesh=body_mesh,
                output_path=output_path,
                original_glb_bytes=glb_bytes,
                original_head_name=scene_meta.get("original_head_name"),
                head_alignment_meta=alignment_meta,
                body_parts=scene_meta.get("body_parts"),
            )

        except Exception as exc:
            log.exception("Pipeline failed: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        # ── Output validation ────────────────────────────────────────────────
        log.info("=== Output validation ===")
        all_pass = True
        for name in _KEY_BLENDSHAPES:
            delta = blendshapes.get(name, np.zeros((1, 3)))
            mean_mag = float(np.linalg.norm(delta, axis=1).mean())
            if mean_mag < 0.001:
                log.warning("VALIDATION FAIL: %s mean=%.6fm < 0.001m", name, mean_mag)
                all_pass = False
            else:
                log.info("VALIDATION OK:   %s mean=%.5fm", name, mean_mag)
        if all_pass:
            log.info("All key blendshapes passed validation (mean > 1mm).")
        else:
            log.warning("One or more key blendshapes failed validation — check alignment and scale.")

        # Copy output outside the TemporaryDirectory before it's cleaned up.
        final_path = Path(tempfile.mktemp(suffix=".glb"))
        final_path.write_bytes(output_path.read_bytes())

    log.info("Returning rigged GLB: %s", final_path)
    return FileResponse(
        path=str(final_path),
        media_type="model/gltf-binary",
        filename="rigged_output.glb",
        background=BackgroundTask(_delete_file, final_path),
    )


def _delete_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass
