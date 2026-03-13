import logging
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from rigger.aligner import align_icp
from rigger.glb_writer import write_rigged_glb
from rigger.separator import separate_head_body
from rigger.transfer import BS_SKIN_PATH, claire_neutral_m, transfer_morph_targets

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
            head_mesh, body_mesh, _scene_meta = separate_head_body(input_path)

            log.info("Aligning head to Claire neutral via ICP...")
            aligned_head = align_icp(head_mesh)

            log.info("Transferring 52 ARKit morph targets...")
            rigged_head, blendshapes = transfer_morph_targets(aligned_head)

            log.info("Writing rigged GLB with pygltflib morph-target accessors...")
            write_rigged_glb(
                head_verts=np.array(rigged_head.vertices),
                head_faces=np.array(rigged_head.faces),
                blendshapes=blendshapes,
                body_mesh=body_mesh,
                output_path=output_path,
            )

        except Exception as exc:
            log.exception("Pipeline failed: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

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
