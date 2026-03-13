import asyncio
import logging
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
from starlette.background import BackgroundTask

from rigger.aligner import align_icp
from rigger.glb_writer import write_rigged_glb
from rigger.separator import _find_jaw_cutoff_geometric, separate_head_body
from rigger.transfer import (
    ARKIT_BLENDSHAPES,
    BS_SKIN_PATH,
    _claire_deltas_m,
    claire_neutral_m,
    transfer_morph_targets,
)

TMP_DIR = Path("tmp")

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
    TMP_DIR.mkdir(exist_ok=True)
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


@app.post("/rig")
async def rig(
    file: Optional[UploadFile] = File(None),
    temp_id: Optional[str] = Form(None),
    y_back: Optional[float] = Form(None),
    y_front: Optional[float] = Form(None),
) -> FileResponse:
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
    log.info("Rig request: source=%s  y_back=%s  y_front=%s", source_name, y_back, y_front)
    log.info("Read %d bytes.", len(glb_bytes))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        input_path = tmp / "input.glb"
        output_path = tmp / "rigged_output.glb"
        input_path.write_bytes(glb_bytes)

        try:
            log.info("Separating head and body meshes...")
            head_mesh, body_mesh, scene_meta = separate_head_body(
                input_path, y_back=y_back, y_front=y_front,
            )

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
                head_vert_indices=scene_meta.get("head_vert_indices"),
                body_vert_indices=scene_meta.get("body_vert_indices"),
                head_uvs=scene_meta.get("head_uvs"),
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

    # Clean up the temp preview file now that rigging is done.
    if temp_id is not None:
        _delete_file(TMP_DIR / f"{temp_id}.glb")

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


async def _delete_after_delay(path: Path, delay: float = 600.0) -> None:
    await asyncio.sleep(delay)
    _delete_file(path)
