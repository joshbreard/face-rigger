# face-rigger

Python FastAPI microservice that auto-rigs a Meshy GLB head with 52 ARKit morph targets for Audio2Face animation.

## What it does

1. Accepts a Meshy-exported GLB via `POST /rig`
2. Detects and separates the head submesh from the body
3. Aligns the head to Claire's neutral mesh using ICP registration (Open3D)
4. Transfers all 52 ARKit blendshapes via nearest-neighbour vertex deformation transfer
5. Re-merges the rigged head with the unmodified body
6. Returns a ready-to-use GLB download (`rigged_output.glb`)

## Requirements

- Python 3.12+
- `assets/bs_skin.npz` — Claire's blendshape skin data (see below)

## Setup

```bash
# 1. Clone the repo
git clone <this-repo>
cd face-rigger

# 2. Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Blendshape skin setup (required)

The pipeline requires `assets/bs_skin.npz`, which contains Claire's neutral
vertex positions and 52 ARKit blendshape displacement vectors (centimetre scale).

```
face-rigger/
└── assets/
    └── bs_skin.npz   ← place it here
```

On first startup the server prints every key found in the file together with
its shape and dtype, so the structure is always visible in the log:

```
=== bs_skin.npz structure ===
  'neutral'                        shape=(V, 3)   dtype=float32
  'bs_deltas'                      shape=(52, V, 3) dtype=float32
=============================
```

The loader recognises the following key names automatically:

| Data | Accepted key names |
|---|---|
| Neutral vertices (cm) | `neutral`, `neutral_verts`, `verts`, `vertices`, `neutral_mesh` |
| Blendshape deltas (cm) | `bs_deltas`, `deltas`, `blendshapes`, `morphs`, `bs` |
| Absolute posed positions (cm) | `poses` (delta = pose − neutral is computed automatically) |

The server logs a clear error on startup if the file is missing, and
`POST /rig` returns HTTP 503 until it is present.

## Scale convention

| Mesh | Unit | Notes |
|---|---|---|
| Claire (`bs_skin.npz`) | centimetres | `templateBBSize ≈ 45.94 cm` |
| Meshy GLB (input) | metres | standard glTF convention |

The loader multiplies Claire's neutral positions and displacement vectors by
`0.01` (cm → m) before building the KD-tree, so the spatial lookup aligns
correctly with the Meshy head.

## Running locally

```bash
uvicorn main:app --reload
```

Open <http://localhost:8000> in your browser to use the drag-and-drop UI.

The API is also directly accessible:

```bash
curl -X POST http://localhost:8000/rig \
  -F "file=@your_model.glb" \
  --output rigged_output.glb
```

## Project structure

```
face-rigger/
├── main.py                  # FastAPI app, routes, startup checks
├── rigger/
│   ├── __init__.py
│   ├── separator.py         # Head/body separation (name-match + bbox fallback)
│   ├── aligner.py           # ICP registration via Open3D (target = Claire neutral)
│   └── transfer.py          # ARKit morph-target transfer (all 52 blendshapes)
├── assets/
│   └── bs_skin.npz          # Claire blendshape skin — place here (not included)
├── static/
│   └── index.html           # Drag-and-drop UI
├── requirements.txt
└── README.md
```

## Deploying to Railway

1. Push this repo to GitHub
2. Create a new Railway project → "Deploy from GitHub repo"
3. Set the start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Include `assets/bs_skin.npz` in the repo (or mount it as a volume)
5. Railway auto-detects Python and installs `requirements.txt`

## Deploying to Render

1. Push this repo to GitHub
2. Create a new Render **Web Service** → connect the repo
3. Runtime: **Python 3**
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Include `assets/bs_skin.npz` in the repo (or mount it as a disk)

## The 52 ARKit blendshapes

The following blendshape names are hardcoded in `rigger/transfer.py` and
preserved exactly in the output GLB so Audio2Face recognises them natively:

```
browDownLeft      browDownRight     browInnerUp       browOuterUpLeft
browOuterUpRight  cheekPuff         cheekSquintLeft   cheekSquintRight
eyeBlinkLeft      eyeBlinkRight     eyeLookDownLeft   eyeLookDownRight
eyeLookInLeft     eyeLookInRight    eyeLookOutLeft    eyeLookOutRight
eyeLookUpLeft     eyeLookUpRight    eyeSquintLeft     eyeSquintRight
eyeWideLeft       eyeWideRight      jawForward        jawLeft
jawOpen           jawRight          mouthClose        mouthDimpleLeft
mouthDimpleRight  mouthFrownLeft    mouthFrownRight   mouthFunnel
mouthLeft         mouthLowerDownLeft mouthLowerDownRight mouthPressLeft
mouthPressRight   mouthPucker       mouthRight        mouthRollLower
mouthRollUpper    mouthShrugLower   mouthShrugUpper   mouthSmileLeft
mouthSmileRight   mouthStretchLeft  mouthStretchRight mouthUpperUpLeft
mouthUpperUpRight noseSneerLeft     noseSneerRight    tongueOut
```
