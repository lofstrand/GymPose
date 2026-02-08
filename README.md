# GymPose – Gymnastics Pose Detection

Full-stack web application that uses YOLO Pose models to detect and analyze gymnastics positions in uploaded videos. Provides per-frame joint angle computation, automatic position classification (Tuck/Pike/Layout), deduction scoring, and grouped snapshot galleries with entry/peak/exit views.

## Features

- Upload gymnastics videos (MP4, MOV, AVI, WebM)
- YOLO Pose inference with support for YOLO26, YOLO11, and YOLOv8 model families
- Inverted pose detection via rotated-frame inference (180°, 90° CW/CCW) for mid-somersault analysis
- 8 joint angles computed per person: elbows, shoulders, hips, knees
- Automatic position classification: Tuck, Pike, Layout, Split, Neutral
- Form deduction scoring with grade (A–F), timeline sparkline, and breakdown chart
- Grouped snapshot segments: entry, sampled intermediates, peak (best form), and exit per position hold
- Region of Interest (ROI) selection for focusing on specific athletes
- Annotated output video with optional skeleton overlay, angle labels, and position names
- Downloadable CSV with per-frame keypoint coordinates, angles, and labels
- Configurable model, confidence threshold, max persons, and overlay toggles

## Prerequisites

- Python 3.9+
- Node.js 18+
- YOLO pose model weights are downloaded automatically on first use by Ultralytics

## Setup

### Backend

```bash
cd backend

# Create and activate a virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start the dev server
npm run dev
```

The frontend runs on http://localhost:5173 and the backend on http://localhost:8000.

## Usage

1. Open http://localhost:5173 in your browser.
2. Drag & drop a gymnastics video or click to browse.
3. Adjust settings in the sidebar (model, confidence, target position, overlays, ROI).
4. Click **Analyze Poses** and wait for processing to complete.
5. Review the form analysis grade, deduction timeline, and snapshot gallery.
6. Click segment cards to browse entry → peak → exit frames; use dot indicators to navigate within a segment.
7. Download the annotated video and/or CSV data.

## Available Models

| Family | Sizes | Notes |
|--------|-------|-------|
| **YOLO26** | n, s, m, l, x | Latest (2026). NMS-free, edge-optimized. Default: `yolo26n-pose.pt` |
| YOLO11 | n, s, m, l, x | Solid all-rounder |
| YOLOv8 | n, s, m, l, x | Legacy, widely tested |

Model weights are downloaded automatically by Ultralytics into `backend/models/` on first use. Larger models are more accurate but slower.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/analyze` | Upload video for pose analysis |
| GET | `/results/{filename}` | Retrieve annotated video, CSV, or snapshot image |

### POST /analyze

Multipart form data fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `video` | file | — | Video file (required) |
| `conf_threshold` | float | 0.10 | Person detection confidence (0.1–1) |
| `max_persons` | int | 5 | Max people to track per frame (1–10) |
| `show_skeleton` | string | "false" | Draw skeleton overlay on output |
| `show_angles` | string | "false" | Draw angle labels on output |
| `show_position` | string | "true" | Draw position name above person |
| `show_deduction` | string | "true" | Draw deduction badge |
| `detect_inverted` | string | "true" | Run rotated-frame detection for mid-air poses |
| `model_name` | string | "yolo26n-pose.pt" | YOLO model to use |
| `target_position` | string | "auto" | Force a specific position (tuck/pike/layout) or auto-classify |
| `roi` | string | "" | Region of interest as "x,y,w,h" fractions (0–1) |

## Snapshot System

Each position hold is captured as a **segment** with multiple frames:

- **Entry** — first frame the position is detected
- **Samples** — evenly-spaced intermediate frames (~10 per segment)
- **Peak** — best form frame (lowest deduction)
- **Exit** — last frame before the position changes

The UI groups these into navigable cards with dot indicators. Filters allow viewing by position type or person.

## Notes

- Processing time depends on video length and GPU availability. All YOLO Pose models run significantly faster on a CUDA-enabled GPU.
- The annotated video is automatically re-encoded to H.264 for browser playback.
- The CSV contains one row per person per frame with all 17 keypoint coordinates, 8 joint angles, and the classified position.

## Project Structure

```
├── backend/
│   ├── main.py          # FastAPI application and endpoints
│   ├── detection.py     # YOLO model loading and inference (singleton cache)
│   ├── angles.py        # Joint angle math, position classification, deduction rules
│   ├── video.py         # Video processing pipeline, annotation, snapshot segments
│   ├── models/          # YOLO .pt weights (auto-downloaded, git-ignored)
│   ├── uploads/         # Uploaded videos (git-ignored)
│   ├── results/         # Annotated videos, CSVs, snapshots (git-ignored)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main application component, settings state
│   │   ├── main.jsx             # React entry point
│   │   ├── index.css            # Tailwind directives and custom styles
│   │   └── components/
│   │       ├── UploadZone.jsx       # Drag-and-drop video upload
│   │       ├── VideoPreview.jsx     # Preview with ROI selection
│   │       ├── SettingsPanel.jsx    # Model picker, overlays, target position
│   │       ├── ProcessingStatus.jsx # Progress bar, cancel support
│   │       └── ResultsPanel.jsx     # Form score, snapshot gallery, video player
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── postcss.config.js
│   └── tailwind.config.js
├── CLAUDE.md
└── README.md
```
