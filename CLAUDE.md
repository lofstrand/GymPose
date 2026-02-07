# CLAUDE.md — GymPose Project

## Project Overview

Gymnastics pose detection tool: YOLO pose estimation → angle math → position classification → deduction scoring. Monorepo with a Python backend and React frontend.

## Architecture

```
gympose/
  backend/          # FastAPI (Python 3.x)
    main.py         # API routes, CORS, file upload/download, job cancellation
    video.py        # Video processing pipeline: detect → annotate → snapshots → CSV
    angles.py       # Joint angle computation, position classification (Tuck/Pike/Layout), deduction rules
    detection.py    # YOLO model loading (singleton cache), inference, keypoint extraction, tracker reset
  frontend/         # React 18 + Vite + TailwindCSS
    src/App.jsx     # Main app shell, settings state, upload → process → results flow
    src/components/
      UploadZone.jsx       # Drag-and-drop video upload
      VideoPreview.jsx     # Preview uploaded video, ROI selection
      SettingsPanel.jsx    # Confidence, max persons, overlays, model picker
      ProcessingStatus.jsx # Progress bar, cancel button
      ResultsPanel.jsx     # Form score, snapshot gallery (segment cards), video player, downloads
```

## Key Conventions

- **Backend**: No type hints on return values of internal helpers. Tuples used for snapshot frame data: `(ded_val, annotated_frame, frame_idx, position, ded_desc, track_id)`.
- **Frontend**: Functional components only, hooks for all state. TailwindCSS utility classes inline (no CSS modules). `useMemo` for derived/filtered data.
- **API base**: Frontend hardcodes `http://localhost:8000`. Backend serves results from `/results/` static path.
- **COCO 17 keypoint format** throughout (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles).

## Build & Run

```bash
# Backend
cd gympose/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend
cd gympose/frontend
npm install
npm run dev          # Vite dev server (port 5173)
npx vite build       # Production build → dist/
```

## Snapshot System (Segment Groups)

Snapshots are grouped into **segments** — each segment captures a continuous stretch where a person holds one scored position (Tuck/Pike/Layout). Each segment buffers all scored frames and on flush picks ~`TARGET_FRAMES` (10) evenly-spaced images:

- **entry**: First frame of the position segment (anchor)
- **sample**: Evenly-spaced frames across the segment (fill slots)
- **peak**: Best frame (lowest deduction) — always included (anchor)
- **exit**: Last frame before position changes (anchor)

If the buffer has fewer than `TARGET_FRAMES`, all frames are emitted. Anchors override sample roles. Backend fields per snapshot: `segment_id` (int), `role` ("entry"|"peak"|"sample"|"exit"). Frontend groups by `segment_id`, sorts by frame number, and shows dot indicators for navigation. Only anchor roles (entry/peak/exit) show a label badge; samples show no label. `MAX_SNAPSHOTS = 50` counts segments. `TARGET_FRAMES` constant is in `video.py` near `_flush_segment`.

## Important Patterns

- **Inverted detection**: Rotated-frame inference (180°, 90° CW/CCW) catches mid-somersault poses. Only runs every 3rd frame for performance. These persons get `from_flip = True`.
- **Motion filter**: When no ROI is set, stationary persons (audience) are filtered by keypoint displacement. ROI mode skips this.
- **Deduction timeline**: Per-frame deduction values streamed to frontend for sparkline chart and form grade (A-F).
- **H.264 re-encode**: After processing, video is re-encoded from mp4v to H.264 via imageio-ffmpeg for browser playback.
- **Cancel support**: `cancel_flag` dict checked each frame; raises `CancelledError` for clean abort.

## Style Notes

- Keep changes minimal and focused — don't refactor surrounding code.
- Backend uses `print()` for progress logging (consumed by frontend polling).
- Frontend uses slate/cyan/violet Tailwind palette. Deduction colors: green (0.0), yellow (0.1), orange (0.2), red (0.3).
