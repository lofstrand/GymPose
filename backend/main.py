"""FastAPI backend for GymPose – gymnastics pose detection."""

import asyncio
import logging
import shutil
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from detection import load_model
from video import process_video, CancelledError

# Suppress the noisy Windows ProactorEventLoop ConnectionResetError
# that fires when browsers disconnect (video seeking, tab close, etc.)
if sys.platform == "win32":
    def _silence_connection_reset(loop, context):
        exc = context.get("exception")
        if isinstance(exc, ConnectionResetError):
            return  # silently ignore
        loop.default_exception_handler(context)

    loop = asyncio.get_event_loop()
    loop.set_exception_handler(_silence_connection_reset)

# Also quiet the noisy uvicorn access logs for partial content requests
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm"}

# Active jobs: job_id -> {"cancelled": bool}
active_jobs: dict = {}

app = FastAPI(title="GymPose – Gymnastics Pose Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Pre-load the YOLO model so the first request isn't slow."""
    try:
        load_model()
        print("YOLO model loaded successfully.")
    except FileNotFoundError as exc:
        print(f"Warning: {exc}")
    except Exception as exc:
        print(f"Warning: could not pre-load model: {exc}")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    conf_threshold: float = Form(0.10),
    max_persons: int = Form(5),
    show_skeleton: str = Form("false"),
    show_angles: str = Form("false"),
    show_position: str = Form("true"),
    show_deduction: str = Form("true"),
    detect_inverted: str = Form("true"),
    model_name: str = Form("yolo26n-pose.pt"),
    target_position: str = Form("auto"),
    roi_x: float = Form(-1),
    roi_y: float = Form(-1),
    roi_w: float = Form(-1),
    roi_h: float = Form(-1),
):
    """Upload a gymnastics video and receive annotated results."""
    # Validate extension
    ext = Path(video.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Clamp settings
    conf_threshold = max(0.1, min(1.0, conf_threshold))
    max_persons = max(1, min(10, max_persons))
    show_skeleton_bool = show_skeleton.lower() in ("true", "1", "yes")
    show_angles_bool = show_angles.lower() in ("true", "1", "yes")
    show_position_bool = show_position.lower() in ("true", "1", "yes")
    show_deduction_bool = show_deduction.lower() in ("true", "1", "yes")
    detect_inverted_bool = detect_inverted.lower() in ("true", "1", "yes")

    # Validate model name (only allow known pose models)
    allowed_models = {
        "yolo26n-pose.pt", "yolo26s-pose.pt", "yolo26m-pose.pt",
        "yolo26l-pose.pt", "yolo26x-pose.pt",
        "yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt",
        "yolo11l-pose.pt", "yolo11x-pose.pt",
        "yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt",
        "yolov8l-pose.pt", "yolov8x-pose.pt",
    }
    if model_name not in allowed_models:
        model_name = "yolo26n-pose.pt"

    # Target position: "auto" means auto-classify, otherwise force a specific position
    target_pos = None
    if target_position.lower() in ("tuck", "pike", "layout"):
        target_pos = target_position.capitalize()

    # ROI: None if not provided, otherwise (x, y, w, h) as 0-1 fractions
    roi = None
    if roi_x >= 0 and roi_y >= 0 and roi_w > 0 and roi_h > 0:
        roi = (
            max(0.0, min(1.0, roi_x)),
            max(0.0, min(1.0, roi_y)),
            max(0.0, min(1.0, roi_w)),
            max(0.0, min(1.0, roi_h)),
        )

    # Save upload
    job_id = uuid.uuid4().hex[:12]
    input_path = UPLOAD_DIR / f"{job_id}{ext}"
    try:
        with open(input_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}")

    # Process
    output_video = RESULTS_DIR / f"{job_id}_annotated.mp4"
    output_csv = RESULTS_DIR / f"{job_id}_data.csv"

    # Register job for cancellation
    active_jobs[job_id] = {"cancelled": False}

    try:
        roi_str = f", roi={roi}" if roi else ""
        print(f"[{job_id}] Starting analysis: {video.filename} (model={model_name}, conf={conf_threshold}, max_persons={max_persons}{roi_str})")

        # Load the requested model
        from detection import load_model
        load_model(model_name)

        # Run in a thread so the event loop stays free for progress/cancel requests
        stats = await asyncio.to_thread(
            process_video,
            str(input_path),
            str(output_video),
            str(output_csv),
            conf_threshold=conf_threshold,
            max_persons=max_persons,
            show_skeleton=show_skeleton_bool,
            show_angles=show_angles_bool,
            show_position=show_position_bool,
            show_deduction=show_deduction_bool,
            roi=roi,
            detect_inverted=detect_inverted_bool,
            target_position=target_pos,
            cancel_flag=active_jobs[job_id],
        )
        print(f"[{job_id}] Done: {stats['frame_count']} frames in {stats['processing_time']}s")
    except CancelledError:
        print(f"[{job_id}] Cancelled by user")
        input_path.unlink(missing_ok=True)
        output_video.unlink(missing_ok=True)
        output_csv.unlink(missing_ok=True)
        raise HTTPException(status_code=499, detail="Cancelled")
    except Exception as exc:
        print(f"[{job_id}] Error: {exc}")
        input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")
    finally:
        active_jobs.pop(job_id, None)

    # Clean up the raw upload
    input_path.unlink(missing_ok=True)

    return {
        "job_id": job_id,
        "video_url": f"/results/{output_video.name}",
        "csv_url": f"/results/{output_csv.name}",
        **stats,
    }


@app.get("/progress/latest")
async def progress_latest():
    """Return progress of the currently running job."""
    if not active_jobs:
        raise HTTPException(status_code=404, detail="No active jobs")
    # Return the most recent job's progress
    job_id = list(active_jobs.keys())[-1]
    info = active_jobs[job_id]
    return {
        "job_id": job_id,
        "frame": info.get("frame", 0),
        "total_frames": info.get("total_frames", 0),
        "elapsed": info.get("elapsed", 0),
        "stage": info.get("stage", "starting"),
    }


@app.post("/cancel/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running analysis job. Use 'latest' to cancel whatever is running."""
    if job_id == "latest":
        # Cancel all active jobs
        for jid, info in active_jobs.items():
            info["cancelled"] = True
            print(f"[{jid}] Cancel requested")
        if active_jobs:
            return {"status": "cancelling"}
        raise HTTPException(status_code=404, detail="No active jobs")

    if job_id in active_jobs:
        active_jobs[job_id]["cancelled"] = True
        print(f"[{job_id}] Cancel requested")
        return {"status": "cancelling"}
    raise HTTPException(status_code=404, detail="Job not found or already finished")


@app.get("/results/{filename}")
async def get_result(filename: str):
    """Serve an annotated video or CSV result file."""
    # Prevent path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    path = RESULTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    type_map = {".mp4": "video/mp4", ".csv": "text/csv", ".jpg": "image/jpeg"}
    media_type = type_map.get(path.suffix, "application/octet-stream")
    return FileResponse(path, media_type=media_type, filename=filename)
