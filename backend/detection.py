"""YOLO pose model loading, inference, and keypoint extraction."""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from ultralytics import YOLO

_model: Optional[YOLO] = None
_model_path: Optional[str] = None

MODELS_DIR = Path(__file__).parent / "models"


def load_model(model_path: str = "yolo26n-pose.pt") -> YOLO:
    """Load the YOLO pose model (cached singleton).

    Resolves model filenames from the models/ directory.
    Ultralytics auto-downloads weights if not already present.
    """
    global _model, _model_path

    # Resolve to models/ directory if just a filename
    resolved = str(MODELS_DIR / model_path) if not Path(model_path).is_absolute() else model_path

    if _model is not None and _model_path == resolved:
        return _model

    _model = YOLO(resolved)
    _model_path = resolved
    return _model


def reset_tracker():
    """Reset the YOLO tracker state (call between videos)."""
    global _model
    if _model is not None and hasattr(_model, "predictor") and _model.predictor is not None:
        _model.predictor = None


def run_inference(frame: np.ndarray, conf_threshold: float = 0.10, use_tracking: bool = False):
    """Run pose inference on a single frame.

    When use_tracking=True, uses YOLO's built-in tracker (BoT-SORT)
    which maintains person identity across frames and is more robust
    to brief detection failures.

    Returns the raw ultralytics Results list.
    """
    model = load_model()
    if use_tracking:
        results = model.track(frame, conf=conf_threshold, persist=True, verbose=False)
    else:
        results = model(frame, conf=conf_threshold, verbose=False)
    return results


def extract_keypoints(results, max_persons: int = 5, min_box_area: float = 0.005) -> List[Dict]:
    """Extract per-person keypoints from YOLO results.

    Sorts detections by bounding box area (largest first) so prominent
    gymnasts are prioritized over small background figures like audience.

    min_box_area: minimum bbox area as fraction of frame area (filters out
                  small/distant people). Default 0.5% of frame.

    Returns a list of dicts, each with:
      - keypoints: np.ndarray of shape (17, 2)
      - confidences: np.ndarray of shape (17,)
    """
    persons: List[Dict] = []

    if not results or results[0].keypoints is None:
        return persons

    r = results[0]
    kpts = r.keypoints
    xy = kpts.xy.cpu().numpy()  # (N, 17, 2)
    conf = kpts.conf.cpu().numpy() if kpts.conf is not None else None

    # Get bounding boxes to sort by size and filter small detections
    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else None
    if boxes is not None:
        frame_h, frame_w = r.orig_shape
        frame_area = frame_w * frame_h
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights
        # Sort indices by area descending (biggest person first)
        order = np.argsort(-areas)
    else:
        order = np.arange(len(xy))
        areas = None
        frame_area = 1

    # Extract track IDs if available (from model.track())
    track_ids = None
    if r.boxes is not None and r.boxes.id is not None:
        track_ids = r.boxes.id.cpu().numpy().astype(int)

    for idx in order:
        # Skip small detections (audience, background people)
        if areas is not None and areas[idx] / frame_area < min_box_area:
            continue
        person = {
            "keypoints": xy[idx],
            "confidences": conf[idx] if conf is not None else np.ones(17),
        }
        if track_ids is not None:
            person["track_id"] = int(track_ids[idx])
        persons.append(person)
        if len(persons) >= max_persons:
            break

    return persons
