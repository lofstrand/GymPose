"""Video I/O, frame-level processing pipeline, and annotation drawing."""

import csv
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class CancelledError(Exception):
    """Raised when a processing job is cancelled by the user."""

import cv2
import numpy as np

from detection import run_inference, extract_keypoints, reset_tracker
from angles import (
    compute_joint_angles,
    classify_position,
    compute_deduction,
    KEYPOINT_NAMES,
    JOINT_ANGLES,
)

# Drawing colours (BGR)
ANGLE_COLOR = (0, 255, 255)      # cyan-ish yellow
POSITION_COLOR = (0, 240, 120)   # green
BG_COLOR = (0, 0, 0)


def _draw_angle_label(frame: np.ndarray, point: np.ndarray, name: str, value: float):
    """Render an angle label next to the given joint."""
    x, y = int(point[0]), int(point[1])
    short = name.replace("left_", "L ").replace("right_", "R ").replace("_", " ").title()
    label = f"{short}: {value:.0f} deg"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.42, 1
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    # background rectangle
    cv2.rectangle(frame, (x + 6, y - th - 6), (x + tw + 12, y + 2), BG_COLOR, -1)
    cv2.putText(frame, label, (x + 8, y - 2), font, scale, ANGLE_COLOR, thickness, cv2.LINE_AA)


DEDUCTION_COLORS = {
    0.0: (0, 200, 0),     # green
    0.1: (0, 220, 220),   # yellow
    0.2: (0, 140, 255),   # orange
    0.3: (0, 0, 255),     # red
}


def _get_label_anchor(keypoints: np.ndarray):
    """Get anchor point above the person's shoulders. Returns (cx, cy) or None."""
    ls, rs = keypoints[5], keypoints[6]
    anchors = [p for p in (ls, rs) if not np.all(p == 0)]
    if not anchors:
        return None
    cx, cy = np.mean(anchors, axis=0).astype(int)
    return cx, cy - 40


def _draw_position_label(frame: np.ndarray, keypoints: np.ndarray, position: str):
    """Draw position name above the person."""
    anchor = _get_label_anchor(keypoints)
    if not anchor:
        return
    cx, cy = anchor
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.6, 2
    (tw, th), _ = cv2.getTextSize(position, font, scale, thickness)
    ox = cx - tw // 2
    cv2.rectangle(frame, (ox - 3, cy - th - 4), (ox + tw + 3, cy + 3), BG_COLOR, -1)
    cv2.putText(frame, position, (ox, cy), font, scale, POSITION_COLOR, thickness, cv2.LINE_AA)


def _draw_deduction_label(frame: np.ndarray, keypoints: np.ndarray,
                          deduction: float, show_position: bool):
    """Draw deduction badge. Offset depends on whether position label is also shown."""
    anchor = _get_label_anchor(keypoints)
    if not anchor:
        return
    cx, cy = anchor
    # Shift down if position label is also drawn
    if show_position:
        cy += 22
    ded_color = DEDUCTION_COLORS.get(deduction, (0, 200, 0))
    font = cv2.FONT_HERSHEY_SIMPLEX
    ded_label = f"{deduction:.1f}"
    scale, thickness = 0.55, 2
    (dw, dh), _ = cv2.getTextSize(ded_label, font, scale, thickness)
    dox = cx - dw // 2
    cv2.rectangle(frame, (dox - 3, cy - dh - 4), (dox + dw + 3, cy + 3), BG_COLOR, -1)
    cv2.putText(frame, ded_label, (dox, cy), font, scale, ded_color, thickness, cv2.LINE_AA)


MIN_MOTION_PX = 5.0  # avg keypoint displacement to count as "moving"


def _reencode_to_h264(input_path: str) -> bool:
    """Re-encode mp4v video to H.264 for browser playback.

    Uses imageio-ffmpeg's bundled ffmpeg binary. Overwrites the original file.
    Returns True on success, False if re-encoding is unavailable.
    """
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg = get_ffmpeg_exe()
    except ImportError:
        return False

    tmp_path = input_path + ".tmp.mp4"
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", input_path,
             "-c:v", "libx264", "-preset", "fast",
             "-crf", "23", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart",
             tmp_path],
            capture_output=True, check=True, timeout=300,
        )
        os.replace(tmp_path, input_path)
        return True
    except Exception as exc:
        print(f"  Re-encode failed: {exc}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False
ROI_COLOR = (238, 211, 34)  # cyan-ish in BGR
ROI_THICKNESS = 2

# COCO skeleton connections for drawing our own skeleton (used for flipped detections)
SKELETON_CONNECTIONS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # shoulders + arms
    (5, 11), (6, 12), (11, 12),                   # torso
    (11, 13), (13, 15), (12, 14), (14, 16),       # legs
]
SKELETON_COLOR = (255, 128, 0)  # orange for flipped detections
KEYPOINT_COLOR = (0, 255, 128)


def _draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, color=SKELETON_COLOR):
    """Draw a skeleton overlay for a person (used for flip-detected persons)."""
    for i, j in SKELETON_CONNECTIONS:
        pt1, pt2 = keypoints[i], keypoints[j]
        if np.all(pt1 == 0) or np.all(pt2 == 0):
            continue
        cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])),
                 color, 2, cv2.LINE_AA)
    for pt in keypoints:
        if np.all(pt == 0):
            continue
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, KEYPOINT_COLOR, -1, cv2.LINE_AA)


def _merge_extra_persons(original: list, extra: list, min_dist: float = 50.0) -> list:
    """Merge extra detections (from rotated frames) into original list, skipping duplicates."""
    merged = list(original)
    for fp in extra:
        fp_valid = fp["keypoints"][fp["keypoints"].sum(axis=1) > 0]
        if len(fp_valid) == 0:
            continue
        fp_c = fp_valid.mean(axis=0)

        is_dup = False
        for op in merged:
            op_valid = op["keypoints"][op["keypoints"].sum(axis=1) > 0]
            if len(op_valid) == 0:
                continue
            if np.linalg.norm(fp_c - op_valid.mean(axis=0)) < min_dist:
                is_dup = True
                break

        if not is_dup:
            fp["from_flip"] = True
            merged.append(fp)
    return merged


def _detect_rotated(frame: np.ndarray, rotation, conf_threshold: float,
                    orig_w: int, orig_h: int, max_persons: int) -> list:
    """Run inference on a rotated frame and transform keypoints back to original coords."""
    rotated = cv2.rotate(frame, rotation)
    rot_results = run_inference(rotated, conf_threshold, use_tracking=False)
    rot_persons = extract_keypoints(rot_results, max_persons=max_persons)

    for p in rot_persons:
        kpts = p["keypoints"]
        valid = kpts.sum(axis=1) > 0

        if rotation == cv2.ROTATE_180:
            kpts[valid, 0] = orig_w - kpts[valid, 0]
            kpts[valid, 1] = orig_h - kpts[valid, 1]
        elif rotation == cv2.ROTATE_90_CLOCKWISE:
            # Rotated frame is (H, W). Point (rx, ry) → original (ry, H-1-rx)
            new_x = kpts[valid, 1].copy()
            new_y = orig_h - 1 - kpts[valid, 0].copy()
            kpts[valid, 0] = new_x
            kpts[valid, 1] = new_y
        elif rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
            # Rotated frame is (H, W). Point (rx, ry) → original (W-1-ry, rx)
            new_x = orig_w - 1 - kpts[valid, 1].copy()
            new_y = kpts[valid, 0].copy()
            kpts[valid, 0] = new_x
            kpts[valid, 1] = new_y

    return rot_persons


def _person_in_roi(keypoints: np.ndarray, roi_px: Tuple[int, int, int, int]) -> bool:
    """Check if a person's keypoint centroid falls inside the ROI rectangle.

    roi_px = (x1, y1, x2, y2) in pixel coordinates.
    """
    valid = keypoints[keypoints.sum(axis=1) > 0]
    if len(valid) == 0:
        return False
    cx, cy = valid.mean(axis=0)
    x1, y1, x2, y2 = roi_px
    return x1 <= cx <= x2 and y1 <= cy <= y2


def _draw_roi(frame: np.ndarray, roi_px: Tuple[int, int, int, int]):
    """Draw the ROI rectangle on the frame."""
    x1, y1, x2, y2 = roi_px
    cv2.rectangle(frame, (x1, y1), (x2, y2), ROI_COLOR, ROI_THICKNESS)
    cv2.putText(frame, "ROI", (x1 + 6, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, ROI_COLOR, 2, cv2.LINE_AA)


def _match_motion(current_kpts: np.ndarray, prev_kpts_list: list) -> float:
    """Compute motion of a person by matching to nearest person in previous frame.

    Returns average keypoint displacement in pixels. Returns inf for the
    first frame (so everyone passes the filter initially).
    """
    if not prev_kpts_list:
        return float("inf")  # first frame — let everyone through

    # Centroid of valid (non-zero) keypoints
    valid_c = current_kpts.sum(axis=1) > 0
    if not valid_c.any():
        return 0.0
    centroid = current_kpts[valid_c].mean(axis=0)

    best_motion = 0.0
    best_dist = float("inf")
    for prev_kpts in prev_kpts_list:
        valid_p = prev_kpts.sum(axis=1) > 0
        if not valid_p.any():
            continue
        prev_centroid = prev_kpts[valid_p].mean(axis=0)
        dist = np.linalg.norm(centroid - prev_centroid)
        if dist < best_dist:
            best_dist = dist
            # Average displacement of mutually-visible keypoints
            both = valid_c & valid_p
            if both.any():
                best_motion = float(np.mean(
                    np.linalg.norm(current_kpts[both] - prev_kpts[both], axis=1)
                ))
    return best_motion


TARGET_FRAMES = 10  # aim for this many images per segment (will use all if buffer is smaller)


def _flush_segment(seg, snapshots, results_dir, job_prefix):
    """Flush a segment, producing ~TARGET_FRAMES evenly-spaced images.

    Guaranteed anchors: entry (first), peak (best deduction), exit (last).
    Remaining slots filled by evenly spacing across the buffer.
    """
    entry = seg["entry"]
    best = seg["best"]
    buffer = seg["buffer"]  # list of frame_tuples, in frame order

    if not buffer:
        return

    # Build the set of frames to emit: {frame_idx: (frame_tuple, role)}
    emit = {}  # frame_idx -> (tuple, role)

    # If buffer is small enough, just emit everything
    if len(buffer) <= TARGET_FRAMES:
        for ft in buffer:
            emit[ft[2]] = (ft, "sample")
    else:
        # Evenly pick TARGET_FRAMES indices from the buffer
        step = (len(buffer) - 1) / (TARGET_FRAMES - 1)
        for i in range(TARGET_FRAMES):
            idx = int(round(i * step))
            ft = buffer[idx]
            if ft[2] not in emit:
                emit[ft[2]] = (ft, "sample")

    # Anchor overrides — entry, peak, exit always get their proper role
    emit[entry[2]] = (entry, "entry")

    if best is not None:
        emit[best[2]] = (best, "peak")

    exit_frame = buffer[-1]
    if exit_frame[2] not in emit:
        emit[exit_frame[2]] = (exit_frame, "exit")
    elif exit_frame[2] != entry[2] and (best is None or exit_frame[2] != best[2]):
        emit[exit_frame[2]] = (exit_frame, "exit")

    # Single-frame segment: just label it peak
    if len(emit) == 1:
        only_key = next(iter(emit))
        ft, _ = emit[only_key]
        emit[only_key] = (ft, "peak")

    # Sort by frame number and write
    for fidx in sorted(emit):
        sf, role = emit[fidx]
        snap_name = f"{job_prefix}_snap_{len(snapshots)}.jpg"
        snap_path = f"{results_dir}/{snap_name}"
        cv2.imwrite(snap_path, sf[1])
        snapshots.append({
            "filename": snap_name,
            "frame": sf[2],
            "position": sf[3],
            "deduction": sf[0],
            "description": sf[4],
            "person_id": sf[5],
            "url": f"/results/{snap_name}",
            "segment_id": seg["seg_id"],
            "role": role,
        })


def process_video(
    input_path: str,
    output_video_path: str,
    output_csv_path: str,
    conf_threshold: float = 0.10,
    max_persons: int = 5,
    show_skeleton: bool = True,
    show_angles: bool = False,
    show_position: bool = True,
    show_deduction: bool = True,
    roi: Optional[Tuple[float, float, float, float]] = None,
    detect_inverted: bool = True,
    target_position: Optional[str] = None,
    cancel_flag: Optional[Dict[str, Any]] = None,
) -> Dict:
    """Full processing pipeline: detect, annotate, write video + CSV.

    Returns a stats dict with frame_count, fps, total_persons_detected,
    processing_time, width, height, snapshots.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Failed to initialise video writer – check OpenCV codec support.")

    # CSV setup
    csv_headers = ["frame", "person_id"]
    for name in KEYPOINT_NAMES:
        csv_headers += [f"{name}_x", f"{name}_y"]
    for name in JOINT_ANGLES:
        csv_headers.append(f"{name}_angle")
    csv_headers.append("position")

    csv_file = open(output_csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(csv_headers)

    # Snapshot tracking: per-person, buffer all frames per segment
    results_dir = str(Path(output_video_path).parent)
    job_prefix = Path(output_video_path).stem.replace("_annotated", "")
    per_person_segments = {}  # track_id -> {"entry": tuple|None, "best": tuple|None, "buffer": list, "seg_id": int, "prev_pos": str}
    segment_counter = 0
    snapshots = []           # list of saved snapshot metadata
    MAX_SNAPSHOTS = 50       # max segments (images per segment varies with interval sampling)

    # Convert ROI from 0-1 fractions to pixel coordinates
    roi_px = None
    if roi:
        rx, ry, rw, rh = roi
        roi_px = (
            int(rx * width),
            int(ry * height),
            int((rx + rw) * width),
            int((ry + rh) * height),
        )
        print(f"  ROI set: ({roi_px[0]},{roi_px[1]}) -> ({roi_px[2]},{roi_px[3]})")

    # Motion tracking: previous frame keypoints for all detected persons
    prev_kpts_list = []

    # Reset YOLO tracker state for this video
    reset_tracker()

    # Per-frame deduction timeline for form analysis
    deduction_timeline = []  # [{frame, deduction, position, hip_avg, knee_avg}]

    frame_idx = 0
    total_persons = 0
    start = time.time()

    while True:
        # Check for cancellation
        if cancel_flag and cancel_flag.get("cancelled"):
            cap.release()
            writer.release()
            csv_file.close()
            raise CancelledError("Job cancelled by user")

        ret, frame = cap.read()
        if not ret:
            break

        # Use tracking mode for temporal consistency
        results = run_inference(frame, conf_threshold, use_tracking=True)
        # Draw skeleton from YOLO (upright detections) if enabled
        if show_skeleton:
            annotated = results[0].plot(boxes=False)
        else:
            annotated = frame.copy()

        # Extract more candidates than max_persons so we can filter
        all_persons = extract_keypoints(results, max_persons=max(max_persons, 10))

        # Run inference on rotated frames to detect mid-somersault poses
        if detect_inverted:
            fetch_max = max(max_persons, 10)
            for rotation in [cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                extra = _detect_rotated(frame, rotation, conf_threshold, width, height, fetch_max)
                all_persons = _merge_extra_persons(all_persons, extra)

        # When ROI is set, it already filters the area — skip motion filter
        if roi_px:
            candidates = [p for p in all_persons if _person_in_roi(p["keypoints"], roi_px)]
        else:
            # No ROI — use motion filter to exclude seated audience
            for person in all_persons:
                person["motion"] = _match_motion(person["keypoints"], prev_kpts_list)
            candidates = [p for p in all_persons if p["motion"] >= MIN_MOTION_PX]

        # Keep prev_kpts for next frame BEFORE filtering
        prev_kpts_list = [p["keypoints"].copy() for p in all_persons]

        # Sort by bounding area (largest/closest first) then cap
        candidates.sort(
            key=lambda p: np.sum(p["keypoints"].sum(axis=1) > 0), reverse=True
        )
        persons = candidates[:max_persons]

        # Collect persons this frame for snapshots & timeline
        frame_inverted = []    # inverted-only (for deduction timeline)
        frame_all_scored = []  # all persons (for snapshot buffer — denser coverage)

        for pid, person in enumerate(persons):
            kpts = person["keypoints"]
            confs = person["confidences"]
            track_id = person.get("track_id", pid)
            is_inverted = person.get("from_flip", False)
            angles = compute_joint_angles(kpts, confs)
            position = classify_position(angles)
            ded_val, ded_desc = compute_deduction(position, angles, target_position)

            # Draw skeleton for flip-detected persons (YOLO didn't draw one)
            if is_inverted and show_skeleton:
                _draw_skeleton(annotated, kpts)

            # Optional overlays — only show for inverted + matching target position
            matches_target = (not target_position) or position == target_position
            if show_angles and is_inverted and matches_target:
                for aname, (_, v_idx, _) in JOINT_ANGLES.items():
                    if angles[aname] is not None and not np.all(kpts[v_idx] == 0):
                        _draw_angle_label(annotated, kpts[v_idx], aname, angles[aname])

            if show_position and is_inverted and matches_target:
                _draw_position_label(annotated, kpts, position)

            if show_deduction and is_inverted and matches_target and ded_desc and ded_desc != "N/A":
                _draw_deduction_label(annotated, kpts, ded_val, show_position)

            # Collect persons for snapshots & timeline
            if is_inverted:
                frame_inverted.append((track_id, position, ded_val, ded_desc, angles, kpts))
            frame_all_scored.append((track_id, position, ded_val, ded_desc, angles, kpts))

            # CSV row
            row = [frame_idx, track_id]
            for kpt in kpts:
                row += [f"{kpt[0]:.1f}", f"{kpt[1]:.1f}"]
            for aname in JOINT_ANGLES:
                val = angles[aname]
                row.append(f"{val:.1f}" if val is not None else "")
            row.append(position)
            csv_writer.writerow(row)

            total_persons += 1

        # --- Per-frame deduction timeline & snapshots (per person) ---
        scored_positions = {target_position} if target_position else {"Tuck", "Pike", "Layout"}

        # Deduction timeline: inverted-only (avoids false positives from upright persons)
        for (tid, position, ded_val, ded_desc, angles, kpts) in frame_inverted:
            if position not in scored_positions:
                continue
            hip_vals = [v for k, v in angles.items() if "hip" in k and v is not None]
            knee_vals = [v for k, v in angles.items() if "knee" in k and v is not None]
            deduction_timeline.append({
                "frame": frame_idx,
                "deduction": ded_val,
                "position": position,
                "person_id": tid,
                "hip_avg": round(float(np.mean(hip_vals)), 1) if hip_vals else None,
                "knee_avg": round(float(np.mean(knee_vals)), 1) if knee_vals else None,
            })

        # Snapshot segments: all persons (inverted + normal) for denser frame buffer
        for (tid, position, ded_val, ded_desc, angles, kpts) in frame_all_scored:
            if position not in scored_positions:
                continue

            # Per-person snapshot segment tracking — buffer all frames per segment
            if tid not in per_person_segments:
                per_person_segments[tid] = {"entry": None, "best": None, "buffer": [], "seg_id": segment_counter, "prev_pos": None}
                segment_counter += 1
            seg = per_person_segments[tid]

            frame_tuple = (ded_val, annotated.copy(), frame_idx, position, ded_desc, tid)

            if position != seg["prev_pos"]:
                # Position changed — flush previous segment
                if seg["best"] is not None and len(snapshots) < MAX_SNAPSHOTS:
                    _flush_segment(seg, snapshots, results_dir, job_prefix)
                # Start new segment
                seg["seg_id"] = segment_counter
                segment_counter += 1
                seg["entry"] = frame_tuple
                seg["best"] = frame_tuple
                seg["buffer"] = [frame_tuple]
            else:
                # Same position — update best if equal or lower deduction
                if seg["best"] is None or ded_val <= seg["best"][0]:
                    seg["best"] = frame_tuple
                # Buffer every frame for interval sampling
                seg["buffer"].append(frame_tuple)
            seg["prev_pos"] = position

        # Draw ROI box on frame
        if roi_px:
            _draw_roi(annotated, roi_px)

        writer.write(annotated)
        frame_idx += 1

        # Update progress for frontend polling (every frame for smooth bar)
        if cancel_flag is not None:
            cancel_flag["frame"] = frame_idx
            cancel_flag["total_frames"] = total_frames
            cancel_flag["elapsed"] = round(time.time() - start, 1)
            cancel_flag["stage"] = "analyzing"

        if frame_idx % 10 == 0 or frame_idx == total_frames:
            elapsed_so_far = time.time() - start
            print(f"  Frame {frame_idx}/{total_frames} ({elapsed_so_far:.1f}s elapsed)")

    # Flush remaining segments for all persons
    for tid, seg in per_person_segments.items():
        if seg["best"] is not None and len(snapshots) < MAX_SNAPSHOTS:
            _flush_segment(seg, snapshots, results_dir, job_prefix)

    # Sort snapshots by frame order
    snapshots.sort(key=lambda s: s["frame"])

    elapsed = time.time() - start
    cap.release()
    writer.release()
    csv_file.close()

    print(f"  Saved {len(snapshots)} position snapshots")

    # Re-encode to H.264 so browsers can play the video
    if cancel_flag is not None:
        cancel_flag["stage"] = "encoding"
    if _reencode_to_h264(output_video_path):
        print("  Re-encoded to H.264 for browser playback")
    else:
        print("  Warning: could not re-encode to H.264 — video may not play in browser")

    # Compute form score summary from deduction timeline
    form_summary = {}
    if deduction_timeline:
        deds = [d["deduction"] for d in deduction_timeline]
        clean_frames = sum(1 for d in deds if d == 0.0)
        scored_frames = len(deds)
        form_summary = {
            "scored_frames": scored_frames,
            "worst_deduction": round(max(deds), 1),
            "avg_deduction": round(float(np.mean(deds)), 3),
            "clean_pct": round(100.0 * clean_frames / scored_frames, 1),
            "deduction_counts": {
                "0.0": sum(1 for d in deds if d == 0.0),
                "0.1": sum(1 for d in deds if d == 0.1),
                "0.2": sum(1 for d in deds if d == 0.2),
                "0.3": sum(1 for d in deds if d == 0.3),
            },
        }
        print(f"  Form: {scored_frames} scored frames, avg ded {form_summary['avg_deduction']}, "
              f"worst {form_summary['worst_deduction']}, {form_summary['clean_pct']}% clean")

    # Downsample timeline for frontend (max ~200 points for chart)
    timeline_out = deduction_timeline
    if len(deduction_timeline) > 200:
        step = len(deduction_timeline) / 200
        timeline_out = [deduction_timeline[int(i * step)] for i in range(200)]

    return {
        "frame_count": frame_idx,
        "fps": round(fps, 2),
        "total_persons_detected": total_persons,
        "processing_time": round(elapsed, 2),
        "width": width,
        "height": height,
        "snapshots": snapshots,
        "form_summary": form_summary,
        "deduction_timeline": timeline_out,
    }
