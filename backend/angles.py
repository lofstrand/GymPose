"""Joint angle computation and gymnastics position classification."""

import numpy as np
from typing import Dict, List, Optional

# COCO 17 keypoint names (index-aligned)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Joint angle definitions: (point_a_idx, vertex_idx, point_b_idx)
JOINT_ANGLES = {
    "left_elbow": (5, 7, 9),       # shoulder -> elbow -> wrist
    "right_elbow": (6, 8, 10),     # shoulder -> elbow -> wrist
    "left_shoulder": (11, 5, 7),   # hip -> shoulder -> elbow
    "right_shoulder": (12, 6, 8),  # hip -> shoulder -> elbow
    "left_hip": (5, 11, 13),       # shoulder -> hip -> knee
    "right_hip": (6, 12, 14),      # shoulder -> hip -> knee
    "left_knee": (11, 13, 15),     # hip -> knee -> ankle
    "right_knee": (12, 14, 16),    # hip -> knee -> ankle
}


def compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute the angle (degrees) at vertex b formed by points a-b-c."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def compute_joint_angles(
    keypoints: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    min_conf: float = 0.3,
) -> Dict[str, Optional[float]]:
    """Compute all 8 joint angles from a (17, 2) keypoints array.

    Returns a dict mapping angle name -> degrees (or None if keypoints
    are missing / below confidence).
    """
    angles: Dict[str, Optional[float]] = {}

    for name, (a_idx, b_idx, c_idx) in JOINT_ANGLES.items():
        # Skip if any involved keypoint has low confidence
        if confidences is not None:
            if (
                confidences[a_idx] < min_conf
                or confidences[b_idx] < min_conf
                or confidences[c_idx] < min_conf
            ):
                angles[name] = None
                continue

        a, b, c = keypoints[a_idx], keypoints[b_idx], keypoints[c_idx]

        # Skip if any point is at origin (undetected)
        if np.all(a == 0) or np.all(b == 0) or np.all(c == 0):
            angles[name] = None
            continue

        angles[name] = compute_angle(a, b, c)

    return angles


def classify_position(angles: Dict[str, Optional[float]]) -> str:
    """Classify a gymnastics position based on joint angle thresholds.

    Categories:
      Tuck   – knees AND hips < 90°
      Pike   – hips < 90°, knees > 150°
      Layout – hips AND knees > 160°
      Split  – large left/right hip angle difference (> 60°)
      Neutral – everything else
    """
    hip_vals = [v for k, v in angles.items() if "hip" in k and v is not None]
    knee_vals = [v for k, v in angles.items() if "knee" in k and v is not None]

    if not hip_vals or not knee_vals:
        return "Unknown"

    avg_hip = np.mean(hip_vals)
    avg_knee = np.mean(knee_vals)

    # Split: large asymmetry in hip angles
    left_hip = angles.get("left_hip")
    right_hip = angles.get("right_hip")
    if left_hip is not None and right_hip is not None:
        if abs(left_hip - right_hip) > 60:
            return "Split"

    # Tuck: everything flexed
    if avg_hip < 100 and avg_knee < 100:
        return "Tuck"

    # Pike: hips flexed, legs more extended
    if avg_hip < 100 and avg_knee > 120:
        return "Pike"

    # Layout: body extended
    if avg_hip > 140 and avg_knee > 140:
        return "Layout"

    return "Neutral"


def _deduction_tuck(angles: Dict[str, Optional[float]]) -> tuple:
    hip_vals = [v for k, v in angles.items() if "hip" in k and v is not None]
    knee_vals = [v for k, v in angles.items() if "knee" in k and v is not None]
    if not hip_vals or not knee_vals:
        return (0.0, "N/A")
    avg_hip, avg_knee = np.mean(hip_vals), np.mean(knee_vals)
    if avg_hip < 40 and avg_knee < 40:
        return (0.0, f"No deduction (hip {avg_hip:.0f}° knee {avg_knee:.0f}°)")
    elif avg_hip < 60 and avg_knee < 60:
        return (0.1, f"0.1 – slightly open (hip {avg_hip:.0f}° knee {avg_knee:.0f}°)")
    elif avg_hip < 80 and avg_knee < 80:
        return (0.2, f"0.2 – open tuck (hip {avg_hip:.0f}° knee {avg_knee:.0f}°)")
    else:
        return (0.3, f"0.3 – very open tuck (hip {avg_hip:.0f}° knee {avg_knee:.0f}°)")


def _deduction_pike(angles: Dict[str, Optional[float]]) -> tuple:
    hip_vals = [v for k, v in angles.items() if "hip" in k and v is not None]
    knee_vals = [v for k, v in angles.items() if "knee" in k and v is not None]
    if not hip_vals:
        return (0.0, "N/A")
    avg_hip = np.mean(hip_vals)
    avg_knee = np.mean(knee_vals) if knee_vals else 180
    # Pike: hips should be closed, knees should be straight
    hip_ded = 0.0
    if avg_hip < 30:
        hip_ded = 0.0
    elif avg_hip < 50:
        hip_ded = 0.1
    elif avg_hip < 70:
        hip_ded = 0.2
    else:
        hip_ded = 0.3
    # Knee bend penalty: knees should be > 160 in a good pike
    knee_ded = 0.0
    if avg_knee < 140:
        knee_ded = 0.1
    ded = min(0.3, hip_ded + knee_ded)
    if ded == 0.0:
        return (0.0, f"No deduction (hip {avg_hip:.0f}° knee {avg_knee:.0f}°)")
    return (ded, f"{ded:.1f} – hip {avg_hip:.0f}° knee {avg_knee:.0f}°")


def _deduction_layout(angles: Dict[str, Optional[float]]) -> tuple:
    hip_vals = [v for k, v in angles.items() if "hip" in k and v is not None]
    knee_vals = [v for k, v in angles.items() if "knee" in k and v is not None]
    if not hip_vals:
        return (0.0, "N/A")
    avg_hip = np.mean(hip_vals)
    avg_knee = np.mean(knee_vals) if knee_vals else 180
    worst = min(avg_hip, avg_knee)  # most bent joint
    if worst > 175:
        return (0.0, f"No deduction (hip {avg_hip:.0f}° knee {avg_knee:.0f}°)")
    elif worst > 165:
        return (0.1, f"0.1 – slight bend (hip {avg_hip:.0f}° knee {avg_knee:.0f}°)")
    elif worst > 155:
        return (0.2, f"0.2 – moderate bend (hip {avg_hip:.0f}° knee {avg_knee:.0f}°)")
    else:
        return (0.3, f"0.3 – significant bend (hip {avg_hip:.0f}° knee {avg_knee:.0f}°)")


def compute_deduction(
    position: str,
    angles: Dict[str, Optional[float]],
    target_position: Optional[str] = None,
) -> tuple:
    """Compute gymnastics deduction based on position and joint angles.

    If target_position is set (e.g. "Pike"), deductions are always scored
    against that position regardless of auto-classification.

    Returns (deduction_value, description).
    """
    eval_position = target_position if target_position else position

    if eval_position == "Tuck":
        return _deduction_tuck(angles)
    elif eval_position == "Pike":
        return _deduction_pike(angles)
    elif eval_position == "Layout":
        return _deduction_layout(angles)

    return (0.0, "N/A")
