"""Shared MediaPipe landmark feature representation for the two-hand model."""
from __future__ import annotations

import numpy as np

LANDMARKS_PER_HAND = 21
VALUES_PER_LANDMARK = 3
FEATURES_PER_HAND = LANDMARKS_PER_HAND * VALUES_PER_LANDMARK
FEATURE_COUNT = 2 * FEATURES_PER_HAND
SINGLE_FEATURE_COUNT = FEATURES_PER_HAND
SINGLE_FEATURE_NAMES = [f"f{i}" for i in range(SINGLE_FEATURE_COUNT)]
FEATURE_NAMES = [f"{side}_{axis}_{i}" for side in ("left", "right")
                 for i in range(LANDMARKS_PER_HAND)
                 for axis in ("x", "y", "z")]


def extract_hand_features(hand_landmarks):
    """Wrist-relative xyz scaled by wrist-to-middle-MCP distance.

    Coordinates stay in MediaPipe's image coordinate system. x/y are centered
    on wrist landmark 0; z is centered and scaled identically. Returns 63
    float32 values, or None for malformed/degenerate input.
    """
    if hand_landmarks is None or len(hand_landmarks) != LANDMARKS_PER_HAND:
        return None
    points = np.asarray([[p.x, p.y, p.z] for p in hand_landmarks], dtype=np.float32)
    if points.shape != (LANDMARKS_PER_HAND, 3) or not np.isfinite(points).all():
        return None
    origin = points[0].copy()
    scale = float(np.linalg.norm(points[9, :2] - origin[:2]))
    if not np.isfinite(scale) or scale < 1e-5:
        return None
    normalized = (points - origin) / scale
    return normalized.reshape(-1).astype(np.float32)


def extract_single_hand_features(hand_landmarks):
    """Single-hand version of the same wrist-relative normalized feature set."""
    return extract_hand_features(hand_landmarks)


def extract_two_hand_features(left_hand, right_hand):
    """Return left-hand then right-hand features, or None if either is absent."""
    left = extract_hand_features(left_hand)
    right = extract_hand_features(right_hand)
    if left is None or right is None:
        return None
    features = np.concatenate((left, right))
    return features if features.size == FEATURE_COUNT else None


def normalize_raw_hand_features(values):
    """Normalize a legacy 63-value x/y/z vector using the live convention."""
    points = np.asarray(values, dtype=np.float32)
    if points.size != FEATURES_PER_HAND or not np.isfinite(points).all():
        return None
    points = points.reshape(LANDMARKS_PER_HAND, 3)
    if np.allclose(points, 0):
        return None
    scale = float(np.linalg.norm(points[9, :2] - points[0, :2]))
    if scale < 1e-5:
        return None
    return ((points - points[0]) / scale).reshape(-1).astype(np.float32)


def assign_handedness(result):
    """Resolve MediaPipe output into stable anatomical Left/Right slots."""
    left = right = None
    for i, classification in enumerate(getattr(result, "handedness", []) or []):
        if i >= len(getattr(result, "hand_landmarks", []) or []) or not classification:
            continue
        label = classification[0].category_name
        if label == "Left" and left is None:
            left = result.hand_landmarks[i]
        elif label == "Right" and right is None:
            right = result.hand_landmarks[i]
    return left, right
