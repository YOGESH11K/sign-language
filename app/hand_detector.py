"""Hand landmark detection using the MediaPipe Tasks API.

Wraps ``mediapipe.tasks.python.vision.HandLandmarker`` (the supported API in
current MediaPipe releases) and returns a plain-Python hand structure that the
rest of the app consumes. The deprecated ``mediapipe.solutions`` module is not
available in recent builds, so this module uses the Tasks API exclusively.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

def _landmark_points(seq) -> np.ndarray:
    """Convert MediaPipe Landmark objects to a (21, 3) float32 array.

    ``np.asarray(seq, dtype=np.float32)`` breaks with the Landmark objects in
    current MediaPipe builds, so we map the x/y/z attributes explicitly.
    """
    pts = np.empty((len(seq), 3), dtype=np.float32)
    for i, p in enumerate(seq):
        pts[i, 0] = p.x
        pts[i, 1] = p.y
        pts[i, 2] = p.z
    return pts

_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass
class Hand:
    """A single detected hand."""

    handedness: str          # "Left", "Right" or "Unknown"
    score: float             # detection confidence (0..1)
    landmarks: np.ndarray    # (21, 3) normalized image-space x, y, z
    world_landmarks: np.ndarray = field(default_factory=lambda: np.zeros((21, 3)))

    def to_dict(self) -> dict:
        return {
            "handedness": self.handedness,
            "score": float(self.score),
            "landmarks": self.landmarks.tolist(),
            "world_landmarks": self.world_landmarks.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Hand":
        return cls(
            handedness=str(data.get("handedness", "Unknown")),
            score=float(data.get("score", 1.0)),
            landmarks=np.asarray(data["landmarks"], dtype=np.float32).reshape(21, 3),
            world_landmarks=np.asarray(
                data.get("world_landmarks", np.zeros((21, 3))), dtype=np.float32
            ).reshape(21, 3),
        )


class HandDetector:
    """Thread-safe wrapper around MediaPipe HandLandmarker.

    Use ``process(frame_bgr)`` to get a list of :class:`Hand`.
    """

    def __init__(
        self,
        model_path: str | Path,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._lock = threading.Lock()
        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Hand landmark model not found: {self._model_path}\n"
                "Download it first (the app does this automatically, or see README)."
            )
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self._model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=int(num_hands),
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        self._frame_ts = 0

    def process(self, frame_bgr: np.ndarray) -> list[Hand]:
        """Detect hands in one BGR frame. Returns a list of Hand (possibly empty)."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        if frame_bgr.ndim == 2:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
        with self._lock:
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)),
            )
            self._frame_ts += 33  # ~30fps, must be strictly monotonic for VIDEO mode
            result = self._detector.detect_for_video(image=mp_image, timestamp_ms=self._frame_ts)

        hands: list[Hand] = []
        for i, lm in enumerate(result.hand_landmarks):
            handed = (
                result.handedness[i][0].category_name
                if result.handedness and result.handedness[i]
                else "Unknown"
            )
            score = float(result.handedness[i][0].score if result.handedness and result.handedness[i] else 1.0)
            world = (
                _landmark_points(result.hand_world_landmarks[i])
                if result.hand_world_landmarks
                else np.zeros((21, 3), dtype=np.float32)
            )
            hands.append(
                Hand(
                    handedness=handed,
                    score=score,
                    landmarks=_landmark_points(lm),
                    world_landmarks=world,
                )
            )
        return hands

    def close(self) -> None:
        with self._lock:
            try:
                self._detector.close()
            except Exception:
                pass

    @staticmethod
    def draw_landmarks(
        frame: np.ndarray,
        hands: list[Hand],
        draw_connections: bool = True,
        color: tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """Draw detected landmarks onto a BGR frame copy and return it."""
        out = frame.copy()
        h, w = frame.shape[:2]
        for hand in hands:
            pts = []
            for x, y, _ in hand.landmarks:
                pts.append((int(x * w), int(y * h)))
                cv2.circle(out, (int(x * w), int(y * h)), 4, (255, 0, 0), -1)
            if draw_connections:
                for a, b in _HAND_CONNECTIONS:
                    cv2.line(out, pts[a], pts[b], color, 2)
        return out


def load_labels_json(path: str | Path) -> dict:
    """Load the vocabulary from ``data/labels.json``."""
    import json

    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)