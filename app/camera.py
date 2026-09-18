"""Webcam wrapper.

Provides a small, forgiving interface over ``cv2.VideoCapture`` with friendly
error messages instead of crashes (feature 14: error handling).
"""

from __future__ import annotations

import cv2
from typing import Optional


class CameraUnavailableError(Exception):
    """Raised when the webcam cannot be opened or read from."""

    def __init__(self, detail: str = ""):
        super().__init__(f"Webcam unavailable. {detail}".strip())


class WebcamCapture:
    """Encapsulates opening/reading/releasing a webcam device."""

    def __init__(self, index: int = 0, width: int = 960, height: int = 540) -> None:
        self.index = index
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            raise CameraUnavailableError(
                f"device index {self.index}. Check that the camera is connected "
                "and not in use by another application, then allow camera access."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read(self):
        """Return ``(ok, frame_bgr)``. ``ok`` is False at end of stream."""
        if not self.is_open:
            return False, None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @staticmethod
    def probe_cameras(max_index: int = 5) -> list[int]:
        """Find which camera indices actually open, for friendly setup."""
        available: list[int] = []
        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
            cap.release()
        return available

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.release()