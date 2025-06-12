"""Camera handling utilities."""

from __future__ import annotations

import cv2


def open_camera(device: int, width: int, height: int) -> cv2.VideoCapture:
    """Open camera and set resolution."""
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {device}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap
