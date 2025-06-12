import cv2
import logging
from typing import Any, Optional


class Camera:
    """Simple camera wrapper for video capture."""

    def __init__(self, source: Any, width: Optional[int] = None, height: Optional[int] = None, fps: Optional[int] = None, flip: bool = False) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.flip = flip
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.source)
        if self.width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not self.cap.isOpened():
            logging.error("Unable to open camera: %s", self.source)
            return False
        logging.info("Camera opened: %s", self.source)
        return True

    def read(self):
        if not self.cap:
            raise RuntimeError("Camera is not opened")
        ret, frame = self.cap.read()
        if not ret:
            logging.error("Failed to read frame from camera")
            return None
        if self.flip:
            frame = cv2.flip(frame, 1)
        return frame

    def release(self) -> None:
        if self.cap:
            self.cap.release()
            logging.info("Camera released")
        self.cap = None
