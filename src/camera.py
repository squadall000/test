from typing import Optional, Tuple
import cv2


class Camera:
    """Simple camera wrapper."""

    def __init__(self, source: str | int, width: int = 640, height: int = 480, fps: int = 30):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        return True

    def read(self) -> Optional[Tuple[bool, any]]:
        if self.cap is None:
            return None
        return self.cap.read()

    def release(self) -> None:
        if self.cap:
            self.cap.release()

