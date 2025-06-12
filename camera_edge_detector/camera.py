import cv2


class Camera:
    """Context manager for cv2.VideoCapture."""

    def __init__(self, source=0) -> None:
        self.source = source
        self.cap = None

    def __enter__(self) -> "Camera":
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open source {self.source}")
        return self

    def read(self):
        if not self.cap:
            raise RuntimeError("Camera not opened")
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera")
        return frame

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.cap:
            self.cap.release()
