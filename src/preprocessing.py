from typing import Tuple
import cv2
import numpy as np


def preprocess(frame: np.ndarray, blur_kernel: int = 5, roi: Tuple[float, float, float, float] | None = None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if blur_kernel > 0:
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    if roi:
        h, w = gray.shape
        x, y, rw, rh = roi
        x1 = int(x * w)
        y1 = int(y * h)
        x2 = int((x + rw) * w)
        y2 = int((y + rh) * h)
        mask = np.zeros_like(gray)
        mask[y1:y2, x1:x2] = 255
        gray = cv2.bitwise_and(gray, mask)
    return gray

