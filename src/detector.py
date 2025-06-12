from __future__ import annotations
from typing import List, Tuple
import cv2
import numpy as np


def detect_defects(gray: np.ndarray, min_area: int = 50, max_area: int = 10000) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Return binary mask and contours of detected defects."""
    edges = cv2.Canny(gray, 100, 200)
    thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    filtered = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            filtered.append(cnt)
    return mask, filtered

