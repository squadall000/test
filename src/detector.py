import cv2
import numpy as np
from typing import List, Tuple, Dict

from . import preprocessing


def detect_defects(img: np.ndarray, roi: Tuple[int, int, int, int] | None = None) -> Dict:
    gray = preprocessing.to_gray(img)
    gray = preprocessing.denoise(gray)
    if roi:
        gray = preprocessing.apply_roi(gray, roi)

    edges = cv2.Canny(gray, 100, 200)
    thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    defects: List[Dict] = []
    overlay = img.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
        defects.append({"bbox": [int(x), int(y), int(w), int(h)], "area": float(area)})

    mask = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
    return {"overlay": overlay, "mask": mask, "defects": defects}
