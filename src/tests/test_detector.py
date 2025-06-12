import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
import cv2
from src.detector import detect_defects


def test_detect_defects_simple():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (30, 30), (255, 255, 255), -1)
    result = detect_defects(img)
    assert len(result["defects"]) >= 1
