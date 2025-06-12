import cv2
import numpy as np


def preprocess(gray: np.ndarray) -> np.ndarray:
    """Apply Gaussian blur to grayscale image."""
    return cv2.GaussianBlur(gray, (5, 5), 0)


def detect_edges(gray: np.ndarray, low: int, high: int, dilate_iter: int) -> np.ndarray:
    """Detect edges and return binary mask with white lines."""
    blurred = preprocess(gray)
    edges = cv2.Canny(blurred, low, high)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=dilate_iter)
    mask = np.where(edges > 0, 255, 0).astype(np.uint8)
    return mask
