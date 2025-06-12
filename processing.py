"""Image processing routines."""

from __future__ import annotations

import cv2
import numpy as np


def replace_background(frame: np.ndarray, lower_hsv: tuple[int, int, int] | None = None,
                       upper_hsv: tuple[int, int, int] | None = None) -> np.ndarray:
    """Replace background with black using HSV mask.

    Parameters
    ----------
    frame : ndarray
        Input BGR frame.
    lower_hsv, upper_hsv : tuple of int, optional
        HSV bounds for background removal. If not provided, returns frame.

    Returns
    -------
    ndarray
        Frame with background replaced by black.
    """
    if lower_hsv is None or upper_hsv is None:
        return frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    result = frame.copy()
    result[mask > 0] = 0
    return result


def detect_irregularities(obj_roi: np.ndarray, thresh: int = 30, mode: str = "laplacian") -> np.ndarray:
    """Detect irregularities in the object region.

    Parameters
    ----------
    obj_roi : ndarray
        Input region of interest.
    thresh : int
        Threshold value.
    mode : {{"laplacian", "canny"}}
        Detection algorithm.

    Returns
    -------
    ndarray
        Binary mask of detected regions.
    """
    gray = cv2.cvtColor(obj_roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    if mode == "canny":
        edges = cv2.Canny(blur, thresh, thresh * 2)
        mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        return mask

    lap = cv2.Laplacian(blur, cv2.CV_16S, ksize=3)
    abs_lap = cv2.convertScaleAbs(lap)
    _, mask = cv2.threshold(abs_lap, thresh, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=2,
    )
    return mask


def apply_overlay(frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    """Apply colored overlay for mask regions."""
    overlay = frame.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
