import cv2
import numpy as np


def overlay_mask(frame: np.ndarray, mask: np.ndarray, color=(0, 0, 255)) -> np.ndarray:
    """Overlay binary mask on frame with given color."""
    colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    colored_mask[:, :, 0] = 0
    colored_mask[:, :, 1] = 0
    colored_mask[:, :, 2] = np.where(mask > 0, 255, 0)
    overlay = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)
    return overlay

