import cv2
import numpy as np


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise(img, method="gaussian", ksize=5):
    if method == "median":
        return cv2.medianBlur(img, ksize)
    if method == "bilateral":
        return cv2.bilateralFilter(img, 9, 75, 75)
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def apply_roi(img, roi):
    if roi is None:
        return img
    x, y, w, h = roi
    mask = np.zeros_like(img)
    mask[y : y + h, x : x + w] = 1
    return img * mask
