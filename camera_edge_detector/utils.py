import cv2
import logging
import numpy as np


def setup_logger() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def save_image(image: np.ndarray, path: str) -> None:
    cv2.imwrite(path, image)


def display(name: str, image: np.ndarray) -> None:
    cv2.imshow(name, image)
