import numpy as np
from preprocessing import preprocess


def test_preprocess_gray():
    dummy = np.zeros((10, 10, 3), dtype=np.uint8)
    gray = preprocess(dummy)
    assert gray.shape == (10, 10)
