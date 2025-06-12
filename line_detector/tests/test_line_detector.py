import os
import tempfile
from unittest import TestCase, mock

import cv2
import numpy as np

from line_detector import line_detector


class TestLineDetector(TestCase):
    def test_camera_unavailable(self):
        with self.assertRaises(RuntimeError):
            line_detector.open_camera("invalid://0")

    def test_detect_lines_in_image(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.line(img, (10, 10), (90, 10), (255, 255, 255), 2)
        cv2.line(img, (10, 20), (90, 20), (255, 255, 255), 2)
        args = mock.Mock(canny_th1=50, canny_th2=150, hough_th=30,
                         min_line_len=10, max_line_gap=5)
        output = line_detector.process_frame(img, args)
        self.assertGreater(np.sum(output > 0), 0)

    def test_single_mode_save(self):
        dummy_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "out.png")
            cap = mock.Mock()
            cap.read.return_value = (True, dummy_frame)
            cap.isOpened.return_value = True
            args = mock.Mock(save=save_path, canny_th1=50, canny_th2=150,
                             hough_th=30, min_line_len=10, max_line_gap=5)
            with mock.patch("cv2.imshow"), mock.patch("cv2.waitKey", return_value=27):
                line_detector.run_single(cap, args)
            self.assertTrue(os.path.exists(save_path))
            self.assertGreater(os.path.getsize(save_path), 0)
