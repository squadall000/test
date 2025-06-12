"""Entry point for irregularity detection application."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from camera import open_camera
from processing import apply_overlay, detect_irregularities, replace_background
from utils import build_arg_parser, load_config


COLORS = [
    (0, 0, 255),  # red
    (0, 255, 0),  # green
    (255, 0, 0),  # blue
    (0, 255, 255),  # yellow
]


def capture_loop(args) -> None:
    """Main capture loop."""
    config = load_config(args.config)
    threshold = args.threshold or config.get("threshold", 30)
    color = args.color or tuple(config.get("mask_color", [0, 0, 255]))
    color_index = 0

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise RuntimeError(f"Cannot read image {args.image}")
        frames = [frame]
    else:
        cap = open_camera(args.camera, args.width, args.height)
        frames = None
        if args.save:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter("output.avi", fourcc, 30, (args.width, args.height))
        else:
            out = None

    paused = False
    last_time = time.time()

    while True:
        if args.image:
            frame = frames[0].copy()
        else:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break

        processed = replace_background(frame)
        mask = detect_irregularities(processed, threshold, args.mode)
        overlay = apply_overlay(processed, mask, color)

        fps = 1.0 / (time.time() - last_time)
        last_time = time.time()
        cv2.putText(
            overlay,
            f"FPS: {fps:.1f} Thr: {threshold}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("original", frame)
        cv2.imshow("processed", overlay)

        if out is not None:
            out.write(overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            paused = not paused
        if key == ord("c"):
            color_index = (color_index + 1) % len(COLORS)
            color = COLORS[color_index]
        if key == 82:  # up arrow
            threshold = min(255, threshold + 1)
        if key == 84:  # down arrow
            threshold = max(0, threshold - 1)

        if paused:
            cv2.waitKey(100)
            continue

        if args.image:
            key = cv2.waitKey(0)
            if key == ord("q"):
                break
            continue

    if not args.image:
        cap.release()
        if out is not None:
            out.release()
    if args.mask_out:
        cv2.imwrite(str(args.mask_out), mask)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = build_arg_parser()
    capture_loop(parser.parse_args())
