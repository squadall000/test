import argparse
import json
import logging
from pathlib import Path

import cv2

from .camera import Camera
from .detector import detect_defects
from .utils import config as config_utils
from .utils import plotting


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Camera index or URL")
    parser.add_argument("--mode", choices=["single", "stream"], default="single")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--save-input", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = config_utils.load_config(args.config)
    cam = Camera(
        args.source,
        width=cfg.get("width"),
        height=cfg.get("height"),
        fps=cfg.get("fps"),
        flip=cfg.get("flip", False),
    )
    if not cam.open():
        return 1

    frame_idx = 0
    try:
        while True:
            frame = cam.read()
            if frame is None:
                break

            if args.mode == "stream" and frame_idx % args.every != 0:
                frame_idx += 1
                continue

            result = detect_defects(frame, cfg.get("roi"))
            overlay = result["overlay"]
            if args.save_input:
                original_path = plotting.save_result(frame, overlay, Path("output"))
                plotting.save_metadata(result["defects"], original_path)
            cv2.imshow("defects", overlay)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if args.mode == "single":
                break
            frame_idx += 1
    finally:
        cam.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
