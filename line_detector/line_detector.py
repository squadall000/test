import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Detect lines from camera feed")
    parser.add_argument("--camera-id", type=str, default="0", help="Camera index or path")
    parser.add_argument("--mode", type=str, default="stream", choices=["single", "stream"], help="Operation mode")
    parser.add_argument("--save", type=Path, default=None, help="Path to save single frame result")
    parser.add_argument("--canny-th1", type=int, default=50, help="Lower threshold for Canny")
    parser.add_argument("--canny-th2", type=int, default=150, help="Upper threshold for Canny")
    parser.add_argument("--hough-th", type=int, default=50, help="Accumulator threshold for HoughLinesP")
    parser.add_argument("--min-line-len", type=int, default=50, help="Minimum line length for HoughLinesP")
    parser.add_argument("--max-line-gap", type=int, default=10, help="Maximum line gap for HoughLinesP")
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def open_camera(camera_id):
    try:
        cam_id = int(camera_id)
    except ValueError:
        cam_id = camera_id
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")
    return cap


def process_frame(frame, args):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, args.canny_th1, args.canny_th2)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, args.hough_th,
                            minLineLength=args.min_line_len,
                            maxLineGap=args.max_line_gap)
    output = np.zeros_like(frame)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(output, (x1, y1), (x2, y2), (255, 255, 255), 2)
    else:
        logging.info("No lines detected")
    return output


def run_single(cap, args):
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read frame")
    output = process_frame(frame, args)
    if args.save:
        cv2.imwrite(str(args.save), output)
        logging.info("Saved result to %s", args.save)
    cv2.imshow("Lines", output)
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()


def run_stream(cap, args):
    fps_counter = cv2.getTickFrequency()
    timings = []
    while True:
        start = cv2.getTickCount()
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to read frame")
            break
        output = process_frame(frame, args)
        cv2.imshow("Lines", output)
        key = cv2.waitKey(1)
        elapsed = (cv2.getTickCount() - start) / fps_counter
        timings.append(elapsed)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    if timings:
        avg_fps = len(timings) / sum(timings)
        logging.info("Average FPS: %.2f", avg_fps)


def main():
    args = parse_args()
    setup_logging()
    logging.info("Opening camera %s", args.camera_id)
    cap = open_camera(args.camera_id)
    try:
        if args.mode == "single":
            run_single(cap, args)
        else:
            run_stream(cap, args)
    finally:
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.error("%s", exc)
        sys.exit(1)
