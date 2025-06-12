import argparse
import logging
from .camera import Camera
from .processor import detect_edges
from .utils import save_image, setup_logger
import cv2


def run(args: argparse.Namespace) -> None:
    setup_logger()
    logger = logging.getLogger(__name__)
    logger.info("Starting edge detection")
    with Camera(args.source) as cam:
        last = None
        try:
            while True:
                frame = cam.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                last = detect_edges(gray, args.canny_thresh[0], args.canny_thresh[1], args.dilate_iter)
                if args.show:
                    cv2.imshow("original", frame)
                    cv2.imshow("processed", last)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27):
                        break
                if args.single_shot:
                    break
        finally:
            if args.show:
                cv2.destroyAllWindows()
        if args.save and last is not None:
            save_image(last, args.save)
            logger.info(f"Saved image to {args.save}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Camera edge detector")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_p = subparsers.add_parser("run", help="run detector")
    run_p.add_argument("--source", default=0, help="camera index or RTSP uri")
    run_p.add_argument("--save", help="output PNG path")
    run_p.add_argument("--show", action="store_true", help="display windows")
    run_p.add_argument("--single-shot", action="store_true", help="process single frame")
    run_p.add_argument("--canny-thresh", nargs=2, type=int, default=[100, 200], metavar=("LOW", "HIGH"))
    run_p.add_argument("--dilate-iter", type=int, default=1, help="dilation iterations")

    args = parser.parse_args()
    if args.command == "run":
        run(args)


if __name__ == "__main__":
    main()
