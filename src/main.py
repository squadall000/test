import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import cv2

from camera import Camera
from detector import detect_defects
from preprocessing import preprocess
from utils.config import load_config
from utils.plotting import overlay_mask

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Surface defect detection')
    parser.add_argument('--source', default=0, help='camera index or URL')
    parser.add_argument('--mode', choices=['single', 'stream'], default='single')
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--every', type=int, default=1, help='process every Nth frame')
    parser.add_argument('--save-input', action='store_true', help='save raw input frame')
    return parser.parse_args()


def save_outputs(frame, mask, contours, out_dir: Path):
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay = overlay_mask(frame, mask)
    side_by_side = cv2.hconcat([frame, overlay])
    frame_path = out_dir / f'frame_{now}.png'
    mask_path = out_dir / f'mask_{now}.png'
    cv2.imwrite(str(frame_path), side_by_side)
    cv2.imwrite(str(mask_path), mask)
    metadata = {
        'num_defects': len(contours),
        'areas': [cv2.contourArea(c) for c in contours],
    }
    with open(out_dir / f'meta_{now}.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)


def main():
    args = parse_args()
    config = load_config(args.config)
    cam_cfg = config.get('camera', {})
    pre_cfg = config.get('preprocessing', {})
    alg_cfg = config.get('algorithm', {})

    camera = Camera(
        source=args.source,
        width=cam_cfg.get('width', 640),
        height=cam_cfg.get('height', 480),
        fps=cam_cfg.get('fps', 30),
    )
    if not camera.open():
        logging.error('Unable to open camera')
        return 1

    frame_count = 0
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                logging.error('Failed to read frame')
                break
            frame_count += 1
            if frame_count % args.every != 0:
                continue
            gray = preprocess(
                frame,
                blur_kernel=pre_cfg.get('blur_kernel', 5),
                roi=tuple(pre_cfg.get('roi', [0, 0, 1, 1])),
            )
            mask, contours = detect_defects(
                gray,
                min_area=alg_cfg.get('min_area', 50),
                max_area=alg_cfg.get('max_area', 10000),
            )
            overlay = overlay_mask(frame, mask)
            cv2.imshow('defects', overlay)
            if args.save_input:
                cv2.imwrite('output/input.png', frame)
            save_outputs(frame, mask, contours, Path('output'))
            if args.mode == 'single':
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    raise SystemExit(main())

