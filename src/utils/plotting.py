import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict


def save_result(original: np.ndarray, overlay: np.ndarray, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined = cv2.hconcat([original, overlay])
    out_path = out_dir / f"frame_{timestamp}.png"
    cv2.imwrite(str(out_path), combined)
    return out_path


def save_metadata(defects: List[Dict], out_path: Path) -> None:
    meta_path = out_path.with_suffix('.json')
    import json
    with meta_path.open('w') as f:
        json.dump({"defects": defects}, f, indent=2)
