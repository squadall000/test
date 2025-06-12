import json
import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix in {".yaml", ".yml"}:
        with p.open("r") as f:
            return yaml.safe_load(f)
    if p.suffix == ".json":
        with p.open("r") as f:
            return json.load(f)
    raise ValueError(f"Unsupported config format: {p.suffix}")
