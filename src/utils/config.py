import json
from dataclasses import dataclass
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML or JSON configuration."""
    with open(path, 'r', encoding='utf-8') as f:
        if path.endswith('.json'):
            return json.load(f)
        return yaml.safe_load(f)

