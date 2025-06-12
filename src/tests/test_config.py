import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import config
from pathlib import Path


def test_load_config_yaml(tmp_path):
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text("a: 1\n")
    cfg = config.load_config(cfg_file)
    assert cfg == {"a": 1}
