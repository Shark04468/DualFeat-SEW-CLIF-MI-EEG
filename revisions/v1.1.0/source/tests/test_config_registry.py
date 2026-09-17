from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.config import load_yaml
from dpc_snn.experiments.runners import RUNNERS


def test_all_experiments_have_runners():
    cfg = load_yaml(ROOT / "configs/experiments/all_experiments.yaml")
    assert len(cfg["experiments"]) == 25
    for exp_id, exp in cfg["experiments"].items():
        assert exp_id.startswith("E")
        assert exp["runner"] in RUNNERS

