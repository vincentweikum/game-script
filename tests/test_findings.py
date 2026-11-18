import json
from pathlib import Path

import pytest

F = Path(__file__).resolve().parents[1] / "reports" / "findings.json"
pytestmark = pytest.mark.skipif(not F.exists(), reason="run `make analysis` first")


@pytest.fixture(scope="module")
def f():
    return json.loads(F.read_text())


def test_model_beats_baselines(f):
    m = f["model"]
    assert m["gradient_boosting"]["log_loss"] < m["logistic_regression"]["log_loss"]
    assert m["gradient_boosting"]["log_loss"] < m["down_and_distance_only"]["log_loss"]


def test_fourth_quarter_game_script(f):
    q4 = f["game_script"]["4th quarter"]
    assert q4["-14"] > q4["+0"] > q4["+14"]
