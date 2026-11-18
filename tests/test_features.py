import numpy as np
import pandas as pd

from gamescript.analysis import neutral
from gamescript.data import add_features
from gamescript.model import FEATURES, down_distance_baseline, gbm, score


def raw(**kw):
    base = {"qb_dropback": 1, "posteam_type": "home", "spread_line": 3.0, "total_line": 44.0,
            "roof": "outdoors", "wind": np.nan, "temp": np.nan, "season_type": "REG"}
    base.update(kw)
    return pd.DataFrame([base])


def test_spread_is_flipped_for_away_offense():
    home = add_features(raw(posteam_type="home", spread_line=3.0))
    away = add_features(raw(posteam_type="away", spread_line=3.0))
    assert home.pregame_spread.iloc[0] == 3.0       # home favored by 3
    assert away.pregame_spread.iloc[0] == -3.0      # so the away offense is a 3 point dog


def test_implied_team_total():
    f = add_features(raw(spread_line=7.0, total_line=47.0))
    assert f.implied_team_total.iloc[0] == 27.0     # 47/2 + 7/2


def test_missing_weather_filled():
    f = add_features(raw())
    assert f.wind.iloc[0] == 0 and f.temp.iloc[0] == 68 and f.outdoors.iloc[0] == 1


def test_neutral_filter():
    d = pd.DataFrame({"wp": [0.5, 0.95, 0.5, 0.5], "down": [1, 1, 4, 2], "qtr": [1, 1, 1, 4],
                      "half_seconds_remaining": [900, 900, 900, 900]})
    assert neutral(d).tolist() == [True, False, False, False]


def synthetic(n=6000, seed=0):
    """Fake plays where passing depends on down and score, to check the model learns it."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "down": rng.integers(1, 4, n), "ydstogo": rng.integers(1, 15, n),
        "yardline_100": rng.integers(1, 99, n), "qtr": rng.integers(1, 5, n),
        "score_differential": rng.integers(-21, 22, n),
        "game_seconds_remaining": rng.integers(0, 3600, n),
        "half_seconds_remaining": rng.integers(0, 1800, n),
        "posteam_timeouts_remaining": 3, "defteam_timeouts_remaining": 3,
        "wp": rng.uniform(0.05, 0.95, n), "pregame_spread": rng.normal(0, 6, n),
        "implied_team_total": 22.0, "is_home": rng.integers(0, 2, n), "outdoors": 1,
        "wind": 0.0, "temp": 68.0, "season": 2024,
    })
    logit = -0.5 + 0.8 * (d.down == 3) + 0.08 * d.ydstogo - 0.06 * d.score_differential
    d["is_pass"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return d


def test_model_beats_down_distance_on_signal():
    tr, te = synthetic(seed=1), synthetic(seed=2)
    p = gbm().fit(tr[FEATURES], tr.is_pass).predict_proba(te[FEATURES])[:, 1]
    ours = score(te.is_pass.to_numpy(), p)
    base = score(te.is_pass.to_numpy(), down_distance_baseline(tr, te))
    assert ours["log_loss"] < base["log_loss"]
    assert ours["auc"] > 0.65
