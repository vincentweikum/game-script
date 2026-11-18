"""Expected pass probability: how often an average team passes in this exact situation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Only game state. Nothing about who the team is, and nothing the offense chooses
# at the snap (shotgun, no-huddle), so "over expected" is left for the team to explain.
FEATURES = [
    "down", "ydstogo", "yardline_100", "qtr", "score_differential",
    "game_seconds_remaining", "half_seconds_remaining",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "wp", "pregame_spread", "implied_team_total", "is_home",
    "outdoors", "wind", "temp", "season",
]


def gbm() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=48, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, random_state=0,
    )


def logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def down_distance_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Pass rate by down and yards-to-go bucket. What a 'common sense' model looks like."""
    def key(d):
        return d["down"].astype(int).astype(str) + "_" + pd.cut(
            d["ydstogo"], [0, 2, 4, 7, 10, 15, 100]).astype(str)
    rates = train.groupby(key(train))["is_pass"].mean()
    return key(test).map(rates).fillna(train["is_pass"].mean()).to_numpy()


def score(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "accuracy": float(np.mean((p > 0.5) == y)),
    }
