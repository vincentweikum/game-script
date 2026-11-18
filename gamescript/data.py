"""Load nflverse play-by-play and keep the plays where a team chose to run or pass."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
SEASONS = range(2019, 2027)

COLUMNS = [
    "game_id", "season", "season_type", "week", "posteam", "defteam", "posteam_type",
    "home_team", "away_team", "play_type", "pass", "rush", "qb_dropback", "qb_kneel", "qb_spike",
    "two_point_attempt", "down", "ydstogo", "yardline_100", "qtr", "score_differential",
    "game_seconds_remaining", "half_seconds_remaining", "posteam_timeouts_remaining",
    "defteam_timeouts_remaining", "wp", "spread_line", "total_line", "roof", "wind", "temp",
    "xpass", "rusher_player_name", "receiver_player_name", "complete_pass", "epa",
]


def download(seasons=SEASONS, force: bool = False) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for s in seasons:
        dest = RAW / f"pbp_{s}.parquet"
        if force or not dest.exists():
            urllib.request.urlretrieve(URL.format(season=s), dest)
            print(f"downloaded {dest.name}")


def load_plays(seasons=SEASONS) -> pd.DataFrame:
    """Every real run/pass decision. Kneels, spikes, 2-point tries and no-plays are removed."""
    frames = [pd.read_parquet(RAW / f"pbp_{s}.parquet", columns=COLUMNS) for s in seasons]
    df = pd.concat(frames, ignore_index=True)
    keep = (
        ((df["pass"] == 1) | (df["rush"] == 1))
        & df["down"].between(1, 4)
        & (df["qb_kneel"] != 1) & (df["qb_spike"] != 1) & (df["two_point_attempt"] != 1)
        & df["wp"].notna() & df["score_differential"].notna() & df["spread_line"].notna()
    )
    df = df.loc[keep].copy()
    return add_features(df)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["is_pass"] = (df["qb_dropback"] == 1).astype(int)   # scrambles count as a called pass
    df["is_home"] = (df["posteam_type"] == "home").astype(int)
    # nflverse spread_line is points the home team is favored by; flip it for the offense
    df["pregame_spread"] = np.where(df["is_home"] == 1, df["spread_line"], -df["spread_line"])
    df["implied_team_total"] = df["total_line"] / 2 + df["pregame_spread"] / 2
    df["outdoors"] = df["roof"].isin(["outdoors", "open"]).astype(int)
    df["wind"] = df["wind"].fillna(0)
    df["temp"] = df["temp"].fillna(68)
    df["is_postseason"] = (df["season_type"] == "POST").astype(int)
    return df
