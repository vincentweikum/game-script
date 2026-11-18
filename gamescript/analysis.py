"""Run the full analysis: model evaluation, pass rate over expected, game script effects."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gamescript.data import load_plays
from gamescript.model import FEATURES, down_distance_baseline, gbm, logistic, score

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
BLUE, ORANGE = "#2563eb", "#ea580c"
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
})


def neutral(df: pd.DataFrame) -> pd.Series:
    """Situations where the score isn't forcing the call: the cleanest read on a team's tendency."""
    return (df.wp.between(0.2, 0.8) & df.down.between(1, 3) & (df.qtr <= 3)
            & (df.half_seconds_remaining > 120))


def evaluate(df: pd.DataFrame) -> dict:
    """Train 2019-2024, test on the full 2025 season."""
    tr, te = df[df.season <= 2024], df[df.season == 2025]
    y = te.is_pass.to_numpy()
    m = gbm().fit(tr[FEATURES], tr.is_pass)
    p = m.predict_proba(te[FEATURES])[:, 1]
    out = {
        "train_plays": int(len(tr)), "test_plays": int(len(te)),
        "gradient_boosting": score(y, p),
        "logistic_regression": score(y, logistic().fit(tr[FEATURES], tr.is_pass)
                                     .predict_proba(te[FEATURES])[:, 1]),
        "down_and_distance_only": score(y, down_distance_baseline(tr, te)),
    }
    has = te.xpass.notna().to_numpy()
    out["vs_nflverse_xpass"] = {
        "plays_compared": int(has.sum()),
        "ours": score(y[has], p[has]),
        "nflverse": score(y[has], te.xpass.to_numpy()[has]),
        "correlation": float(np.corrcoef(p[has], te.xpass.to_numpy()[has])[0, 1]),
    }

    # calibration chart
    bins = np.clip((p * 10).astype(int), 0, 9)
    cal = pd.DataFrame({"b": bins, "p": p, "y": y}).groupby("b").agg(p=("p", "mean"), y=("y", "mean"),
                                                                      n=("y", "size"))
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1, label="Perfect calibration")
    ax.plot(cal.p, cal.y, marker="o", color=BLUE, label="Model (2025 holdout)")
    ax.set(xlabel="Predicted pass probability", ylabel="Actual pass rate",
           title=f"Calibration on {len(te):,} unseen 2025 plays")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "calibration.png")
    plt.close(fig)
    return out


def out_of_sample_xpass(df: pd.DataFrame) -> pd.Series:
    """Score every season with a model that never saw it (leave one season out)."""
    xp = pd.Series(np.nan, index=df.index)
    for s in sorted(df.season.unique()):
        test = df.season == s
        m = gbm().fit(df.loc[~test, FEATURES], df.loc[~test, "is_pass"])
        xp[test] = m.predict_proba(df.loc[test, FEATURES])[:, 1]
    return xp


def team_proe(df: pd.DataFrame, season: int) -> pd.DataFrame:
    d = df[(df.season == season) & neutral(df) & (df.season_type == "REG")]
    t = d.groupby("posteam").agg(plays=("is_pass", "size"), pass_rate=("is_pass", "mean"),
                                 xpass=("xpass_oos", "mean"))
    t["proe"] = t.pass_rate - t.xpass
    t = t.sort_values("proe", ascending=False)

    fig, ax = plt.subplots(figsize=(6.5, 8))
    colors = [BLUE if v > 0 else ORANGE for v in t.proe]
    ax.barh(t.index[::-1], t.proe[::-1] * 100, color=colors[::-1])
    ax.axvline(0, color=INK, lw=0.8)
    ax.set(xlabel="Neutral situations, percentage points",
           title=f"Pass rate over expected, {season} through Week {d.week.max()}\n"
                 "Positive = throws more than an average team would")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / f"proe_{season}.png")
    plt.close(fig)
    return t


def game_script_curve(df: pd.DataFrame) -> dict:
    """Early-down pass rate by score differential, split by game phase."""
    d = df[df.down.isin([1, 2])].copy()
    d["diff"] = d.score_differential.clip(-21, 21)
    phases = {"1st half": (1, 2), "3rd quarter": (3, 3), "4th quarter": (4, 4)}
    colors = {"1st half": MUTED, "3rd quarter": BLUE, "4th quarter": ORANGE}
    out = {}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, (a, b) in phases.items():
        c = d[d.qtr.between(a, b)].groupby("diff").agg(rate=("is_pass", "mean"), n=("is_pass", "size"))
        c = c[c.n >= 150]
        ax.plot(c.index, c.rate * 100, color=colors[name], lw=2, label=name)
        out[name] = {f"{k:+d}": float(c.rate.get(k, np.nan)) for k in (-14, -7, 0, 7, 14)}
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set(xlabel="Offense's score differential (negative = trailing)",
           ylabel="1st and 2nd down pass rate (%)",
           title="Game script is mostly a 4th quarter story\n"
                 "In the 1st half, teams with a lead keep throwing (2019-2026)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "game_script_curve.png")
    plt.close(fig)
    return out


def spread_to_volume(df: pd.DataFrame) -> dict:
    """The betting/fantasy angle: what does the pregame spread say about a team's play mix?"""
    reg = df[df.season_type == "REG"]
    g = reg.groupby(["game_id", "posteam"]).agg(
        spread=("pregame_spread", "first"), plays=("is_pass", "size"),
        dropbacks=("is_pass", "sum"),
        time_leading_7=("score_differential", lambda s: float((s >= 7).mean())),
        time_trailing_7=("score_differential", lambda s: float((s <= -7).mean())),
    )
    g["rushes"] = g.plays - g.dropbacks
    g["bucket"] = pd.cut(g.spread, [-30, -6.5, -2.5, 2.5, 6.5, 30],
                         labels=["Dog 7+", "Dog 3-6", "Pick'em", "Fav 3-6", "Fav 7+"])
    t = g.groupby("bucket", observed=True)[["plays", "dropbacks", "rushes", "time_leading_7",
                                            "time_trailing_7"]].mean()
    t["pass_rate"] = t.dropbacks / t.plays
    t["games"] = g.groupby("bucket", observed=True).size()

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(t))
    ax.bar(x - 0.2, t.dropbacks, 0.4, color=BLUE, label="Dropbacks")
    ax.bar(x + 0.2, t.rushes, 0.4, color=ORANGE, label="Designed runs")
    for i, (db, ru) in enumerate(zip(t.dropbacks, t.rushes, strict=True)):
        ax.text(i - 0.2, db + 0.4, f"{db:.1f}", ha="center", fontsize=8, color=INK)
        ax.text(i + 0.2, ru + 0.4, f"{ru:.1f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [f"{b}\n(n={n:,})" for b, n in zip(t.index, t.games, strict=True)])
    ax.set(ylabel="Per team per game", xlabel="Pregame point spread",
           title="Big favorites don't throw less, they run more\n"
                 "Average play mix by pregame spread, 2019-2026")
    ax.set_ylim(0, 46)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "spread_to_volume.png")
    plt.close(fig)
    t.round(3).to_csv(REPORTS / "spread_to_volume.csv")
    return {str(k): {c: float(v) for c, v in row.items()} for k, row in t.iterrows()}


def stickiness(df: pd.DataFrame) -> dict:
    """Is PROE a real team trait or noise? Compare weeks 1-9 against weeks 10-18."""
    d = df[neutral(df) & (df.season_type == "REG") & (df.season <= 2025)].copy()
    d["half"] = np.where(d.week <= 9, "first", "second")
    t = d.groupby(["season", "posteam", "half"]).agg(rate=("is_pass", "mean"), x=("xpass_oos", "mean"))
    t["proe"] = t.rate - t.x
    w = t.unstack("half")
    within_proe = w[("proe", "first")].corr(w[("proe", "second")])
    within_rate = w[("rate", "first")].corr(w[("rate", "second")])

    s = d.groupby(["season", "posteam"]).agg(rate=("is_pass", "mean"), x=("xpass_oos", "mean"))
    s["proe"] = s.rate - s.x
    s = s.reset_index()
    nxt = s.assign(season=s.season - 1)
    yoy = s.merge(nxt, on=["season", "posteam"], suffixes=("", "_next"))
    return {
        "within_season_proe_r": float(within_proe),
        "within_season_raw_pass_rate_r": float(within_rate),
        "year_over_year_proe_r": float(yoy.proe.corr(yoy.proe_next)),
        "team_halves": int(len(w)), "team_season_pairs": int(len(yoy)),
    }


def main() -> None:
    df = load_plays()
    findings = {"plays": int(len(df)), "seasons": f"{df.season.min()}-{df.season.max()}",
                "current_week": int(df[df.season == df.season.max()].week.max())}
    findings["model"] = evaluate(df)
    df["xpass_oos"] = out_of_sample_xpass(df)
    cur = int(df.season.max())
    t = team_proe(df, cur)
    t.round(4).to_csv(REPORTS / f"team_proe_{cur}.csv")
    findings["proe_current"] = {
        "season": cur,
        "top3": {k: float(v) for k, v in t.proe.head(3).items()},
        "bottom3": {k: float(v) for k, v in t.proe.tail(3).items()},
    }
    findings["game_script"] = game_script_curve(df)
    findings["spread_to_volume"] = spread_to_volume(df)
    findings["stickiness"] = stickiness(df)
    (REPORTS / "findings.json").write_text(json.dumps(findings, indent=2))
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
