"""
Stage 1 of the pipeline: predict wait time by ride and time of day.

The optimizer needs w[i,t] for every ride and slot. Observations are sparse
and noisy, so this module turns raw polls into a dense forecast.

Design notes
------------
The zero question. A reading of wait_minutes == 0 on an open ride is
ambiguous: it may be a genuine walk-on, or the park reporting a placeholder
before it posts a real number. These are not interchangeable -- averaging in
a morning of placeholder zeros will drag a ride's whole curve down and the
optimizer will route you to a ride that only LOOKS empty.

This module does not guess. Run diagnose_zeros() on your own data, look at
how the zero fraction moves through the day, then set `zero_policy` to
whichever reading your data supports. Whatever you choose goes in the README.

Model selection. baseline() is the historical mean by (ride, slot). Any
fitted model has to beat it on held-out days or it is not worth the
complexity -- report that comparison honestly either way.
"""

import numpy as np
import pandas as pd

from optimize import SLOT_MIN

DATA_PATH = "data/observations.csv"
SLOTS_PER_DAY = 24 * 60 // SLOT_MIN


# --- loading ---------------------------------------------------------------

def load(path: str = DATA_PATH) -> pd.DataFrame:
    """Read observations and attach local-time columns."""
    df = pd.read_csv(path, parse_dates=["polled_at"])
    local = df["polled_at"].dt.tz_convert("America/Chicago")
    df["hour"] = local.dt.hour
    df["date"] = local.dt.date
    df["dow"] = local.dt.dayofweek
    df["month"] = local.dt.month
    df["slot"] = (local.dt.hour * 60 + local.dt.minute) // SLOT_MIN
    return df


def diagnose_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run this BEFORE choosing a zero policy.

    Returns, by hour: the share of open-ride readings that are zero, and the
    mean wait among non-zero readings.

    Read it like this. If the zero share is near 1.0 in the first operating
    hour and falls off a cliff by mid-morning while non-zero waits stay
    stable, the early zeros are placeholders -- the park had not started
    posting. If the zero share declines smoothly alongside a smoothly rising
    wait, they are real walk-ons and should be kept.
    """
    open_rides = df[df["is_open"] == 1]
    return (
        open_rides.groupby("hour")
        .agg(
            n=("wait_minutes", "size"),
            zero_share=("wait_minutes", lambda s: (s == 0).mean()),
            mean_nonzero=("wait_minutes", lambda s: s[s > 0].mean()),
        )
        .round(3)
    )


# --- cleaning --------------------------------------------------------------

def clean(
    df: pd.DataFrame,
    zero_policy: str = "keep",
    drop_first_hours: float = 0.0,
    max_wait: float = 300.0,
) -> pd.DataFrame:
    """
    Drop rows that would corrupt the forecast.

    zero_policy:
        "keep"        treat zeros as real walk-ons
        "drop"        discard every zero reading
        "drop_early"  discard zeros only in the first `drop_first_hours`
                      of each day, keep the rest

    Closed rides are always dropped -- a closed ride has no wait, and
    including it as a zero is the same error in a different costume.
    """
    if zero_policy not in {"keep", "drop", "drop_early"}:
        raise ValueError(f"unknown zero_policy {zero_policy!r}")

    out = df[df["is_open"] == 1].copy()
    out = out[out["wait_minutes"].notna()]
    out = out[out["wait_minutes"] <= max_wait]   # guard against bad readings

    if zero_policy == "drop":
        out = out[out["wait_minutes"] > 0]
    elif zero_policy == "drop_early":
        opening = out.groupby("date")["slot"].transform("min")
        cutoff = opening + drop_first_hours * 60 / SLOT_MIN
        early_zero = (out["wait_minutes"] == 0) & (out["slot"] < cutoff)
        out = out[~early_zero]

    return out


def operating_slots(df: pd.DataFrame, coverage: float = 0.5) -> tuple[int, int]:
    """
    Infer the operating window from data rather than assuming it.

    Returns the first and last slot in which at least `coverage` of the
    park's rides were open. Slots outside this are park-closed, not quiet.
    """
    by_slot = df.groupby("slot")["ride_name"].nunique()
    n_rides = df["ride_name"].nunique()
    live = by_slot[by_slot >= coverage * n_rides]
    if live.empty:
        raise ValueError("no slot meets the coverage threshold")
    return int(live.index.min()), int(live.index.max())


# --- baseline --------------------------------------------------------------

def baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Historical mean wait by (ride, slot). The forecast to beat.

    Returns a frame indexed by ride_name with one column per slot.
    """
    return df.pivot_table(
        index="ride_name", columns="slot", values="wait_minutes", aggfunc="mean"
    )


# --- features and models ---------------------------------------------------

FEATURES = ["ride_code", "slot", "slot_sin", "slot_cos", "dow", "is_weekend", "month"]


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Small, deliberate feature set.

    With a few weeks of data, more features than this will overfit before
    they help. Slot enters as sin/cos as well as raw so a tree does not have
    to spend splits rediscovering that the day is a smooth curve.
    """
    out = df.copy()
    angle = 2 * np.pi * out["slot"] / SLOTS_PER_DAY
    out["slot_sin"] = np.sin(angle)
    out["slot_cos"] = np.cos(angle)
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["ride_code"] = out["ride_name"].astype("category").cat.codes
    return out


class MeanModel:
    """Wraps baseline() in the same interface as the fitted model."""

    def __init__(self, table: pd.DataFrame):
        self.table = table
        self.global_mean = float(np.nanmean(table.values))

    def predict_frame(self, df: pd.DataFrame) -> np.ndarray:
        def lookup(row):
            try:
                v = self.table.at[row["ride_name"], row["slot"]]
            except KeyError:
                return self.global_mean
            return self.global_mean if pd.isna(v) else v
        return df.apply(lookup, axis=1).to_numpy(dtype=float)


class GBModel:
    """Gradient-boosted trees over FEATURES."""

    def __init__(self, model, categories: dict):
        self.model = model
        self.categories = categories   # ride_name -> code, fixed at fit time

    def predict_frame(self, df: pd.DataFrame) -> np.ndarray:
        X = df.copy()
        X["ride_code"] = X["ride_name"].map(self.categories).fillna(-1)
        angle = 2 * np.pi * X["slot"] / SLOTS_PER_DAY
        X["slot_sin"], X["slot_cos"] = np.sin(angle), np.cos(angle)
        X["is_weekend"] = (X["dow"] >= 5).astype(int)
        return np.clip(self.model.predict(X[FEATURES]), 0, None)


def fit(train: pd.DataFrame) -> GBModel:
    """Fit gradient boosting. Only worth it with enough distinct days."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    feats = featurize(train)
    categories = dict(zip(feats["ride_name"], feats["ride_code"]))
    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.08, max_depth=6, random_state=0
    )
    model.fit(feats[FEATURES], feats["wait_minutes"])
    return GBModel(model, categories)


def evaluate(model, holdout: pd.DataFrame) -> dict:
    """MAE and RMSE on held-out rows."""
    pred = model.predict_frame(holdout)
    actual = holdout["wait_minutes"].to_numpy(dtype=float)
    err = pred - actual
    return {
        "n": len(actual),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
    }


def split_by_date(df: pd.DataFrame, n_holdout_days: int = 3):
    """
    Hold out the most recent days -- never a random row split.

    A random split leaks: rows five minutes apart on the same day are nearly
    the same observation, so the model would be scored on data it has
    effectively already seen. Splitting by date is the honest test.
    """
    days = sorted(df["date"].unique())
    if len(days) <= n_holdout_days:
        raise ValueError(f"only {len(days)} days of data; need more to hold out")
    cut = set(days[-n_holdout_days:])
    return df[~df["date"].isin(cut)], df[df["date"].isin(cut)]


def compare(df: pd.DataFrame, n_holdout_days: int = 3) -> pd.DataFrame:
    """Baseline vs fitted model on the same held-out days."""
    train, hold = split_by_date(df, n_holdout_days)
    rows = [
        {"model": "historical mean", **evaluate(MeanModel(baseline(train)), hold)},
        {"model": "gradient boosting", **evaluate(fit(train), hold)},
    ]
    return pd.DataFrame(rows).round(2)


# --- handoff to the optimizer ----------------------------------------------

def to_wait_dict(
    model,
    rides: list[str],
    slot_range: tuple[int, int],
    context: dict,
) -> dict[tuple[str, int], float]:
    """
    Build the {(ride, slot): minutes} dict optimize.Instance expects.

    `slot_range` is (first_slot, last_slot) from operating_slots(); slots are
    re-indexed to 0 at park open, which is the convention optimize.py uses.
    `context` supplies the day being planned, e.g. {"dow": 5, "month": 8}.

    Missing predictions are interpolated ALONG each ride's own curve and then
    filled at the edges. Defaulting a gap to zero would invent a walk-on that
    the optimizer would eagerly route into.
    """
    first, last = slot_range
    abs_slots = list(range(first, last + 1))

    grid = pd.DataFrame(
        [{"ride_name": r, "slot": s, **context} for r in rides for s in abs_slots]
    )
    grid["pred"] = model.predict_frame(grid)

    wide = grid.pivot(index="ride_name", columns="slot", values="pred")
    wide = wide.interpolate(axis=1, limit_direction="both")
    wide = wide.fillna(float(np.nanmean(wide.values)))

    return {
        (r, s - first): float(max(0.0, wide.at[r, s]))
        for r in rides
        for s in abs_slots
    }
