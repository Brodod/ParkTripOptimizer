"""
Evaluation: does the optimizer actually beat what a guest would do anyway?

The only honest test. Forecast on training days, optimize a plan, then score
that plan against ACTUAL observed waits on a held-out day, alongside the
greedy baselines.

Why replay is not just "add up the planned waits". A plan is a SEQUENCE, not
a fixed schedule. If the first ride's real wait runs 20 minutes over
forecast, everything downstream shifts and you meet different waits than the
plan assumed. Scoring a plan against its own forecast is circular -- of
course the optimizer wins on the numbers it was handed. Replaying the
sequence forward against reality is the whole point of this file.
"""

import math
import random

import pandas as pd

import forecast as F
from optimize import SLOT_MIN, Instance, solve
from synthetic import greedy_nearest, greedy_shortest_wait, popularity_order


def actual_wait_lookup(day_df: pd.DataFrame, first_slot: int) -> dict:
    """
    {(ride, slot_since_open): observed wait} for one real day.

    Slots are re-indexed to 0 at park open to match optimize.py.
    """
    grid = day_df.pivot_table(
        index="ride_name", columns="slot", values="wait_minutes", aggfunc="mean"
    ).interpolate(axis=1, limit_direction="both")
    return {
        (r, s - first_slot): float(grid.at[r, s])
        for r in grid.index
        for s in grid.columns
        if not pd.isna(grid.at[r, s])
    }


def replay(sequence: list[str], inst: Instance, actual: dict) -> dict:
    """
    Walk a ride ORDER forward against real observed waits.

    Takes the order the method chose and re-times it: at each step you arrive
    when you arrive, and you meet whatever wait was actually posted then --
    not the one that was forecast. Rides that no longer fit before close are
    counted as not completed rather than silently dropped.
    """
    n_slots = len(inst.slots)
    now, here = 0.0, None
    total_queue, completed, dropped = 0.0, [], []

    for ride in sequence:
        travel = inst.walk[here, ride] if here else 0.0
        arrive = now + travel
        slot = min(int(arrive // SLOT_MIN), n_slots - 1)

        # Fall back to the forecast only if that slot was never observed.
        w = actual.get((ride, slot), inst.wait.get((ride, slot), 0.0))
        exit_at = arrive + w + inst.duration[ride]

        if exit_at > inst.horizon:
            dropped.append(ride)
            continue

        total_queue += w
        completed.append({"ride": ride, "join": arrive, "wait": w, "exit": exit_at})
        now, here = exit_at, ride

    return {
        "queue_min": round(total_queue, 1),
        "finish_min": round(completed[-1]["exit"], 1) if completed else 0.0,
        "rides_done": len(completed),
        "dropped": dropped,
        "plan": completed,
    }


def build_instance(model, rides: list[str], slot_range: tuple[int, int],
                   context: dict, walk: dict, duration: dict) -> Instance:
    """Forecast -> Instance, ready for the optimizer."""
    first, last = slot_range
    wait = F.to_wait_dict(model, rides, slot_range, context)
    return Instance(
        rides=rides,
        horizon=(last - first + 1) * SLOT_MIN,
        wait=wait,
        duration=duration,
        walk=walk,
    )


def evaluate_day(day_df: pd.DataFrame, inst: Instance, first_slot: int,
                 objective: str = "finish") -> list[dict]:
    """
    Score every method on one held-out day. Returns one row per method.
    """
    actual = actual_wait_lookup(day_df, first_slot)
    rows = []

    opt = solve(inst, objective=objective)
    if opt["plan"]:
        order = [p["ride"] for p in opt["plan"]]
        rows.append({"method": f"MILP ({objective})", **replay(order, inst, actual)})

    for label, fn in [
        ("greedy nearest", greedy_nearest),
        ("greedy shortest wait", greedy_shortest_wait),
        ("popularity order", popularity_order),
    ]:
        order = [p["ride"] for p in fn(inst)["plan"]]
        rows.append({"method": label, **replay(order, inst, actual)})

    return rows


def make_geometry(rides: list[str], seed: int = 1) -> tuple[dict, dict]:
    """
    Placeholder walk times and ride durations.

    TODO: replace with real coordinates from the park map and real durations
    from RCDB. Until then every result here carries that caveat, and the
    README must say so.
    """
    rng = random.Random(seed)
    pos = {r: (rng.uniform(0, 900), rng.uniform(0, 900)) for r in rides}
    walk = {
        (a, b): round(math.hypot(pos[a][0] - pos[b][0],
                                 pos[a][1] - pos[b][1]) * 1.3 / 80, 1)
        for a in rides for b in rides if a != b
    }
    duration = {r: 3.0 for r in rides}
    return walk, duration


def run_all(rides: list[str], n_holdout_days: int = 3,
            zero_policy: str = "drop_early", drop_first_hours: float = 1.0,
            objective: str = "finish") -> pd.DataFrame:
    """
    Full pipeline over held-out days. Prints a comparison table.

    Reports the spread as well as the mean: a method that wins on average but
    loses badly on a third of days is worth knowing about, and the mean alone
    hides it.
    """
    df = F.clean(F.load(), zero_policy=zero_policy,
                 drop_first_hours=drop_first_hours)
    first, last = F.operating_slots(df)
    train, hold = F.split_by_date(df, n_holdout_days)
    model = F.fit(train)

    walk, duration = make_geometry(rides)
    all_rows = []

    for day in sorted(hold["date"].unique()):
        day_df = hold[hold["date"] == day]
        dow = pd.Timestamp(day).dayofweek
        inst = build_instance(model, rides, (first, last),
                              {"dow": dow, "month": pd.Timestamp(day).month},
                              walk, duration)
        for row in evaluate_day(day_df, inst, first, objective):
            all_rows.append({"date": day, **row})

    raw = pd.DataFrame(all_rows)
    summary = (
        raw.groupby("method")
        .agg(queue_mean=("queue_min", "mean"), queue_sd=("queue_min", "std"),
             finish_mean=("finish_min", "mean"),
             rides_mean=("rides_done", "mean"), days=("date", "nunique"))
        .round(1)
        .sort_values("finish_mean")
    )
    return summary
