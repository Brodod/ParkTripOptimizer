"""
Operator-side analysis of collected queue data.

Two outputs:
  1. Wait profile by ride by hour  -> where and when the park is congested
  2. Little's Law throughput check -> which rides are the real constraints

Little's Law:  L = lambda * W
  L      = people in queue
  lambda = effective arrival rate (people/hour)
  W      = wait time (hours)

You cannot see L directly from the API. But if you know a ride's design
capacity C (riders/hour, published for most major coasters), then for a
ride running at capacity, lambda ~= C, and:

      expected queue length L = C * (wait_minutes / 60)

Comparing observed wait against design capacity tells you whether a long
wait is a DEMAND problem (popular ride, running fine) or a THROUGHPUT
problem (ride running below spec). That distinction is the whole point:
the second one is fixable by operations, the first one is not.

Usage:
    python analyze.py                       # uses data/observations.csv + capacities.csv
"""

import pandas as pd

DATA_PATH = "data/observations.csv"
CAPACITY_PATH = "capacities.csv"  # columns: ride_name, design_capacity_hr


def load(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["polled_at"])
    df = df[df["is_open"] == 1].copy()
    local = df["polled_at"].dt.tz_convert("America/Chicago")
    df["hour"] = local.dt.hour
    df["date"] = local.dt.date
    return df


def wait_profile(df):
    """Average posted wait by ride and hour of day."""
    profile = (
        df.pivot_table(
            index="ride_name", columns="hour", values="wait_minutes", aggfunc="mean"
        )
        .round(1)
    )
    return profile


def bottlenecks(df, capacity_path=CAPACITY_PATH):
    """Rank rides by implied queue length -- your congestion constraints."""
    try:
        caps = pd.read_csv(capacity_path)
    except FileNotFoundError:
        print(f"No {capacity_path} found. Create it with columns: "
              "ride_name, design_capacity_hr")
        return None

    summary = (
        df.groupby("ride_name")["wait_minutes"]
        .agg(mean_wait="mean", p90_wait=lambda s: s.quantile(0.90), n="count")
        .reset_index()
    )
    summary = summary.merge(caps, on="ride_name", how="left")

    # --- merge check ---
    #Only warn about rides that actually generate waits (flat/kiddie rides do not hurt)
    missing = summary[summary["design_capacity_hr"].isna() & (summary["mean_wait"] > 5)]["ride_name"].tolist()
    if missing:
        print (f"WARNING: {len(missing)} rides with real waits have no capacity:")
        for name in missing:
            print(f" - {name!r}")

    # Guest-hours of waiting generated per operating hour. This is the number
    # an operations manager actually cares about minimizing park-wide.
    summary["guest_hours_per_hr"] = (
        summary["design_capacity_hr"] * summary["mean_wait"] / 60
    ).round(1)

    return summary.sort_values("guest_hours_per_hr", ascending=False)


if __name__ == "__main__":
    df = load()
    print(f"Loaded {len(df):,} open-ride observations "
          f"across {df['date'].nunique()} days\n")

    print("=== Average posted wait by hour ===")
    print(wait_profile(df).to_string())

    print("\n=== Congestion ranking ===")
    result = bottlenecks(df)
    if result is not None:
        print(result.to_string(index=False))
