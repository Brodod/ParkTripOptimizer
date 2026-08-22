

import sys

import numpy as np
import pandas as pd

import forecast as F

BAR = "=" * 64


def section(title: str) -> None:
    print(f"\n{BAR}\n{title}\n{BAR}")


def collection_health(df: pd.DataFrame) -> None:
    """Did the poller run when it was supposed to?"""
    section("1. COLLECTION HEALTH")

    polls = df["polled_at"].drop_duplicates().sort_values()
    days = df["date"].nunique()
    span = (polls.max() - polls.min()).days + 1

    print(f"rows                {len(df):,}")
    print(f"distinct polls      {len(polls):,}")
    print(f"days with data      {days} (calendar span {span})")
    print(f"first poll          {polls.min()}")
    print(f"last poll           {polls.max()}")

    if days < span:
        print(f"\n  ** {span - days} calendar day(s) have NO data. "
              f"Check the Actions tab for failed or skipped runs.")

    # Real sampling interval, not the one the cron requests.
    gaps = polls.diff().dt.total_seconds().div(60).dropna()
    intraday = gaps[gaps < 120]          # ignore overnight gaps
    if len(intraday):
        print(f"\nsampling interval (intraday gaps under 2h):")
        print(f"  median            {intraday.median():.1f} min")
        print(f"  mean              {intraday.mean():.1f} min")
        print(f"  p90               {intraday.quantile(0.90):.1f} min")
        print(f"  max               {intraday.max():.1f} min")
        print("\n  Report the MEDIAN in your README. GitHub's scheduler is")
        print("  queued, not punctual, so this will exceed your cron interval.")

    big = gaps[(gaps >= 120) & (gaps < 60 * 10)]
    if len(big):
        print(f"\n  ** {len(big)} mid-day gap(s) over 2h — poller outages, "
              f"largest {big.max() / 60:.1f}h")


def ride_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Which rides have enough observations to model?"""
    section("2. RIDE COVERAGE")

    cov = (
        df.groupby("ride_name")
        .agg(n=("wait_minutes", "size"),
             days=("date", "nunique"),
             open_share=("is_open", "mean"),
             mean_wait=("wait_minutes", lambda s: s[s > 0].mean()),
             max_wait=("wait_minutes", "max"))
        .sort_values("mean_wait", ascending=False)
        .round(2)
    )
    print(cov.to_string())

    thin = cov[cov["days"] < 5]
    if len(thin):
        print(f"\n  ** {len(thin)} ride(s) seen on fewer than 5 days — "
              f"too thin to forecast, consider excluding")
    return cov


def zero_analysis(df: pd.DataFrame) -> None:
    """The decision that changes every downstream number."""
    section("3. ZERO ANALYSIS  -> pick your zero_policy from this")

    z = F.diagnose_zeros(df)
    print(z.to_string())

    open_rides = df[df["is_open"] == 1]
    by_hour = open_rides.groupby("hour")["wait_minutes"].apply(lambda s: (s == 0).mean())
    if by_hour.empty:
        return

    first, rest = by_hour.iloc[0], by_hour.iloc[1:4].mean()
    print(f"\nfirst operating hour zero share : {first:.0%}")
    print(f"next three hours                : {rest:.0%}")

    print("\nREAD IT LIKE THIS:")
    if first > 0.6 and rest < first / 2:
        print("  Sharp cliff after opening -> those early zeros are")
        print("  PLACEHOLDERS, not walk-ons. Use:")
        print('      zero_policy="drop_early", drop_first_hours=1.0')
    elif first < 0.35:
        print("  No opening spike -> zeros look like real walk-ons. Use:")
        print('      zero_policy="keep"')
    else:
        print("  Ambiguous. Plot a single ride's full day and look at it")
        print("  before deciding. Whatever you choose, say so in the README.")


def wait_profile(df: pd.DataFrame, top_n: int = 8) -> None:
    """Does the day have the shape you'd expect?"""
    section("4. WAIT PROFILE BY HOUR (top rides, zeros excluded)")

    nz = df[(df["is_open"] == 1) & (df["wait_minutes"] > 0)]
    if nz.empty:
        print("  no non-zero waits recorded")
        return

    top = nz.groupby("ride_name")["wait_minutes"].mean().nlargest(top_n).index
    table = (
        nz[nz["ride_name"].isin(top)]
        .pivot_table(index="ride_name", columns="hour",
                     values="wait_minutes", aggfunc="mean")
        .round(0)
    )
    print(table.to_string())
    print("\n  Expect a rise through midday and a taper toward close.")
    print("  A flat profile means either a quiet period or a data problem.")


def weekday_effect(df: pd.DataFrame) -> None:
    """Is there a weekend signal for the model to learn?"""
    section("5. DAY-OF-WEEK EFFECT")

    nz = df[(df["is_open"] == 1) & (df["wait_minutes"] > 0)]
    if nz.empty:
        return
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_dow = nz.groupby("dow")["wait_minutes"].agg(["mean", "size"]).round(1)
    by_dow.index = [names[i] for i in by_dow.index]
    print(by_dow.to_string())

    wknd = nz[nz["dow"] >= 5]["wait_minutes"].mean()
    week = nz[nz["dow"] < 5]["wait_minutes"].mean()
    if week and not np.isnan(wknd):
        print(f"\nweekend / weekday ratio: {wknd / week:.2f}x")
        if wknd / week < 1.1:
            print("  Weak weekend signal — day-of-week may not help the model.")


def capacity_join(df: pd.DataFrame, path: str = "capacities.csv") -> None:
    """Do capacities.csv ride names actually match the observations?"""
    section("6. CAPACITY FILE JOIN")

    try:
        caps = pd.read_csv(path)
    except FileNotFoundError:
        print(f"  {path} not found — skipping")
        return

    obs_names = set(df["ride_name"].unique())
    cap_names = set(caps["ride_name"].dropna())

    missing = sorted(obs_names - cap_names)
    orphan = sorted(cap_names - obs_names)

    print(f"rides observed      {len(obs_names)}")
    print(f"rides in capacities {len(cap_names)}")
    print(f"matched             {len(obs_names & cap_names)}")

    if missing:
        print(f"\n  ** in data but NOT in capacities.csv ({len(missing)}):")
        for n in missing[:10]:
            print(f"       {n!r}")
    if orphan:
        print(f"\n  ** in capacities.csv but NOT in data ({len(orphan)}):")
        for n in orphan[:10]:
            print(f"       {n!r}")
        print("     Usually a typo or a stray character — the repr above shows it.")

    if "source_type" in caps.columns:
        print("\nprovenance:")
        print(caps["source_type"].fillna("(blank)").value_counts().to_string())


def readiness(df: pd.DataFrame) -> None:
    """Is there enough here to move on to forecasting?"""
    section("7. READINESS")

    days = df["date"].nunique()
    weekends = df[df["dow"] >= 5]["date"].nunique()
    rides_ok = (df.groupby("ride_name")["date"].nunique() >= 5).sum()

    checks = [
        (days >= 10, f"{days} days of data (want 10+ before fitting a model)"),
        (weekends >= 2, f"{weekends} weekend days (want 2+ for a dow signal)"),
        (rides_ok >= 5, f"{rides_ok} rides with 5+ days of coverage"),
        (days >= 6, f"{days} days — enough to hold out 3 and still train"),
    ]
    for ok, msg in checks:
        print(f"  [{'OK ' if ok else '   '}] {msg}")

    if all(ok for ok, _ in checks):
        print("\n  Ready. Next: python -c \"import forecast as F; "
              "print(F.compare(F.clean(F.load()), 3))\"")
    else:
        print("\n  Keep collecting. The pipeline will run on what you have,")
        print("  but held-out results will be too noisy to mean much.")


def main(path: str = F.DATA_PATH) -> int:
    try:
        df = F.load(path)
    except FileNotFoundError:
        print(f"No data at {path}.")
        print("Pull it down first:")
        print("  git fetch origin data")
        print("  mkdir -p data && git show origin/data:data/observations.csv "
              "> data/observations.csv")
        return 1

    print(f"auditing {path}")
    collection_health(df)
    ride_coverage(df)
    zero_analysis(df)
    wait_profile(df)
    weekday_effect(df)
    capacity_join(df)
    readiness(df)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else F.DATA_PATH))
