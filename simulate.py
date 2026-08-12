"""
Discrete-event model of a single attraction. SimPy.

Why this exists. The MILP treats waits as deterministic constants. Reality
has random arrivals and batch dispatch, and queues behave non-linearly as
utilization approaches capacity -- a system at 95% of capacity does not have
a wait 5% worse than one at 90%, it has one several times worse. This model
answers two questions an LP structurally cannot:

    1. Does an optimizer plan survive realistic variability, or does one
       longer-than-forecast wait cascade through the rest of the day?
    2. How does standby wait degrade as paid-priority share increases?

Units are MINUTES throughout, matching optimize.py.
"""

import math
import random
import statistics
from dataclasses import dataclass, field

import simpy


@dataclass
class Rider:
    arrived: float
    priority: bool
    boarded: float | None = None

    @property
    def wait(self) -> float:
        return self.boarded - self.arrived


@dataclass
class Attraction:
    """
    Batch-service queue: a vehicle of `train_size` departs roughly every
    `dispatch_min`, drawing from the priority line first, then standby.

    priority_share is the fraction of each train's seats HELD for priority
    riders. Unclaimed priority seats fall through to standby rather than
    dispatching empty -- real operations do this, and modelling it otherwise
    would overstate the harm of the priority lane.
    """

    env: simpy.Environment
    train_size: int
    dispatch_min: float
    priority_share: float = 0.0
    dispatch_cv: float = 0.15          # coefficient of variation on interval
    rng: random.Random = field(default_factory=random.Random)

    standby: list = field(default_factory=list)
    priority: list = field(default_factory=list)
    done: list = field(default_factory=list)
    balked: int = 0

    def __post_init__(self):
        self.env.process(self.dispatch_loop())

    def arrive(self, is_priority: bool) -> None:
        rider = Rider(arrived=self.env.now, priority=is_priority)
        (self.priority if is_priority else self.standby).append(rider)

    def _interval(self) -> float:
        """
        Lognormal dispatch interval. Real intervals vary with loading -- a
        fixed interval would make the queue far better behaved than reality
        and quietly invalidate every conclusion drawn from the model.
        """
        sigma = math.sqrt(math.log(1 + self.dispatch_cv ** 2))
        mu = math.log(self.dispatch_min) - sigma ** 2 / 2
        return self.rng.lognormvariate(mu, sigma)

    def dispatch_loop(self):
        while True:
            yield self.env.timeout(self._interval())

            seats = self.train_size
            held = int(seats * self.priority_share)

            # priority first, up to its held allocation
            for _ in range(min(held, len(self.priority))):
                r = self.priority.pop(0)
                r.boarded = self.env.now
                self.done.append(r)
                seats -= 1

            # standby fills everything left, including unclaimed priority seats
            for _ in range(min(seats, len(self.standby))):
                r = self.standby.pop(0)
                r.boarded = self.env.now
                self.done.append(r)


def arrival_process(env, attraction: Attraction, rate_by_hour: dict[int, float],
                    priority_share_of_demand: float, rng: random.Random):
    """
    Non-stationary Poisson arrivals: rate changes hour to hour.

    Implemented as piecewise-constant exponential interarrivals. A constant
    rate would badly understate peak congestion, which is the only part of
    the day anyone cares about.
    """
    while True:
        hour = int(env.now // 60)
        rate = rate_by_hour.get(hour, 0.0)          # riders per hour
        if rate <= 0:
            yield env.timeout(60 - (env.now % 60))  # skip to next hour
            continue
        yield env.timeout(rng.expovariate(rate / 60.0))
        attraction.arrive(rng.random() < priority_share_of_demand)


def run(train_size: int, dispatch_min: float, rate_by_hour: dict[int, float],
        priority_share: float = 0.0, priority_demand: float = 0.0,
        hours: int = 12, seed: int = 0) -> dict:
    """
    One replication. Returns wait distribution stats by queue type.

    Percentiles matter more than the mean here -- the mean hides the tail
    that guests actually complain about.
    """
    rng = random.Random(seed)
    env = simpy.Environment()
    ride = Attraction(env=env, train_size=train_size, dispatch_min=dispatch_min,
                      priority_share=priority_share, rng=rng)
    env.process(arrival_process(env, ride, rate_by_hour, priority_demand, rng))
    env.run(until=hours * 60)

    def stats(riders):
        waits = sorted(r.wait for r in riders)
        if not waits:
            return {"n": 0}
        return {
            "n": len(waits),
            "mean": round(statistics.mean(waits), 1),
            "p50": round(waits[len(waits) // 2], 1),
            "p90": round(waits[int(len(waits) * 0.90)], 1),
            "p99": round(waits[int(len(waits) * 0.99)], 1),
        }

    served = ride.done
    return {
        "standby": stats([r for r in served if not r.priority]),
        "priority": stats([r for r in served if r.priority]),
        "unserved_standby": len(ride.standby),
        "unserved_priority": len(ride.priority),
        "throughput": len(served),
    }


def sweep_priority_share(shares: list[float], reps: int = 20, **kwargs) -> list[dict]:
    """
    Run `reps` replications at each priority share and report mean standby
    wait with a 95% CI.

    Multiple replications are not optional. A single run of a stochastic
    model tells you nothing about whether a difference between two shares is
    real or is just the seed.
    """
    rows = []
    for share in shares:
        means = []
        for rep in range(reps):
            out = run(priority_share=share, seed=1000 + rep, **kwargs)
            if out["standby"].get("n"):
                means.append(out["standby"]["mean"])
        if not means:
            continue
        m = statistics.mean(means)
        half = (1.96 * statistics.stdev(means) / math.sqrt(len(means))
                if len(means) > 1 else 0.0)
        rows.append({
            "priority_share": share,
            "standby_mean": round(m, 1),
            "ci_low": round(m - half, 1),
            "ci_high": round(m + half, 1),
            "reps": len(means),
        })
    return rows


def validate(observed_waits: list[float], simulated_waits: list[float]) -> dict:
    """
    Does the model reproduce reality? Compare the simulated wait distribution
    against observed posted waits for the same ride and hours.

    An unvalidated simulation is an animation. Report this either way -- if
    the distributions do not match, say so and say what you think is missing
    (breakdowns, capacity changes, posted-wait padding).
    """
    if not observed_waits or not simulated_waits:
        return {"status": "insufficient data"}

    obs, sim = sorted(observed_waits), sorted(simulated_waits)

    def pct(xs, q):
        return xs[min(int(len(xs) * q), len(xs) - 1)]

    return {
        "obs_mean": round(statistics.mean(obs), 1),
        "sim_mean": round(statistics.mean(sim), 1),
        "obs_p90": round(pct(obs, 0.90), 1),
        "sim_p90": round(pct(sim, 0.90), 1),
        "mean_gap_pct": round(
            100 * (statistics.mean(sim) - statistics.mean(obs))
            / max(statistics.mean(obs), 1e-9), 1
        ),
    }
