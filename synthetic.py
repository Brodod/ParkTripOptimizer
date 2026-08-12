"""
Synthetic instance generator + greedy baselines.

Lets you develop and test optimize.py before enough real observations have
accumulated. The wait curve here is a rough caricature of a real park day:
low at open, peak early afternoon, tapering toward close.
"""

import math
import random

from optimize import SLOT_MIN, Instance


def make_instance(n_rides=8, horizon=720, seed=0) -> Instance:
    rng = random.Random(seed)
    rides = [f"ride_{k:02d}" for k in range(n_rides)]
    slots = list(range(horizon // SLOT_MIN))

    # Scatter rides over a ~1km square park, then convert distance to walking
    # minutes at 3 mph with a 1.3x detour factor for paths that aren't straight.
    pos = {i: (rng.uniform(0, 1000), rng.uniform(0, 1000)) for i in rides}
    walk = {}
    for a in rides:
        for b in rides:
            if a == b:
                continue
            dx = pos[a][0] - pos[b][0]
            dy = pos[a][1] - pos[b][1]
            meters = math.hypot(dx, dy) * 1.3
            walk[a, b] = round(meters / 80.0, 1)  # 80 m/min ~= 3 mph

    duration = {i: rng.uniform(2, 6) for i in rides}

    # popularity scales the whole curve; peak is early afternoon
    wait = {}
    for i in rides:
        popularity = rng.uniform(0.3, 1.0)
        for t in slots:
            minutes = t * SLOT_MIN
            shape = math.exp(-((minutes - 300) ** 2) / (2 * 150 ** 2))
            base = 90 * popularity * shape
            wait[i, t] = round(max(0.0, base + rng.gauss(0, 4)), 1)

    return Instance(rides=rides, horizon=horizon, wait=wait,
                    duration=duration, walk=walk)


# --- baselines -------------------------------------------------------------
# A solver result is meaningless without something to compare it against.
# These are what an actual guest does.

def _greedy(inst: Instance, pick) -> dict:
    """Shared walk: repeatedly pick the next ride by some rule."""
    remaining = set(inst.rides)
    now, here, plan = 0.0, None, []

    while remaining:
        choice = pick(inst, remaining, here, now)
        if choice is None:
            break
        travel = inst.walk[here, choice] if here else 0.0
        arrive = now + travel
        t = min(int(arrive // SLOT_MIN), len(inst.slots) - 1)
        w = inst.wait[choice, t]
        exit_at = arrive + w + inst.duration[choice]
        if exit_at > inst.horizon:
            break
        plan.append({"ride": choice, "join_queue": arrive, "wait": w,
                     "board": arrive + w, "exit": exit_at})
        now, here = exit_at, choice
        remaining.discard(choice)

    return {
        "status": "Complete" if not remaining else "Partial",
        "plan": plan,
        "total_queue_min": sum(p["wait"] for p in plan),
        "finish_min": plan[-1]["exit"] if plan else 0,
        "rides_done": len(plan),
    }


def greedy_nearest(inst: Instance) -> dict:
    """Always walk to the closest unvisited ride."""
    def pick(inst, remaining, here, now):
        if here is None:
            return min(remaining)
        return min(remaining, key=lambda r: inst.walk[here, r])
    return _greedy(inst, pick)


def greedy_shortest_wait(inst: Instance) -> dict:
    """Always go to whatever has the lowest wait right now."""
    def pick(inst, remaining, here, now):
        t = min(int(now // SLOT_MIN), len(inst.slots) - 1)
        return min(remaining, key=lambda r: inst.wait[r, t])
    return _greedy(inst, pick)


def popularity_order(inst: Instance) -> dict:
    """Big rides first -- what most guests actually do."""
    order = sorted(inst.rides,
                   key=lambda r: -max(inst.wait[r, t] for t in inst.slots))
    seq = iter(order)

    def pick(inst, remaining, here, now):
        for r in order:
            if r in remaining:
                return r
        return None
    return _greedy(inst, pick)
