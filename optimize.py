"""
Time-dependent orienteering with mandatory visits.

Given a set of rides the guest wants to ride and a forecast of wait times by
time of day, choose the start time for each ride so that total time spent in
queues is minimized, subject to walking times and the park's operating hours.

Formulation
-----------
Sets
    R   selected rides
    T   time slots (SLOT_MIN-minute bins across the operating day)

Decision variables
    x[i,t] in {0,1}   join the queue for ride i at slot t
    y[i,j] in {0,1}   ride i is done before ride j   (defined for i < j)

Derived (linear in x)
    s[i] = sum_t (t * SLOT_MIN) * x[i,t]              start of queueing
    f[i] = s[i] + sum_t w[i,t] * x[i,t] + r[i]        exit from the ride

Objective
    min sum_{i,t} w[i,t] * x[i,t]

Constraints
    (1) sum_t x[i,t] == 1                             each ride exactly once
    (2) s[j] >= f[i] + d[i,j] - M(1 - y[i,j])         disjunctive sequencing
        s[i] >= f[j] + d[j,i] - M*y[i,j]
    (3) f[i] <= horizon                               finish before close
    (4) x[i,t] == 0 where infeasible or ride closed

All times are in MINUTES SINCE PARK OPEN. This is the single most common
source of bugs in this model -- see assert_units() below.
"""

from dataclasses import dataclass, field

import pulp

SLOT_MIN = 15  # minutes per time slot


@dataclass
class Instance:
    """One solvable problem: which rides, what waits, how far apart."""

    rides: list[str]
    horizon: int                          # minutes from open to close
    wait: dict[tuple[str, int], float]    # (ride, slot) -> predicted wait
    duration: dict[str, float]            # ride -> on-ride minutes
    walk: dict[tuple[str, str], float]    # (from, to) -> walking minutes
    closed: set[tuple[str, int]] = field(default_factory=set)

    @property
    def slots(self) -> list[int]:
        return list(range(self.horizon // SLOT_MIN))

    def assert_units(self) -> None:
        """Catch the unit errors that make this model silently wrong."""
        assert self.horizon > 60, "horizon looks like slots, not minutes"
        for i in self.rides:
            assert 0 < self.duration[i] < 30, f"{i}: duration {self.duration[i]} not in minutes"
        for (a, b), v in self.walk.items():
            assert 0 <= v < 60, f"walk {a}->{b} = {v} not in minutes"
        for (i, t), v in self.wait.items():
            assert 0 <= v < 400, f"wait {i}@{t} = {v} not in minutes"


def build(inst: Instance, objective: str = "queue") -> tuple:
    """Construct the MILP. objective: 'queue' or 'finish'."""
    inst.assert_units()
    R, T = inst.rides, inst.slots

    # Tightest valid big-M: nothing can be pushed past the horizon plus the
    # longest walk. Using 1e6 here would make the LP relaxation useless and
    # branch-and-bound crawl.
    M = inst.horizon + max(inst.walk.values(), default=0)

    prob = pulp.LpProblem("touring_plan", pulp.LpMinimize)

    x = pulp.LpVariable.dicts("x", (R, T), cat="Binary")
    # One ordering variable per unordered pair -- (i,j) and (j,i) describe the
    # same relationship, so only i < j is needed.
    pairs = [(i, j) for n, i in enumerate(R) for j in R[n + 1:]]
    y = pulp.LpVariable.dicts("y", pairs, cat="Binary")

    def start(i):
        return pulp.lpSum(t * SLOT_MIN * x[i][t] for t in T)

    def finish(i):
        queued = pulp.lpSum(inst.wait[i, t] * x[i][t] for t in T)
        return start(i) + queued + inst.duration[i]

    # --- objective ---
    if objective == "queue":
        prob += pulp.lpSum(inst.wait[i, t] * x[i][t] for i in R for t in T)
    elif objective == "finish":
        # min max_i finish[i], linearized with an epigraph variable
        z = pulp.LpVariable("makespan", lowBound=0)
        for i in R:
            prob += z >= finish(i)
        prob += z
    else:
        raise ValueError(f"unknown objective {objective!r}")

    # --- (1) each ride exactly once ---
    for i in R:
        prob += pulp.lpSum(x[i][t] for t in T) == 1, f"once_{i}"

    # --- (2) disjunctive sequencing ---
    for i, j in pairs:
        prob += start(j) >= finish(i) + inst.walk[i, j] - M * (1 - y[i, j]), f"seq_{i}_{j}"
        prob += start(i) >= finish(j) + inst.walk[j, i] - M * y[i, j], f"seq_{j}_{i}"

    # --- (3) finish before close ---
    for i in R:
        prob += finish(i) <= inst.horizon, f"close_{i}"

    # --- (4) forbid infeasible or closed start slots ---
    # Pre-filtering here shrinks the search space a lot; the solver never has
    # to consider starts that could not possibly finish before closing.
    for i in R:
        for t in T:
            too_late = t * SLOT_MIN + inst.wait[i, t] + inst.duration[i] > inst.horizon
            if too_late or (i, t) in inst.closed:
                prob += x[i][t] == 0, f"block_{i}_{t}"

    return prob, x, y


def solve(inst: Instance, objective: str = "queue", verbose: bool = False):
    prob, x, _ = build(inst, objective)
    prob.solve(pulp.PULP_CBC_CMD(msg=verbose))

    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        return {"status": status, "plan": None}

    plan = []
    for i in inst.rides:
        t = next(t for t in inst.slots if x[i][t].value() > 0.5)
        s = t * SLOT_MIN
        w = inst.wait[i, t]
        plan.append({
            "ride": i,
            "join_queue": s,
            "wait": w,
            "board": s + w,
            "exit": s + w + inst.duration[i],
        })
    plan.sort(key=lambda p: p["join_queue"])

    return {
        "status": status,
        "objective": pulp.value(prob.objective),
        "total_queue_min": sum(p["wait"] for p in plan),
        "finish_min": max(p["exit"] for p in plan),
        "plan": plan,
    }


def verify(inst: Instance, plan: list[dict]) -> list[str]:
    """Independently re-check the solution. Never trust a solver blindly."""
    errors = []
    for a, b in zip(plan, plan[1:]):
        gap = b["join_queue"] - a["exit"]
        need = inst.walk[a["ride"], b["ride"]]
        if gap < need - 1e-6:
            errors.append(
                f"{a['ride']} -> {b['ride']}: {gap:.0f} min gap, needs {need:.0f}"
            )
    if len({p["ride"] for p in plan}) != len(inst.rides):
        errors.append("ride count mismatch")
    if plan and plan[-1]["exit"] > inst.horizon + 1e-6:
        errors.append("finishes after close")
    return errors


def show(result: dict, open_hour: int = 10) -> None:
    if result["plan"] is None:
        print(f"No solution: {result['status']}")
        return

    def clock(m):
        return f"{open_hour + int(m) // 60:2d}:{int(m) % 60:02d}"

    print(f"{'ride':<34}{'queue':>7}{'board':>7}{'exit':>7}{'wait':>7}")
    print("-" * 62)
    for p in result["plan"]:
        print(f"{p['ride']:<34}{clock(p['join_queue']):>7}"
              f"{clock(p['board']):>7}{clock(p['exit']):>7}{p['wait']:>6.0f}m")
    print("-" * 62)
    print(f"total queueing: {result['total_queue_min']:.0f} min   "
          f"done by {clock(result['finish_min'])}")
