# Theme Park Capacity & Congestion Analysis

> Operations analysis of attraction throughput and guest wait times at
> [PARK NAME], using [N] observations collected over [X] weeks.

*Powered by [Queue-Times.com](https://queue-times.com/en-US)*

---

## Problem

Fill this in once you have data. The shape you want:

> At [park], posted standby waits regularly exceed [X] minutes at [N]
> attractions during peak hours, while [M] attractions sit below [Y]
> minutes. This project quantifies where the park's throughput constraints
> actually are, and evaluates two operational levers against them.

State a problem an operations manager has. Not "I wanted to practice
optimization."

## Data

- Source: Queue-Times.com real-time API, polled every 10 minutes
- Coverage: [date range], [N] attractions, [N] observations
- Ride design capacities: compiled from manufacturer specs / park published
  figures (see `capacities.csv`, with sources cited)

Note the limitation honestly: these are *posted* waits, which parks round
and pad. Say so, and say how you accounted for it.

## Analysis

### 1. Congestion profile
Where and when guest waiting accumulates, by attraction and hour.

### 2. Throughput vs. demand (Little's Law)
Separates attractions that are long-wait because they're popular from
attractions that are long-wait because they're running below design
capacity. Only the second category is fixable by operations.

### 3. Capacity allocation model
LP/MILP: given a fixed staffing budget in labor-hours, allocate crew across
attractions to minimize total guest-hours of waiting park-wide.

- Decision variables: crew assigned to each attraction per hour
- Objective: minimize sum of (throughput-implied queue length) across rides
- Constraints: total labor budget, minimum crew per open attraction,
  maximum capacity per attraction

**Result: [X]% reduction in total guest waiting versus observed baseline.**

### 4. Priority-lane simulation (SimPy)
Discrete-event model of a single attraction with batch service and two
merging arrival streams (standby + paid priority). Sweeps the share of
hourly capacity allocated to the paid lane and reports the effect on
standby wait.

**Result: standby wait degrades non-linearly beyond [X]% allocation.**

## Limitations

List them. Every real analysis has them, and naming them yourself is the
difference between a student project and an engineering document.

## Running it

```bash
pip install -r requirements.txt
python poller.py --list-parks
python poller.py --park-id <ID>     # schedule this every 10 minutes
python analyze.py
```
