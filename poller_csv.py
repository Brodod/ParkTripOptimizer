"""
Polls the Queue-Times.com real-time API and appends observations to a CSV.

CSV (not SQLite) so that git diffs stay readable and GitHub Actions can
commit results without rewriting a binary blob every run.

Data attribution required: "Powered by Queue-Times.com"

Usage:
    python poller.py --list-parks       # find your park's ID
    python poller.py --park-id 57       # poll once and append
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timezone

import requests

BASE = "https://queue-times.com"
DATA_PATH = os.path.join("data", "observations.csv")

FIELDS = [
    "park_id",
    "ride_id",
    "ride_name",
    "land_name",
    "is_open",
    "wait_minutes",
    "last_updated",
    "polled_at",
]


def list_parks():
    groups = requests.get(f"{BASE}/parks.json", timeout=20).json()
    for group in groups:
        print(f"\n{group['name']}")
        for park in group["parks"]:
            print(f"  {park['id']:>4}  {park['name']}  ({park['country']})")


def existing_keys(path):
    """(ride_id, last_updated) pairs already recorded, so reruns don't duplicate."""
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {(r["ride_id"], r["last_updated"]) for r in csv.DictReader(f)}


def poll(park_id, path=DATA_PATH):
    payload = requests.get(
        f"{BASE}/parks/{park_id}/queue_times.json", timeout=20
    ).json()
    polled_at = datetime.now(timezone.utc).isoformat()

    # Rides appear either nested under lands or in a flat top-level list.
    pairs = [(land["name"], ride)
             for land in payload.get("lands", [])
             for ride in land.get("rides", [])]
    pairs += [(None, ride) for ride in payload.get("rides", [])]

    seen = existing_keys(path)
    rows = []
    for land_name, ride in pairs:
        key = (str(ride["id"]), str(ride.get("last_updated")))
        if key in seen:
            continue
        rows.append({
            "park_id": park_id,
            "ride_id": ride["id"],
            "ride_name": ride["name"],
            "land_name": land_name or "",
            "is_open": int(ride["is_open"]),
            "wait_minutes": ride.get("wait_time"),
            "last_updated": ride.get("last_updated"),
            "polled_at": polled_at,
        })

    os.makedirs(os.path.dirname(path), exist_ok=True)
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)

    print(f"{polled_at}  {len(pairs)} rides seen, {len(rows)} new rows appended")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-parks", action="store_true")
    parser.add_argument("--park-id", type=int)
    args = parser.parse_args()

    if args.list_parks:
        list_parks()
    elif args.park_id:
        poll(args.park_id)
    else:
        parser.print_help()
        sys.exit(1)
