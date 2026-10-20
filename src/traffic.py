"""Send realistic traffic to the API so the monitoring dashboard has data.

Requests replay real patients from the cleaned dataset (missing ca/thal stay
null); a small share is deliberately broken to exercise validation (422).

Usage:
    python -m src.traffic --url http://localhost --rate 5 --duration 300
"""
import argparse
import json
import random
import time
import urllib.error
import urllib.request
from collections import Counter

import pandas as pd

from src.config import CLEAN_DATA_PATH, FEATURES

# ways a real client might send bad data; each breaks one field
BAD_INPUTS = [
    ("age", 250),        # outside plausible range
    ("cp", 0),           # invalid chest-pain code
    ("thal", 5),         # invalid thal code
    ("chol", None),      # required field missing
    ("sex", "M"),        # wrong type
]


def load_patients() -> list[dict]:
    df = pd.read_csv(CLEAN_DATA_PATH)[FEATURES]
    records = []
    for row in df.to_dict(orient="records"):
        # JSON-friendly: NaN -> null, whole-number floats -> int (codes like cp=4)
        records.append({k: (None if pd.isna(v) else int(v) if float(v).is_integer() else v)
                        for k, v in row.items()})
    return records


def send(url: str, payload: dict) -> int:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status
    except urllib.error.HTTPError as err:
        return err.code
    except (urllib.error.URLError, TimeoutError):
        return 0      # connection problem


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost")
    parser.add_argument("--rate", type=float, default=5, help="requests per second")
    parser.add_argument("--duration", type=float, default=120, help="seconds")
    parser.add_argument("--bad-share", type=float, default=0.05,
                        help="fraction of deliberately invalid requests")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    patients = load_patients()
    endpoint = args.url.rstrip("/") + "/predict"
    statuses = Counter()
    end = time.monotonic() + args.duration
    print(f"Sending ~{args.rate}/s to {endpoint} for {args.duration:.0f}s "
          f"({len(patients)} patients, {args.bad_share:.0%} invalid)")

    while time.monotonic() < end:
        payload = dict(rng.choice(patients))
        if rng.random() < args.bad_share:
            field, value = rng.choice(BAD_INPUTS)
            payload[field] = value
        statuses[send(endpoint, payload)] += 1
        total = sum(statuses.values())
        if total % 50 == 0:
            print(f"  {total} requests  {dict(statuses)}")
        time.sleep(1 / args.rate)

    print(f"Done: {sum(statuses.values())} requests  {dict(statuses)}")


if __name__ == "__main__":
    main()
