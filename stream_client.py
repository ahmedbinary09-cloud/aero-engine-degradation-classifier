"""Replay a TEST engine cycle by cycle into the API, like a live sensor feed.

Test engines (data/test_FD001.txt) stop before they fail; data/RUL_FD001.txt says how many
cycles they still had left after the last recorded cycle, so we can show the true RUL
next to every prediction.

Usage:  python stream_client.py                      # engine closest to failure, 0.2 s/cycle
        python stream_client.py --unit 7 --delay 0.5
        python stream_client.py --url https://<app>.fly.dev/predict
"""

import argparse
import time

import requests

from health import load_test

ICON = {"healthy": "[ OK ]", "warning": "[WARN]", "failure": "[FAIL]"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:9696/predict")
    p.add_argument("--data", default="data/test_FD001.txt")
    p.add_argument("--rul", default="data/RUL_FD001.txt")
    p.add_argument("--unit", type=int, default=None,
                   help="engine id (default: the test engine with the smallest final RUL)")
    p.add_argument("--delay", type=float, default=0.2, help="seconds between cycles")
    p.add_argument("--every", type=int, default=5, help="print every N-th cycle")
    args = p.parse_args()

    df = load_test(args.data, args.rul)
    unit = args.unit
    if unit is None:
        unit = int(df.loc[df.groupby("unit")["rul"].transform("min") == df["rul"], :]
                   .sort_values("rul").iloc[0]["unit"])
    engine = df[df["unit"] == unit].sort_values("cycle")
    last = int(engine["cycle"].max())
    print(f"streaming test engine {unit}: {last} cycles recorded, "
          f"{int(engine['rul'].iloc[-1])} cycles left after the last one")

    correct = total = 0
    for _, row in engine.iterrows():
        payload = {k: float(v) for k, v in row.drop(["rul", "rul_capped"]).items()}
        payload["unit"], payload["cycle"] = int(payload["unit"]), int(payload["cycle"])
        r = requests.post(args.url, json=payload, timeout=10)
        r.raise_for_status()
        out = r.json()
        true_rul = int(row["rul"])
        true_state = "failure" if true_rul <= 50 else "warning" if true_rul <= 100 else "healthy"
        total += 1
        correct += out["state"] == true_state
        if payload["cycle"] % args.every == 0 or payload["cycle"] == last:
            print(f"cycle {payload['cycle']:>3}  predicted RUL {out['rul_cycles']:>6.1f}  "
                  f"true RUL {true_rul:>3}  {ICON[out['state']]} {out['state']}"
                  f"{'' if out['state'] == true_state else '   (true: ' + true_state + ')'}")
        time.sleep(args.delay)
    print(f"state matched the truth on {correct}/{total} cycles")


if __name__ == "__main__":
    main()
