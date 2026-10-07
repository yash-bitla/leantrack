"""Experiment 2: fixed interval against confidence trigger.

Both policies use optical flow between detector runs. The comparison is HOTA at an equal
share of frames with a detector run.

    python -m bench.trigger yolox_s --jobs 3
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from bench._eval import METRICS, evaluate, markdown_table, mot17_sequences, run_policy
from leantrack.schedule.policy import ConfidenceTrigger, FixedInterval, SchedulePolicy

# name -> (policy class, keyword arguments, series label for the chart)
CONFIGS: dict[str, tuple[type, dict[str, Any], str]] = {
    **{f"fixed-{n:02d}": (FixedInterval, {"interval": n}, "fixed") for n in (2, 3, 5, 10, 20)},
    **{
        f"motion-{m:.2f}": (ConfidenceTrigger, {"max_motion": m}, "motion")
        for m in (0.1, 0.2, 0.4, 0.8)
    },
    **{
        f"reliability-{f:.2f}": (ConfidenceTrigger, {"max_unreliable_fraction": f}, "reliability")
        for f in (0.05, 0.1, 0.2, 0.4)
    },
    **{
        f"both-{m:.2f}-{f:.2f}": (
            ConfidenceTrigger,
            {"max_motion": m, "max_unreliable_fraction": f},
            "both",
        )
        for m, f in ((0.2, 0.1), (0.4, 0.2), (0.8, 0.4))
    },
}


class _Factory:
    """A policy factory that a worker process can receive."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self) -> SchedulePolicy:
        policy_type, kwargs, _ = CONFIGS[self.name]
        policy: SchedulePolicy = policy_type(**kwargs)
        return policy


def _job(name: str, data: Path, cache: Path, out: Path) -> tuple[str, dict[str, float]]:
    return name, run_policy(_Factory(name), mot17_sequences(data), cache, out / name, flow=True)


def main() -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("model", help="model name, a folder in the detection cache")
    p.add_argument("--data", type=Path, default=Path("data/MOT17/train"))
    p.add_argument("--cache", type=Path, default=Path("runs/detections"))
    p.add_argument("--out", type=Path, default=Path("runs/trigger"))
    p.add_argument("--only", nargs="+", choices=sorted(CONFIGS), help="run these configs only")
    p.add_argument("--jobs", type=int, default=1, help="worker processes; more than 1 adds noise")
    args = p.parse_args()

    names = args.only or list(CONFIGS)
    out = args.out / args.model
    cache = args.cache / args.model
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(_job, name, args.data, cache, out) for name in names]
        costs = dict(f.result() for f in futures)

    sequences = mot17_sequences(args.data)
    scores = evaluate(names, sequences, args.data, out)
    rows = {name: {**costs[name], **scores[name]} for name in names}
    summary = {name: {"series": CONFIGS[name][2], **rows[name]} for name in names}

    print(f"MOT17 train, {args.model}, Kalman + optical flow, {len(sequences)} sequences")
    columns = ["detector_runs_pct", "mean_ms", "p99_ms", *METRICS]
    print(markdown_table(rows, columns, "policy"))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
