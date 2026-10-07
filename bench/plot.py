"""Plot HOTA against the mean frame time for one or more interval experiments.

python -m bench.plot assets/interval.png "YOLOX-s, Kalman=runs/interval/yolox_s"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter

X_LABELS = {
    "mean_ms": "Mean time for each frame (ms)",
    "detector_runs_pct": "Frames with a detector run (%)",
}


def main() -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("out", type=Path)
    p.add_argument("series", nargs="+", help="LABEL=DIRECTORY that contains summary.json")
    p.add_argument("--x", choices=sorted(X_LABELS), default="mean_ms")
    p.add_argument(
        "--title", default="MOT17 train. The number at each point is the detection interval N."
    )
    p.add_argument("--log-x", action="store_true")
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    low, high = float("inf"), 0.0
    for item in args.series:
        label, _, directory = item.partition("=")
        rows = list(json.loads((Path(directory) / "summary.json").read_text()).values())
        # A summary with a "series" field holds more than one curve.
        groups: dict[str, list[dict[str, float]]] = {}
        for row in rows:
            name = f"{label}, {row['series']}" if "series" in row else label
            groups.setdefault(name, []).append(row)
        for name, group in groups.items():
            group.sort(key=lambda r: r[args.x])
            low = min(low, group[0][args.x])
            high = max(high, group[-1][args.x])
            ax.plot([r[args.x] for r in group], [r["HOTA"] for r in group], marker="o", label=name)
            for r in group:
                if "interval" in r:
                    ax.annotate(
                        f"{r['interval']:.0f}",
                        (r[args.x], r["HOTA"]),
                        textcoords="offset points",
                        xytext=(5, -11),
                        fontsize=8,
                    )
    if args.log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50, 100, 200])
        ax.set_xlim(low * 0.8, high * 1.25)
    ax.set_xlabel(X_LABELS[args.x])
    ax.set_ylabel("HOTA")
    ax.set_title(args.title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
