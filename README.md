# leantrack

A multi-object tracking runtime that uses the detector as a limited resource.

Most tracking libraries run the detector on every frame. `leantrack` will decide, frame by
frame, between a detector run and a low-cost track propagation. It will do this against a
latency budget, and it will report the accuracy cost of each decision.

## Status

Phase 1 of 6 is complete, and Phase 2 is in progress. The fixed-interval scheduler and
the detector backends exist. The interval experiment has no published results yet. The
confidence monitor and the recovery logic do not exist yet.

| Phase | Content | Status |
|---|---|---|
| 1 | Tracker core, MOT input and output, evaluation, baseline | Complete |
| 2 | Detection scheduler, track propagation between detections | In progress |
| 3 | Failure detection, recovery, re-identification | Not started |
| 4 | Budget controller, ONNX detector, benchmark report | Not started |
| 5 | Stream service, metrics endpoint, container | Not started |
| 6 | Demo assets, decision records | Not started |

## What exists

- A constant-velocity Kalman filter on `(cx, cy, w, h)` with noise that scales with box size.
- Optimal assignment with a cost gate (`scipy.optimize.linear_sum_assignment`).
- Two-pass IoU association in the style of ByteTrack.
- An explicit track lifecycle: `tentative -> confirmed -> lost -> removed`. An illegal
  transition raises an error.
- A reader and a writer for the MOTChallenge format.
- A synthetic scene generator with occlusion intervals, for deterministic tests.
- An evaluation script that uses TrackEval and compares against BoxMOT ByteTrack.
- A fixed-interval schedule policy. Between detector runs, the tracker reports its
  Kalman prediction.
- Two detector backends behind one `Detector` protocol (see "Detector backends").

## Baseline results

MOT17 train, 7 sequences, public detections from the dataset. Each tracker gets the same
detection file, so the scores compare association only. Both trackers use their default
parameters. No parameter was tuned on this data.

| Detections | Tracker | HOTA | AssA | DetA | MOTA | IDF1 | IDSW | Step p50 (ms) | Step p99 (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FRCNN | BoxMOT ByteTrack 25.0.0 | 47.32 | 50.71 | 44.37 | 49.43 | 55.17 | 657 | 1.15 | 2.82 |
| FRCNN | leantrack | 47.11 | 49.88 | 44.72 | 49.39 | 54.22 | 609 | 0.42 | 1.22 |
| SDP | BoxMOT ByteTrack 25.0.0 | 54.05 | 53.24 | 55.08 | 64.10 | 64.30 | 920 | 1.25 | 3.32 |
| SDP | leantrack | 53.94 | 53.51 | 54.54 | 63.89 | 64.28 | 846 | 0.48 | 1.50 |

The step time is the tracker update only, on an Apple M3 Pro, one run. It does not include
detection. Do not compare these scores with published MOT17 scores, because published
scores use the test set and stronger private detectors.

## Detector backends

| Backend | Install | License of the backend |
|---|---|---|
| YOLOX through ONNX Runtime (default) | included | Apache-2.0 |
| Ultralytics YOLO (optional) | `pip install -e ".[ultralytics]"` | AGPL-3.0 |

The core of `leantrack` does not import Ultralytics, and the MIT license applies to the
code in this repository. If you install the optional backend, the AGPL-3.0 terms of
Ultralytics apply to the combination that you make.

The YOLOX backend uses the ONNX files of the
[YOLOX 0.1.1 release](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0).
Put them in `models/`.

## Install

```sh
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

## Use

Track one MOTChallenge sequence from its detection file:

```sh
.venv/bin/leantrack track data/MOT17/train/MOT17-02-FRCNN --out runs/MOT17-02.txt
```

Reproduce the table (this needs MOT17 in `data/MOT17`):

```sh
.venv/bin/pip install -e ".[bench]"
.venv/bin/python -m bench.mot17 --detector FRCNN
.venv/bin/python -m bench.mot17 --detector SDP --out runs/mot17-sdp
```

Run the interval experiment (this needs `models/yolox_s.onnx`):

```sh
.venv/bin/python bench/cache_detections.py models/yolox_s.onnx
.venv/bin/python -m bench.interval yolox_s
```

## Prior work

The association follows ByteTrack (Zhang et al., 2022). Detection at intervals with
tracking between detections is not a new idea. See Confidence-Triggered Detection
(arXiv 1902.00615) and SDOF-Tracker (arXiv 2106.14259). This project adds a tested
implementation and published measurements of the trade-off.

## License

MIT
