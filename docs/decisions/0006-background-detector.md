# 6. Run a slow detector in the background and correct its late result

## Context

On a live stream, a detector that is slower than the frame period makes a blocking loop
late. YOLOX-s takes about 80 ms on this CPU, and a frame at 30 frames for each second
arrives each 33 ms.

## Decision

`run_realtime` runs the detector in a worker thread. Each frame gets an output from the
Kalman prediction and the optical flow. When a detector result arrives, optical flow
moves its boxes from the frame that the detector saw to the current frame. Then the
association runs.

A frame time budget controls the wait. The loop waits for the detector only if it
expects the result inside the budget.

## Evidence

MOT17 train, live stream at the frame rate of each sequence:

| Model | Mode | Output latency, mean (ms) | HOTA |
|---|---|---:|---:|
| YOLOX-s | Blocking | 101.44 | 32.90 |
| YOLOX-s | Background | 5.12 | 34.49 |
| YOLOX-s | Background, no correction | 2.81 | 31.55 |
| YOLOX-tiny | Blocking | 22.94 | 31.94 |
| YOLOX-tiny | Background | 4.04 | 31.34 |
| YOLOX-tiny | Background, budget 33 ms | 24.66 | 31.86 |

## Consequences

- The correction of late results is necessary. Without it, the background mode is worse
  than the blocking loop.
- For a detector that fits in the frame period, the loop must wait. The budget does that.
- The propagator stores 16 frames. A result that is older gets a partial correction.
