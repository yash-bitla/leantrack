# 1. Write the tracker core, and use BoxMOT as the reference

## Context

Mature tracking libraries exist, for example BoxMOT. Their interface accepts the
detections of a frame and returns the tracks. They have no step for a frame without a
detector run. This project is about those frames.

## Decision

Write the tracker core in this repository: a Kalman filter, optimal assignment, a
two-pass association in the style of ByteTrack, and an explicit track lifecycle. Use
SciPy for the assignment solver. Use BoxMOT ByteTrack only in the benchmark, as the
reference.

## Evidence

On the same detection files and with default parameters, the two trackers have almost
equal scores on MOT17 train: 47.11 against 47.32 HOTA with the FRCNN detections, and
53.94 against 54.05 with the SDP detections.

## Consequences

- The tracker has separate `advance`, `associate` and `coast` steps. The optical flow,
  the schedule policies and the appearance recovery use those steps.
- BoxMOT has the AGPL-3.0 license. It is not a dependency of the package.
- The core is about 500 lines in four files and has its own tests.
