# 5. Request appearance vectors only when necessary

## Context

A ReID network takes about 1.6 ms for each box on this CPU. A frame of MOT17 can have
more than 20 detections. A vector for each detection on each frame would cost more than
the tracker and the optical flow together.

## Decision

The tracker gets a function that returns vectors for given detections. It calls the
function only for these detections:

- A detection that can match a lost track, by position or by overlap.
- The matched detection of a track, on each 5th match, to update the stored vector.

A detection that overlaps another detection does not update a stored vector.

## Evidence

In the live pipeline with YOLOX-s, OSNet took 4.45 ms for each detector run, which is
about 3 boxes. It added 6.5% to the mean frame time at N = 1 and gave +0.99 HOTA.

A color histogram gave +0.84 HOTA for 0.11 ms. In the occlusion experiment it was equal
to OSNet up to a gap of 30 frames.

## Consequences

- The histogram is a sensible first choice. OSNet is better only for long gaps.
- The stored vector must follow the object quickly. A momentum of 0.5 gave 68.49%
  recovery at a gap of 60 frames on the tune split, against 57.53% for 0.9.
