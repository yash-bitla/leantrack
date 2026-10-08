# 4. Let a detection replace the flow measurement

## Context

The pipeline moves each track to the current frame before the schedule decides. That
move includes a measurement from the optical flow. On a frame with a detector run, the
tracker then has two measurements of the same frame.

## Decision

On a frame with a detector run, the tracker discards the flow measurement and uses only
the detection. `Tracker.advance` stores the flow measurement as pending. `coast` applies
it, and `associate` discards it.

## Evidence

When the filter used the two measurements, HOTA was 1.00 lower at N = 10 and 1.70 lower
at N = 20 on MOT17 train. The interval experiment found this, because its scores did not
agree with the scores of the pull request before.

## Consequences

- The filter gives a flow measurement the same trust as a detection. A separate noise
  level for the flow is a possible improvement and is not tested.
- After each later change of the tracker, two reference scores were checked by hand.
  This check is not automatic.
