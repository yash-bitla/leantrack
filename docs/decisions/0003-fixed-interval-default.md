# 3. Use a fixed interval as the default schedule

## Context

A fixed interval runs the detector on each N-th frame. A confidence trigger runs it when
the tracks look uncertain. The trigger has two signals for each track: the reliability
of the optical flow, and the motion since the last detection.

## Decision

The default schedule is a fixed interval. The confidence trigger stays in the code as a
policy, but it is not the default.

## Evidence

The two signals do predict a bad box. A box with flow reliability below 0.25 had an IoU
below 0.5 in 77.8% of cases, against 15.6% for reliability above 0.9.

But at an equal share of frames with a detector run, the trigger is not better. The
motion trigger is within 0.5 HOTA of the fixed interval, above it at two points and
below it at two. The reliability trigger is 0.7 to 1.6 points lower.

## Consequences

- An uncertain track needs recovery logic and not more detector runs. Recovery by
  appearance addresses that case.
- A maximum interval is necessary in each policy, because no track signal can show a
  new object.
