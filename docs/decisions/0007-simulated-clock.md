# 7. Use a simulated clock for the live stream experiments

## Context

A live stream experiment with a real thread and a real clock is not repeatable. The
result changes with the load of the machine, and one run of each configuration takes
the full duration of the video.

## Decision

The experiments use `SimulatedExecutor`. A detector run that starts on frame `i` is
ready after the latency that the model had on that frame in a stored measurement. The
times of the optical flow and the tracker are real measurements. `ThreadedExecutor` has
the same interface and uses a real thread.

## Consequences

- A run is repeatable, and the detector results are identical between configurations.
- The simulation assumes that the detector and the frame loop do not slow each other.
  On a machine with few cores, that is not true.
- One manual run of the stream service used the real thread on a real sequence. It held
  30 frames for each second with a p99 latency of 32 ms. No experiment compares the
  simulated and the real numbers in a systematic way.
