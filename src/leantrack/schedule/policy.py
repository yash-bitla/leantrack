from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from leantrack.tracks.track import Track, TrackState


@dataclass(frozen=True, slots=True)
class ScheduleContext:
    """What a policy knows when it decides on a frame.

    The tracks are already at this frame: prediction and propagation are complete.
    """

    frame_index: int
    """1-based index of this frame."""
    frames_since_detection: int | None
    """Frames since the last detector run. None before the first run."""
    tracks: Sequence[Track]


class SchedulePolicy(Protocol):
    def should_detect(self, context: ScheduleContext) -> bool:
        """Decide if the detector runs on this frame."""
        ...


class FixedInterval:
    """Run the detector on frame 1 and then on each `interval`-th frame after it."""

    def __init__(self, interval: int = 1) -> None:
        if interval < 1:
            raise ValueError("interval must be at least 1")
        self.interval = interval

    def should_detect(self, context: ScheduleContext) -> bool:
        return (context.frame_index - 1) % self.interval == 0


class ConfidenceTrigger:
    """Run the detector when the tracks become uncertain.

    The detector runs if one of these conditions is true:

    - No detector run occurred yet, or `max_interval` frames passed since the last run.
    - The fraction of confirmed tracks with a propagator reliability below
      `min_reliability` is larger than `max_unreliable_fraction`.
    - A confirmed track moved more than `max_motion` box sizes since its last detection.

    The detector does not run more often than each `min_interval`-th frame. The
    `max_interval` bound is necessary because no track signal can show a new object.
    """

    def __init__(
        self,
        *,
        min_interval: int = 2,
        max_interval: int = 20,
        min_reliability: float = 0.5,
        max_unreliable_fraction: float | None = None,
        max_motion: float | None = None,
    ) -> None:
        if not 1 <= min_interval <= max_interval:
            raise ValueError("intervals must satisfy 1 <= min_interval <= max_interval")
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.min_reliability = min_reliability
        self.max_unreliable_fraction = max_unreliable_fraction
        self.max_motion = max_motion

    def should_detect(self, context: ScheduleContext) -> bool:
        since = context.frames_since_detection
        if since is None or since >= self.max_interval:
            return True
        if since < self.min_interval:
            return False

        confirmed = [t for t in context.tracks if t.state is TrackState.CONFIRMED]
        if not confirmed:
            return False
        if self.max_unreliable_fraction is not None:
            unreliable = sum(t.reliability < self.min_reliability for t in confirmed)
            if unreliable / len(confirmed) > self.max_unreliable_fraction:
                return True
        return self.max_motion is not None and any(
            t.motion_since_update > self.max_motion for t in confirmed
        )
