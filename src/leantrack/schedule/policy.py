from __future__ import annotations

from typing import Protocol


class SchedulePolicy(Protocol):
    def should_detect(self, frame_index: int) -> bool:
        """Decide if the detector runs on this frame. Frame indices are 1-based."""
        ...


class FixedInterval:
    """Run the detector on frame 1 and then on each `interval`-th frame after it."""

    def __init__(self, interval: int = 1) -> None:
        if interval < 1:
            raise ValueError("interval must be at least 1")
        self.interval = interval

    def should_detect(self, frame_index: int) -> bool:
        return (frame_index - 1) % self.interval == 0
