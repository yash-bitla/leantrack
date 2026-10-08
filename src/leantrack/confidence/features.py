from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from leantrack._types import FloatArray
from leantrack.boxes import iou_matrix
from leantrack.tracks.track import Track

FEATURE_NAMES = (
    "frames_since_update",
    "reliability",
    "motion_since_update",
    "score",
    "log_hits",
    "position_sigma",
    "crowding",
    "log_size",
)


def track_features(tracks: Sequence[Track]) -> FloatArray:
    """One row of features for each track, in the order of FEATURE_NAMES.

    Each feature is available at run time. None of them needs the ground truth.

    - position_sigma: standard deviation of the predicted center, in box sizes.
    - crowding: largest IoU with one of the other tracks. A high value means that the
      object is possibly behind a different object.
    """
    if not tracks:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    boxes = np.stack([t.box for t in tracks])
    overlap = iou_matrix(boxes, boxes)
    np.fill_diagonal(overlap, 0.0)

    rows = []
    for track, crowding in zip(tracks, overlap.max(axis=1), strict=True):
        size = float(np.sqrt(max(track.mean[2] * track.mean[3], 1.0)))
        sigma = float(np.sqrt(max(track.covariance[0, 0] + track.covariance[1, 1], 0.0))) / size
        rows.append(
            [
                float(track.frames_since_update),
                track.reliability,
                track.motion_since_update,
                track.score,
                float(np.log(track.hits)),
                sigma,
                float(crowding),
                float(np.log(size)),
            ]
        )
    return np.array(rows, dtype=np.float64)
