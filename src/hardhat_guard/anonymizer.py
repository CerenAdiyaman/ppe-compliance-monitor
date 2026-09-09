"""Face blurring applied before any frame leaves the process.

Snapshots of workers are personal data. The system's purpose is to document a
rule breach, not to identify a person, so faces are obscured before a frame is
written to disk or attached to an email - not afterwards, and not optionally in
the storage layer where a future caller could forget.

Regions come from the detector rather than a dedicated face detector. The model
already localises every head in the frame, bare (`head`) or covered (`helmet`),
and it does so in profile, from behind and under a hard hat - the cases a
frontal face detector is worst at and this system produces constantly. It also
keeps the privacy guarantee free of a second model to ship and load.

Note the consequence: a head the detector misses is a head that stays sharp.
Recall of the model is therefore a privacy property, not only an accuracy one.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import cv2
import numpy as np

from hardhat_guard.config import settings
from hardhat_guard.detector import Detection

logger = logging.getLogger(__name__)

Box = tuple[int, int, int, int]

# Blur kernel as a fraction of the region's size. Scaling with the region keeps
# the effect irreversible whether the worker is near the camera or far from it;
# a fixed kernel would leave distant faces recognisable.
_KERNEL_RATIO = 0.25
_MIN_KERNEL = 15

# Boxes are grown before blurring: a tight box leaves the jaw, ears and hairline
# sharp, and those identify a person nearly as well as the face.
_MARGIN = 0.15

# A `helmet` box frames the hard hat, and the face sits below it. Extending it
# downwards by its own height covers what the box itself never contained.
_HELMET_FACE_DROP = 1.0


def _odd(value: int) -> int:
    """GaussianBlur requires odd, positive kernel dimensions."""
    value = max(value, _MIN_KERNEL)
    return value if value % 2 == 1 else value + 1


class FaceAnonymizer:
    def __init__(self, enabled: bool | None = None) -> None:
        self.enabled = settings.anonymize_faces if enabled is None else enabled
        if not self.enabled:
            logger.warning("Face anonymisation is disabled; stored frames identify workers")

    def anonymize(self, frame: np.ndarray, detections: Sequence[Detection] = ()) -> np.ndarray:
        """Return a copy of ``frame`` with every detected head blurred.

        The input is never modified: the pipeline keeps the clean frame for the
        live preview while the redacted copy is what gets persisted or emailed.
        """
        if not self.enabled:
            return frame

        redacted = frame.copy()
        for detection in detections:
            self._blur_region(redacted, self._region_for(detection))
        return redacted

    def _region_for(self, detection: Detection) -> Box:
        x1, y1, x2, y2 = detection.box
        width, height = x2 - x1, y2 - y1

        margin_x = int(width * _MARGIN)
        margin_y = int(height * _MARGIN)
        drop = int(height * _HELMET_FACE_DROP) if detection.class_name == "helmet" else 0

        return (x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y + drop)

    def _blur_region(self, frame: np.ndarray, box: Box) -> None:
        frame_height, frame_width = frame.shape[:2]
        # Boxes come from a model and, once grown, may run past the frame edge;
        # clamping avoids an empty slice, which GaussianBlur rejects.
        x1 = max(0, min(box[0], frame_width))
        y1 = max(0, min(box[1], frame_height))
        x2 = max(0, min(box[2], frame_width))
        y2 = max(0, min(box[3], frame_height))
        if x2 <= x1 or y2 <= y1:
            return

        region = frame[y1:y2, x1:x2]
        kernel = (
            _odd(int((x2 - x1) * _KERNEL_RATIO)),
            _odd(int((y2 - y1) * _KERNEL_RATIO)),
        )
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(region, kernel, 0)
