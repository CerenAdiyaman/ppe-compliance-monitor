"""Turns per-frame detections into confirmed violations.

Two filters sit between a detection and an alert:

* **Confirmation** - a violation must be present in ``confirmation_frames``
  consecutive frames. A single frame of motion blur or a momentary
  misclassification never raises an alert.
* **Cooldown** - once a violation is recorded, the same type is suppressed for
  ``cooldown_seconds``. Without it, one worker standing in view would produce a
  record per frame and the alerts would be ignored as noise.

The engine holds all its state in memory and is deliberately free of I/O, so it
can be reasoned about and tested without a model, a camera or a database.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

from hardhat_guard.config import settings
from hardhat_guard.detector import Detection

logger = logging.getLogger(__name__)

ViolationType = Literal["missing_helmet"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class Violation:
    """A violation that survived confirmation and cooldown."""

    violation_type: ViolationType
    camera_id: str
    location: str
    detected_at: datetime
    confidence: float
    box: tuple[int, int, int, int]


class RuleEngine:
    """Confirms violations for a single camera.

    One instance per camera: the streak and cooldown state describe one scene,
    and sharing them across cameras would let activity on one suppress alerts
    on another.
    """

    def __init__(
        self,
        camera_id: str = "CAM_01",
        location: str = "",
        *,
        confirmation_frames: int | None = None,
        cooldown_seconds: int | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.camera_id = camera_id
        self.location = location
        self._confirmation_frames = (
            settings.confirmation_frames if confirmation_frames is None else confirmation_frames
        )
        self._cooldown = timedelta(
            seconds=settings.cooldown_seconds if cooldown_seconds is None else cooldown_seconds
        )
        self._clock = clock

        # Consecutive frames each violation type has been seen in, and when it
        # was last recorded. Keyed by type so a future `missing_vest` tracks
        # independently of `missing_helmet`.
        self._streaks: dict[ViolationType, int] = {}
        self._last_recorded: dict[ViolationType, datetime] = {}

    @property
    def streaks(self) -> dict[ViolationType, int]:
        """Current consecutive-frame counts, for logging and diagnostics."""
        return dict(self._streaks)

    def evaluate(self, detections: Sequence[Detection]) -> list[Violation]:
        """Feed one frame's detections in; get back the violations to act on.

        Must be called once per frame, including frames with no detections -
        that is how a streak gets broken.
        """
        now = self._clock()
        violations: list[Violation] = []

        offenders = [d for d in detections if d.is_violation]
        self._advance("missing_helmet", offenders)

        confirmed = self._confirm("missing_helmet", offenders, now)
        if confirmed is not None:
            violations.append(confirmed)

        return violations

    def reset(self) -> None:
        """Drop all state - used when a video source reconnects and the scene
        may have changed entirely."""
        self._streaks.clear()
        self._last_recorded.clear()

    def _advance(self, violation_type: ViolationType, offenders: Iterable[Detection]) -> None:
        if any(True for _ in offenders):
            self._streaks[violation_type] = self._streaks.get(violation_type, 0) + 1
        else:
            self._streaks[violation_type] = 0

    def _confirm(
        self,
        violation_type: ViolationType,
        offenders: Sequence[Detection],
        now: datetime,
    ) -> Violation | None:
        if self._streaks.get(violation_type, 0) < self._confirmation_frames:
            return None

        last = self._last_recorded.get(violation_type)
        if last is not None and now - last < self._cooldown:
            return None

        # The most confident offender represents the frame: it is the box most
        # likely to actually contain a bare head, so it is what gets stored and
        # blurred downstream.
        best = max(offenders, key=lambda d: d.confidence)

        # The streak is intentionally not reset. A worker who stays uncovered
        # keeps the streak alive and is re-reported once the cooldown expires,
        # which is the behaviour a safety officer expects from a standing
        # hazard. Resetting would restart confirmation and delay every repeat.
        self._last_recorded[violation_type] = now

        violation = Violation(
            violation_type=violation_type,
            camera_id=self.camera_id,
            location=self.location,
            detected_at=now,
            confidence=best.confidence,
            box=best.box,
        )
        logger.info(
            "%s confirmed on %s (confidence %.2f)",
            violation_type,
            self.camera_id,
            best.confidence,
        )
        return violation
