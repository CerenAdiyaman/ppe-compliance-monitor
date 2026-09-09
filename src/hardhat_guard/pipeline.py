"""Wires the components together and runs the monitoring loop.

Everything this module does is orchestration: it owns no rules, no queries and
no I/O of its own. The order below is the privacy guarantee in code - the frame
is redacted before it reaches anything that stores or transmits it, and the
clean frame only ever goes to the local preview window.

    frame -> detect -> evaluate -> [anonymise -> store -> record -> notify]

The components are constructor arguments rather than globals so a test can
substitute any of them; ``tests/integration`` exercises this whole path with a
stub detector and video source, no GPU and no camera.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import numpy as np

from hardhat_guard.anonymizer import FaceAnonymizer
from hardhat_guard.config import settings
from hardhat_guard.detector import Detection, HardHatDetector
from hardhat_guard.notifier import EmailNotifier
from hardhat_guard.rules import RuleEngine, Violation
from hardhat_guard.storage import FileStore, ViolationRepository
from hardhat_guard.video_source import VideoSource

logger = logging.getLogger(__name__)

_BOX_COLOURS = {"head": (0, 0, 255), "helmet": (0, 200, 0)}  # BGR: red, green


class MonitoringPipeline:
    def __init__(
        self,
        source: VideoSource | Iterable[np.ndarray] | None = None,
        detector: HardHatDetector | None = None,
        rules: RuleEngine | None = None,
        anonymizer: FaceAnonymizer | None = None,
        store: FileStore | None = None,
        repository: ViolationRepository | None = None,
        notifier: EmailNotifier | None = None,
        *,
        camera_id: str = "CAM_01",
        location: str = "",
        preview: bool = False,
    ) -> None:
        self.source = source if source is not None else VideoSource()
        self.detector = detector or HardHatDetector()
        self.rules = rules or RuleEngine(camera_id=camera_id, location=location)
        self.anonymizer = anonymizer or FaceAnonymizer()
        self.store = store or FileStore()
        self.repository = repository or ViolationRepository()
        self.notifier = notifier or EmailNotifier()
        self.preview = preview

        self.frames_seen = 0
        self.violations_recorded = 0

    def run(self) -> None:
        """Consume the source until it ends or the user interrupts."""
        logger.info("Monitoring started on %s", self.rules.camera_id)
        try:
            for frame in self.source:
                self.process(frame)
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        finally:
            self._close()
            logger.info(
                "Processed %d frames, recorded %d violation(s)",
                self.frames_seen,
                self.violations_recorded,
            )

    def process(self, frame: np.ndarray) -> list[Violation]:
        """Run one frame through the whole path. Returns what was recorded."""
        self.frames_seen += 1

        detections = self.detector.detect(frame)
        violations = self.rules.evaluate(detections)

        for violation in violations:
            self._handle(frame, detections, violation)

        if self.preview:
            self._show(frame, detections)
        return violations

    def _handle(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
        violation: Violation,
    ) -> None:
        # Redaction first, and on a copy: everything downstream of this line
        # leaves the process, and none of it may carry an identifiable face.
        redacted = self.anonymizer.anonymize(frame, detections)

        snapshot = self.store.save(redacted, violation)
        self.repository.add(violation, snapshot)
        self.violations_recorded += 1

        # The result is ignored on purpose - a mail failure is logged inside
        # the notifier and must not interrupt monitoring.
        self.notifier.notify(violation, snapshot)

    def _show(self, frame: np.ndarray, detections: Sequence[Detection]) -> None:
        """Draw boxes on a copy and display it. Local only, never stored."""
        import cv2

        canvas = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            colour = _BOX_COLOURS.get(detection.class_name, (200, 200, 200))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(
                canvas,
                f"{detection.class_name} {detection.confidence:.0%}",
                (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colour,
                1,
                cv2.LINE_AA,
            )

        cv2.imshow(f"Hardhat Guard - {self.rules.camera_id}", canvas)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):  # q or Esc
            raise KeyboardInterrupt

    def _close(self) -> None:
        release = getattr(self.source, "release", None)
        if callable(release):
            release()
        if self.preview:
            import cv2

            cv2.destroyAllWindows()


def build_pipeline(
    source: str | None = None,
    camera_id: str = "CAM_01",
    preview: bool = False,
) -> MonitoringPipeline:
    """Construct a pipeline from configuration, resolving the camera table."""
    from hardhat_guard.config import CAMERAS

    camera = CAMERAS.get(camera_id, {})
    return MonitoringPipeline(
        source=VideoSource(source or camera.get("source") or settings.video_source),
        camera_id=camera_id,
        location=camera.get("location", ""),
        preview=preview,
    )
