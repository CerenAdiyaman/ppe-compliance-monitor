"""Frame ingestion for webcams, files and network cameras.

The three behave differently in the one way that matters: a file ends, and a
live source is not supposed to. So a dropped frame from an RTSP camera means
"reconnect", while the same from a file means "we are done". This module hides
that difference behind one iterator.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from hardhat_guard.config import settings

logger = logging.getLogger(__name__)

# Reconnect backoff. Retrying a downed camera every few milliseconds floods the
# log and the network for no benefit; a few seconds is fast enough that a brief
# switch reboot costs only a handful of frames.
_RECONNECT_DELAY_SECONDS = 3.0
_MAX_CONSECUTIVE_FAILURES = 5

# A device can open and then deliver nothing - a webcam index with no camera
# behind it, or one already held by another application. Reopening it succeeds
# every time, so "did open() work?" is not enough to detect the situation.
#
# Retries are therefore unbounded by default but backed off exponentially: a
# monitoring process is meant to outlive a router reboot, a camera power-cycle
# or an overnight network cut, and a limit that stops it permanently turns a
# transient outage into a silent gap in the safety log. The backoff is what
# keeps unbounded retrying cheap - the delay grows to a minute, so a dead
# camera costs one attempt per minute rather than a hot loop.
_MAX_BARREN_RECONNECTS = 0  # 0 = keep trying
_MAX_RECONNECT_DELAY_SECONDS = 60.0

# Capture backends by configuration name. On Windows, OpenCV prefers Media
# Foundation, which on plenty of laptops opens a webcam successfully and then
# fails to start the stream (MF_E_HW_MFT_FAILED_START_STREAMING) - the device
# is fine, the backend is not. DirectShow is the reliable fallback there, so
# "auto" tries OpenCV's own choice and then DirectShow.
_BACKENDS = {
    "any": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
}
_AUTO_ORDER = ("any", "dshow")


def _parse(source: str) -> int | str:
    """Webcams are opened by index, everything else by string.

    OpenCV treats ``VideoCapture(0)`` and ``VideoCapture("0")`` differently -
    the second looks for a file named "0" - so the conversion has to happen
    here rather than being left to the caller.
    """
    return int(source) if source.isdigit() else source


class VideoSource:
    """An iterable of BGR frames.

    Usable as a context manager so the capture handle is released even if the
    pipeline raises; a webcam left open stays locked against other processes.
    """

    def __init__(
        self,
        source: str | None = None,
        *,
        reconnect_delay: float = _RECONNECT_DELAY_SECONDS,
        max_failures: int = _MAX_CONSECUTIVE_FAILURES,
        max_barren_reconnects: int = _MAX_BARREN_RECONNECTS,
        max_reconnect_delay: float = _MAX_RECONNECT_DELAY_SECONDS,
        backend: str | None = None,
    ) -> None:
        self.source = source if source is not None else settings.video_source
        self._target = _parse(self.source)
        self._capture: cv2.VideoCapture | None = None
        self._reconnect_delay = reconnect_delay
        self._max_failures = max_failures
        self._max_barren_reconnects = max_barren_reconnects
        self._max_reconnect_delay = max_reconnect_delay
        self._backend = backend or settings.capture_backend

    @property
    def is_live(self) -> bool:
        """Whether a dropped frame should trigger a reconnect.

        A path that exists on disk is a recording and is allowed to end;
        anything else is a camera or a stream and is expected to keep going.
        """
        return not (isinstance(self._target, str) and Path(self._target).exists())

    def _candidates(self) -> tuple[str, ...]:
        """Backend names to try, in order.

        Only webcams get a fallback: files and streams are decoded rather than
        captured from a device, so the backend is not where they go wrong.
        """
        if self._backend != "auto":
            return (self._backend,)
        if isinstance(self._target, int):
            return _AUTO_ORDER
        return ("any",)

    def open(self) -> None:
        errors = []
        for name in self._candidates():
            capture = cv2.VideoCapture(self._target, _BACKENDS[name])

            if not capture.isOpened():
                capture.release()
                errors.append(f"{name}: device did not open")
                continue

            # Opening is not proof of anything on a webcam. Take one frame to
            # find out whether this backend can actually stream, and discard
            # it - a single frame at startup is not worth complicating the
            # read loop for.
            if isinstance(self._target, int) and not capture.read()[0]:
                capture.release()
                errors.append(f"{name}: opened but delivered no frame")
                continue

            self._capture = capture
            logger.info(
                "Opened %s source: %s (backend: %s)",
                "live" if self.is_live else "recorded",
                self.source,
                name,
            )
            return

        raise RuntimeError(
            f"Could not open video source: {self.source} ({'; '.join(errors)})"
        )

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> VideoSource:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    def __iter__(self) -> Iterator[np.ndarray]:
        if self._capture is None:
            self.open()

        failures = 0
        barren_reconnects = 0
        while True:
            if self._capture is None:  # a reconnect attempt failed; try again
                barren_reconnects += 1
                if 0 < self._max_barren_reconnects < barren_reconnects:
                    logger.error("Giving up on %s", self.source)
                    return
                self._reconnect(barren_reconnects)
                continue

            ok, frame = self._capture.read()

            if ok:
                failures = 0
                barren_reconnects = 0  # the source is genuinely alive again
                yield frame
                continue

            if not self.is_live:
                logger.info("End of recording: %s", self.source)
                return

            failures += 1
            logger.warning("Dropped frame %d/%d from %s",
                           failures, self._max_failures, self.source)
            if failures < self._max_failures:
                continue

            barren_reconnects += 1
            if 0 < self._max_barren_reconnects < barren_reconnects:
                logger.error(
                    "%s opens but delivers no frames after %d reconnects; giving up. "
                    "Is the camera present and not held by another application?",
                    self.source,
                    self._max_barren_reconnects,
                )
                return

            if barren_reconnects == 1:
                logger.error(
                    "Lost %s. Retrying until it comes back - no frames are being "
                    "monitored until then.",
                    self.source,
                )
            self._reconnect(barren_reconnects)
            failures = 0

    def _backoff(self, attempt: int) -> float:
        """Exponential, capped. Attempt 1 waits the base delay, 2 waits double,
        and so on up to the ceiling, so a long outage is retried patiently
        rather than hammered."""
        return min(self._reconnect_delay * 2 ** (attempt - 1), self._max_reconnect_delay)

    def _reconnect(self, attempt: int = 1) -> bool:
        """Reopen a live source. Returns whether the stream is usable again.

        A failure is not fatal: the caller keeps trying, because the reason a
        camera cannot be opened right now is usually the reason it will open
        again in a minute.
        """
        delay = self._backoff(attempt)
        logger.warning(
            "Reconnecting to %s in %.0fs (attempt %d)", self.source, delay, attempt
        )
        self.release()
        time.sleep(delay)

        try:
            self.open()
        except RuntimeError as error:
            logger.warning("Reconnect to %s failed: %s", self.source, error)
            return False
        logger.info("Recovered %s after %d attempt(s)", self.source, attempt)
        return True
