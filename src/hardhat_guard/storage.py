"""Persistence: a row per violation and its redacted snapshot on disk.

The two are deliberately separate. Images in a database bloat every query and
every backup; a path in a row keeps the table small and lets the operating
system serve the file. The row is the record of truth - a missing image
degrades the evidence but never breaks a query.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import DateTime, Engine, String, TypeDecorator, create_engine, func, select
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

from hardhat_guard.config import settings
from hardhat_guard.rules import Violation

logger = logging.getLogger(__name__)

# JPEG, not PNG: a snapshot is evidence for a human to look at, and a lossless
# frame costs roughly ten times the space for no forensic gain here.
_JPEG_QUALITY = 90


class UtcDateTime(TypeDecorator[datetime]):
    """A datetime column that is always timezone-aware UTC in Python.

    SQLite has no native timezone: an aware datetime goes in and a naive one
    comes back, and comparing that naive value against an aware one raises
    TypeError somewhere far from the cause. Normalising in both directions
    keeps the rest of the code free of the question.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            # A naive value from a caller is taken as UTC rather than rejected;
            # the alternative is a crash deep in the write path.
            return value
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=timezone.utc)


class Base(DeclarativeBase):
    pass


class ViolationRecord(Base):
    """One confirmed violation.

    Times are stored in UTC. A site spanning time zones, or a server that moves
    to summer time mid-shift, would otherwise produce records that cannot be
    ordered.
    """

    __tablename__ = "violations"

    id: Mapped[int] = mapped_column(primary_key=True)
    violation_type: Mapped[str] = mapped_column(String(32), index=True)
    camera_id: Mapped[str] = mapped_column(String(32), index=True)
    location: Mapped[str] = mapped_column(String(128), default="")
    detected_at: Mapped[datetime] = mapped_column(UtcDateTime, index=True)
    confidence: Mapped[float]
    snapshot_path: Mapped[str | None] = mapped_column(String(512), default=None)

    def __repr__(self) -> str:
        return (
            f"<ViolationRecord {self.id} {self.violation_type} "
            f"{self.camera_id} {self.detected_at:%Y-%m-%d %H:%M:%S}>"
        )


class FileStore:
    """Writes snapshots under ``violations_dir/<date>/``.

    Dated subdirectories keep any single directory small enough for a file
    manager to open, and make retention a matter of deleting whole folders.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or settings.violations_dir

    def save(self, frame: np.ndarray, violation: Violation) -> Path | None:
        """Persist one already-anonymised frame. Returns ``None`` on failure.

        A disk error must not stop monitoring: losing an image is bad, losing
        the next hour of detections because the disk filled up is worse.
        """
        directory = self.root / violation.detected_at.strftime("%Y-%m-%d")
        filename = (
            f"{violation.detected_at:%H%M%S}_{violation.camera_id}_{violation.violation_type}.jpg"
        )
        path = directory / filename

        try:
            directory.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        except OSError:
            logger.exception("Could not write snapshot to %s", path)
            return None

        if not ok:
            logger.error("OpenCV declined to encode a snapshot at %s", path)
            return None
        return path

    def purge_older_than(self, days: int) -> int:
        """Delete snapshot folders older than ``days``; returns how many went.

        Retention is not automatic - call this from a scheduled job to enforce
        the policy your lawful basis for storing the images depends on.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date()
        removed = 0

        for directory in sorted(p for p in self.root.glob("*") if p.is_dir()):
            try:
                folder_date = datetime.strptime(directory.name, "%Y-%m-%d").date()
            except ValueError:
                continue  # not one of ours; leave it alone
            if folder_date >= cutoff:
                continue

            for file in directory.iterdir():
                file.unlink()
            directory.rmdir()
            removed += 1

        if removed:
            logger.info("Purged %d snapshot folder(s) older than %d days", removed, days)
        return removed


def _engine_options(url: str) -> dict[str, object]:
    """SQLite needs help to survive being read from more than one thread.

    The monitor writes from its own loop while the API serves requests from a
    thread pool, so both of the driver's threading defaults get in the way:

    * ``check_same_thread`` rejects a connection used off its creating thread.
    * The default pool hands each thread its own connection, which for an
      in-memory database means each thread gets its own empty database - the
      schema is created in one and queried in another.

    Neither applies to a real server database, so the options are scoped to
    SQLite URLs only.
    """
    if not url.startswith("sqlite"):
        return {}

    options: dict[str, object] = {"connect_args": {"check_same_thread": False}}
    if ":memory:" in url or url in ("sqlite://", "sqlite:///:memory:"):
        options["poolclass"] = StaticPool  # one shared connection, one database
    return options


class ViolationRepository:
    """The only place that talks to the database."""

    def __init__(self, database_url: str | None = None, engine: Engine | None = None) -> None:
        url = database_url or settings.database_url
        self._engine = engine or create_engine(url, **_engine_options(url))
        self._session_factory = sessionmaker(bind=self._engine)
        Base.metadata.create_all(self._engine)

    def add(self, violation: Violation, snapshot_path: Path | None = None) -> ViolationRecord:
        record = ViolationRecord(
            violation_type=violation.violation_type,
            camera_id=violation.camera_id,
            location=violation.location,
            detected_at=violation.detected_at,
            confidence=violation.confidence,
            snapshot_path=str(snapshot_path) if snapshot_path else None,
        )
        with self._session_factory.begin() as session:
            session.add(record)
            session.flush()
            session.expunge(record)  # usable after the session closes
        return record

    def get(self, violation_id: int) -> ViolationRecord | None:
        with self._session_factory() as session:
            return session.get(ViolationRecord, violation_id)

    def list(
        self,
        violation_type: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[ViolationRecord]:
        """Newest first, because that is what an operator opens the log for."""
        query = select(ViolationRecord).order_by(ViolationRecord.detected_at.desc())

        if violation_type is not None:
            query = query.where(ViolationRecord.violation_type == violation_type)
        if since is not None:
            query = query.where(ViolationRecord.detected_at >= since)
        if until is not None:
            query = query.where(ViolationRecord.detected_at <= until)

        with self._session_factory() as session:
            return session.scalars(query.limit(limit).offset(offset)).all()

    def stats(self) -> dict[str, int]:
        """Counts per violation type."""
        query = select(ViolationRecord.violation_type, func.count()).group_by(
            ViolationRecord.violation_type
        )
        with self._session_factory() as session:
            return dict(session.execute(query).all())  # type: ignore[arg-type]
