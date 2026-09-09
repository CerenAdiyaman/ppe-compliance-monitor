"""Read-only HTTP access to recorded violations.

Read-only on purpose: the monitoring process is the only writer, and an API
that could insert or delete rows would turn a safety log into something a
dispute could be argued about. Run it beside the monitor, or on another machine
pointed at the same database.

    uvicorn hardhat_guard.api:app --reload
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from hardhat_guard import __version__
from hardhat_guard.storage import ViolationRecord, ViolationRepository

app = FastAPI(
    title="Hardhat Guard",
    description="Recorded PPE violations.",
    version=__version__,
)

_repository: ViolationRepository | None = None


def get_repository() -> ViolationRepository:
    """Lazily built and reused.

    A FastAPI dependency rather than a module-level object so tests can
    override it with a repository pointed at an in-memory database.
    """
    global _repository
    if _repository is None:
        _repository = ViolationRepository()
    return _repository


# FastAPI resolves this per request; the Annotated form keeps the call out
# of a default argument, where it would be evaluated once at import time.
Repo = Annotated[ViolationRepository, Depends(get_repository)]


class ViolationOut(BaseModel):
    """The wire shape of a violation.

    Explicit rather than serialising the ORM row: the response then stays
    stable when a column is added, and internals are not published by accident.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    violation_type: str
    camera_id: str
    location: str
    detected_at: datetime
    confidence: float
    has_snapshot: bool

    @classmethod
    def of(cls, record: ViolationRecord) -> ViolationOut:
        return cls(
            id=record.id,
            violation_type=record.violation_type,
            camera_id=record.camera_id,
            location=record.location,
            detected_at=record.detected_at,
            confidence=record.confidence,
            # The path is deliberately not exposed: it is a filesystem detail,
            # and clients should fetch the image through the endpoint below.
            has_snapshot=record.snapshot_path is not None,
        )


class Health(BaseModel):
    status: str
    version: str


@app.get("/health", response_model=Health, tags=["meta"])
def health() -> Health:
    return Health(status="ok", version=__version__)


@app.get("/violations", response_model=list[ViolationOut], tags=["violations"])
def list_violations(
    repository: Repo,
    violation_type: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> list[ViolationOut]:
    """Newest first. `limit` is capped so one request cannot pull the archive."""
    records = repository.list(
        violation_type=violation_type,
        since=since,
        until=until,
        limit=limit,
        offset=offset,
    )
    return [ViolationOut.of(record) for record in records]


@app.get("/violations/stats", tags=["violations"])
def violation_stats(
    repository: Repo,
) -> dict[str, int]:
    """Counts per violation type.

    Declared before `/violations/{violation_id}` because FastAPI matches routes
    in order; the other way round, "stats" would be parsed as an id.
    """
    return repository.stats()


@app.get("/violations/{violation_id}", response_model=ViolationOut, tags=["violations"])
def get_violation(
    violation_id: int,
    repository: Repo,
) -> ViolationOut:
    record = repository.get(violation_id)
    if record is None:
        raise HTTPException(status_code=404, detail="No such violation")
    return ViolationOut.of(record)


@app.get("/violations/{violation_id}/snapshot", tags=["violations"])
def get_snapshot(
    violation_id: int,
    repository: Repo,
) -> FileResponse:
    """The stored JPEG, which is always the face-blurred version."""
    record = repository.get(violation_id)
    if record is None:
        raise HTTPException(status_code=404, detail="No such violation")
    if record.snapshot_path is None:
        raise HTTPException(status_code=404, detail="This violation has no snapshot")

    path = Path(record.snapshot_path)
    if not path.exists():
        # The row outlives the image: retention may have purged it.
        raise HTTPException(status_code=410, detail="Snapshot no longer on disk")

    return FileResponse(path, media_type="image/jpeg", filename=path.name)
