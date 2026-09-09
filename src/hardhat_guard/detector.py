"""YOLO inference wrapper.

Loads the fine-tuned model once and exposes a single ``detect`` call that
returns plain dataclasses, so the rest of the pipeline never touches
Ultralytics types.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from hardhat_guard.config import settings

logger = logging.getLogger(__name__)

ClassName = Literal["head", "helmet"]


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected object in a single frame."""

    class_name: ClassName
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixel coordinates

    @property
    def is_violation(self) -> bool:
        """A bare head means the worker is not wearing a hard hat."""
        return self.class_name == "head"


class HardHatDetector:
    def __init__(self) -> None:
        if not settings.model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {settings.model_path}")

        logger.info("Loading model from %s on %s", settings.model_path, settings.device)
        self._model = YOLO(str(settings.model_path))
        self._class_names: dict[int, ClassName] = {
            settings.class_head: "head",
            settings.class_helmet: "helmet",
        }

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on one BGR frame."""
        # `predict` is annotated as possibly returning a generator (stream=True),
        # which this call never does; narrow it so the boxes below are typed.
        results = cast(
            list[Results],
            self._model.predict(
                frame,
                conf=settings.confidence_threshold,
                classes=list(self._class_names),  # person is filtered out here
                device=settings.device,
                verbose=False,
            ),
        )

        detections: list[Detection] = []
        boxes = results[0].boxes
        if boxes is None:  # the model ran a task that produces no boxes
            return detections

        # Indexed rather than iterated: `Boxes` supports the legacy __getitem__
        # protocol but declares no __iter__, which type checkers reject.
        for i in range(len(boxes)):
            box = boxes[i]
            class_id = int(box.cls.item())
            name = self._class_names.get(class_id)
            if name is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    class_name=name,
                    confidence=float(box.conf.item()),
                    box=(x1, y1, x2, y2),
                )
            )
        return detections
