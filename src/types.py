from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import TypedDict

import torch
from timecode import Timecode


class RecognitionStatus(str, Enum):
    """Статус ответа модели. Взят из фронтенда."""
    SUCCESS = 'success'
    ERROR = 'error'


class RecognitionResponse(TypedDict):
    """Формат ответа эндпоинта. Взята из фронтенда."""
    status: RecognitionStatus
    timestrings: list[str]  # 'HH:MM:SS - HH:MM:SS', '...'


@dataclass(slots=True, frozen=True)
class ModelPrediction:
    """Snow leopard prediction results for a single image."""
    peak_conf: float
    img: torch.Tensor


@dataclass(slots=True, frozen=True)
class ProcessedFrame:
    """A single video frame with metadata and prediction results."""
    number: int
    timecode: Timecode
    prediction: ModelPrediction


@dataclass(slots=True, frozen=True)
class ProcessedVideo:
    fps: float
    width: int
    height: int
    frame_count: int
    frames: Iterator[ProcessedFrame]


@dataclass(slots=True, frozen=True)
class LetterboxParams:
    r: float
    new_w: int
    new_h: int
    pad_w: int
    pad_h: int

@dataclass(slots=True)
class TimeInterval:
    """An interval which covers one particular snow leopard appearance in a given video."""
    start: Timecode
    end: Timecode | None

    def __str__(self) -> str:
        return f"{self.start}-{self.end}"
