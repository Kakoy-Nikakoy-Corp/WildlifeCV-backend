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
    conf: list[float]
    bbox_coords: torch.Tensor


@dataclass(slots=True, frozen=True)
class ProcessedFrame:
    """Video frame with metadata and prediction results."""
    number: int
    timecode: Timecode
    prediction: ModelPrediction


@dataclass(slots=True)
class TimeInterval:
    """An interval which covers one particular snow leopard appearance in a given video."""
    start: Timecode
    end: Timecode | None

    def __str__(self) -> str:
        return f"{self.start}-{self.end}"
