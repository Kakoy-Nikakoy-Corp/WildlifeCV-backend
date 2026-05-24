from dataclasses import dataclass
from enum import Enum
from typing import TypedDict

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
class FrameResults:
    """Snow leopard detection results for a single video frame."""
    number: int
    timecode: Timecode
    confs: list[float]
    bbox_coords: list[list[float] | None]


@dataclass(slots=True)
class TimeInterval:
    """An interval which covers one particular snow leopard appearance in a given video."""
    start: Timecode
    end: Timecode | None

    def __str__(self) -> str:
        return f"{self.start}-{self.end}"
