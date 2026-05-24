from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypedDict, Protocol

from timecode import Timecode


class RecognitionStatus(str, Enum):
    """Статус ответа модели. Взят из фронтенда."""
    SUCCESS = 'success'
    ERROR = 'error'


class VideoResponse(TypedDict):
    """Формат ответа эндпоинта. Взята из фронтенда."""
    status: RecognitionStatus
    timestrings: list[str]  # 'HH:MM:SS - HH:MM:SS', '...'
    path: Path


@dataclass(slots=True)
class VideoRecognitionOutput:
    """
    Результат работы модели на видео.

    Parameters:
        timestrings: Список таймкодов (длинных)
        path: Путь до видео с bbox'ами
    """
    timestrings: list[str]
    path: Path


@dataclass(slots=True, frozen=True)
class FrameResults:
    """Snow leopard detection results for a single video frame."""
    number: int
    timecode: Timecode
    confs: list[float]
    bbox_coords: list[list[float]]


@dataclass(slots=True)
class TimeInterval:
    """An interval which covers one particular snow leopard appearance in a given video."""
    start: Timecode
    end: Timecode | None

    def __str__(self) -> str:
        return f"{self.start}-{self.end}"


class ModelProtocol(Protocol):
    def detect_video_timeintervals(
        self,
        video_path: Path,
        window_coef: float,
        threshold: float,
        smoothing_interval: float,
        gap: int,
        batch_size: int
    ) -> VideoRecognitionOutput:
        ...

    def detect_image(self, image_path: Path) -> Path:
        ...

    # Обработка картинок в архиве:
    #   - Собираем картинки в объект видео
    #   - Сохраняем видео на диск
    #   - Юзаем detect_video
