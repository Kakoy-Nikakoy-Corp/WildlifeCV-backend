from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypedDict, Protocol, NamedTuple

import torch
from timecode import Timecode


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


class RecognitionStatus(StrEnum):
    """
    Статус ответа эндпоинта.

    OK: Эндпоинт отработал по стандартному сценарию.
    SIZE_LIMIT: Файл, переданное в эндпоинт, превышает максимальный размер.
    DOWNLOAD_ERROR: Произошла ошибка во время загрузки файла
    """
    OK = 'OK'
    SIZE_LIMIT = "SIZE_LIMIT"
    DOWNLOAD_ERROR = "DOWNLOAD_ERROR"


class VideoSuccessResponse(TypedDict):
    """
    Успешный ответ эндпоинта обработки видео.

    Parameters:
        status: Значение RecognitionStatus
        data: Результат от модели
    """
    status: RecognitionStatus
    data: VideoRecognitionOutput


class ImageSuccessResponse(TypedDict):
    """
    Успешный ответ эндпоинта обработки изображений.

    Parameters:
        status: Значение RecognitionStatus
        path: Путь к обработанному изображению
    """
    status: RecognitionStatus
    path: Path


class DownloadErrorResponse(TypedDict):
    """
    Формат ответа эндпоинта, если на нем выброшена ошибка.

    Parameters:
        status: Значения RecognitionStatus
        detail: Комментарий к ошибке
    """
    status: RecognitionStatus
    detail: str


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
