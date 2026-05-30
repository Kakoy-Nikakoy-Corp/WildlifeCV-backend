from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, TypedDict

import torch
from timecode import Timecode


class RecognitionStatus(StrEnum):
    """
    Статус ответа эндпоинта.

    IRBIS_FOUND: Эндпоинт отработал по стандартному сценарию, барсы найдены в медиа
    SIZE_LIMIT: Файл, переданное в эндпоинт, превышает максимальный размер
    DOWNLOAD_ERROR: Произошла ошибка во время загрузки файла
    NO_IRBIS_FOUND: В медиа не найдено ни одного барса
    """
    IRBIS_FOUND = "IRBIS_FOUND"
    SIZE_LIMIT = "SIZE_LIMIT"
    DOWNLOAD_ERROR = "DOWNLOAD_ERROR"
    NO_IRBIS_FOUND = "NO_IRBIS_FOUND"


class LoadingErrorResponse(TypedDict):
    """
    Формат ответа эндпоинта, если на нем выброшена ошибка.

    Parameters:
        status: Значения RecognitionStatus
        detail: Комментарий к ошибке
    """
    status: RecognitionStatus
    detail: str


class VideoSuccessResponse(TypedDict):
    """
    Успешный ответ эндпоинта обработки видео.

    Parameters:
        status: Значение RecognitionStatus
        timestrings: Список таймкодов (длинных)
        link: Путь до видео с bbox'ами
    """
    status: RecognitionStatus
    timestrings: list[str]
    link: str


class ImageSuccessResponse(TypedDict):
    """
    Успешный ответ эндпоинта обработки изображений.

    Parameters:
        status: Значение RecognitionStatus
        link: Ссылка на обработанное изображение
    """
    status: RecognitionStatus
    link: str


class MultiImageSuccessResponse(TypedDict):
    """
    Успешный ответ эндпоинта обработки архивов с изображениями.

    Parameters:
        status: Значение RecognitionStatus
        link: Ссылка на обработанный архив
        collage_images: Ссылки на картинки для превью архива
    """
    status: RecognitionStatus
    link: str
    collage_images: list[str]


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
        output_path: Path,
        window_coef: float = 1.5,
        window_threshold: float = 0.4,
        threshold: float = 0.25,
        smoothing_interval: float = 2,
        gap: int = 2,
        batch_size: int = 16
    ) -> list[str]:
        ...

    def detect_image(
        self,
        image_path: Path,
        output_path: Path,
        threshold: float = 0.25
    ) -> bool:
        ...

    # Обработка картинок в архиве:
    #   - Картинки отправляются в detect_image
