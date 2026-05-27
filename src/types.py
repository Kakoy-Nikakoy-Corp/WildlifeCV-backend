from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypedDict, Protocol, NamedTuple

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
