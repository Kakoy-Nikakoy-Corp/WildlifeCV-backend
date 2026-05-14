from dataclasses import dataclass
from enum import Enum
from typing import TypedDict


@dataclass(slots=True, frozen=True)
class Recognition:
    """
    Результат работы модели.

    Parameters:
        proba: Вероятность наличия барса на фото
        bbox_coords: Координаты рамки
    """
    proba: float
    bbox_coords: tuple[float, float, float, float]

    def __str__(self) -> str:
        return f"Proba: {self.proba}. Bbox coords: {self.bbox_coords}"


class RecognitionStatus(str, Enum):
    """
    Статус ответа модели. Взят из фронтенда.
    """
    SUCCESS = 'success'
    ERROR = 'error'


class RecognitionResponse(TypedDict):
    """
    Формат ответа эндпоинта. Взята из фронтенда.
    """
    status: RecognitionStatus
    timestamps: list[str]  # 'HH:MM:SS - HH:MM:SS', '...'
