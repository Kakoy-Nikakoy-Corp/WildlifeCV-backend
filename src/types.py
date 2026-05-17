from dataclasses import dataclass
from enum import Enum
from typing import TypedDict


class RecognitionStatus(str, Enum):
    """Статус ответа модели. Взят из фронтенда."""
    SUCCESS = 'success'
    ERROR = 'error'


class RecognitionResponse(TypedDict):
    """Формат ответа эндпоинта. Взята из фронтенда."""
    status: RecognitionStatus
    timestrings: list[str]  # 'HH:MM:SS - HH:MM:SS', '...'
