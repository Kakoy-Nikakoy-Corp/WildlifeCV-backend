from pathlib import Path
from typing import Protocol, TypeAlias

from cv2.typing import MatLike

from src.types import Recognition


OpenCVFrame: TypeAlias = MatLike


class ModelInterface(Protocol):
    """Интерфейс класса модели."""
    def recognise(self, frame) -> Recognition:
        ...


class ModelDouble:
    """Заглушка модели."""
    def recognise(self, frame: OpenCVFrame) -> Recognition:
        """
        Ищет барса на изображении.

        Parameters:
            frame: Изображение
        """
        ...


def rolling_window_double(recognitions: list[Recognition]) -> list[str]:
    """
    Заглушка функции rolling window.

    Parameters:
        recognitions: Список пар 'вероятность+координаты bbox'
    Returns:
        Список таймкодов
    """
    ...
