from pathlib import Path
from typing import Protocol, TypeAlias

from cv2.typing import MatLike
from numpy import ndarray

from src.types import Recognition


OpenCVFrame: TypeAlias = MatLike


class ModelInterface(Protocol):
    """Интерфейс класса модели."""
    def recognise(self, frame) -> Recognition:
        ...


class ModelDouble:
    """Заглушка модели."""
    def recognise(self, frame: OpenCVFrame = ndarray([])) -> Recognition:
        """
        Ищет барса на изображении.

        Parameters:
            frame: Изображение
        """
        return Recognition(proba=0.5, bbox_coords=(1., 2., 3., 4.))


def rolling_window_double(recognitions: list[Recognition]) -> list[str]:
    """
    Заглушка функции rolling window.

    Parameters:
        recognitions: Список пар 'вероятность+координаты bbox'
    Returns:
        Список таймкодов
    """
    ...
