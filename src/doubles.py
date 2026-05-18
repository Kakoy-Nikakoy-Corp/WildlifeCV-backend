from typing import Protocol, TypeAlias

from cv2.typing import MatLike
from numpy import ndarray


OpenCVFrame: TypeAlias = MatLike


class ModelInterface(Protocol):
    """Интерфейс класса модели."""
    def recognise(self, frame):
        ...


class ModelDouble:
    """Заглушка модели."""
    def recognise(self, frame: OpenCVFrame = ndarray([])):
        """
        Ищет барса на изображении.

        Parameters:
            frame: Изображение
        """
        return None


def rolling_window_double(recognitions: list) -> list[str]:
    """
    Заглушка функции rolling window.

    Parameters:
        recognitions: Список пар 'вероятность+координаты bbox'
    Returns:
        Список таймкодов
    """
    ...
