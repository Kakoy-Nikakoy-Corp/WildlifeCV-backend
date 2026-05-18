from typing import Protocol


class ModelInterface(Protocol):
    """Интерфейс класса модели."""
    def recognise(self, frame):
        ...


class ModelDouble:
    """Заглушка модели."""
    def recognise(self, frame):
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
