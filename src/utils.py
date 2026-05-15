from pathlib import Path

from cv2 import VideoCapture

from src.doubles import ModelInterface
from src.types import Recognition


def get_recognitions(model: ModelInterface, video_path: Path) -> list[Recognition]:
    """
    Разбирает видео на кадры и отправляет их в модель.

    Parameters:
        model: Объект модели
        video_path: Путь до модели
    Returns:
        Список Recognition
    """
    # Открываем видео
    video = VideoCapture(str(video_path))
    if not video.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео '{video_path}'")

    # Читаем по кадру и передаем кадр в модель
    frames: list[Recognition] = []
    while True:
        success, frame = video.read()
        if not success:
            break
        # Обращаемся к модели
        result = model.recognise(frame)
        frames.append(result)

    video.release()
    return frames
