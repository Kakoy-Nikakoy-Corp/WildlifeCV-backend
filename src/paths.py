from pathlib import Path
from typing import Callable


def fixdir(func: Callable[[], Path]):
    """Создает директорию, если её не существует."""
    def wrapper():
        dir_path = func()
        if dir_path:
            dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    return wrapper


def get_project_root():
    """Возвращает путь к корню проекта."""
    current_path = Path('.').resolve()
    return current_path


@fixdir
def get_uploads_dpath() -> Path:
    """Возвращает путь к директории с файлами, загруженными пользователями."""
    return get_project_root() / 'uploads'


@fixdir
def get_videos_dpath() -> Path:
    """Возвращает путь к директории с загруженными видео."""
    return get_uploads_dpath() / 'videos'


@fixdir
def get_images_dpath() -> Path:
    """Возвращает путь к директории с загруженными фото."""
    return get_uploads_dpath() / 'images'


@fixdir
def get_archives_dpath() -> Path:
    """Возвращает путь к директории с загруженными архивами."""
    return get_uploads_dpath() / 'archives'


def get_tests_dpath() -> Path:
    """Возвращает путь к директории с тестами."""
    return get_project_root() / 'tests'


def get_test_videos_dpath() -> Path:
    """Возвращает путь к директории с тестовыми видео."""
    return get_tests_dpath() / 'videos'


@fixdir
def get_model_weights_dpath() -> Path:
    """Возвращает путь к директории с весамим моделей."""
    return get_project_root() / 'weights'


def get_yolo_weights_path() -> Path:
    """Возвращает путь к актуальным весам YOLO26."""
    return get_model_weights_dpath() / 'best.pt'
