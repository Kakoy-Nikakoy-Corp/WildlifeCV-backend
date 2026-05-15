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
    PROJECT_NAME = "WildlifeCV-backend"
    current_path = Path(__file__).resolve()
    while current_path.parts[-1] != PROJECT_NAME:
        if current_path.parent == current_path:
            raise FileNotFoundError(f"Корень проекта {PROJECT_NAME} не найден.")
        current_path = current_path.parent
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
