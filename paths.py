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
