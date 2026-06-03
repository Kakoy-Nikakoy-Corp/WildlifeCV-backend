from collections.abc import Callable
from pathlib import Path

from torch import cuda


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
def get_output_dpath() -> Path:
    """Возвращает путь к директории с файлами, обработанными YOLO."""
    return get_project_root() / 'output'


@fixdir
def get_output_videos_dpath() -> Path:
    """Возвращает путь к директории с обработанными видео."""
    return get_output_dpath() / 'videos'


@fixdir
def get_output_images_dpath() -> Path:
    """Возвращает путь к директории с обработанными фото."""
    return get_output_dpath() / 'images'


@fixdir
def get_output_images_with_irbis_dpath() -> Path:
    """Возвращает путь к директории с обработанными фото с барсами."""
    return get_output_images_dpath() / 'snow_leopards'


@fixdir
def get_output_images_with_bg_dpath() -> Path:
    """Возвращает путь к директории с обработанными фото без барсов."""
    return get_output_images_dpath() / 'backgrounds'


@fixdir
def get_output_archives_dpath() -> Path:
    """Возвращает путь к директории с обработанными архивами."""
    return get_output_dpath() / 'archives'


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
    if cuda.is_available():
        return get_model_weights_dpath() / 'best.engine'

    onnx_path = get_model_weights_dpath() / 'best.onnx'
    if onnx_path.exists():
        return onnx_path
    return get_model_weights_dpath() / 'best.pt'


@fixdir
def get_logs_dpath() -> Path:
    """Возвращает путь к логам проекта."""
    return get_project_root() / 'logs'


def get_backend_logs_path() -> Path:
    """Возвращает путь к файлу с логами бэкенда."""
    return get_logs_dpath() / 'backend_inf.log'


def get_model_logs_path() -> Path:
    return get_logs_dpath() / 'model_inf.log'
