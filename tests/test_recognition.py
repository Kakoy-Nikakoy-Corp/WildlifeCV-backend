from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from src.app import app
from src.doubles import ModelDouble, ModelInterface
from src.paths import get_test_videos_dpath
from src.types import Recognition


@pytest.fixture(scope='session')
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope='session')
def model() -> ModelDouble:
    return ModelDouble()


@pytest.fixture(scope='session')
def video_with_irbis() -> Path:
    """Видео с барсом."""
    return get_test_videos_dpath() / 'Ирбис.mkv'


@pytest.fixture(scope='session')
def video_without_irbis() -> Path:
    """Видео без барса."""
    return get_test_videos_dpath() / 'заяц-беляк.mkv'


def test_model_recognition(
    model: ModelDouble,
    video_with_irbis: Path,
):
    sut = get_recognitions

    predictions = sut(model, video_with_irbis)

    assert predictions[0] == model.recognise()
