from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.app import app
from src.model import Model
from src.paths import get_test_videos_dpath


@pytest.fixture(scope='session')
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope='session')
def model() -> Model:
    return Model()


@pytest.fixture(scope='session')
def video_with_irbis() -> Path:
    """Видео с барсом."""
    return get_test_videos_dpath() / 'Ирбис.mkv'


@pytest.fixture(scope='session')
def video_without_irbis() -> Path:
    """Видео без барса."""
    return get_test_videos_dpath() / 'заяц-беляк.mkv'


def test_model_recognition():
    ...
