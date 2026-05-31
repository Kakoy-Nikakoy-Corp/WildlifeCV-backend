import sys
from collections.abc import Generator
from typing import Any

import pytest
from pytest_mock import MockerFixture
from fastapi.testclient import TestClient

from src.paths import (
    get_output_archives_dpath,
    get_output_images_dpath,
    get_output_videos_dpath
)


@pytest.fixture(scope='session')
def client(session_mocker: MockerFixture) -> TestClient:
    mock_module = session_mocker.MagicMock()
    session_mocker.patch.dict(
        sys.modules, {
            'torchcodec': mock_module,
            'torchcodec.decoders': mock_module,
            'torchcodec.encoders': mock_module,
        }
    )
    session_mocker.patch('src.model.Model', return_value="MODEL")
    from src.app import app
    return TestClient(app)


@pytest.fixture(scope='session')
def output_video() -> Generator[str, Any, None]:
    video_name = "video.mp4"
    video_path = get_output_videos_dpath() / video_name
    with open(video_path, "wb"):
        yield video_name

    video_path.unlink()


@pytest.fixture(scope='session')
def output_image() -> Generator[str, Any, None]:
    image_name = "image.jpg"
    image_path = get_output_images_dpath() / image_name
    with open(image_path, "wb"):
        yield image_name

    image_path.unlink()


@pytest.fixture(scope='session')
def output_archive() -> Generator[str, Any, None]:
    archive_name = "archive.zip"
    archive_path = get_output_archives_dpath() / archive_name
    with open(archive_path, "wb"):
        yield archive_name

    archive_path.unlink()


def test_video_distribution(client: TestClient, output_video: str):
    video_path = get_output_videos_dpath() / output_video
    sut = client

    response = sut.get(
        f"/output/videos/{output_video}",
        params={
            "video": f"file:///{video_path}"
        }
    )

    assert response.status_code == 200


def test_image_distribution(client: TestClient, output_image: str):
    image_path = get_output_images_dpath() / output_image
    sut = client

    response = sut.get(
        f"/output/images/{output_image}",
        params={
            "video": f"file:///{image_path}"
        }
    )

    assert response.status_code == 200


def test_archive_distribution(client: TestClient, output_archive: str):
    archive_path = get_output_archives_dpath() / output_archive
    sut = client

    response = sut.get(
        f"/output/images/{output_archive}",
        params={
            "video": f"file:///{archive_path}"
        }
    )

    assert response.status_code == 200


def test_distribution_limits(client: TestClient):
    sut = client
    urls = (
        "/output/videos/video.mp4",
        "/output/images/image.jpg",
        "/output/archive/archive.zip",
        "/output/model/"
    )

    for url in urls:
        response = sut.get(url)
        assert response.status_code == 404
