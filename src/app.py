from pathlib import Path
from typing import Final
from uuid import uuid4
from urllib.parse import urljoin
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
import puremagic
from sympy.external.ntheory import j

from src.model import Model
from src.paths import get_output_dpath, get_output_videos_dpath, get_output_images_dpath, get_output_archives_dpath, get_project_root
from src.types import RecognitionStatus, LoadingErrorResponse
from src.types import VideoSuccessResponse, VideoRecognitionOutput
from src.types import ImageSuccessResponse, MultiImageSuccessResponse
from src.utils import download_file

app = FastAPI()
model = Model()

origins = [
    "http://localhost:5173",
    "https://irbis.wild1.net",
    "https://api.irbis.wild1.net"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"]
)
app.mount("/output", StaticFiles(directory=get_output_dpath()), "outputs")

MAX_SIZE_MIB: Final = 500
CHUNK_SIZE: Final = 8 * 1024 * 1024  # 8 мебибайт
ROOT_LINK: Final = 'https://api.irbis.wild1.net'
ALLOWED_VIDEO_MIMES = {'video/mp4', 'video/x-matroska', 'video/matroska'}
ALLOWED_IMAGE_MIMES = {'image/jpeg', 'image/png'}
ALLOWED_ARCHIVE_MIMES = {'application/zip', 'application/x-7z-compressed'}


@app.post("/recognise/video/")
async def recognise_video(video: UploadFile) -> VideoSuccessResponse | LoadingErrorResponse:
    """
    Запускает пайплайн на видеофайле.

    Parameters:
        video (UploadFile): Видеофайл

    Returns:
        Словарь со статусом и таймкодами.
    """

    if video.filename is None:
        raise HTTPException(422, detail='Сервис не обрабатывает файл без имени :(')

    if video.content_type not in ALLOWED_VIDEO_MIMES:
        raise HTTPException(400, detail='Поддерживаются только видео, фото и архивы с фото')

    # Получаем расширение файла
    video_name = Path(video.filename)
    if video_name.suffix is None:
        ext = puremagic.from_file(video_name)
    else:
        ext = video_name.suffix

    with NamedTemporaryFile(suffix=ext) as video_file:
        download_status = await download_file(video_file, video)

        match download_status:
            case RecognitionStatus.DOWNLOAD_ERROR as error:
                return LoadingErrorResponse(
                    status=error,
                    detail="Во время загрузки файла произошла ошибка :("
                )

            case RecognitionStatus.SIZE_LIMIT as error:
                return LoadingErrorResponse(
                    status=error,
                    detail=f"Файл слишком большой, лимит - {MAX_SIZE_MIB:.2f}"
                )

            case _:  # case IRBIS_FOUND
                # Собираем путь для сохранения видео от модели и вызываем модель
                output_name = f'{uuid4()}.mp4'
                output_path = get_output_videos_dpath() / output_name
                video_path = Path(video_file.name)
                timestrings = model.detect_video_intervals(
                    video_path, output_path,
                    window_threshold=0.5,
                    smoothing_interval=3,
                    gap=10,
                    batch_size=32
                )
                if not timestrings:
                    return LoadingErrorResponse(
                        status=RecognitionStatus.NO_IRBIS_FOUND,
                        detail="Снежный барс не обнаружен :("
                    )

                relative_output_path = str(output_path.relative_to(get_project_root()))
                bboxed_video_link = urljoin(ROOT_LINK, relative_output_path)
                return VideoSuccessResponse(
                    status=RecognitionStatus.IRBIS_FOUND,
                    data=VideoRecognitionOutput(
                        timestrings=timestrings,
                        link=bboxed_video_link
                    )
                )




@app.post("/recognise/image/")
async def recognise_image(image: UploadFile) -> ImageSuccessResponse | LoadingErrorResponse:
    """
    Запускает пайплайн на изображении.

    Parameters:
        image (UploadFile): Изображение
    """

    if image.filename is None:
        raise HTTPException(422, detail='Сервис не обрабатывает файл без имени :(')

    if image.content_type not in ALLOWED_IMAGE_MIMES:
        raise HTTPException(status_code=400, detail='Поддерживаются только видео, фото и архивы с фото!')

    # Получаем суффикс из файла
    image_name = Path(image.filename)
    if not image_name.suffix:
        ext = puremagic.from_file(image_name)
    else:
        ext = image_name.suffix

    # Загружаем файл
    with NamedTemporaryFile('wb', suffix=ext) as image_file:
        download_status = await download_file(image_file, image)
        match download_status:
            case RecognitionStatus.DOWNLOAD_ERROR as error:
                return LoadingErrorResponse(
                    status=error,
                    detail="Во время загрузки файла произошла ошибка :("
                )

            case RecognitionStatus.SIZE_LIMIT as error:
                return LoadingErrorResponse(
                    status=error,
                    detail=f"Файл слишком большой, лимит - {MAX_SIZE_MIB:.2f}"
                )

            case _:
                # Собираем путь для сохранения видео от модели и вызываем модель
                image_path = Path(image_file.name)
                output_name = f'{uuid4()}.jpg'
                output_path = get_output_images_dpath() / output_name
                is_found = model.detect_image(image_path, output_path)
                if not is_found:
                    return LoadingErrorResponse(
                        status=RecognitionStatus.NO_IRBIS_FOUND,
                        detail="Снежный барс не обнаружен :("
                    )

                relative_output_path = str(output_path.relative_to(get_project_root()))
                bboxed_image_link = urljoin(ROOT_LINK, relative_output_path)
                return ImageSuccessResponse(
                    status=RecognitionStatus.IRBIS_FOUND,
                    link=bboxed_image_link
                )
