from pathlib import Path
from typing import Final

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import puremagic

from src.model import Model
from src.paths import get_videos_dpath, get_images_dpath, get_archives_dpath
from src.types import RecognitionStatus, DownloadErrorResponse
from src.types import VideoSuccessResponse, ImageSuccessResponse
from src.utils import download_file, get_uuid4


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

MAX_SIZE_BYTES: Final = 500*1024*1024
CHUNK_SIZE: Final = 8*1024*1024  # 8 мебибайт
ALLOWED_VIDEO_MIMES = {'video/mp4', 'video/x-matroska', 'video/matroska'}
ALLOWED_IMAGE_MIMES = {'image/jpeg', 'image/png'}
ALLOWED_ARCHIVE_MIMES = {'application/zip', 'application/x-7z-compressed'}

# Есть middlware, который шарит директорию как роуты в бэке
@app.get("/video/")


@app.post("/recognise/video/")
async def recognise_video(video: UploadFile) -> VideoSuccessResponse | DownloadErrorResponse:
    """
    Запускает пайплайн на видеофайле.

    Parameters:
        file (UploadFile): Видеофайл

    Returns:
        Словарь со статусом и таймкодами.
    """

    if video.filename is None:
        raise HTTPException(422, detail='Сервис не обрабатывает файл без имени :(')

    if video.content_type not in ALLOWED_VIDEO_MIMES:
        raise HTTPException(status_code=400, detail='Поддерживаются только видео, фото и архивы с фото!')

    # Получаем расширение файла
    video_name = Path(video.filename)
    if video_name.suffix is None:
        ext = puremagic.from_file(video_name)
    else:
        ext = video_name.suffix

    video_path = get_videos_dpath() / f"{get_uuid4()}{ext}"
    # Загружаем файл
    download_status = await download_file(video, video_path)
    match download_status:
        case RecognitionStatus.DOWNLOAD_ERROR as error:
            return VideoErrorResponse(
                status=error,
                detail="Во время загрузки произошла ошибка :("
            )

        case RecognitionStatus.SIZE_LIMIT as error:
            max_size_mebibytes = MAX_SIZE_BYTES / 1024 / 1024
            return VideoErrorResponse(
                status=error,
                detail=f"Файл слишком большой, лимит - {(max_size_mebibytes):.2f}"
            )

        case RecognitionStatus.OK as success:
            # Предикты + rolling window
            model_response = model.detect_video_timeintervals(video_path)
            return VideoSuccessResponse(
                status=success,
                data=model_response
            )


@app.post("/recognise/image/")
async def recognise_image(image: UploadFile) -> ImageSuccessResponse | ImageErrorResponse:
    """
    Запускает пайплайн на изображении.

    Parameters:
        image (UploadFile): Изображение

    Returns:
        Словарь со статусом и таймкодами.
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

    image_path = get_images_dpath() / f"{get_uuid4()}{ext}"
    # Загружаем файл
    download_status = await download_file(image, image_path)
    match download_status:
        case RecognitionStatus.DOWNLOAD_ERROR as error:
            return DownloadErrorResponse(
                status=error,
                detail="Во время загрузки файла произошла ошибка :("
            )

        case RecognitionStatus.SIZE_LIMIT as error:
            max_size_mebibytes = MAX_SIZE_BYTES / 1024 / 1024
            return DownloadErrorResponse(
                status=error,
                detail=f"Файл слишком большой, лимит - {(max_size_mebibytes):.2f}"
            )

        case RecognitionStatus.OK as success:
            bboxed_image_path = model.find_intervals(
                image_path, threshold=0.5, smoothing_interval=3, gap=10
            )
            return ImageSuccessResponse(
                status=success,
                path=bboxed_image_path
            )
