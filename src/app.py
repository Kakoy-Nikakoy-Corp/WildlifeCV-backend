from pathlib import Path
from typing import Final
from tempfile import NamedTemporaryFile
import uuid

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import puremagic

from src.model import Model
from src.paths import get_videos_dpath
from src.types import DownloadStatus, VideoRecognitionResponse, VideoResponse, RecognitionStatus
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

MAX_SIZE_BYTES: Final = 500*1024*1024
CHUNK_SIZE: Final = 8*1024*1024  # 8 мебибайт
ALLOWED_VIDEO_MIMES = {'video/mp4', 'video/x-matroska', 'video/matroska'}
ALLOWED_IMAGE_MIMES = {'image/jpeg', 'image/png'}
ALLOWED_ARCHIVE_MIMES = {'application/zip', 'application/x-7z-compressed'}

# Есть middlware, который шарит директорию как роуты в бэке
@app.get("/video/")


@app.post("/recognise/video/")
async def recognise_video(video: UploadFile) -> VideoResponse:
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

    video_path = get_videos_dpath() / f"{uuid.uuid4()}{ext}"

    # Загружаем видеофайл
    download_status = await download_file(video, video_path)
    match download_status:
        case DownloadStatus.ERROR:
            return VideoRecognitionResponse(status='')


        # Предикты + rolling window
        video_path = Path(video_file.name)
        timestrings = model.detect_video_timeintervals(video_path)
    return VideoResponse(
        status=RecognitionStatus.SUCCESS,
        timestrings=timestrings,
        path=Path()  # wait for model class update
    )


@app.post("/recognise/image/")
async def recognise_image(image: UploadFile):
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
    file_name = Path(image.filename)
    if file_name.suffix is None:
        ext = puremagic.from_file(file_name)
    else:
        ext = file_name.suffix

    # Загружаем видеофайл чанками
    with NamedTemporaryFile('wb', suffix=ext) as image_file:

        # Предикты + rolling window
        video_path = Path(image_file.name)
        timestrings = model.find_intervals(video_path, threshold=0.5, smoothing_interval=3, gap=10)
    return {
        'status': RecognitionStatus.SUCCESS,
        'timestrings': timestrings,
        'path': Path()  # wait for model class update
    }
