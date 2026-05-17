from pathlib import Path
from typing import Final
import uuid

import cv2
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.model import Model
from src.paths import get_videos_dpath
from src.types import RecognitionResponse, RecognitionStatus


app = FastAPI()
model = Model()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"]
)

MAX_SIZE_BYTES: Final = 500*1024*1024
CHUNK_SIZE: Final = 8*1024*1024  # 8 мебибайт


@app.post("/recognise")
async def recognise(file: UploadFile) -> RecognitionResponse:
    """
    Запускает пайплайн на файле.

    Parameters:
        video (UploadFile): Видеофайл

    Returns:
        Словарь со статусом и таймкодами.
    """
    # Проверка есть на фронтенде, бэкенд ее дублирует
    # Она нужна для роутинга внутри эндпоинта
    ALLOWED_VIDEO_MIMES = {'video/mp4', 'video/x-matroska', 'video/matroska'}
    ALLOWED_IMAGE_MIMES = {'image/jpeg', 'image/png'}
    ALLOWED_ARCHIVE_MIMES = {'application/zip', 'application/x-7z-compressed'}
    if file.content_type in ALLOWED_VIDEO_MIMES:
        route = 'video'
    elif file.content_type in ALLOWED_IMAGE_MIMES:
        route = 'image'
    elif file.content_type in ALLOWED_ARCHIVE_MIMES:
        route = 'multi-image'
    else:
        raise HTTPException(status_code=400, detail='Поддерживаются только видео, фото и архивы с фото!')

    match route:
        case 'video':
            # Собираем файлу уникальное имя
            ext = Path(file.filename).suffix if file.filename else ".mp4"
            name = f'{uuid.uuid4()}{ext}'
            video_path = get_videos_dpath() / name

            # Загружаем видеофайл чанками
            try:
                total_size = 0
                with open(video_path, "wb") as f:
                    while chunk := await file.read(CHUNK_SIZE):
                        total_size += len(chunk)
                        # Проверка есть на фронтенде, бэкенд ее дублирует
                        if total_size > MAX_SIZE_BYTES:
                            raise HTTPException(413, detail='Файл слишком большой, лимит - 500 MB')
                        f.write(chunk)
            except Exception:
                if video_path.exists():
                    video_path.unlink()
                raise HTTPException(status_code=500, detail="Ошибка при сохранении файла")

            # Предикты + rolling window
            timestrings = model.recognise(video_path, threshold=0.6, smoothing_interval=1)
            return {'status': RecognitionStatus.SUCCESS, 'timestrings': timestrings}

        case 'image':
            # Собираем файлу уникальное имя
            ext = Path(file.filename).suffix if file.filename else ".jpg"
            name = f'{uuid.uuid4()}{ext}'
            image_path = get_videos_dpath() / name

            # Скачиваем фото по чанкам
            try:
                total_size = 0
                with open(image_path, "wb") as f:
                    while chunk := await file.read(CHUNK_SIZE):
                        total_size += len(chunk)
                        # Проверка есть на фронтенде, бэкенд ее дублирует
                        if total_size > MAX_SIZE_BYTES:
                            raise HTTPException(413, detail='Файл слишком большой, лимит - 500 MB')
                        f.write(chunk)
            except Exception:
                if image_path.exists():
                    image_path.unlink()
                raise HTTPException(status_code=500, detail="Ошибка при сохранении файла")

            image = cv2.imread(str(image_path))
            if image is not None:
                recognition = model.recognise(image)
                # Дальше нужно вернуть что-то фронтенду.
                # Формата ответа пока нет.
