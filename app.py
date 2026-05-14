from enum import Enum
from pathlib import Path
from typing import Final
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from paths import get_videos_dpath
from utils import ModelInterface, ModelDouble


class PredictiontStatus(str, Enum):
    """
    Статус ответа модели.
    Взят из фронтенда.
    """
    SUCCESS = 'success'
    ERROR = 'error'


class Prediction(BaseModel):
    """
    Модель ответа модели.
    Взята из фронтенда.
    """
    timestamps: list[str]  # 'HH:MM:SS - HH:MM:SS', '...'
    status: PredictionStatus



app = FastAPI()
model: ModelInterface = ModelDouble()

origins = [
    "http://localhost:5173/",
    "https://localhost:5173/",
    "https://irbis.wild1.net/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.post("/predict")
async def predict(video: UploadFile) -> Prediction:
    """
    Запускает пайплайн на файле.

    Parameters:
        video (UploadFile): Видеофайл
    """
    ALLOWED_MIMES = {'video/mp4', 'video/mkv'}
    if video.content_type not in ALLOWED_MIMES:
        raise HTTPException(status_code=400, detail='Поддерживаются только видео!')

    # Собираем файлу уникальное имя
    ext = Path(video.filename).suffix if video.filename else ".mp4"
    name = f'{uuid.uuid4()}{ext}'
    video_path = get_videos_dpath() / name

    # Загружаем видеофайл чанками
    try:
        MAX_SIZE_BYTES: Final = 500*1024*1024
        CHUNK_SIZE: Final = 8*1024*1024  # 8 мебибайт
        total_size = 0
        with open(video_path, "wb") as f:
            while chunk := await video.read(CHUNK_SIZE):
                total_size += len(chunk)
                # Проверка есть на фронтенде, бэкенд ее дублирует
                if total_size > MAX_SIZE_BYTES:
                    raise HTTPException(413, detail='Файл слишком большой, лимит - 500 MB')
                f.write(chunk)
    except Exception:
        if video_path.exists():
            video_path.unlink()
        raise HTTPException(status_code=500, detail="Ошибка при сохранении файла")

    # Нарезчик видео тащит кадры из видео, покадрово сует их в модель
    # Генераторы?...
    # Затем **список** из предсказаний передается в функцию-помощник
    # (идея rolling window, Максим еще проектировал)
