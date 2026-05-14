from pathlib import Path
from typing import Final
import uuid

import cv2
from cv2 import VideoCapture
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.paths import get_videos_dpath
from src.types import Recognition, RecognitionResponse, RecognitionStatus
from src.utils import ModelInterface, ModelDouble, rolling_window_double


app = FastAPI()
# В аннотации временно протокол, а не обычный класс, т.к.
# модель - сфера ответственности Максима,
# я в нее лезу по минимуму.
# Бэкенду нужна единственная гарантия -
# класс модели соответствует протоколу ModelInterface.
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
    # Она понадобится для роутинга внутри эндпоинта
    ALLOWED_MIMES = {'video/mp4', 'video/mkv'}
    if file.content_type not in ALLOWED_MIMES:
        raise HTTPException(status_code=400, detail='Поддерживаются только видео!')

    # Собираем файлу уникальное имя
    ext = Path(file.filename).suffix if file.filename else ".mp4"
    name = f'{uuid.uuid4()}{ext}'
    video_path = get_videos_dpath() / name

    # Загружаем видеофайл чанками
    try:
        MAX_SIZE_BYTES: Final = 500*1024*1024
        CHUNK_SIZE: Final = 8*1024*1024  # 8 мебибайт
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

    # Открываем видео
    video = VideoCapture(str(video_path))
    if not video.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео '{video_path}'")

    # Читаем по кадру и передаем кадр в модель
    frames: list[Recognition] = []
    while True:
        success, frame = video.read()
        if not success:
            break
        # Обращаемся к модели
        result = model.recognise(frame)
        frames.append(result)

    video.release()

    # Передаем кадры в функцию rolling window
    timestamps = rolling_window_double(frames)
    return {'status': RecognitionStatus.SUCCESS, 'timestamps': timestamps}
