from typing import Protocol

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware


class ModelInterface(Protocol):
    """Интерфейс класса модели."""
    def predict(self, image) -> tuple[float, float, float, float]:
        """
        Ищет барса на фотографии.
        Возвращает координаты bbox'а
        """
        ...


app = FastAPI()

origins = [
    "http://localhost/",
    "https://localhost/",
    "https://irbis.wild1.net/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"]
)


def predict_video(file: UploadFile) -> str:
    """
    Ищет барса на видео.
    Нарезает видео на кадры и отправляет их в модель.


    Барс найден в кадре - +таймкод.
    В итоге получаем файл с таймкодами, форматируем его в текст.
    """
    ...


@app.post("/predict")
async def predict(file: UploadFile = File(media_type='video/mp4')):
    """
    Запускает пайплайн на файле.

    Parameters:
        file (UploadFile): Фото или видео
    """
    # Наличие MIME-типа проверяется на фронте
    mime = file.content_type
    if mime and "video" in mime.lower():
        content_type = "video"
    else:
        raise HTTPException(status_code=400)

    if content_type == "video":
        return {'message': 'Video, hooray!'}
        #  return predict_video(file)
