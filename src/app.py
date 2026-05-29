from pathlib import Path
from typing import Final
from uuid import uuid4
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
import puremagic

from src.model import Model
from src.paths import get_output_dpath
from src.types import RecognitionStatus, DownloadErrorResponse
from src.types import VideoSuccessResponse, VideoRecognitionOutput


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
ALLOWED_VIDEO_MIMES = {'video/mp4', 'video/x-matroska', 'video/matroska'}
ALLOWED_IMAGE_MIMES = {'image/jpeg', 'image/png'}
ALLOWED_ARCHIVE_MIMES = {'application/zip', 'application/x-7z-compressed'}


@app.post("/recognise/video/")
async def recognise_video(video: UploadFile) -> VideoSuccessResponse | DownloadErrorResponse:
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

    total_size = 0
    try:
        with NamedTemporaryFile(suffix=ext) as media_file:
            while chunk := await video.read(CHUNK_SIZE):
                total_size += len(chunk)

                # Проверка есть на фронтенде, бэкенд ее дублирует
                if total_size > MAX_SIZE_MIB:
                    return DownloadErrorResponse(
                        status=RecognitionStatus.SIZE_LIMIT,
                        detail=f"Файл слишком большой, лимит - {MAX_SIZE_MIB} MiB"
                    )

                media_file.write(chunk)

            # Предикты + rolling window
            video_path = Path(media_file.name)
            output_path = f'output/{uuid4()}.mp4'
            timestrings = model.detect_video_intervals(video_path, Path(output_path), threshold=0.5, smoothing_interval=3, gap=10, batch_size=32)

            return VideoSuccessResponse(
                status=RecognitionStatus.OK,
                data=VideoRecognitionOutput(
                    timestrings=timestrings,
                    link='https://api.irbis.wild1.net/' + output_path
                )
            )

    except Exception as e:
        logger.opt(exception=True).error(e)

        # Проверять существование частично загруженного файла не нужно,
        # контекстный менеджер корректно закрывается
        return DownloadErrorResponse(
            status=RecognitionStatus.DOWNLOAD_ERROR,
            detail="Во время загрузки произошла ошибка :("
        )


# @app.post("/recognise/image/")
# async def recognise_image(image: UploadFile) -> ImageSuccessResponse | DownloadErrorResponse:
#     """
#     Запускает пайплайн на изображении.

#     Parameters:
#         image (UploadFile): Изображение

#     Returns:
#         Словарь со статусом и таймкодами.
#     """
#     if image.filename is None:
#         raise HTTPException(422, detail='Сервис не обрабатывает файл без имени :(')

#     if image.content_type not in ALLOWED_IMAGE_MIMES:
#         raise HTTPException(status_code=400, detail='Поддерживаются только видео, фото и архивы с фото!')

#     # Получаем суффикс из файла
#     image_name = Path(image.filename)
#     if not image_name.suffix:
#         ext = puremagic.from_file(image_name)
#     else:
#         ext = image_name.suffix

#     # Загружаем файл
#     download_status = await download_file(image, ext)
#     match download_status:
#         case RecognitionStatus.DOWNLOAD_ERROR as error:
#             return DownloadErrorResponse(
#                 status=error,
#                 detail="Во время загрузки файла произошла ошибка :("
#             )

#         case RecognitionStatus.SIZE_LIMIT as error:
#             max_size_mebibytes = MAX_SIZE_BYTES / 1024 / 1024
#             return DownloadErrorResponse(
#                 status=error,
#                 detail=f"Файл слишком большой, лимит - {(max_size_mebibytes):.2f}"
#             )

#         case RecognitionStatus.OK as success:
#             bboxed_image_path = model.find_intervals(
#                 image_path, threshold=0.5, smoothing_interval=3, gap=10
#             )
#             return ImageSuccessResponse(
#                 status=success,
#                 path=bboxed_image_path
#             )
