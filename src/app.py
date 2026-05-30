from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Final
from urllib.parse import urljoin

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from jinja2.ext import i18n
import patoolib
import puremagic

from src.model import Model
from src.paths import (
    get_output_dpath, get_output_videos_dpath, get_output_images_dpath,
    get_output_archives_dpath, get_project_root
)
from src.templates import TemplateException, TEMPLATE_RESPONSES
from src.types import (
    RecognitionStatus, LoadingErrorResponse,
    VideoSuccessResponse, VideoRecognitionOutput,
    ImageSuccessResponse, MultiImageSuccessResponse
)
from src.utils import download_file, get_uuid4, MAX_SIZE_MIB

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

ROOT_LINK: Final = 'https://api.irbis.wild1.net'
ALLOWED_VIDEO_MIMES = {'video/mp4', 'video/x-matroska', 'video/matroska'}
ALLOWED_IMAGE_MIMES = {'image/jpeg', 'image/png'}
ALLOWED_ARCHIVE_MIMES = {
    'application/zip',
    'application/x-zip-compressed',
    'application/x-7z-compressed',
    'application/x-rar-compressed',
    'application/x-rar',
    'application/rar',
    'application/vnd.rar',
}
SUPPORTED_IMAGE_TYPES = {'jpg', 'jpeg', 'png'}


@app.post("/recognise/video/")
async def recognise_video(video: UploadFile) -> VideoSuccessResponse | LoadingErrorResponse:
    """
    Запускает пайплайн на видеофайле.

    Parameters:
        video (UploadFile): Видеофайл
    """

    if video.filename is None:
        raise TemplateException.NO_FILENAME.value

    if video.content_type not in ALLOWED_VIDEO_MIMES:
        raise TemplateException.UNSUPPORTED_MIME_TYPE.value

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
                return TEMPLATE_RESPONSES[error]

            case RecognitionStatus.SIZE_LIMIT as error:
                return TEMPLATE_RESPONSES[error]

            case _:  # case IRBIS_FOUND
                # Собираем путь для сохранения видео от модели и вызываем модель
                output_name = f'{get_uuid4()}.mp4'
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
                    return TEMPLATE_RESPONSES[RecognitionStatus.NO_IRBIS_FOUND]

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
        raise TemplateException.NO_FILENAME.value

    if image.content_type not in ALLOWED_IMAGE_MIMES:
        raise TemplateException.UNSUPPORTED_MIME_TYPE.value

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
                return TEMPLATE_RESPONSES[error]

            case RecognitionStatus.SIZE_LIMIT as error:
                return TEMPLATE_RESPONSES[error]

            case _:
                # Собираем путь для сохранения видео от модели и вызываем модель
                image_path = Path(image_file.name)
                output_name = f'{get_uuid4()}.jpg'
                output_path = get_output_images_dpath() / output_name
                is_found = model.detect_image(image_path, output_path)
                if not is_found:
                    return TEMPLATE_RESPONSES[RecognitionStatus.NO_IRBIS_FOUND]

                relative_output_path = str(output_path.relative_to(get_project_root()))
                bboxed_image_link = urljoin(ROOT_LINK, relative_output_path)
                return ImageSuccessResponse(
                    status=RecognitionStatus.IRBIS_FOUND,
                    link=bboxed_image_link
                )


@app.post("/recognise/multi-image")
async def recognise_archive(archive: UploadFile) -> MultiImageSuccessResponse | LoadingErrorResponse:
    """
    Запускает пайплайн на архиве изображений.

    Parameters:
        image (UploadFile): Архив
    """

    if archive.filename is None:
        raise TemplateException.NO_FILENAME.value

    if archive.content_type not in ALLOWED_ARCHIVE_MIMES:
        raise TemplateException.UNSUPPORTED_MIME_TYPE.value

    # Получаем расширение из файла
    archive_name = Path(archive.filename)
    if not archive_name.suffix:
        ext = puremagic.from_file(archive_name)
    else:
        ext = archive_name.suffix

    with NamedTemporaryFile(suffix=ext) as archive_file:
        download_status = await download_file(archive_file, archive)
        match download_status:
            case RecognitionStatus.DOWNLOAD_ERROR as error:
                return TEMPLATE_RESPONSES[error]

            case RecognitionStatus.SIZE_LIMIT as error:
                return TEMPLATE_RESPONSES[error]

            case _:
                # Извлекаем архив в temp-директорию
                with TemporaryDirectory() as extracted_images_dir:
                    patoolib.extract_archive(archive_file.name, outdir=extracted_images_dir)

                    # Записываем в видеофайл все изображения из извлеченного архива
                    images = []
                    for file in Path(extracted_images_dir).iterdir():
                        if file.is_file() and file.suffix in SUPPORTED_IMAGE_TYPES:
                            images.append(...)
