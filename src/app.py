from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Final
from urllib.parse import urljoin

import patoolib
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.model import Model
from src.paths import (
    get_output_archives_dpath,
    get_output_dpath,
    get_output_images_dpath,
    get_output_videos_dpath,
    get_project_root,
)
from src.templates import TEMPLATE_RESPONSES, TemplateException
from src.types import (
    ImageSuccessResponse,
    LoadingErrorResponse,
    MultiImageSuccessResponse,
    RecognitionStatus,
    VideoSuccessResponse,
)
from src.utils import download_file, get_file_extension, get_uuid4

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

# `torchvision.io.decode_image` documentation page:
# > Currently supported image formats are
# > jpeg, png, gif and webp
SUPPORTED_IMAGE_TYPES: Final = {'jpg', 'jpeg', 'png'}
ARCHIVE_COLLAGE_SIZE: Final = 4


@app.get("/output/{filename}")
async def download_video(filename: str) -> FileResponse:
    file_path = get_output_dpath() / filename

    if 'videos' in filename:
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="video/mp4"
        )


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
    ext = get_file_extension(video.filename)
    with NamedTemporaryFile(suffix=ext, delete_on_close=False) as video_file:
        download_status = await download_file(video_file, video)
        video_file.close()

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
                    timestrings=timestrings,
                    link=bboxed_video_link
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

    # Получаем расширение изображения
    ext = get_file_extension(image.filename)
    # Загружаем файл
    with NamedTemporaryFile('wb', suffix=ext, delete_on_close=False) as image_file:
        download_status = await download_file(image_file, image)
        image_file.close()

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

    # Получаем расширение архива
    ext = get_file_extension(archive.filename)
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

                    bboxed_image_paths: list[Path] = []
                    is_any_irbis = False
                    collage_images: list[str] = []
                    # Итерируемся по изображениям в извелеченном архиве
                    for file in Path(extracted_images_dir).iterdir():
                        file_ext = file.suffix.lower()
                        # Проверяем расширение изображения
                        if file.is_file() and file_ext in SUPPORTED_IMAGE_TYPES:
                            output_path = get_output_archives_dpath() / f'{get_uuid4()}{file_ext}'
                            is_any_irbis = model.detect_image(file, output_path) or is_any_irbis
                            # Отдавать пользователю исходный набор фото
                            # или только фото с обнаруженным барсом?
                            bboxed_image_paths.append(output_path)
                            # Заполняем список ссылок для превью архива в фронтенде
                            if len(collage_images) < ARCHIVE_COLLAGE_SIZE:
                                relative_image_path = str(output_path.relative_to(get_project_root()))
                                bboxed_image_link = urljoin(ROOT_LINK, relative_image_path)
                                collage_images.append(bboxed_image_link)

                    # FIXME!!!! Comment me!!!
                    if not is_any_irbis or not collage_images:
                        return TEMPLATE_RESPONSES[RecognitionStatus.NO_IRBIS_FOUND]

                    while len(collage_images) < ARCHIVE_COLLAGE_SIZE:
                        collage_images.append(collage_images[-1])


                    output_archive_path = get_output_archives_dpath() / f'{get_uuid4()}.zip'
                    patoolib.create_archive(str(output_archive_path), bboxed_image_paths)
                    relative_archive_path = str(output_archive_path.relative_to(get_project_root()))
                    output_archive_link = urljoin(ROOT_LINK, relative_archive_path)

                    return MultiImageSuccessResponse(
                        status=RecognitionStatus.IRBIS_FOUND,
                        link=output_archive_link,
                        collage_images=collage_images
                    )
