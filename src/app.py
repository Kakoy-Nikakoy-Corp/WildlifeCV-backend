from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Final

import patoolib
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from loguru import logger

from src.model import Model
from src.paths import (
    get_output_archives_dpath,
    get_output_dpath,
    get_output_images_dpath,
    get_output_videos_dpath,
)
from src.templates import TEMPLATE_RESPONSES, TemplateException
from src.types import (
    ImageSuccessResponse,
    LoadingErrorResponse,
    MultiImageSuccessResponse,
    RecognitionStatus,
    VideoSuccessResponse,
)
from src.utils import (
    download_file,
    get_file_extension,
    get_uuid4,
    register_link_on_file,
)

app = FastAPI()
logger.add('backend_inf.log', level='INFO')
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

ALLOWED_VIDEO_MIMES = {'video/mp4', 'video/x-matroska', 'video/matroska'}
ALLOWED_IMAGE_MIMES = {'image/jpeg', 'image/png'}
ALLOWED_ARCHIVE_MIMES = {
    'application/zip',
    'application/x-zip-compressed',
    'application/x-7z-compressed',
    'application/x-rar-compressed',
    'applicaton/x-compressed'
    'application/x-rar',
    'application/rar',
    'application/vnd.rar',
}

# `torchvision.io.decode_image` documentation page:
# > Currently supported image formats are
# > jpeg, png, gif and webp
SUPPORTED_IMAGE_TYPES: Final = {'jpg', 'jpeg', 'png'}
ARCHIVE_COLLAGE_SIZE: Final = 4


@app.get("/output/{folder}/{filename}")
async def download_video(folder: str, filename: str) -> FileResponse:
    file_path = get_output_dpath() / folder / filename
    if not file_path.exists():
        raise TemplateException.NOT_FOUND.value

    match folder:
        case 'videos':
            media_type = "video/mp4"
        case 'images':
            media_type = "image/jpeg"
        case 'archives':
            media_type = "application/zip"
        # Этот кейс отлавливается проверкой на существование пути,
        # но без него media_type считается 'possibly unbound' :(
        case _:
            raise TemplateException.NOT_FOUND.value

    return FileResponse(
        file_path,
        media_type=media_type,
        filename=filename
    )


@app.post("/recognise/video/")
async def recognise_video(file: UploadFile) -> VideoSuccessResponse | LoadingErrorResponse:
    """
    Запускает пайплайн на видеофайле.

    Parameters:
        video (UploadFile): Видеофайл
    """

    if file.filename is None:
        raise TemplateException.NO_FILENAME.value

    if file.content_type not in ALLOWED_VIDEO_MIMES:
        raise TemplateException.UNSUPPORTED_MIME_TYPE.value

    # Получаем расширение файла
    ext = get_file_extension(file.filename)
    with NamedTemporaryFile(suffix=ext, delete_on_close=False) as video_file:
        download_status = await download_file(video_file, file)
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

                bboxed_video_link = register_link_on_file(output_path)
                return VideoSuccessResponse(
                    status=RecognitionStatus.IRBIS_FOUND,
                    timestrings=timestrings,
                    link=bboxed_video_link
                )


@app.post("/recognise/image/")
async def recognise_image(file: UploadFile) -> ImageSuccessResponse | LoadingErrorResponse:
    """
    Запускает пайплайн на изображении.

    Parameters:
        image (UploadFile): Изображение
    """

    if file.filename is None:
        raise TemplateException.NO_FILENAME.value

    if file.content_type not in ALLOWED_IMAGE_MIMES:
        raise TemplateException.UNSUPPORTED_MIME_TYPE.value

    ext = get_file_extension(file.filename)
    with NamedTemporaryFile('wb', suffix=ext, delete_on_close=False) as image_file:
        download_status = await download_file(image_file, file)
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

                bboxed_image_link = register_link_on_file(output_path)
                return ImageSuccessResponse(
                    status=RecognitionStatus.IRBIS_FOUND,
                    link=bboxed_image_link
                )


@app.post("/recognise/multi-image/")
async def recognise_archive(file: UploadFile) -> MultiImageSuccessResponse | LoadingErrorResponse:
    """
    Запускает пайплайн на архиве изображений.

    Parameters:
        image (UploadFile): Архив
    """

    if file.filename is None:
        raise TemplateException.NO_FILENAME.value

    logger.debug(f"Content type: {file.content_type}")

    if file.content_type not in ALLOWED_ARCHIVE_MIMES:
        raise TemplateException.UNSUPPORTED_MIME_TYPE.value

    # Получаем расширение архива
    ext = get_file_extension(file.filename)
    with NamedTemporaryFile(suffix=ext, delete_on_close=False) as archive_file:
        download_status = await download_file(archive_file, file)
        archive_file.close()

        match download_status:
            case RecognitionStatus.DOWNLOAD_ERROR as error:
                return TEMPLATE_RESPONSES[error]

            case RecognitionStatus.SIZE_LIMIT as error:
                return TEMPLATE_RESPONSES[error]

            case _:
                # Извлекаем архив в temp-директорию
                with TemporaryDirectory() as extracted_images_dir:
                    patoolib.extract_archive(archive_file.name, outdir=extracted_images_dir)

                    is_any_irbis = False
                    bboxed_image_paths: list[Path] = []
                    collage_images: list[str] = []
                    # Итерируемся по изображениям в извелеченном архиве
                    for extracted_file in Path(extracted_images_dir).glob('**/*'):
                        logger.debug(f"Extracted file: {extracted_file}")
                        file_ext = extracted_file.suffix.lower()
                        if file_ext not in SUPPORTED_IMAGE_TYPES:
                            continue

                        output_path = get_output_archives_dpath() / f'{get_uuid4()}{file_ext}'
                        is_any_irbis = model.detect_image(extracted_file, output_path) or is_any_irbis
                        # Отдаем пользователю исходный набор фото,
                        # сохраняем в результат **все** фотографии
                        bboxed_image_paths.append(output_path)
                        # Заполняем список ссылок для превью архива в фронтенде
                        if len(collage_images) < ARCHIVE_COLLAGE_SIZE:
                            bboxed_image_link = register_link_on_file(output_path)
                            collage_images.append(bboxed_image_link)

                    logger.debug(f"""
                        is_any_irbis: {is_any_irbis}
                        collage_images: {collage_images}
                    """)
                    if not is_any_irbis or not collage_images:
                        return TEMPLATE_RESPONSES[RecognitionStatus.NO_IRBIS_FOUND]

                    # Boostrap'аем коллаж, если он не смог набраться
                    # из обработанных изображений
                    while len(collage_images) < ARCHIVE_COLLAGE_SIZE:
                        collage_images.append(collage_images[-1])

                    # Собираем свой архив с обработанными изображениями
                    output_archive_path = get_output_archives_dpath() / f'{get_uuid4()}.zip'
                    patoolib.create_archive(str(output_archive_path), bboxed_image_paths)
                    # Удаляем обработанные изображения, исключая фотки для коллажа
                    for path in bboxed_image_paths:
                        if path not in collage_images:
                            path.unlink()

                    output_archive_link = register_link_on_file(output_archive_path)
                    return MultiImageSuccessResponse(
                        status=RecognitionStatus.IRBIS_FOUND,
                        link=output_archive_link,
                        collage_images=collage_images
                    )
