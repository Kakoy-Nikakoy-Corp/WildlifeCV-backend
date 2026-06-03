import shutil
from pathlib import Path
import subprocess
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
    get_project_root,
    get_backend_logs_path,
    get_output_images_with_irbis_dpath,
    get_output_images_with_bg_dpath
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
    ROOT_LINK
)

logger.add(get_backend_logs_path(), level='DEBUG')
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

ALLOWED_VIDEO_MIMES = {'video/mp4', 'video/x-matroska', 'video/matroska'}
ALLOWED_IMAGE_MIMES = {'image/jpeg', 'image/png'}
ALLOWED_ARCHIVE_MIMES = {
    'application/zip',
    'application/x-zip-compressed',
    'application/octet-stream',
    'application/x-7z-compressed',
    'application/x-rar-compressed',
    'application/x-compressed',
    'application/x-rar',
    'application/rar',
    'application/vnd.rar',
}

# `torchvision.io.decode_image` documentation page:
# > Currently supported image formats are
# > jpeg, png, gif and webp
SUPPORTED_IMAGE_TYPES: Final = {'.jpg', '.jpeg', '.png'}
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

                    bboxed_image_paths: list[Path] = []
                    no_bboxed_image_paths: list[Path] = []
                    collage_image_paths: list[Path] = []
                    # Итерируемся по изображениям в извелеченном архиве
                    for extracted_file in Path(extracted_images_dir).glob('**/*'):
                        logger.debug(f"Extracted file: {extracted_file}")
                        logger.debug(f"Relative extracted file: {extracted_file.relative_to(extracted_images_dir)}")
                        logger.debug(f"Relative suffix path: {extracted_file.relative_to(extracted_images_dir).parent}")
                        file_ext = extracted_file.suffix.lower()
                        if file_ext not in SUPPORTED_IMAGE_TYPES:
                            continue

                        # /.../output/images/1234.jpg
                        output_path = get_output_images_dpath() / f'{get_uuid4()}{file_ext}'
                        logger.debug(f"Absolute new-file path: {output_path}")
                        is_irbis = model.detect_image(extracted_file, output_path)
                        # ./output/images/1234.jpg
                        output_relative_path = output_path.relative_to(get_project_root())
                        # Заполняем коллаж фотками с барсом
                        if len(collage_image_paths) < ARCHIVE_COLLAGE_SIZE and is_irbis:
                            collage_image_paths.append(output_relative_path)
                        # Продолжаем сохранять фото с архивом
                        elif is_irbis:  # коллаж заполнен
                            bboxed_image_paths.append(output_relative_path)
                        # Фото без барса тоже сохраняем
                        else:  # not is_irbis
                            no_bboxed_image_paths.append(output_relative_path)

                    logger.debug(f"collage_images:\n{collage_image_paths}")
                    logger.debug(f"bboxed_images:\n{bboxed_image_paths}")
                    logger.debug(f"no_bboxed_images:\n {no_bboxed_image_paths}")
                    if not collage_image_paths:
                        return TEMPLATE_RESPONSES[RecognitionStatus.NO_IRBIS_FOUND]

                    # Boostrap'аем коллаж, если он не смог набраться
                    # из обработанных изображений
                    while len(collage_image_paths) < ARCHIVE_COLLAGE_SIZE:
                        collage_image_paths.append(collage_image_paths[-1])

                    # ./output/images/snow_leopards/1234.jpg
                    archive_bboxed_images: list[Path] = []
                    archive_bg_images: list[Path] = []
                    for path in collage_image_paths + bboxed_image_paths:
                        # cp ./output/images/1234.jpg /.../output/images/snow_leopards
                        shutil.copy(path, get_output_images_with_irbis_dpath())
                        #
                        archive_bboxed_images.append(get_output_images_with_irbis_dpath().relative_to(get_output_images_dpath()) / path.name)
                    for path in no_bboxed_image_paths:
                        shutil.copy(path, get_output_images_with_bg_dpath())
                        archive_bg_images.append(get_output_images_with_bg_dpath().relative_to(get_output_images_dpath()) / path.name)

                    # Собираем свой архив с обработанными изображениями
                    archive_name = f"{get_uuid4()}.zip"
                    output_archive_path = get_output_images_dpath() / archive_name
                    logger.debug(f"Output archive path: {output_archive_path}")
                    output_image_paths = [str(path) for path in archive_bboxed_images + archive_bg_images]
                    logger.debug(f"Files to archive: {output_image_paths}")
                    create_archive = subprocess.Popen(
                        f'patool create {archive_name} {' '.join(output_image_paths)}',
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        cwd=get_output_images_dpath()
                    )
                    create_archive.communicate()

                    # Удаляем обработанные изображения, исключая фотки для коллажа
                    for path in bboxed_image_paths:
                        path.unlink()

                    collage_links: list[str] = [register_link_on_file(path.resolve()) for path in collage_image_paths]

                    output_archive_link = register_link_on_file(output_archive_path)
                    return MultiImageSuccessResponse(
                        status=RecognitionStatus.IRBIS_FOUND,
                        link=output_archive_link,
                        collage_images=collage_links
                    )
