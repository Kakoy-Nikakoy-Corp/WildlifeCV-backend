from enum import Enum

from fastapi import HTTPException

from src.types import LoadingErrorResponse
from src.utils import MAX_SIZE_MIB, RecognitionStatus


class TemplateException(Enum):
    """
    Шаблонные исключения эндпоинта.

    UNSUPPORTED_MIME_TYPE: Эндпоинт получил файл, тип которого отсутствует в списке поддерживаемых типов
    NO_FILENAME: Эндпоинт получил файл без имени
    """
    UNSUPPORTED_MIME_TYPE = HTTPException(
        400,
        """
        Поддерживаются только следующие виды файлов:
        - Видео: MP4, MKV, MOV
        - Изображения: JPG, PNG
        - Архивы: RAR, 7z, ZIP
          (в архивах могут быть только изображения форматов выше)
        """
    )
    NO_FILENAME = HTTPException(
        422,
        'Сервис не обрабатывает файл без имени :('
    )


TEMPLATE_RESPONSES = {
    RecognitionStatus.DOWNLOAD_ERROR: LoadingErrorResponse(
        status=RecognitionStatus.DOWNLOAD_ERROR,
        detail="Во время загрузки файла произошла ошибка :("
    ),
    RecognitionStatus.SIZE_LIMIT: LoadingErrorResponse(
        status=RecognitionStatus.SIZE_LIMIT,
        detail=f"Файл слишком большой, лимит - {MAX_SIZE_MIB:.2f} МиБ"
    ),
    RecognitionStatus.NO_IRBIS_FOUND: LoadingErrorResponse(
        status=RecognitionStatus.NO_IRBIS_FOUND,
        detail="Снежный барс не обнаружен :("
    )
}
