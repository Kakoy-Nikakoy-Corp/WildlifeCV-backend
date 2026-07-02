# WildlifeCV Backend
Бэкенд проекта (REST API сервер + алгоритмы инференса CV-модели)

> Для работы бэкенда требуется **nightly-версия PyTorch**.  
> При использовании стабильной версии библиотеки возникает [ошибка в деструкторе декодировщика видео в torchcodec](https://github.com/meta-pytorch/torchcodec/issues/1438)  
> Issue закрыт ошибочно, ошибку решает не фикс мейнтейнера, а использование **pytorch-nightly**

## Структура репозитория
- `fonts` - шрифты, использующиеся в проекте
- `src` - исходники бэкенда:
  - `аpp.py` - обработчики маршрутов
  - `model.py` - функционал модели компьютерного зрения
  - `overlay.py` - рендер итогового видео
  - `paths.py` - пути к файлам в репозитории
  - `templates.py` - шаблонные ответы бэкенда
  - `types.py` - типы, использующиеся в проекте
  - `utils.py` - вспомогательные функции
- `tests`:
  - `helpers.py` - вспомогательные функции для тестов
  - `test_rolling_window` - песочница для механизма Rolling Window.  
    Требовалась при создании алгоритма RADIC для проверки гипотез и итоговой реализации на базовых кейсах.  
    **Legacy, не используется как юнит-тест**
  - `test_static_distribution` - тесты эндпоинта раздачи статики (результатов работы модели)

# Использование
## Клонирование и установка зависимостей
```shell
git clone https://github.com/Kakoy-Nikakoy-Corp/WildlifeCV-backend.git
cd WildlifeCV-backend
uv sync 
```

## Запуск
```shell 
uv run uvicorn src.app:app --host 0.0.0.0 --port 3001
```

## Запуск через Docker Compose
Смотрите `docker-compose.yaml`
