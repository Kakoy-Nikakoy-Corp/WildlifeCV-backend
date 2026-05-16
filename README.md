# WildlifeCV Backend
## Клонирование и установка зависимостей
```shell
git clone https://github.com/Kakoy-Nikakoy-Corp/WildlifeCV-backend.git
cd WildlifeCV-backend
uv sync 
```

## Запуск
```shell 
uv run uvicorn src.main:app
```

## Запуск через Docker Compose
Добавьте сервис в `docker-compose.yaml`:
```yaml
services:
  api:
    build: /<path_to_package>/WildlifeCV-backend
    container_name: wcv-backend
    ports:
      - "3001:3001"
    restart: unless-stopped
```
Запускайте контейнеры:
`docker compose up -d`
