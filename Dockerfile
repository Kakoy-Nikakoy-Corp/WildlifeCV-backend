FROM ultralytics/ultralytics:latest

WORKDIR /app

RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY ./pyproject.toml ./

RUN pip install . --group cuda

COPY . .

EXPOSE 3001
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "3001"]
