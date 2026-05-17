FROM ultralytics/ultralytics:latest

WORKDIR /app

COPY ./pyproject.toml ./

RUN pip install .

COPY . .

EXPOSE 3001
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "3001"]
