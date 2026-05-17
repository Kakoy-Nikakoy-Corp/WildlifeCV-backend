FROM ultralytics/ultralytics:latest

WORKDIR /app

COPY ./pyproject.toml ./

# libxcb1 - for opecv-python in slim python image
RUN apt install libxcb1 && \
    pip install .

COPY . .

EXPOSE 3001
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "3001"]
