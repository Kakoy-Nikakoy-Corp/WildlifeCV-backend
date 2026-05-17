FROM ultralytics/ultralytics:latest

WORKDIR /app

COPY ./pyproject.toml ./

ENV CMAKE_ARGS="-D WITH_FFMPEG=ON"

# libxcb1 - for opecv-python in slim python image
RUN pip install . && \
    pip install --reinstall --no-binary opencv-python --no-deps opencv-python

COPY . .

EXPOSE 3001
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "3001"]
