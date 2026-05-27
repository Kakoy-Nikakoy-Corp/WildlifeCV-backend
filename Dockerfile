FROM nvcr.io/nvidia/pytorch:26.04-py3

WORKDIR /app

RUN apt-get update && \
    apt-get install -y cmake ffmpeg libavdevice-dev libavfilter-dev libavformat-dev \
    libavcodec-dev libavutil-dev libswresample-dev libswscale-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*

RUN pip install fastapi[standard]>=0.136.1 loguru>=0.7.3 numpy>=2.4.4 pydantic>=2.13.4 timecode>=1.5.1 uvicorn>=0.46.0 pillow>=12.2.0 puremagic>=2.2.0 --break-system-packages
RUN pip install ultralytics>=8.4.51 --break-system-packages

COPY torchcodec/ torchcodec/
RUN pip install --force-reinstall --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu132

ENV ENABLE_CUDA=1
RUN pip install -e torchcodec --no-build-isolation

COPY src/ src/
COPY weights/ weights/
COPY fonts/ fonts/

EXPOSE 3001
ENV IRBIS_DEBUG=0
ENV USE_PROFILER=0
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "3001"]
