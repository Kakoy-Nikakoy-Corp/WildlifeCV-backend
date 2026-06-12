FROM nvcr.io/nvidia/pytorch:26.04-py3

WORKDIR /app

RUN apt-get update && \
    apt-get install -y ffmpeg p7zip rar unrar && \
    rm -rf /var/lib/apt/lists/*

RUN pip install fastapi[standard]>=0.136.1 loguru>=0.7.3 numpy>=2.4.4 pydantic>=2.13.4 timecode>=1.5.1 uvicorn>=0.46.0 pillow>=12.2.0 puremagic>=2.2.0 patool>=4.0.5
RUN pip install ultralytics>=8.4.51
RUN pip install torchcodec>=0.14

COPY src/ src/
COPY weights/ weights/
COPY fonts/ fonts/

EXPOSE 3001
ENV IRBIS_DEBUG=0
ENV USE_PROFILER=0
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "3001"]
