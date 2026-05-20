FROM nvcr.io/nvidia/pytorch:26.04-py3

WORKDIR /app

RUN apt-get update && \
    apt-get install -y valgrind ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN pip install torchcodec --index-url=https://download.pytorch.org/whl/cu132 --break-system-packages
RUN pip install ultralytics>=8.4.51 --break-system-packages
RUN pip install fastapi[standard]>=0.136.1 loguru>=0.7.3 numpy>=2.4.4 pydantic>=2.13.4 timecode>=1.5.1 uvicorn>=0.46.0 --break-system-packages

COPY . .

EXPOSE 3001
ENV IRBIS_PROD=1
#ENV PYTHONMALLOC=malloc
#CMD ["valgrind", "--tool=memcheck", "--track-origins=yes", "--trace-children=yes", "--show-mismatched-frees=yes", "python", "test_dec.py"]
#CMD ["valgrind", "--leak-check=full", "--show-leak-kinds=all", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "3001"]
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "3001"]
