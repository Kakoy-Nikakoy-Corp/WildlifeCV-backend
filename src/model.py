from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Callable
import os
import sys

import torch
from torchcodec.decoders import VideoDecoder
from loguru import logger
from timecode import Timecode
from ultralytics import YOLO

from src.paths import get_yolo_weights_path


type FrameGeneratorFactory = Callable[[], Iterator[Frame]]
logger.remove()


@dataclass(slots=True, frozen=True)
class Frame:
    number: int
    timecode: Timecode
    confs: list[float]
    bbox_coords: list[list[float] | None]


@dataclass(slots=True)
class TimeInterval:
    """Один длинный таймкод."""
    start: Timecode
    end: Timecode | None

    def __str__(self) -> str:
        return f"{self.start}-{self.end}"


class Model:
    def __init__(self, weights_path: Path = get_yolo_weights_path()) -> None:
        self.verbose = os.getenv('IRBIS_PROD') is None
        self.model = YOLO(weights_path, verbose=self.verbose)
        self.device = '0' if torch.cuda.is_available() else 'cpu'

        loglevel = 'INFO' if self.verbose else 'WARNING'
        logger.add('model_inf.log', level=loglevel)
        logger.add(sys.stderr, level=loglevel)

    def predict_frame(self, frame_tensor: torch.Tensor) -> tuple:
        frame_np = frame_tensor.permute(1, 2, 0).contiguous().cpu().numpy()
        boxes = self.model(frame_np, verbose=self.verbose)[0].boxes

        confs: list[float] = boxes.conf.tolist()
        if not confs:
            confs.append(0.)
        bbox_coords: list = boxes.xyxy.tolist()

        logger.info(f"Confs: {boxes.conf.round(decimals=3)}")

        return confs, bbox_coords

    def predict_video(self, video_path: Path, gap: int = 2) -> tuple[float, int, FrameGeneratorFactory]:
        video = VideoDecoder(video_path, seek_mode="exact")

        fps = video.metadata.average_fps
        total_frames = video.metadata.num_frames
        logger.info(f"FPS: {fps}")

        def frame_gen() -> Iterator[Frame]:
            for frame_number in range(1, total_frames + 1, gap):
                frame_tensor = video[frame_number - 1]

                confs, bbox_coords = self.predict_frame(frame_tensor)
                timecode = Timecode(fps, frames=frame_number)

                logger.info(f"Frame {frame_number}/{total_frames}")

                yield Frame(frame_number, timecode, confs, bbox_coords)

        return total_frames, fps, frame_gen

    def recognise(self, video_path: Path, window_coef: float = 1.5, threshold: float = 0.4, smoothing_interval: float = 2, gap: int = 2) -> list[str]:
        """
        Передает видео в модель и rolling window.

        Parameters:
            video_path: Путь до файла с видео
            window_coef: Коэффициент длины окна поиска.

            Длина окна = FPS * window_coef

            smoothing_interval: Максимальный промежуток между интервалами для их слияния (в секундах)
            threshold: Порог обнаружения барса в rolling window.
            gap: Промежуток между "значимыми" кадрами, на которых делает предсказания модель.

            Увеличение значения этого параметра даёт кратный прирост к производительности ценой точности временных интервалов.

        Returns:
            Список TimeInterval
        """
        total_frames, fps, frame_gen = self.predict_video(video_path, gap)
        window_size = int(fps * window_coef / gap)
        smoothing_tc = Timecode(fps, start_seconds=smoothing_interval)

        # Алгоритм Rolling Window для сглаживания и уточнения предсказаний модели на временных интервалах
        avg_window_conf = 0.
        is_recording = False
        last_interval: TimeInterval | None = None

        window: deque[Frame] = deque()
        intervals: list[TimeInterval] = []

        for frame in frame_gen():
            window.append(frame)
            avg_window_conf += max(frame.confs) / window_size

            # deque len -> O(1)
            if len(window) < window_size:
                continue

            left_frame: Frame = window.popleft()

            # Если превысили порог обнаружения и запись интервала не идёт, начинаем его отслеживать
            if avg_window_conf >= threshold and not is_recording:
                is_recording = True
                start = left_frame.timecode

                # Если с конца предыдущей записи не прошло smoothing_interval секунд
                # (и такая запись вообще имеется), то вместо добавления нового
                # интервала будем "двигать" конец предыдущего, чтобы слить их в один
                if last_interval is None or start >= last_interval.end + smoothing_tc:
                    last_interval = TimeInterval(start=start, end=None)
                    intervals.append(last_interval)

            # Если ведём запись и упали ниже порога ЛИБО дошли до конца видео, заканчиваем интервал
            elif is_recording and (avg_window_conf < threshold or frame.number + gap >= total_frames):
                is_recording = False
                last_interval.end = frame.timecode

            avg_window_conf -= max(left_frame.confs) / window_size

        return [str(interval) for interval in intervals]
