from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Callable
import os
import time
import cProfile

import torch
from torchcodec.decoders import VideoDecoder
from loguru import logger
from timecode import Timecode
from ultralytics import YOLO
from ultralytics.engine.results import Results

from src.paths import get_yolo_weights_path
from src.utils import preprocess


type FrameGeneratorFactory = Callable[[], Iterator[Frame]]
type ParsedResults = tuple[list[float], list]

pr = cProfile.Profile()  # Virtually no performance impact until enabled


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
        # Debug-related fields
        self.verbose: bool = os.getenv('IRBIS_PROD') is None
        self.use_profiler: bool = os.getenv('USE_PROFILER') is not None

        # Instantiate a model and determine a primary computational unit we're using (CUDA/CPU)
        self.model = YOLO(weights_path, verbose=self.verbose, task='detect')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        logger.add('model_inf.log', level='INFO')

    def parse_results(self, results: Results) -> tuple[list[float], list]:
        boxes = results.boxes  # Contains data for 'detection' task models

        # Model confidence must not be empty to avoid sorting issues
        confs: list[float] = boxes.conf.tolist()
        if not confs:
            confs.append(0.)

        bbox_coords: list = boxes.xyxy.tolist()

        if self.verbose:
            logger.info(f"Confs: {boxes.conf.round(decimals=3)}")

        return confs, bbox_coords

    def make_prediction(self, frames: torch.Tensor, single: bool = False) -> Iterator[ParsedResults] | ParsedResults:
        # Letterbox and normalize tensors
        preprocessed_frames: torch.Tensor = preprocess(frames)

        # Feed tensors to YOLO to obtain an actual prediction
        results_list: list[Results] = self.model(preprocessed_frames, verbose=self.verbose)

        # Return a single prediction
        if single:
            return self.parse_results(results_list[0])

        # Generate results one by one in case of batch prediction
        def results_iterator() -> Iterator[ParsedResults]:
            for results in results_list:
                yield self.parse_results(results)

        return results_iterator()

    def predict_video(self, video_path: Path, gap: int = 2, batch_size=16) -> tuple[float, int, FrameGeneratorFactory]:
        # This gives substantial performance boosts only on CPU
        threads = 1 if self.device == 'cuda' else 8

        # We are intentionally NOT applying any on-decode transformations to retain original frames for further encoding
        video = VideoDecoder(video_path, seek_mode="approximate", num_ffmpeg_threads=threads, device=self.device)

        fps = video.metadata.average_fps
        total_frames = video.metadata.num_frames
        logger.info(f"FPS: {fps}")

        def frame_gen() -> Iterator[Frame]:
            batch = []
            for i in range(1, total_frames + 1, gap):
                batch.append(i - 1)

                if len(batch) == batch_size:
                    frames = video.get_frames_in_range(min(batch), max(batch) + 1, gap).data

                    for frame_number, (confs, bbox_coords) in zip(batch, self.make_prediction(frames)):
                        timecode = Timecode(fps, frames=frame_number + 1)

                        if self.verbose:
                            logger.info(f"Frame {frame_number + 1}/{total_frames}")

                        yield Frame(frame_number + 1, timecode, confs, bbox_coords)

                    batch.clear()

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
        t = time.time()

        if self.use_profiler:
            pr.enable()

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

        if self.use_profiler:
            pr.disable()
            pr.print_stats(sort='cumtime')

        logger.info(f'Execution time: {time.time() - t}')

        return [str(interval) for interval in intervals]
