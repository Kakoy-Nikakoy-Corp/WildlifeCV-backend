import cProfile
import os
import time
from collections import deque
from collections.abc import Iterator
from pathlib import Path

import torch
from loguru import logger
from timecode import Timecode
from torchcodec.decoders import VideoDecoder
from ultralytics import YOLO
from ultralytics.engine.results import Results

from src.paths import get_yolo_weights_path
from src.types import FrameResults, TimeInterval
from src.utils import preprocess

type ParsedResults = tuple[list[float], list]

pr = cProfile.Profile()  # Virtually no performance impact until enabled


class Model:
    def __init__(self) -> None:
        # Debug-related fields
        self.verbose: bool = os.getenv('IRBIS_DEBUG') == '1'
        self.use_profiler: bool = os.getenv('USE_PROFILER') == '1'

        # Determine a primary computational unit we're using (CUDA/CPU)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Instantiate a model
        weights_path: Path = get_yolo_weights_path()
        self.model = YOLO(weights_path, verbose=self.verbose, task='detect')
        self.model.to(self.device)

        logger.add('model_inf.log', level='INFO')

    @staticmethod
    def collect_results(results: Results) -> tuple[list[float], list]:
        boxes = results.boxes  # Contains data for 'detection' task models

        # Model confidences must not be empty to avoid sorting issues
        confs: list[float] = boxes.conf.tolist()
        if not confs:
            confs.append(0.)

        bbox_coords: list[list[float] | None] = boxes.xyxy.tolist()

        return confs, bbox_coords

    def make_prediction(self, frames: torch.Tensor, single: bool = False) -> Iterator[ParsedResults] | ParsedResults:
        # Letterbox and normalize tensors
        preprocessed_frames: torch.Tensor = preprocess(frames)

        # Feed tensors to YOLO to obtain an actual prediction
        results_list: list[Results] = self.model(preprocessed_frames, verbose=self.verbose)

        # Return a single prediction
        if single:
            return self.collect_results(results_list[0])

        # Generate results one by one in case of batch prediction
        def results_iterator() -> Iterator[ParsedResults]:
            for results in results_list:
                yield self.collect_results(results)

        return results_iterator()

    def process_video(self, video_path: Path, gap: int = 2, batch_size: int = 16) -> tuple[float, int, Iterator[FrameResults]]:
        # Arbitrary thread count gives substantial performance boosts only on CPU
        threads = 1 if self.device == 'cuda' else 0

        # We are intentionally NOT applying any on-decode transformations to retain original frames for further encoding
        video = VideoDecoder(video_path, seek_mode="approximate", num_ffmpeg_threads=threads, device=self.device)

        fps = video.metadata.average_fps
        total_frames = video.metadata.num_frames
        logger.info(f"FPS: {fps}, Total frames: {total_frames}")

        def frame_results_iterator() -> Iterator[FrameResults]:
            batch_length = gap * batch_size  # Length is measured in actual frames

            # Iterate over all possible batch starting points
            for offset in range(0, total_frames, batch_length):
                frames = video.get_frames_in_range(offset, offset + batch_length, gap).data

                if self.verbose:
                    logger.info(f"\nBatch {offset // batch_length}/{total_frames // batch_length}: {len(frames)} frames")

                frame_number = offset + 1 # May equal 1...total_frames

                # Every batch yields 'batch_size' results with an interval of 'gap' frames between them
                for confs, bbox_coords in self.make_prediction(frames):
                    timecode = Timecode(fps, frames=frame_number)

                    yield FrameResults(frame_number, timecode, confs, bbox_coords)
                    frame_number += gap

        return total_frames, fps, frame_results_iterator()

    def find_intervals(
            self,
            video_path: Path,
            window_coef: float = 1.5,
            threshold: float = 0.4,
            smoothing_interval: float = 2,
            gap: int = 2,
            batch_size: int = 16
    ) -> list[str]:
        """
        Search for snow leopard appearances using YOLO26 predictions enhanced with rolling window algorithm.

        Parameters:
            video_path: Путь до файла с видео.
            window_coef: Коэффициент длины окна поиска.

            Длина окна = FPS * window_coef

            smoothing_interval: Максимальный промежуток между интервалами для их слияния (в секундах).
            threshold: Порог обнаружения барса в rolling window.
            gap: Промежуток между "значимыми" кадрами, на которых модель делает предсказания.
            batch_size: Количество кадров внутри одного пакета.

            Увеличение значения этого параметра даёт кратный прирост к производительности ценой точности временных интервалов.

        Returns:
            Список TimeInterval
        """
        t = time.time()

        if self.use_profiler:
            pr.enable()

        total_frames, fps, frame_results = self.process_video(video_path, gap, batch_size)
        window_size = int(fps * window_coef / gap)
        smoothing_tc = Timecode(fps, start_seconds=smoothing_interval)

        # Алгоритм Rolling Window для сглаживания и уточнения предсказаний модели на временных интервалах
        avg_window_conf = 0.
        is_recording = False
        last_interval: TimeInterval | None = None

        window: deque[FrameResults] = deque()
        intervals: list[TimeInterval] = []
        for frame in frame_results:
            window.append(frame)
            avg_window_conf += max(frame.confs) / window_size

            # deque len -> O(1)
            if len(window) < window_size:
                continue

            left_frame: FrameResults = window.popleft()

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
