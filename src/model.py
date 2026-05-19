from collections import deque
from dataclasses import dataclass
from pathlib import Path
import os
import sys

from torchcodec.decoders import VideoDecoder
from loguru import logger
from timecode import Timecode
from ultralytics import YOLO

from src.paths import get_yolo_weights_path


logger.remove()


@dataclass(slots=True, frozen=True)
class Frame:
    id: int
    index: int
    timecode: Timecode
    max_conf: float
    bboxes_coords: list[list[float] | None]


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

        loglevel = 'INFO' if self.verbose else 'WARNING'
        logger.add('model_inf.log', level=loglevel)
        logger.add(sys.stderr, level=loglevel)

    def decode_video(self, video_path, gap):
        video = VideoDecoder(
            video_path,
            seek_mode="exact"
        )

        fps = video.metadata.average_fps
        num_real_frames = video.metadata.num_frames
        logger.info(f"FPS: {fps}; Real frames: {num_real_frames}")

        def frame_gen():
            current_id = 0
            for frame_index in range(1, num_real_frames + 1, gap):
                current_id += 1
                frame_obj = video.get_frame_at(frame_index - 1)
                frame_np = frame_obj.data.permute(1, 2, 0).contiguous().cpu().numpy()

                boxes = self.model(frame_np, verbose=self.verbose)[0].boxes

                confs: list[float] = boxes.conf.tolist()
                bbox_coords: list = boxes.xyxy.tolist()
                logger.info(f"Frame index {frame_index}/{num_real_frames}\nConfs: {boxes.conf.round(decimals=3)}, Bboxes: {boxes.xyxy.round()}")

                max_conf: float = max(confs) if confs else 0.
                curr_timecode = Timecode(fps, frames=frame_index)

                yield Frame(current_id, frame_index, curr_timecode, max_conf, bbox_coords)

        return fps, num_real_frames, frame_gen


    def recognise(self, video_path: Path, window_coef: float = 1.5, threshold: float = 0.4, smoothing_interval: float = 2, gap: int = 2) -> list[str]:
        """
        Передает видео в модель и rolling window.

        Parameters:
            video_path: Путь до файла с видео
            window_coef: Коэффициент длины окна поиска.

            Длина окна = FPS * window_coef

            smoothing_interval: Максимальный промежуток между интервалами для их слияния (в секундах)
            threshold: Порог обнаружения барса в rolling window.

        Returns:
            Список TimeInterval
        """
        fps, num_real_frames, frame_gen = self.decode_video(video_path, gap)
        window_size = int(fps * window_coef / gap)

        # Rolling Window Mechanism
        avg_window_conf = 0.
        is_recording = False
        last_ending: Timecode | None = None

        window: deque[Frame] = deque()
        intervals: list[TimeInterval] = []

        for frame in frame_gen():
            window.append(frame)
            avg_window_conf += frame.max_conf / window_size

            if frame.id < window_size:
                continue

            left_frame: Frame = window.popleft()

            # Если превысили порог обнаружения и запись интервала не идёт, начинаем его отслеживать
            if avg_window_conf >= threshold and not is_recording:
                is_recording = True
                start = left_frame.timecode

                if (last_ending is None or
                    start >= last_ending + Timecode(fps, start_seconds=smoothing_interval)):
                    timeinterval = TimeInterval(start=start, end=None)
                    intervals.append(timeinterval)

            # Если ведём запись и упали ниже порога ЛИБО дошли до конца видео, заканчиваем интервал
            elif is_recording and (avg_window_conf < threshold or frame.index + gap >= num_real_frames):
                is_recording = False
                intervals[-1].end = frame.timecode
                last_ending = frame.timecode

            avg_window_conf -= left_frame.max_conf / window_size

        return [str(interval) for interval in intervals]
