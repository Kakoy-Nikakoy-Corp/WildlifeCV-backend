from collections import deque
from dataclasses import dataclass
from pathlib import Path

from torchcodec.decoders import VideoDecoder
from loguru import logger
from timecode import Timecode
from ultralytics import YOLO

from src.paths import get_yolo_weights_path


logger.add('model_inf.log')


@dataclass(slots=True, frozen=True)
class Frame:
    timecode: Timecode
    confs: list[float]
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
        self.model = YOLO(weights_path)

    def recognise(self, video_path: Path, window_coef: float = 1.5, threshold: float = 0.4, smoothing_interval: float = 2) -> list[str]:
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
        video = VideoDecoder(video_path, seek_mode="approximate")

        FPS = video.metadata.average_fps
        SAMPLE_FRAME = Timecode(FPS)
        WINDOW_SIZE = int(FPS * window_coef)
        logger.info(f"FPS: {FPS}")

        # Rolling Window Mechanism
        curr_timecode = SAMPLE_FRAME
        window: deque[Frame] = deque()
        avg_window_conf = 0.
        is_recording = False
        intervals: list[TimeInterval] = []
        last_ending: Timecode | None = None
        for frame in video:
            # Convert frame from CHW to float32 normalized BCWH
            frame = frame.unsqueeze(0).float() / 255.0

            result = self.model(frame)

            boxes = result.boxes
            confs: list[float] = boxes.conf.tolist()
            bboxes_coords: list = boxes.xyxy.tolist()
            if not confs:  # Пустой confs
                confs.append(0)
            logger.info(f"Confs: {boxes.conf.round(decimals=3)}, Bboxes: {boxes.xyxy.round()}")

            frame = Frame(curr_timecode, confs, bboxes_coords)
            window.append(frame)
            avg_window_conf += max(confs) / WINDOW_SIZE

            if len(window) == WINDOW_SIZE:
                gunbye_frame = window.popleft()
                if avg_window_conf >= threshold and not is_recording:
                    is_recording = True
                    start = gunbye_frame.timecode
                    if (last_ending is None or
                        start >= last_ending + Timecode(FPS, start_seconds=smoothing_interval)):
                        timeinterval = TimeInterval(start=start, end=None)
                        intervals.append(timeinterval)

                elif avg_window_conf < threshold and is_recording:
                    is_recording = False
                    intervals[-1].end = curr_timecode
                    last_ending = curr_timecode

                avg_window_conf -= max(gunbye_frame.confs) / WINDOW_SIZE

            curr_timecode += SAMPLE_FRAME

        if is_recording:
            intervals[-1].end = curr_timecode

        return [str(interval) for interval in intervals]
