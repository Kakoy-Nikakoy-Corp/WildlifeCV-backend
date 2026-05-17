from collections import deque
from dataclasses import dataclass
from pathlib import Path

from cv2 import VideoCapture, CAP_PROP_FPS
from loguru import logger
from timecode import Timecode
from ultralytics import YOLO

from src.paths import get_yolo_weights_path


logger.add('model_inf.log')


@dataclass(slots=True, frozen=True)
class Frame:
    timecode: Timecode
    confs: list[float] | None
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

    def recognise(self, video_path: Path, window_coef: float = 1.5, threshold: float = 0.4) -> list[str]:
        """
        Передает видео в модель и rolling window.

        Parameters:
            video_path: Путь до файла с видео
            window_coef: Коэффициент длины окна поиска.

            Длина окна = FPS * window_coef
            threshold: Порог обнаружения барса в rolling window.

        Returns:
            Список TimeInterval
        """
        results = self.model(video_path, stream=True)

        # Milliseconds per frame
        video = VideoCapture(str(video_path))
        FPS = video.get(CAP_PROP_FPS)
        logger.info(f"FPS: {FPS}")
        video.release()
        SAMPLE_FRAME = Timecode(FPS)

        # Rolling Window Mechanism
        curr_timecode = SAMPLE_FRAME
        window_size = int(FPS * window_coef)
        window: deque[Frame] = deque()
        avg_window_conf = 0.
        ongoing_timeinterval = False
        timeintervals: list[TimeInterval] = []
        last_ending = SAMPLE_FRAME
        for result in results:
            boxes = result.boxes
            confs: list[float] = boxes.conf.tolist()
            bboxes_coords: list = boxes.xyxy.tolist()
            if not confs:  # Пустой confs
                confs.append(0)
            logger.info(f"Confs: {boxes.conf.round(decimals=3)}, Bboxes: {boxes.xyxy.round()}")

            frame = Frame(curr_timecode, confs, bboxes_coords)
            window.append(frame)
            avg_window_conf += max(confs) / window_size

            if len(window) == window_size:
                gunbye_frame = window.popleft()
                if avg_window_conf >= threshold and not ongoing_timeinterval:
                    ongoing_timeinterval = True
                    start = gunbye_frame.timecode
                    if start >= last_ending:
                        timeinterval = TimeInterval(start=start, end=None)
                        timeintervals.append(timeinterval)

                elif avg_window_conf < threshold and ongoing_timeinterval:
                    ongoing_timeinterval = False
                    timeintervals[-1].end = curr_timecode
                    last_ending = curr_timecode

                avg_window_conf -= max(gunbye_frame.confs) / window_size

            curr_timecode += SAMPLE_FRAME

        if ongoing_timeinterval:
            timeintervals[-1].end = curr_timecode
            ongoing_timeinterval = False

        return [str(interval) for interval in timeintervals]
