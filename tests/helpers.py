from collections.abc import Generator
from typing import Any

from timecode import Timecode
from torch import zeros

from src.types import ProcessedFrame, ModelPrediction


def get_frame_stream(confs: list[float], fps: int = 30) -> Generator[ProcessedFrame, Any, None]:
    """
    Эмулирует 'поток' кадров из видео.

    Parameters:
        confs: Список уверенностей на кадрах.
        fps: FPS 'видео'
    """
    img = zeros(1)
    framerate = str(fps)
    video_stream_len = len(confs)
    for i, conf in zip(range(1, video_stream_len+1), confs):
        yield ProcessedFrame(i, Timecode(framerate, frames=i), ModelPrediction(conf, img))


def get_frame_list(conf_pairs: list[tuple[int, float]], fps: int = 30) -> list[ProcessedFrame]:
    """
    Эмулирует возврат 'видео' как набора кадров.
    Нужна для генерации ожидаемого дампа.

    Parameters:
        conf_pairs: Список из пар вида 'номер-кадра, conf'
        fps: FPS 'видео'
    """
    img = zeros(1)
    framerate = str(fps)

    frame_nums = [num for num, conf in conf_pairs]
    confs = [conf for num, conf in conf_pairs]

    frames = []
    for i, conf in zip(frame_nums, confs, strict=True):
        frame = ProcessedFrame(i, Timecode(framerate, frames=i), ModelPrediction(conf, img))
        frames.append(frame)
    return frames
