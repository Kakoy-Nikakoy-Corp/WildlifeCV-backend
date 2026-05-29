from collections import deque
from typing import Iterator

from timecode import Timecode
from torch import Tensor

from src.types import ProcessedFrame, TimeInterval, ModelPrediction

def rolling_window(
    frames: Iterator[ProcessedFrame],
    fps: int,
    frame_count: int,
    window_coef: float = 1.5,
    window_threshold: float = 0.4,
    smoothing_interval: float = 2,
    gap: int = 2,
) -> list[ProcessedFrame]:
    window_size = int(fps * window_coef / gap)
    smoothing_tc = Timecode(fps, start_seconds=smoothing_interval)

    # Алгоритм Rolling Window для сглаживания и уточнения предсказаний модели на временных интервалах
    avg_window_conf = 0.
    is_recording = False
    last_interval: TimeInterval | None = None

    window: deque[ProcessedFrame] = deque()
    intervals: list[TimeInterval] = []
    encoder_input: list[ProcessedFrame] = []

    buffer: list[ProcessedFrame] = []
    all_frames: list[ProcessedFrame] = []
    for frame in frames:
        all_frames.append(frame)
        window.append(frame)
        avg_window_conf += frame.prediction.peak_conf / window_size

        # deque len -> O(1)
        if len(window) < window_size:
            continue

        left_frame: ProcessedFrame = window.popleft()

        # Если превысили порог обнаружения и запись интервала не идёт, начинаем его отслеживать
        if avg_window_conf >= window_threshold and not is_recording:
            is_recording = True
            buffer.append(left_frame)
            start = left_frame.timecode

            # Если с конца предыдущей записи не прошло smoothing_interval секунд
            # (и такая запись вообще имеется), то вместо добавления нового
            # интервала будем "двигать" конец предыдущего, чтобы слить их в один
            if last_interval is None or start >= last_interval.end + smoothing_tc:
                last_interval = TimeInterval(start=start, end=None)
                intervals.append(last_interval)

        # Если ведём запись и упали ниже порога ЛИБО дошли до конца видео, заканчиваем интервал
        elif is_recording and (avg_window_conf < window_threshold or frame.number + gap >= frame_count):
            is_recording = False
            last_interval.end = frame.timecode
            encoder_input.extend(buffer)
            buffer.clear()

        avg_window_conf -= left_frame.prediction.peak_conf / window_size

    return encoder_input



def test_rolling_window_behaviour():
    sut = rolling_window

    frames = [
        ProcessedFrame(1, Timecode('30', frames=1), ModelPrediction(0.8, Tensor())),
        ProcessedFrame(2, Timecode('30', frames=2), ModelPrediction(0.8, Tensor())),
        ProcessedFrame(3, Timecode('30', frames=3), ModelPrediction(0.8, Tensor())),
        ProcessedFrame(4, Timecode('30', frames=4), ModelPrediction(0.8, Tensor()))
    ]

    dump_frames = rolling_window(
        frames, 30, len(frames), window_coef=0.1, window_threshold=0.8
    )

    assert dump_frames == frames[:3]
