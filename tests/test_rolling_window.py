from collections import deque
from typing import Any, Generator, Iterator

import pytest
from pytest_mock import MockerFixture
from timecode import Timecode
from torch import zeros

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


def get_frame_stream(confs: list[float], fps: int = 30):
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


def get_frame_list(conf_pairs: list[tuple[int, float]], fps: int =30) -> list[ProcessedFrame]:
    """
    Эмулирует возврат 'видео' как набор кадров.
    Нужен, чтобы сгенерировать ожидаемый дамп.

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


def test_rolling_window_behaviour(mocker: MockerFixture):
    # Управление настройками видео.
    # Каждая уверенность - это кадр
    # Список уверенностей - это как бы видео
    FPS = 30
    confs = [
        0.8,
        0.6,
        0.8,
        0.2
    ]
    # Очень простое представление ожидаемых кадров в дампе.
    # Пары вида 'Номер кадра + уверенность в кадре'
    expected_conf_pairs = [
        (1, 0.8),
        (2, 0.6),
        (3, 0.8),
        (4, 0.2)
    ]

    frame_amount = len(confs)
    frame_iterator = get_frame_stream(confs=confs, fps=FPS)
    expected_dump = get_frame_list(expected_conf_pairs, FPS)
    sut = rolling_window

    frames_dump = sut(
        frame_iterator, FPS, frame_amount,
        window_coef=0.1, window_threshold=0.8
    )

    assert frames_dump == expected_dump
