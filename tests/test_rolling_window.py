from collections import deque
from typing import Any, Generator, Iterator

import pytest
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

    last_time_written: Timecode | None = None
    def write_frame_to_file(f: ProcessedFrame) -> Timecode:
        if last_time_written is None or f.timecode > last_time_written:
            encoder_input.append(f)
            return f.timecode

        return last_time_written

    buffer: deque[ProcessedFrame] = deque()
    for frame in frames:
        window.append(frame)
        avg_window_conf += frame.prediction.peak_conf / window_size

        # deque len -> O(1)
        if len(window) < window_size:
            continue

        left_frame: ProcessedFrame = window.popleft()

        if last_interval and last_interval.end and last_interval.end < frame.timecode < last_interval.end + smoothing_tc:
            buffer.append(frame)

        # Если превысили порог обнаружения и запись интервала не идёт, начинаем его отслеживать
        if avg_window_conf >= window_threshold and not is_recording:
            is_recording = True
            start = left_frame.timecode

            # Если с конца предыдущей записи не прошло smoothing_interval секунд
            # (и такая запись вообще имеется), то вместо добавления нового
            # интервала будем "двигать" конец предыдущего, чтобы слить их в один
            if last_interval is None or start >= last_interval.end + smoothing_tc:
                last_interval = TimeInterval(start=start, end=None)
                intervals.append(last_interval)
            elif last_interval.end < start:
                while len(buffer) != 0:
                    last_time_written = write_frame_to_file(buffer.popleft())

            buffer.clear()

            # Записываем окно
            last_time_written = write_frame_to_file(left_frame)
            for fr in window:
                last_time_written = write_frame_to_file(fr)

        elif is_recording:
            last_time_written = write_frame_to_file(frame)

        # Если ведём запись и упали ниже порога ЛИБО дошли до конца видео, заканчиваем интервал
        if is_recording and (avg_window_conf < window_threshold or frame.number + gap >= frame_count):
            is_recording = False
            last_interval.end = frame.timecode

        avg_window_conf -= left_frame.prediction.peak_conf / window_size

    return encoder_input


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


def test_rolling_window_two_diff_intervals():
    """
    Тесткейс механизма rolling window.

    Два полностью разных интервала, не сливающиеся в один.
    """
    # Управление настройками видео.
    # Каждая уверенность - это кадр
    # Список уверенностей - это как бы видео
    FPS = 30
    confs = [
        0.8,
        0.8,
        0.8,
        0.8,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.8,
        0.8,
        0.8
    ]
    # Очень простое представление ожидаемых кадров в дампе.
    # Пары вида 'Номер кадра + уверенность в кадре'
    expected_conf_pairs = [
        (1,  0.8),
        (2,  0.8),
        (3,  0.8),
        (4,  0.8),
        (5,  0.2),
        (10,  0.8),
        (11,  0.8),
        (12,  0.8)
    ]

    frame_amount = len(confs)
    frame_iterator = get_frame_stream(confs=confs, fps=FPS)
    expected_dump = get_frame_list(expected_conf_pairs, FPS)
    sut = rolling_window

    frames_dump = sut(
        frame_iterator, FPS, frame_amount,
        window_coef=0.1, window_threshold=0.8, smoothing_interval=1/FPS, gap=1
    )

    assert frames_dump == expected_dump


def test_rolling_window_two_overlapped_intervals():
    """
    Тесткейс механизма rolling window.

    Два интервала, сливающихся в один из-за наложения
    """
    # Управление настройками видео.
    # Каждая уверенность - это кадр
    # Список уверенностей - это как бы видео
    FPS = 30
    confs = [
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.2,
        0.2,
        1.0,
        1.0
    ]
    # Очень простое представление ожидаемых кадров в дампе.
    # Пары вида 'Номер кадра + уверенность в кадре'
    expected_conf_pairs = [
        (1, 0.6),
        (2, 0.6),
        (3, 0.6),
        (4, 0.6),
        (5, 0.6),
        (6, 0.2),
        (7, 0.2),
        (8, 1.0),
        (9, 1.0)
    ]

    frame_amount = len(confs)
    frame_iterator = get_frame_stream(confs=confs, fps=FPS)
    expected_dump = get_frame_list(expected_conf_pairs, FPS)
    sut = rolling_window

    frames_dump = sut(
        frame_iterator, FPS, frame_amount,
        window_coef=0.17, window_threshold=0.6, smoothing_interval=1/FPS, gap=1
    )

    assert frames_dump == expected_dump


def test_rolling_window_two_merged_intervals():
    """
    Тесткейс механизма rolling window.

    Два интервала, сливающихся в один из-за механизма smoothing_tc
    """
    # Управление настройками видео.
    # Каждая уверенность - это кадр
    # Список уверенностей - это как бы видео
    FPS = 30
    confs = [
        0.8,
        0.8,
        0.8,
        0.1,
        0.1,
        0.1,
        0.4,
        0.4,
        1.0
    ]
    # Очень простое представление ожидаемых кадров в дампе.
    # Пары вида 'Номер кадра + уверенность в кадре'
    expected_conf_pairs = [
        (1, 0.8),
        (2, 0.8),
        (3, 0.8),
        (4, 0.1),
        (5, 0.1),
        (6, 0.1),
        (7, 0.4),
        (8, 0.4),
        (9, 1.0)
    ]

    frame_amount = len(confs)
    frame_iterator = get_frame_stream(confs=confs, fps=FPS)
    expected_dump = get_frame_list(expected_conf_pairs, FPS)
    sut = rolling_window

    frames_dump = sut(
        frame_iterator, FPS, frame_amount,
        window_coef=0.1, window_threshold=0.6, smoothing_interval=4/FPS, gap=1
    )

    assert frames_dump == expected_dump
