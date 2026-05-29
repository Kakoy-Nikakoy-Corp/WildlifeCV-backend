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
from torchcodec.encoders import Encoder
from torchvision.ops import nms
from torchvision.io import decode_image, write_jpeg
from ultralytics import YOLO
from ultralytics.engine.results import Results

from src.paths import get_yolo_weights_path, get_output_images_dpath
from src.types import ProcessedFrame, TimeInterval, ModelPrediction, ProcessedVideo
from src.utils import preprocess, rescale_bboxes, make_glyph_atlas, blit_text, draw_bboxes

pr = cProfile.Profile()  # Virtually no performance impact until enabled


class Model:
    def __init__(self) -> None:
        # Debug-related fields
        self.__verbose: bool = os.getenv('IRBIS_DEBUG') == '1'
        self.__use_profiler: bool = os.getenv('USE_PROFILER') == '1'

        # Determine a primary computational unit we're using (CUDA/CPU)
        self.__device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Instantiate a model
        weights_path: Path = get_yolo_weights_path()
        self.__model = YOLO(weights_path, verbose=self.__verbose, task='detect')

        # Set up the Glyph Atlas
        self.glyph_atlas = make_glyph_atlas(device=self.__device)

        logger.add('model_inf.log', level='INFO')

    def __postprocess(self, results: Results, original_image: torch.Tensor) -> ModelPrediction:
        """
        Obtain all necessary fields from a `ultralytics.engine.results.Results` object.

        Parameters:
            results: an object containing single prediction results.

        Returns:
            A single ModelPrediction object.
        """
        boxes = results.boxes  # Contains data for 'detection' task models

        conf: torch.Tensor = boxes.conf
        bboxes: torch.Tensor = boxes.xyxy

        if conf.numel() == 0:
            return ModelPrediction(0.0, original_image)

        # Non-maximum suppression
        idx = nms(bboxes, conf, 0.45)
        conf, bboxes = conf[idx], bboxes[idx]

        # Rescaling
        orig_shape = original_image.shape[-2:]
        scaled_bboxes = rescale_bboxes(bboxes, orig_shape, 736)

        peak_conf = float(conf.max())

        labels = list(map(lambda x: str(round(x, 2)), conf.tolist()))
        # brown, seagreen, orange, darkviolet
        colors = [(165, 42, 42), (46, 139, 87), (255, 165, 0), (148, 0, 211)]
        label_colors = [(230, 230, 250)]  # lavender
        img = draw_bboxes(original_image, scaled_bboxes, labels, self.glyph_atlas, colors, label_colors, 5)

        return ModelPrediction(peak_conf, img)

    def __make_prediction(self,
        frames: torch.Tensor,
        single: bool = False,
        threshold: float = 0.25) -> Iterator[ModelPrediction] | ModelPrediction:
        """
        Make a prediction on a single (C, H, W) image or a batch (B, C, H, W) of images.

        Parameters:
            frames: input data in PyTorch canonical tensor format.
            single: whether a prediction should be singular or not. This method behaves as an iterator for the latter case.
            threshold: FIXME!!!

        Returns:
            A single ModelPrediction or a sequence of them via an Iterator.
        """
        # Letterbox and normalize tensors
        preprocessed_frames: torch.Tensor = preprocess(frames, 736)

        # Feed tensors to YOLO to obtain an actual prediction
        results_list: list[Results] = self.__model(preprocessed_frames, conf=threshold, verbose=self.__verbose, device=self.__device)

        # Return a single prediction
        if single:
            return self.__postprocess(results_list[0], frames[0])

        # Generate results one by one in case of batch prediction
        def results_iterator() -> Iterator[ModelPrediction]:
            for i, results in enumerate(results_list):
                yield self.__postprocess(results, frames[i])

        return results_iterator()

    def __process_video(self, video_path: Path, gap: int = 2, batch_size: int = 16, threshold: float = 0.25) -> ProcessedVideo:
        """
        Given path to a particular video, makes a series of predictions on its frames.

        Parameters:
            video_path: Путь до файла с видео.
            gap: Промежуток между "значимыми" кадрами, на которых модель делает предсказания.

            Увеличение значения этого параметра даёт кратный прирост к производительности ценой точности временных интервалов.

            batch_size: Количество кадров внутри одного пакета.

        Returns:
            Total video frames, frames per second and an iterator over ProcessedFrame objects.
        """
        # Arbitrary thread count gives substantial performance boosts only on CPU
        threads = 1 if self.__device == 'cuda' else 0

        # We are intentionally NOT applying any on-decode transformations to retain original frames for further encoding
        decoder = VideoDecoder(video_path, seek_mode="approximate", num_ffmpeg_threads=threads, device=self.__device)

        fps = decoder.metadata.average_fps
        total_frames = decoder.metadata.num_frames
        height = decoder.metadata.height
        width = decoder.metadata.width

        logger.info(f"FPS: {fps:.2f}, Total frames: {total_frames}, Shape: ({width}, {height})")

        def frames_iterator() -> Iterator[ProcessedFrame]:
            batch_length = gap * batch_size  # Length is measured in actual frames

            # Iterate over all possible batch starting points
            for offset in range(0, total_frames, batch_length):
                frames = decoder.get_frames_in_range(offset, offset + batch_length, gap).data

                if self.__verbose:
                    logger.info(f"\nBatch {offset // batch_length}/{total_frames // batch_length}: {len(frames)} frames")

                frame_number = offset + 1 # May equal 1...total_frames

                # Every batch yields 'batch_size' results with an interval of 'gap' frames between them
                for prediction in self.__make_prediction(frames, threshold=threshold):
                    timecode = Timecode(fps, frames=frame_number)

                    yield ProcessedFrame(frame_number, timecode, prediction)
                    frame_number += gap

        return ProcessedVideo(fps, width, height, total_frames, frames_iterator())

    def detect_video_intervals(
            self,
            video_path: Path,
            output_path: Path,
            window_coef: float = 1.5,
            window_threshold: float = 0.4,
            threshold: float = 0.25,
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
            window_threshold: Порог обнаружения барса в rolling window.
            threshold: FIXME!!!!
            gap: Промежуток между "значимыми" кадрами, на которых модель делает предсказания.

            Увеличение значения этого параметра даёт кратный прирост к производительности ценой точности временных интервалов.

            batch_size: Количество кадров внутри одного пакета.

        Returns:
            Список TimeInterval
        """
        t = time.time()

        if self.__use_profiler:
            pr.enable()

        video = self.__process_video(video_path, gap, batch_size, threshold)
        window_size = int(video.fps * window_coef / gap)
        smoothing_tc = Timecode(video.fps, start_seconds=smoothing_interval)

        # Алгоритм Rolling Window для сглаживания и уточнения предсказаний модели на временных интервалах
        avg_window_conf = 0.
        is_recording = False
        last_interval: TimeInterval | None = None

        window: deque[ProcessedFrame] = deque()
        intervals: list[TimeInterval] = []

        all_frames: list[ProcessedFrame] = []
        for frame in video.frames:
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
                start = left_frame.timecode

                # Если с конца предыдущей записи не прошло smoothing_interval секунд
                # (и такая запись вообще имеется), то вместо добавления нового
                # интервала будем "двигать" конец предыдущего, чтобы слить их в один
                if last_interval is None or start >= last_interval.end + smoothing_tc:
                    last_interval = TimeInterval(start=start, end=None)
                    intervals.append(last_interval)

            # Если ведём запись и упали ниже порога ЛИБО дошли до конца видео, заканчиваем интервал
            elif is_recording and (avg_window_conf < window_threshold or frame.number + gap >= video.frame_count):
                is_recording = False
                last_interval.end = frame.timecode

            avg_window_conf -= left_frame.prediction.peak_conf / window_size

        encoder = Encoder()
        if self.__device == 'cpu':
            params = {
                'codec': 'libx264',
                'pixel_format': "yuv420p",
                'crf': 30,
                'preset': 'ultrafast',
                'extra_options': {
                    "tune": "zerolatency",
                },
            }
        else:
            params = {
                'codec': 'h264_nvenc',
                'device': 'cuda',
                'crf': 30,
                'preset': 'p1',
                'extra_options': {
                    "tune": 3,
                    "qp": 30
                }
            }

        stream = encoder.add_video(
            height=video.height,
            width=video.width,
            frame_rate=video.fps / gap,
            **params
        )
        encoder.open_file(output_path)

        for interval in intervals:
            start = (interval.start.frames - 1) // gap
            end = (interval.end.frames - 1) // gap

            for i in range(start, end + 1):
                frame = all_frames[i]
                img = frame.prediction.img
                final_img = blit_text(img, str(frame.timecode), self.glyph_atlas, 30, video.height - 80)
                stream.add_frames(final_img.unsqueeze(0))

        encoder.close()

        if self.__use_profiler:
            pr.disable()
            pr.print_stats(sort='cumtime')

        logger.info(f'Execution time: {time.time() - t}')

        return [str(interval) for interval in intervals]

    def detect_image(self, image_path: Path, output_path: Path, threshold: float = 0.25) -> None:
        image_tensor = decode_image(str(image_path))
        pred: ModelPrediction = self.__make_prediction(image_tensor, single=True, threshold=threshold)
        write_jpeg(pred.img, output_path)
