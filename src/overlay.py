from pathlib import Path
from timecode import Timecode
from torchcodec.encoders import Encoder, VideoStream

from src.utils import make_glyph_atlas, blit_text
from src.types import ProcessedVideo, ProcessedFrame


class OverlayEncoder:
    def __init__(self, device: str) -> None:
        self.__encoder: Encoder | None = None
        self.glyph_atlas = make_glyph_atlas(device=device)

        if device == 'cpu':
            self.__params = {
                'codec': 'libx264',
                'pixel_format': "yuv420p",
                'crf': 30,
                'preset': 'ultrafast',
                'extra_options': {
                    "tune": "zerolatency",
                },
            }
        else:
            self.__params = {
                'codec': 'h264_nvenc',
                'device': 'cuda',
                'crf': 30,
                'preset': 'p1',
                'extra_options': {
                    "tune": 3,
                    "qp": 30
                }
            }

        self.__stream: VideoStream | None = None
        self.__last_timecode: Timecode | None = None
        self.video: ProcessedVideo | None = None

    def new_video(self, video: ProcessedVideo, output_path: Path) -> None:
        self.__encoder = Encoder()
        self.__stream = self.__encoder.add_video(
            height=video.height,
            width=video.width,
            frame_rate=video.fps,
            **self.__params
        )
        self.__encoder.open_file(output_path)

        self.video = video

    def close(self) -> None:
        self.__encoder.close()
        self.__last_timecode = None

    def add_frame(self, frame: ProcessedFrame) -> None:
        if self.__last_timecode is None or frame.timecode > self.__last_timecode:
            img = frame.prediction.img
            final_img = blit_text(img, str(frame.timecode), self.glyph_atlas, self.video.width - 300, 30)
            self.__stream.add_frames(final_img.unsqueeze(0))

            self.__last_timecode = frame.timecode
