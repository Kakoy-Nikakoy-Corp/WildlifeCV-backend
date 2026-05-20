import torch
from torchcodec.decoders import VideoDecoder
from pathlib import Path

def test_lifecycle():
    video_path = Path("video.mkv")
    print("Creating Decoder...")
    decoder = VideoDecoder(str(video_path), device='cuda')
    print("Decoder Created.")

    print("Deleting Decoder...")
    del decoder
    print("Decoder Deleted.")

    torch.cuda.empty_cache()
    print("Done.")

if __name__ == "__main__":
    test_lifecycle()