import torch
import torchvision.transforms.v2.functional as f
from torchvision.transforms import InterpolationMode


def preprocess(img: torch.Tensor, target_size: int = 640) -> torch.Tensor:
    """
    Letterboxing and normalization for a raw batch of images (B, C, H, W) or a single image (C, H, W).

    Args:
        img: Input tensor (B, C, H, W) or (C, H, W) in uint8 [0, 255]
        target_size: Target square size (default 640)

    Returns:
        YOLO inference-ready tensor (B, C, target_size, target_size)
    """
    if img.ndim == 3:
        img = img.unsqueeze(0)

    B, C, H, W = img.shape

    # Calculate scale ratio
    r = min(target_size / H, target_size / W)

    # New dimensions
    new_h = int(round(H * r))
    new_w = int(round(W * r))

    # Resize
    resized = f.resize(img, size=[new_h, new_w], antialias=False, interpolation=InterpolationMode.BILINEAR)

    # Create target canvas
    canvas_dim = (B, C, target_size, target_size)
    letterboxed = torch.full(canvas_dim, fill_value=114.0, dtype=img.dtype, device=img.device)

    # Compute padding
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2

    # Paste resized image in center
    letterboxed[:, :, pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    return letterboxed / 255.0
