import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from typing import Tuple


def yolo_letterbox(
    img: torch.Tensor, 
    target_size: int = 640, 
    pad_value: float = 114.0
) -> torch.Tensor:
    """
    YOLO-style letterbox for batch of images (B, C, H, W).
    
    Args:
        img: Input tensor (B, C, H, W) in float [0, 1] or uint8 [0, 255]
        target_size: Target square size (default 640)
        pad_value: Padding value (114 is YOLO default)
    
    Returns:
        Letterboxed tensor (B, C, target_size, target_size)
    """
    B, C, H, W = img.shape
    
    # Calculate scale ratio
    r = min(target_size / H, target_size / W)
    
    # New dimensions
    new_h = int(round(H * r))
    new_w = int(round(W * r))
    
    # Resize
    resized = F.resize(img, size=[new_h, new_w], antialias=True)
    
    # Create target canvas
    letterboxed = torch.full((B, C, target_size, target_size), 
                           fill_value=pad_value,
                           dtype=img.dtype,
                           device=img.device)
    
    # Compute padding
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    
    # Paste resized image in center
    letterboxed[:, :, pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    
    return letterboxed / 255.0

YOLOLetterBox = T.Lambda(
    lambda x: letterbox_func(x, target_size=640, pad_value=114.0)
)
