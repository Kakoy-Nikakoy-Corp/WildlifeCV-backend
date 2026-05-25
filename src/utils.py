import torch
import torchvision.transforms.v2.functional as f
from torchvision.transforms import InterpolationMode

from src.types import ScalingParams


def calculate_scaling(orig_shape: torch.Size | tuple[int, int], target_size: int = 640) -> ScalingParams:
    h, w = orig_shape

    # Calculate scale ratio
    r = min(target_size / h, target_size / w)

    # New dimensions
    new_h = int(round(h * r))
    new_w = int(round(w * r))

    # Compute padding
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2

    return ScalingParams(r, new_w, new_h, pad_w, pad_h)


def preprocess(img: torch.Tensor, target_size: int = 640) -> torch.Tensor:
    """
    Letterboxing and normalization for a raw batch of images (B, C, H, W) or a single image (C, H, W).

    Parameters:
        img: Input tensor (B, C, H, W) or (C, H, W) in uint8 [0, 255]
        target_size: Target square size (default 640)

    Returns:
        YOLO inference-ready tensor (B, C, target_size, target_size)
    """
    if img.ndim == 3:
        img = img.unsqueeze(0)

    b, c, h, w = img.shape
    p: ScalingParams = calculate_scaling((h, w))

    # Resize
    resized = f.resize(img, size=[p.new_h, p.new_w], antialias=False, interpolation=InterpolationMode.BILINEAR)

    # Create target canvas
    canvas_dim = (b, c, target_size, target_size)
    letterboxed = torch.full(canvas_dim, fill_value=114.0, dtype=img.dtype, device=img.device)

    top, bottom = p.pad_h, p.pad_h + p.new_h
    left, right = p.pad_w, p.pad_w + p.new_w

    # Paste resized image in center
    letterboxed[:, :, top:bottom, left:right] = resized

    return letterboxed / 255.0


def rescale_bboxes(
        bboxes: torch.Tensor,
        orig_shape: torch.Size,
        target_size: int = 640
) -> torch.Tensor:
    """
    Scale bounding boxes from original image coordinates to letterboxed coordinates.

    Args:
        bboxes: Tensor of shape (N, 4) in xyxy format (x1, y1, x2, y2)
                in absolute pixel coordinates of the original image.
        orig_shape: Original image shape as (height, width)
        target_size: Target square size used in preprocess()

    Returns:
        Scaled bboxes in the letterboxed (target_size, target_size) coordinate system.
    """
    p: ScalingParams = calculate_scaling(orig_shape, target_size)

    # Clone to avoid modifying original
    bboxes = bboxes.clone()

    # Remove padding
    bboxes[..., 0] -= p.pad_w  # x1
    bboxes[..., 1] -= p.pad_h  # y1
    bboxes[..., 2] -= p.pad_w  # x2
    bboxes[..., 3] -= p.pad_h  # y2

    # Scale back to original size
    bboxes /= p.r

    return bboxes
