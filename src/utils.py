from uuid import uuid4

import torch
import torchvision.transforms.v2.functional as f
from torchvision.transforms import InterpolationMode
from PIL import ImageDraw, ImageFont, Image

from src.types import LetterboxParams

CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', ':', ';']


def get_uuid4() -> str:
    """Возвращает уникальный идентификатор."""
    return str(uuid4())


def calculate_letterbox_params(orig_shape: torch.Size | tuple[int, int], target_size: int = 640) -> LetterboxParams:
    h, w = orig_shape

    # Calculate scale ratio
    r = min(target_size / h, target_size / w)

    # New dimensions
    new_h = int(round(h * r))
    new_w = int(round(w * r))

    # Compute padding
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2

    return LetterboxParams(r, new_w, new_h, pad_w, pad_h)


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
    p: LetterboxParams = calculate_letterbox_params((h, w), target_size)

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
    p: LetterboxParams = calculate_letterbox_params(orig_shape, target_size)

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


def make_glyph_atlas(font_size: int = 40, device: str = 'cpu') -> dict[str, torch.Tensor]:
    font = ImageFont.truetype("fonts/droidsans.ttf", font_size)

    height = sum(font.getmetrics())
    width = int(font.getlength(CHARS[0]))

    glyph_atlas = {}

    for c in CHARS:
        stroke_img = Image.new("L", (width, height))
        stroke_draw = ImageDraw.Draw(stroke_img)

        stroke_draw.text((0, 0), c, fill=255, font=font, stroke_width=2, stroke_fill=255)

        char_img = Image.new("L", (width, height))
        char_draw = ImageDraw.Draw(char_img)

        char_draw.text((0, 0), c, fill=255, font=font)

        char_tensor = f.pil_to_tensor(char_img).to(device).squeeze(0) / 255.0
        stroke_tensor = f.pil_to_tensor(stroke_img).to(device).squeeze(0) / 255.0

        glyph_atlas[c] = {
            'fill': char_tensor,
            'stroke': stroke_tensor - char_tensor
        }

    return glyph_atlas


def blit_text(frame: torch.Tensor, text: str, atlas: dict, pos_x: int, pos_y: int, text_color: tuple = (255, 255, 255), stroke_color: tuple = (0, 0, 0)):
    current_x = pos_x
    color_tensor = torch.tensor(text_color, dtype=frame.dtype, device=frame.device).view(-1, 1, 1)
    stroke_tensor = torch.tensor(stroke_color, dtype=frame.dtype, device=frame.device).view(-1, 1, 1)

    for c in text:
        fill_mask: torch.Tensor = atlas[c]['fill'].unsqueeze(0)
        stroke_mask: torch.Tensor = atlas[c]['stroke'].unsqueeze(0)

        _, gh, gw = fill_mask.shape
        roi = frame[:, pos_y:pos_y + gh, current_x:current_x + gw]

        roi = roi * (1 - fill_mask) + color_tensor * fill_mask
        roi = roi * (1 - stroke_mask) + stroke_tensor * stroke_mask

        frame[:, pos_y:pos_y + gh, current_x:current_x + gw] = roi.to(frame.device)

        current_x += gw

    return frame


def draw_bboxes(frame: torch.Tensor, boxes: torch.Tensor, labels: list[str], atlas: dict, colors: list[tuple], label_colors: list[tuple], width: int = 2):
    _, h, w = frame.shape

    for i, box in enumerate(boxes):
        color_tensor = torch.tensor(colors[i % len(colors)], dtype=torch.uint8, device=frame.device).view(-1, 1, 1)

        # Округляем координаты до целых чисел и ограничиваем их размерами картинки
        xmin, ymin, xmax, ymax = box.long()
        xmin, xmax = torch.clamp(xmin, 0, w - 1), torch.clamp(xmax, 0, w - 1)
        ymin, ymax = torch.clamp(ymin, 0, h - 1), torch.clamp(ymax, 0, h - 1)

        # Настраиваем границы с учетом толщины линии (width)
        xmin_end = torch.clamp(xmin + width, 0, w)
        xmax_start = torch.clamp(xmax - width, 0, w)
        ymin_end = torch.clamp(ymin + width, 0, h)
        ymax_start = torch.clamp(ymax - width, 0, h)

        # Рисуем 4 стороны прямоугольника
        frame[:, ymin:ymax, xmin:xmin_end] = color_tensor  # Левая вертикальная линия
        frame[:, ymin:ymax, xmax_start:xmax] = color_tensor  # Правая вертикальная линия
        frame[:, ymin:ymin_end, xmin:xmax] = color_tensor  # Верхняя горизонтальная линия
        frame[:, ymax_start:ymax, xmin:xmax] = color_tensor  # Нижняя горизонтальная линия

        frame = blit_text(frame, labels[i], atlas, int(xmin + 10), int(ymin + 10), text_color=label_colors[i % len(label_colors)])

    return frame
