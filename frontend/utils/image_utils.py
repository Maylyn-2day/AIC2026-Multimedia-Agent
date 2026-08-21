"""
Frontend Image Utilities.

Pure-PIL image helpers used by Streamlit components.  No Streamlit imports
here — these functions are unit-testable in isolation.

Exported API:
- ``base64_encode``   : PIL Image → Base64 string.
- ``base64_decode``   : Base64 string → PIL Image.
- ``draw_bboxes``     : Overlay grounding bounding boxes on a keyframe.
- ``create_thumbnail``: Resize maintaining aspect ratio.
"""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image as PILImage

_BBOX_DEFAULT_COLOR: tuple[int, int, int] = (255, 80, 0)
_BBOX_DEFAULT_ALPHA: int = 180
_BBOX_LINE_WIDTH: int = 3
_LABEL_FONT_SIZE: int = 14
_LABEL_PAD: int = 4


def base64_encode(image: "PILImage.Image", fmt: str = "JPEG") -> str:
    """Encode a PIL Image to a Base64 string.

    Args:
        image: PIL Image (any mode).
        fmt: Output format — ``"JPEG"`` or ``"PNG"``. Defaults to ``"JPEG"``.

    Returns:
        Base64-encoded string (no data-URI prefix).

    Raises:
        ValueError: If ``fmt`` is not a supported PIL format.
    """
    buf = io.BytesIO()
    rgb = image.convert("RGB")
    try:
        rgb.save(buf, format=fmt)
    except (KeyError, OSError) as exc:
        raise ValueError(f"Unsupported image format '{fmt}': {exc}") from exc
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_decode(data: str) -> "PILImage.Image":
    """Decode a Base64 string to a PIL Image in RGB mode.

    Accepts both raw Base64 and ``data:image/...;base64,`` prefixed strings.

    Args:
        data: Base64-encoded image string.

    Returns:
        PIL Image in ``"RGB"`` mode.

    Raises:
        ValueError: If ``data`` is empty, invalid, or not a recognised image.
    """
    from PIL import Image, UnidentifiedImageError

    if not data:
        raise ValueError("Base64 data must not be empty.")

    if "," in data:
        data = data.split(",", 1)[1]

    try:
        raw = base64.b64decode(data)
    except Exception as exc:
        raise ValueError(f"Invalid Base64 encoding: {exc}") from exc

    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Unrecognised image format: {exc}") from exc


def draw_bboxes(
    image: "PILImage.Image",
    bboxes: list[Any],
    color: tuple[int, int, int] = _BBOX_DEFAULT_COLOR,
    line_width: int = _BBOX_LINE_WIDTH,
    show_labels: bool = True,
) -> "PILImage.Image":
    """Overlay bounding boxes on a keyframe image.

    Bounding boxes are expected in normalised ``[x1, y1, x2, y2]`` format
    (values in [0, 1]).  Accepts both ``GroundingResult`` Pydantic objects
    and plain dicts with ``label``, ``confidence``, ``bbox`` keys.

    Args:
        image: Source PIL image.
        bboxes: List of grounding results.
        color: RGB tuple for box stroke and label background.
        line_width: Pixel width of box border.
        show_labels: Draw ``label (confidence%)`` text above each box.

    Returns:
        New PIL Image with bounding boxes overlaid.
    """
    from PIL import Image, ImageDraw, ImageFont

    result = image.convert("RGBA").copy()
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = image.size
    fill_color = (*color, _BBOX_DEFAULT_ALPHA)

    for item in bboxes:
        if hasattr(item, "bbox"):
            bbox, label, confidence = item.bbox, item.label, item.confidence
        else:
            bbox, label, confidence = item["bbox"], item["label"], item["confidence"]

        x1, y1, x2, y2 = bbox
        px1, py1, px2, py2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)

        draw.rectangle([px1, py1, px2, py2], fill=fill_color)
        for t in range(line_width):
            draw.rectangle([px1 + t, py1 + t, px2 - t, py2 - t], outline=(*color, 255))

        if show_labels:
            text = f"{label} {confidence:.0%}"
            try:
                font = ImageFont.truetype("arial.ttf", _LABEL_FONT_SIZE)
            except (IOError, OSError):
                font = ImageFont.load_default()

            bbox_text = draw.textbbox((px1, py1), text, font=font)
            tw, th = bbox_text[2] - bbox_text[0], bbox_text[3] - bbox_text[1]
            label_y = max(0, py1 - th - _LABEL_PAD * 2)
            draw.rectangle(
                [px1, label_y, px1 + tw + _LABEL_PAD * 2, label_y + th + _LABEL_PAD * 2],
                fill=(*color, 220),
            )
            draw.text(
                (px1 + _LABEL_PAD, label_y + _LABEL_PAD),
                text,
                fill=(255, 255, 255, 255),
                font=font,
            )

    return Image.alpha_composite(result, overlay).convert("RGB")


def create_thumbnail(
    image: "PILImage.Image",
    size: tuple[int, int] = (320, 180),
) -> "PILImage.Image":
    """Resize an image to ``size`` preserving aspect ratio.

    The resized image is centred on a black canvas of exactly ``size``.

    Args:
        image: Source PIL image.
        size: Target ``(width, height)`` in pixels. Defaults to ``(320, 180)``.

    Returns:
        New PIL Image of exactly ``size`` in ``"RGB"`` mode.
    """
    from PIL import Image

    target_w, target_h = size
    img = image.convert("RGB")
    img.thumbnail(size, Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    paste_x = (target_w - img.width) // 2
    paste_y = (target_h - img.height) // 2
    canvas.paste(img, (paste_x, paste_y))
    return canvas
