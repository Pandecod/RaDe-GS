"""Readable comparison labels outside the original reference/render pixel regions."""
from __future__ import annotations

from pathlib import Path
import os

from PIL import Image, ImageDraw, ImageFont


def label_font(size, *, bold=False):
    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    windows = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    names += [str(windows / ("arialbd.ttf" if bold else "arial.ttf"))]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError as exc:
        raise RuntimeError("Label besar memerlukan Pillow >= 10.1; perbarui requirements-research.txt.") from exc


def wrapped_lines(text, font, available):
    """Wrap by measured pixel width, including long filenames without spaces."""
    lines, line = [], ""
    for word in str(text).split():
        candidate = f"{line} {word}" if line else word
        box = font.getbbox(candidate)
        if box[2] - box[0] <= available:
            line = candidate
            continue
        if line:
            lines.append(line)
            line = ""
        for char in word:
            candidate = line + char
            box = font.getbbox(candidate)
            if line and box[2] - box[0] > available:
                lines.append(line)
                line = char
            else:
                line = candidate
    if line:
        lines.append(line)
    return lines or [""]


def comparison_canvas(left, right, *, scene_name, split, render_method, method_name, filename):
    if left.mode != "RGB" or right.mode != "RGB" or left.size != right.size:
        raise ValueError("Comparison inputs must be equal-size RGB images")
    width, height = left.size
    # Relative sizing stays legible when a full-width pair is fitted to a screen.
    title_size = max(1, round(width * 0.035))
    detail_size = max(1, round(title_size * 0.62))
    padding = max(1, round(width * 0.018))
    gap = max(1, round(title_size * 0.24))
    title_font = label_font(title_size, bold=True)
    detail_font = label_font(detail_size)
    available = max(1, width - 2 * padding)
    iteration = render_method.removeprefix("ours_")
    labels = [
        ("Foto referensi", f"{scene_name} | {split} | {filename}"),
        (f"Hasil {method_name}", f"{scene_name} | iterasi {iteration} | {filename}"),
    ]
    positioned, bottoms = [], []
    for panel, (title, subtitle) in enumerate(labels):
        top = padding
        for text, font, color in ((title, title_font, "#122535"), (subtitle, detail_font, "#374a59")):
            for line in wrapped_lines(text, font, available):
                box = font.getbbox(line)
                ink_height = max(1, box[3] - box[1])
                x = panel * width + padding
                positioned.append({"text":line,"font":font,"color":color,
                    "point":(x-box[0],top-box[1]),
                    "ink_box":(x,top,x+box[2]-box[0],top+ink_height),"panel":panel})
                top += max(ink_height, round(font.size * 1.2))
            top += gap
        bottoms.append(top)
    header = max(bottoms) + padding
    canvas = Image.new("RGB", (2 * width, height + header), "white")
    # Labels and divider never cover, rescale or alter the two source images.
    canvas.paste(left, (0, header))
    canvas.paste(right, (width, header))
    draw = ImageDraw.Draw(canvas)
    for item in positioned:
        draw.text(item["point"], item["text"], font=item["font"], fill=item["color"])
    draw.line((width, 0, width, header-1), fill="#d2d9df", width=max(1,round(width/1000)))
    metadata = {"layout_version":2,"header_height":header,"title_font_size":title_size,
                "detail_font_size":detail_size,"font_name":title_font.getname()[0],
                "image_width":width,"image_height":height,
                "labels":[{"text":item["text"],"panel":item["panel"],"ink_box":list(item["ink_box"])}
                          for item in positioned]}
    return canvas, metadata
