import argparse
import os

from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def generate_demo_chart(
    out_path: str,
    title: str = "Demo Chart",
    legend_items=None,
    width: int = 900,
    height: int = 600,
):
    if legend_items is None:
        legend_items = [("Series A", (46, 134, 193)), ("Series B", (231, 76, 60))]

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    title_font = _load_font(28)
    body_font = _load_font(18)

    # Title (top)
    tw, th = draw.textsize(title, font=title_font)
    draw.text(((width - tw) // 2, 15), title, fill=(20, 20, 20), font=title_font)

    # Plot area
    plot_x1, plot_y1 = 80, 90
    plot_x2, plot_y2 = width - 80, height - 80

    # Axes
    draw.line([(plot_x1, plot_y2), (plot_x2, plot_y2)], fill=(0, 0, 0), width=3)
    draw.line([(plot_x1, plot_y1), (plot_x1, plot_y2)], fill=(0, 0, 0), width=3)

    # Y ticks
    for i in range(6):
        y = plot_y2 - int((plot_y2 - plot_y1) * (i / 5.0))
        draw.line([(plot_x1 - 6, y), (plot_x1, y)], fill=(0, 0, 0), width=2)
        label = str(i * 20)
        lw, lh = draw.textsize(label, font=body_font)
        draw.text((plot_x1 - 10 - lw, y - lh // 2), label, fill=(0, 0, 0), font=body_font)

    # Bars
    bar_colors = [legend_items[0][1], legend_items[1][1], legend_items[0][1]]
    values = [30, 55, 40]  # out of 100
    labels = ["A", "B", "C"]
    n = len(values)
    bar_w = int((plot_x2 - plot_x1) / (n * 2))
    gap = bar_w

    for i, (v, lab) in enumerate(zip(values, labels)):
        x1 = plot_x1 + gap + i * (bar_w + gap)
        x2 = x1 + bar_w
        y2 = plot_y2
        y1 = plot_y2 - int((plot_y2 - plot_y1) * (v / 100.0))
        draw.rectangle([x1, y1, x2, y2], fill=bar_colors[i], outline=None)
        lw, lh = draw.textsize(lab, font=body_font)
        draw.text(((x1 + x2 - lw) // 2, plot_y2 + 10), lab, fill=(0, 0, 0), font=body_font)

    # Legend (top-right)
    legend_padding = 12
    item_h = 26
    marker = 16
    legend_w = 0
    for text, _ in legend_items:
        legend_w = max(legend_w, draw.textsize(text, font=body_font)[0])
    legend_box_w = legend_padding * 3 + marker + legend_w
    legend_box_h = legend_padding * 2 + item_h * len(legend_items)

    lx2 = width - 20
    lx1 = lx2 - legend_box_w
    ly1 = plot_y1
    ly2 = ly1 + legend_box_h
    draw.rectangle([lx1, ly1, lx2, ly2], outline=(0, 0, 0), width=2, fill=(255, 255, 255))

    for i, (text, color) in enumerate(legend_items):
        y = ly1 + legend_padding + i * item_h
        mx1 = lx1 + legend_padding
        my1 = y + (item_h - marker) // 2
        draw.rectangle([mx1, my1, mx1 + marker, my1 + marker], fill=color, outline=(0, 0, 0))
        draw.text((mx1 + marker + legend_padding, y + 3), text, fill=(0, 0, 0), font=body_font)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="Output path for the demo chart PNG")
    ap.add_argument("--title", default="Demo Chart")
    args = ap.parse_args()
    generate_demo_chart(args.output, title=args.title)
    print(args.output)


if __name__ == "__main__":
    main()

