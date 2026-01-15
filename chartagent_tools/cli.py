import argparse
import json
import os
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from .metadata import CHART_METADATA_PROMPT, extract_chart_metadata
from .tools import (
    annotate_legend,
    arithmetic,
    _axis_localizer_with_boxes,
    axis_localizer,
    clean_chart_image,
    compute_bar_height,
    compute_boxplot_entity,
    compute_segment_area,
    debug_clean_chart_ocr,
    detect_clean_chart_regions,
    get_edgepoints,
    get_boxplot,
    get_bar,
    get_marker_rgb,
    get_radial,
    analyze_radial_geometry,
    estimate_radial_value,
    interpolate_pixel_to_value,
    remove_regions,
    segment_and_mark,
)


BboxXyxy = Tuple[int, int, int, int]


def _parse_bbox_list(raw: Optional[Sequence[Sequence[str]]]) -> List[BboxXyxy]:
    if not raw:
        return []
    out: List[BboxXyxy] = []
    for item in raw:
        if len(item) != 4:
            raise ValueError("bbox must have 4 integers: x1 y1 x2 y2")
        x1, y1, x2, y2 = [int(v) for v in item]
        out.append((x1, y1, x2, y2))
    return out


def _save_preview(
    image: Image.Image,
    title_bbox: Optional[BboxXyxy],
    legend_bbox: Optional[BboxXyxy],
    extra_bboxes: Sequence[BboxXyxy],
    path: str,
) -> None:
    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)
    if title_bbox is not None:
        draw.rectangle(list(title_bbox), outline=(0, 128, 255), width=3)
    if legend_bbox is not None:
        draw.rectangle(list(legend_bbox), outline=(0, 200, 0), width=3)
    for b in extra_bboxes:
        draw.rectangle(list(b), outline=(255, 0, 0), width=3)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img.save(path)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="chartagent-tools")
    sub = parser.add_subparsers(dest="command")

    p_clean = sub.add_parser("clean_chart_image", help="Remove title/legend regions from a chart image.")
    p_clean.add_argument("--image", required=True, help="Input chart image path")
    p_clean.add_argument("--output", required=True, help="Output cleaned image path (e.g., out.png)")
    p_clean.add_argument(
        "--metadata",
        default=None,
        help="Optional metadata JSON path (output of extract_metadata). Uses its title/legend unless overridden.",
    )
    p_clean.add_argument(
        "--title",
        default=argparse.SUPPRESS,
        help="Title text to remove. If omitted, auto-detects title. Use --skip-title to disable.",
    )
    p_clean.add_argument(
        "--legend",
        action="append",
        default=argparse.SUPPRESS,
        help="Legend entry to remove (repeatable). If omitted, auto-detects legend. Use --skip-legend to disable.",
    )
    p_clean.add_argument("--skip-title", action="store_true", help="Skip title removal entirely.")
    p_clean.add_argument("--skip-legend", action="store_true", help="Skip legend removal entirely.")
    p_clean.add_argument(
        "--extra-remove",
        nargs=4,
        action="append",
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Extra manual removal bbox in XYXY (repeatable).",
    )
    p_clean.add_argument(
        "--preview",
        default=None,
        help="Optional path to save a preview image with removal boxes drawn.",
    )
    p_clean.add_argument(
        "--debug-ocr-json",
        default=None,
        help="Optional path to save OCR debug JSON (shows text boxes and match scores).",
    )
    p_clean.add_argument(
        "--debug-ocr-max",
        type=int,
        default=120,
        help="Max number of OCR candidate lines to include in debug JSON (default: 120).",
    )
    p_clean.add_argument(
        "--print-metadata",
        action="store_true",
        help="Print JSON with removed boxes and settings (best-effort).",
    )

    p_annot = sub.add_parser("annotate_legend", help="Crop legend and label legend entries with numeric ids.")
    p_annot.add_argument("--image", required=True, help="Input chart image path")
    p_annot.add_argument(
        "--metadata",
        default=None,
        help="Optional metadata JSON path (output of extract_metadata). Uses its legend unless overridden.",
    )
    p_annot.add_argument(
        "--legend",
        action="append",
        default=argparse.SUPPRESS,
        help="Legend entry to label (repeatable). If omitted, uses --metadata legend.",
    )
    p_annot.add_argument("--legend-output", required=True, help="Output path for cropped legend image")
    p_annot.add_argument("--labeled-output", required=True, help="Output path for labeled legend image")
    p_annot.add_argument("--mapping-output", required=True, help="Output path for bbox mapping JSON")
    p_annot.add_argument(
        "--debug-dir",
        default=None,
        help="Optional directory to write debug artifacts (legend crop, ROI/mask on failure, debug JSON).",
    )
    p_annot.add_argument(
        "--print-metadata",
        action="store_true",
        help="Print JSON with bbox mapping (best-effort).",
    )

    p_meta = sub.add_parser("extract_metadata", help="Extract chart metadata JSON (L.1.3) using OpenAI Responses API.")
    p_meta.add_argument("--image", required=True, help="Input chart image path")
    p_meta.add_argument("--output", default=None, help="Optional output JSON path")
    p_meta.add_argument(
        "--model",
        default="gpt-5-nano",
        help="Model name (default: gpt-5-nano).",
    )
    p_meta.add_argument(
        "--base-url",
        default=None,
        help="Optional OpenAI base URL override (e.g., https://api.openai.com).",
    )
    p_meta.add_argument(
        "--no-strict",
        action="store_true",
        help="Disable strict validation (fills missing keys with defaults).",
    )
    p_meta.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print the metadata extraction prompt template and exit.",
    )

    p_rgb = sub.add_parser("get_marker_rgb", help="Get the dominant RGB color of a legend marker.")
    p_rgb.add_argument(
        "--image",
        required=True,
        help="Input legend image path (output of annotate_legend --legend-output).",
    )
    p_rgb.add_argument(
        "--mapping",
        required=True,
        help="Path to bbox mapping JSON (output of annotate_legend --mapping-output).",
    )
    p_rgb.add_argument(
        "--text-of-interest",
        default=None,
        help="Legend text to select (fuzzy match). Optional if --label-of-interest is provided.",
    )
    p_rgb.add_argument(
        "--label-of-interest",
        type=int,
        default=None,
        help="Numeric label id to select (odd=marker, even=text). Optional if --text-of-interest is provided.",
    )
    p_rgb.add_argument(
        "--distance-between-text-and-marker",
        type=int,
        default=5,
        help="Legacy fallback distance in pixels (default: 5).",
    )
    p_rgb.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path to save the result.",
    )

    p_seg = sub.add_parser("segment_and_mark", help="Segment an image (SAM1) and label cleaned masks.")
    p_seg.add_argument("--image", required=True, help="Input chart image path")
    p_seg.add_argument("--output", required=True, help="Output labeled image path (e.g., out/labeled.png)")
    p_seg.add_argument(
        "--metadata",
        default=None,
        help="Optional metadata JSON path (output of extract_metadata). Used to pre-clean title/legend.",
    )
    p_seg.add_argument(
        "--masks-json",
        default=None,
        help="Optional path to save mask summaries as JSON (id/bbox/area/score).",
    )
    p_seg.add_argument(
        "--masks-npz",
        default=None,
        help="Optional path to save masks as a compressed NPZ (masks/bboxes/scores/ids).",
    )
    p_seg.add_argument("--segmentation-model", default="SAM", help='Segmentation model (default: "SAM").')
    p_seg.add_argument("--min-area", type=int, default=5000, help="Minimum mask area (default: 5000).")
    p_seg.add_argument(
        "--iou-thresh-unique",
        type=float,
        default=0.9,
        help="IoU threshold for duplicate removal (default: 0.9).",
    )
    p_seg.add_argument(
        "--iou-thresh-composite",
        type=float,
        default=0.98,
        help="IoU threshold for composite removal (default: 0.98).",
    )
    p_seg.add_argument(
        "--white-ratio-thresh",
        type=float,
        default=0.95,
        help="Background/white ratio threshold (default: 0.95).",
    )
    p_seg.add_argument(
        "--remove-background-color",
        action="store_true",
        help="If set, remove background-colored pixels from masks before filtering.",
    )
    p_seg.add_argument(
        "--max-points",
        type=int,
        default=256,
        help="Ignored (legacy SAM3 option).",
    )
    p_seg.add_argument(
        "--text-prompt",
        default=None,
        help="Ignored (legacy SAM3 option).",
    )
    p_seg.add_argument(
        "--debug-dir",
        default=None,
        help="Optional directory to save debug artifacts (seg_input.png, original_input.png, segment_and_mark_debug.json).",
    )
    p_seg.add_argument(
        "--print-metadata",
        action="store_true",
        help="Print JSON with mask summaries (best-effort).",
    )

    p_axis = sub.add_parser(
        "axis_localizer",
        help="Localize one axis and map tick values to pixel positions (OCR-based).",
    )
    p_axis.add_argument("--image", required=True, help="Input chart image path")
    p_axis.add_argument(
        "--axis",
        required=True,
        choices=["x", "top_x", "y", "right_y"],
        help="Axis to localize: x (bottom), top_x (top), y (left), right_y.",
    )
    p_axis.add_argument(
        "--axis-threshold",
        type=float,
        default=0.2,
        help="Fraction of image to scan near the axis (default: 0.2).",
    )
    p_axis.add_argument(
        "--axis-ticker",
        action="append",
        default=None,
        help="Optional axis ticker string (repeatable). If omitted, tries metadata tickers if --metadata is provided.",
    )
    p_axis.add_argument(
        "--metadata",
        default=None,
        help="Optional metadata JSON path (output of extract_metadata). Used to supply axis tickers when --axis-ticker is omitted.",
    )
    p_axis.add_argument("--output", default=None, help="Optional output JSON path (axis mapping).")
    p_axis.add_argument("--preview", default=None, help="Optional debug preview image path.")

    p_interp = sub.add_parser(
        "interpolate_pixel_to_value",
        help="Map a pixel coordinate to an axis value using linear interpolation.",
    )
    p_interp.add_argument("--pixel", required=True, type=float, help="Pixel coordinate to map.")
    p_interp.add_argument(
        "--axis-json",
        default=None,
        help="Axis mapping JSON path (output of axis_localizer). Uses keys: axis_values, axis_pixel_positions.",
    )
    p_interp.add_argument(
        "--axis-value",
        action="append",
        type=float,
        default=None,
        help="Axis tick value (repeatable). Use with --axis-pixel if --axis-json is not provided.",
    )
    p_interp.add_argument(
        "--axis-pixel",
        action="append",
        type=int,
        default=None,
        help="Axis tick pixel position (repeatable). Use with --axis-value if --axis-json is not provided.",
    )
    p_interp.add_argument("--output", default=None, help="Optional output JSON path.")

    p_arith = sub.add_parser(
        "arithmetic",
        help="Perform an arithmetic operation between two numbers.",
    )
    p_arith.add_argument("--a", required=True, type=float, help="First operand.")
    p_arith.add_argument("--b", required=True, type=float, help="Second operand.")
    p_arith.add_argument(
        "--operation",
        default="percentage",
        help='Operation: add/subtract/multiply/divide/percentage/ratio (default: "percentage").',
    )
    p_arith.add_argument("--output", default=None, help="Optional output JSON path.")

    p_area = sub.add_parser(
        "compute_segment_area",
        help="Compute segment area by pixel count or discrete-dot counting.",
    )
    p_area.add_argument("--image", required=True, help="Input chart image path")
    p_area.add_argument(
        "--measure",
        required=True,
        choices=["pixels", "discrete-dots"],
        help='Area measure method: "pixels" or "discrete-dots".',
    )
    p_area.add_argument(
        "--filter-rgb",
        nargs=3,
        type=int,
        default=None,
        metavar=("R", "G", "B"),
        help="Optional RGB filter (e.g., --filter-rgb 255 0 0).",
    )
    p_area.add_argument(
        "--masks-npz",
        default=None,
        help="Optional masks NPZ from segment_and_mark (contains masks + ids). Enables --filter-segment.",
    )
    p_area.add_argument(
        "--filter-segment",
        action="append",
        type=int,
        default=None,
        help="Segment id/index to include (repeatable). Requires --masks-npz.",
    )
    p_area.add_argument("--vis-output", default=None, help="Optional visualization image output path.")
    p_area.add_argument("--output", default=None, help="Optional output JSON path (area).")

    p_bar = sub.add_parser(
        "get_bar",
        help="Detect a bar bbox by color and/or ticker label (uses SAM1 + OCR).",
    )
    p_bar.add_argument("--image", required=True, help="Input chart image path")
    p_bar.add_argument(
        "--rgb",
        nargs=3,
        type=int,
        default=None,
        metavar=("R", "G", "B"),
        help="Optional RGB of interest (e.g., --rgb 255 0 0).",
    )
    p_bar.add_argument(
        "--ticker-label",
        default=None,
        help="Optional axis/category label text to localize (e.g., '2019' or 'Asia').",
    )
    p_bar.add_argument(
        "--segmentation-model",
        default="SAM",
        help='Segmentation model (default: "SAM").',
    )
    p_bar.add_argument(
        "--bar-orientation",
        default="vertical",
        choices=["vertical", "horizontal", "vertical-right"],
        help='Bar orientation (default: "vertical").',
    )
    p_bar.add_argument("--output", default=None, help="Optional output JSON path.")
    p_bar.add_argument("--preview", default=None, help="Optional preview image path with bbox drawn.")

    p_barh = sub.add_parser(
        "compute_bar_height",
        help="Compute a bar value from its bbox using axis_localizer + interpolation.",
    )
    p_barh.add_argument("--image", required=True, help="Input chart image path")
    p_barh.add_argument(
        "--bar",
        nargs=4,
        type=float,
        required=True,
        metavar=("A", "B", "C", "D"),
        help="Bar bbox as 4 numbers. Accepts either xyxy (x1 y1 x2 y2) or xywh (x y w h).",
    )
    p_barh.add_argument(
        "--bar-orientation",
        default="vertical",
        choices=["vertical", "horizontal", "vertical-right"],
        help='Bar orientation (default: "vertical").',
    )
    p_barh.add_argument(
        "--axis-threshold",
        type=float,
        default=0.15,
        help="Fraction of image scanned for tick labels during axis localization (default: 0.15).",
    )
    p_barh.add_argument(
        "--x-axis-ticker",
        action="append",
        default=None,
        help="Optional x-axis tick string (repeatable).",
    )
    p_barh.add_argument(
        "--y-axis-ticker",
        action="append",
        default=None,
        help="Optional y-axis tick string (repeatable). Used for left y and right y.",
    )
    p_barh.add_argument(
        "--metadata",
        default=None,
        help="Optional metadata JSON path (to auto-fill axis tickers when not provided).",
    )
    p_barh.add_argument("--output", default=None, help="Optional output JSON path.")

    p_box = sub.add_parser(
        "get_boxplot",
        help="Select boxplot segments from segmentation masks (filter by label/color/ids).",
    )
    p_box.add_argument("--image", required=True, help="Input chart image path")
    p_box.add_argument(
        "--masks-npz",
        required=True,
        help="Masks NPZ from segment_and_mark (contains masks/bboxes/scores/ids).",
    )
    p_box.add_argument(
        "--rgb",
        nargs=3,
        type=int,
        default=None,
        metavar=("R", "G", "B"),
        help="Optional RGB of interest to filter segments (e.g., --rgb 106 184 209).",
    )
    p_box.add_argument(
        "--ticker-label",
        default=None,
        help='Optional axis/category label to select a group (e.g., "Tuesday").',
    )
    p_box.add_argument(
        "--box-label",
        action="append",
        type=int,
        default=None,
        help="Optional segmentation id to include (repeatable). Overrides other filters.",
    )
    p_box.add_argument(
        "--boxplot-orientation",
        default="vertical",
        choices=["vertical", "horizontal"],
        help='Boxplot orientation (default: "vertical").',
    )
    p_box.add_argument(
        "--axis-threshold",
        type=float,
        default=0.15,
        help="Fraction of image scanned for category labels (default: 0.15).",
    )
    p_box.add_argument("--output", default=None, help="Optional output JSON path.")
    p_box.add_argument("--preview", default=None, help="Optional preview image path with bboxes drawn.")

    p_boxent = sub.add_parser(
        "compute_boxplot_entity",
        help="Compute a boxplot entity value (median/q1/q3/min/max/range/iqr) from selected bboxes.",
    )
    p_boxent.add_argument("--image", required=True, help="Input chart image path")
    p_boxent.add_argument(
        "--boxplot-json",
        default=None,
        help="Optional get_boxplot output JSON path (uses key: bboxes_xyxy).",
    )
    p_boxent.add_argument(
        "--bbox",
        action="append",
        nargs=4,
        type=int,
        default=None,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Boxplot segment bbox (repeatable). Use when --boxplot-json is not provided.",
    )
    p_boxent.add_argument(
        "--boxplot-orientation",
        default="vertical",
        choices=["vertical", "horizontal"],
        help='Boxplot orientation (default: "vertical").',
    )
    p_boxent.add_argument(
        "--entity",
        default="median",
        help='Entity to compute: median/min/max/q1/q3/q2/range/iqr (default: "median").',
    )
    p_boxent.add_argument(
        "--axis-threshold",
        type=float,
        default=0.15,
        help="Fraction of image scanned for tick labels during axis localization (default: 0.15).",
    )
    p_boxent.add_argument(
        "--x-axis-ticker",
        action="append",
        default=None,
        help="Optional x-axis tick string (repeatable).",
    )
    p_boxent.add_argument(
        "--y-axis-ticker",
        action="append",
        default=None,
        help="Optional y-axis tick string (repeatable).",
    )
    p_boxent.add_argument(
        "--metadata",
        default=None,
        help="Optional metadata JSON path (to auto-fill axis tickers when not provided).",
    )
    p_boxent.add_argument("--output", default=None, help="Optional output JSON path.")

    p_edge = sub.add_parser(
        "get_edgepoints",
        help="Get edge points (or dot centroid) for a segment at a given axis label.",
    )
    p_edge.add_argument("--image", required=True, help="Input chart image path")
    p_edge.add_argument(
        "--masks-npz",
        default=None,
        help="Optional masks NPZ from segment_and_mark (contains masks/bboxes/scores/ids). Required when --mask-label is used.",
    )
    p_edge.add_argument(
        "--rgb",
        nargs=3,
        type=int,
        default=None,
        metavar=("R", "G", "B"),
        help="Optional RGB of interest to filter pixels.",
    )
    p_edge.add_argument(
        "--ticker-label",
        default=None,
        help='Optional axis label to localize (e.g., "Q2", "2019", "Tuesday").',
    )
    p_edge.add_argument(
        "--mask-label",
        action="append",
        type=int,
        default=None,
        help="Optional segmentation id to union (repeatable). Requires --masks-npz.",
    )
    p_edge.add_argument(
        "--chart-orientation",
        default="vertical",
        choices=["vertical", "horizontal"],
        help='Chart orientation (default: "vertical").',
    )
    p_edge.add_argument(
        "--lineplot-get-dot",
        action="store_true",
        help="If set, return dot centroid(s) instead of edge endpoints.",
    )
    p_edge.add_argument(
        "--axis-threshold",
        type=float,
        default=0.15,
        help="Fraction of image scanned for axis label OCR (default: 0.15).",
    )
    p_edge.add_argument("--output", default=None, help="Optional output JSON path.")
    p_edge.add_argument("--preview", default=None, help="Optional preview image path with points drawn.")

    p_radial = sub.add_parser(
        "get_radial",
        help="Detect a radial segment bbox by color and/or ticker label (color mask or SAM1 + OCR).",
    )
    p_radial.add_argument("--image", required=True, help="Input chart image path")
    p_radial.add_argument(
        "--rgb",
        nargs=3,
        type=int,
        default=None,
        metavar=("R", "G", "B"),
        help="Optional RGB of interest (e.g., --rgb 255 0 0). Required for --segmentation-model color.",
    )
    p_radial.add_argument(
        "--ticker-label",
        default=None,
        help="Optional label text to localize with OCR (used as a tie-breaker / selector).",
    )
    p_radial.add_argument(
        "--segmentation-model",
        default="color",
        choices=["color", "SAM"],
        help='Segmentation model (default: "color"). Use "SAM" to use SAM1 masks.',
    )
    p_radial.add_argument("--output", default=None, help="Optional output JSON path.")
    p_radial.add_argument("--preview", default=None, help="Optional preview image path with bbox drawn.")

    p_radgeo = sub.add_parser(
        "analyze_radial_geometry",
        help="Estimate radial chart geometry (center, r_outer, r_max) from a segment bbox/contour.",
    )
    p_radgeo.add_argument("--image", required=True, help="Input chart image path")
    p_radgeo.add_argument(
        "--radial-json",
        default=None,
        help="Optional get_radial output JSON path (uses key: bbox_xyxy).",
    )
    p_radgeo.add_argument(
        "--bbox",
        nargs=4,
        type=int,
        default=None,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Segment bbox in XYXY. Used to derive a contour within the bbox.",
    )
    p_radgeo.add_argument(
        "--rgb",
        nargs=3,
        type=int,
        default=None,
        metavar=("R", "G", "B"),
        help="Optional RGB of the segment (improves contour extraction inside bbox).",
    )
    p_radgeo.add_argument("--output", default=None, help="Optional output JSON path.")
    p_radgeo.add_argument(
        "--preview",
        default=None,
        help="Optional preview image path with center/radii drawn.",
    )

    p_rval = sub.add_parser(
        "estimate_radial_value",
        help="Estimate a radial segment value from (center_x, center_y, r_outer, r_max).",
    )
    p_rval.add_argument("--image", required=True, help="Input chart image path")
    p_rval.add_argument(
        "--geometry-json",
        default=None,
        help="Optional analyze_radial_geometry output JSON path (uses keys: center_x, center_y, r_outer, r_max).",
    )
    p_rval.add_argument("--center-x", type=int, default=None, help="Circle center x (required if --geometry-json absent).")
    p_rval.add_argument("--center-y", type=int, default=None, help="Circle center y (required if --geometry-json absent).")
    p_rval.add_argument("--r-outer", type=float, default=None, help="Outer circle radius (required if --geometry-json absent).")
    p_rval.add_argument("--r-max", type=float, default=None, help="Max radius to segment (required if --geometry-json absent).")
    p_rval.add_argument(
        "--reference-circle-value",
        type=float,
        default=100.0,
        help="Value corresponding to outer reference circle (default: 100).",
    )
    p_rval.add_argument("--output", default=None, help="Optional output JSON path.")

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        raise SystemExit(2)

    if args.command == "clean_chart_image":
        img = Image.open(args.image)

        metadata_json = None
        if getattr(args, "metadata", None):
            with open(args.metadata, "r") as f:
                metadata_json = json.load(f)

        meta_title = None
        meta_legend_list = None
        if isinstance(metadata_json, dict):
            meta_title = metadata_json.get("title")
            meta_legend = metadata_json.get("legend")
            if isinstance(meta_legend, list):
                meta_legend_list = [str(x) for x in meta_legend if str(x).strip()]
            elif isinstance(meta_legend, dict):
                meta_legend_list = [str(k) for k in meta_legend.keys() if str(k).strip()]

        # Title input: auto by default (argument omitted), explicit string if provided, or None if skip flag.
        title_mode = "auto"
        title_arg = None
        if args.skip_title:
            title_mode = "skip"
            title_arg = None
        elif hasattr(args, "title"):
            title_mode = "explicit"
            title_arg = getattr(args, "title")
        elif meta_title is not None and str(meta_title).strip():
            title_mode = "metadata"
            title_arg = str(meta_title).strip()

        # Legend input: auto by default, explicit list if provided, or None if skip flag.
        legend_mode = "auto"
        legend_list = []
        legend_arg = None
        if args.skip_legend:
            legend_mode = "skip"
            legend_arg = None
        elif hasattr(args, "legend"):
            legend_mode = "explicit"
            legend_list = getattr(args, "legend") or []
            legend_arg = [str(s) for s in legend_list if str(s).strip()] if legend_list else []
        elif meta_legend_list:
            legend_mode = "metadata"
            legend_list = list(meta_legend_list)
            legend_arg = list(meta_legend_list)

        detect_kwargs = {}
        if title_mode == "skip":
            detect_kwargs["title"] = None
        elif title_mode in ("explicit", "metadata"):
            detect_kwargs["title"] = title_arg
        # auto => omit to trigger auto in tool

        if legend_mode == "skip":
            detect_kwargs["legend"] = None
        elif legend_mode in ("explicit", "metadata"):
            detect_kwargs["legend"] = legend_arg

        debug_payload = None
        if getattr(args, "debug_ocr_json", None):
            debug_payload = debug_clean_chart_ocr(
                img,
                title=title_arg if title_mode in ("explicit", "metadata") else None,
                legend=legend_list if legend_mode in ("explicit", "metadata") else None,
                max_candidates=int(getattr(args, "debug_ocr_max", 120) or 120),
            )

        try:
            regions = detect_clean_chart_regions(img, **detect_kwargs)
        except Exception as e:
            if getattr(args, "debug_ocr_json", None) and debug_payload is not None:
                debug_payload["error"] = {"type": type(e).__name__, "message": str(e)}
                os.makedirs(os.path.dirname(args.debug_ocr_json) or ".", exist_ok=True)
                with open(args.debug_ocr_json, "w") as f:
                    json.dump(debug_payload, f, ensure_ascii=False, indent=2)
            raise

        # Run the tool (auto title/legend removal).
        cleaned_kwargs = {}
        if title_mode == "skip":
            cleaned_kwargs["title"] = None
        elif title_mode in ("explicit", "metadata"):
            cleaned_kwargs["title"] = title_arg
        if legend_mode == "skip":
            cleaned_kwargs["legend"] = None
        elif legend_mode in ("explicit", "metadata"):
            cleaned_kwargs["legend"] = legend_arg

        cleaned = clean_chart_image(img, **cleaned_kwargs)

        # Optional manual removals after tool execution.
        extra_remove = _parse_bbox_list(args.extra_remove)
        if extra_remove:
            cleaned = remove_regions(cleaned, extra_remove)

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        cleaned.save(args.output)

        if args.preview:
            _save_preview(
                img,
                title_bbox=regions.get("title_bbox_xyxy"),
                legend_bbox=regions.get("legend_bbox_xyxy"),
                extra_bboxes=extra_remove,
                path=args.preview,
            )

        if getattr(args, "debug_ocr_json", None) and debug_payload is not None:
            debug_payload["detected_regions"] = {
                "title_bbox_xyxy": regions.get("title_bbox_xyxy"),
                "legend_bbox_xyxy": regions.get("legend_bbox_xyxy"),
                "title_bbox_source": regions.get("title_bbox_source"),
                "legend_bbox_source": regions.get("legend_bbox_source"),
            }
            os.makedirs(os.path.dirname(args.debug_ocr_json) or ".", exist_ok=True)
            with open(args.debug_ocr_json, "w") as f:
                json.dump(debug_payload, f, ensure_ascii=False, indent=2)

        if args.print_metadata:
            meta = {
                "tool": "clean_chart_image",
                "image": args.image,
                "output": args.output,
                "metadata_path": getattr(args, "metadata", None),
                "title_mode": title_mode,
                "title": title_arg if title_mode in ("explicit", "metadata") else None,
                "legend_mode": legend_mode,
                "legend": legend_list if legend_mode in ("explicit", "metadata") else None,
                "extra_remove": extra_remove,
                "title_bbox_xyxy": regions.get("title_bbox_xyxy"),
                "legend_bbox_xyxy": regions.get("legend_bbox_xyxy"),
                "title_bbox_source": regions.get("title_bbox_source"),
                "legend_bbox_source": regions.get("legend_bbox_source"),
                "used_tesseract": regions.get("used_tesseract"),
            }
            print(json.dumps(meta, ensure_ascii=False, indent=2))
        return

    if args.command == "annotate_legend":
        img = Image.open(args.image)

        metadata_json = None
        if getattr(args, "metadata", None):
            with open(args.metadata, "r") as f:
                metadata_json = json.load(f)

        legend_list = None
        if hasattr(args, "legend"):
            legend_list = [str(s) for s in (getattr(args, "legend") or []) if str(s).strip()]
        elif isinstance(metadata_json, dict):
            meta_legend = metadata_json.get("legend")
            if isinstance(meta_legend, list):
                legend_list = [str(x) for x in meta_legend if str(x).strip()]
            elif isinstance(meta_legend, dict):
                legend_list = [str(k) for k in meta_legend.keys() if str(k).strip()]

        if not legend_list:
            raise SystemExit(
                "annotate_legend requires legend entries via --legend ... or --metadata with a legend list."
            )

        legend_image, labeled_legend, bbox_mapping = annotate_legend(
            img, legend_list, debug_dir=getattr(args, "debug_dir", None)
        )

        os.makedirs(os.path.dirname(args.legend_output) or ".", exist_ok=True)
        legend_image.save(args.legend_output)
        os.makedirs(os.path.dirname(args.labeled_output) or ".", exist_ok=True)
        labeled_legend.save(args.labeled_output)

        os.makedirs(os.path.dirname(args.mapping_output) or ".", exist_ok=True)
        with open(args.mapping_output, "w") as f:
            json.dump(bbox_mapping, f, ensure_ascii=False, indent=2)

        if args.print_metadata:
            print(json.dumps(bbox_mapping, ensure_ascii=False, indent=2))
        return

    if args.command == "extract_metadata":
        if args.print_prompt:
            print(CHART_METADATA_PROMPT)
            return

        img = Image.open(args.image)
        meta = extract_chart_metadata(
            img,
            strict=not args.no_strict,
            model=args.model,
            base_url=args.base_url,
        )
        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        return

    if args.command == "get_marker_rgb":
        img = Image.open(args.image)
        with open(args.mapping, "r") as f:
            mapping_json = json.load(f)

        rgb = get_marker_rgb(
            img,
            mapping_json,
            text_of_interest=getattr(args, "text_of_interest", None),
            label_of_interest=getattr(args, "label_of_interest", None),
            distance_between_text_and_marker=getattr(args, "distance_between_text_and_marker", 5),
        )
        payload = {"rgb": [int(rgb[0]), int(rgb[1]), int(rgb[2])]}
        if getattr(args, "output", None):
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "segment_and_mark":
        img = Image.open(args.image)
        meta_json = None
        if getattr(args, "metadata", None):
            with open(args.metadata, "r") as f:
                meta_json = json.load(f)
        labeled, cleaned_masks = segment_and_mark(
            img,
            segmentation_model=getattr(args, "segmentation_model", "SAM"),
            min_area=getattr(args, "min_area", 5000),
            iou_thresh_unique=getattr(args, "iou_thresh_unique", 0.9),
            iou_thresh_composite=getattr(args, "iou_thresh_composite", 0.98),
            white_ratio_thresh=getattr(args, "white_ratio_thresh", 0.95),
            remove_background_color=bool(getattr(args, "remove_background_color", False)),
            max_points=getattr(args, "max_points", 256),
            text_prompt=getattr(args, "text_prompt", None),
            metadata=meta_json if isinstance(meta_json, dict) else None,
            debug_dir=getattr(args, "debug_dir", None),
        )

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        labeled.save(args.output)

        summary = []
        for m in cleaned_masks:
            bb = m.get("bbox_xyxy")
            bb_list = [int(v) for v in bb] if isinstance(bb, tuple) and len(bb) == 4 else None
            summary.append(
                {
                    "id": int(m.get("id", 0) or 0),
                    "bbox_xyxy": bb_list,
                    "area": int(m.get("area", 0) or 0),
                    "score": float(m.get("score", 0.0) or 0.0),
                }
            )

        if getattr(args, "masks_json", None):
            os.makedirs(os.path.dirname(args.masks_json) or ".", exist_ok=True)
            with open(args.masks_json, "w") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

        if getattr(args, "masks_npz", None):
            import numpy as np

            masks = [m.get("segmentation") for m in cleaned_masks]
            mask_stack = None
            if masks and all(isinstance(mm, np.ndarray) for mm in masks):
                mask_stack = np.stack([mm.astype(np.uint8) for mm in masks], axis=0)
            else:
                mask_stack = np.zeros((0, img.size[1], img.size[0]), dtype=np.uint8)

            bboxes = np.asarray(
                [(s.get("bbox_xyxy") or [-1, -1, -1, -1]) for s in summary],
                dtype=np.int32,
            )
            scores = np.asarray([s.get("score", 0.0) for s in summary], dtype=np.float32)
            ids = np.asarray([s.get("id", 0) for s in summary], dtype=np.int32)

            os.makedirs(os.path.dirname(args.masks_npz) or ".", exist_ok=True)
            np.savez_compressed(args.masks_npz, masks=mask_stack, bboxes=bboxes, scores=scores, ids=ids)

        if args.print_metadata:
            raw_mask_count = None
            final_mask_count = int(len(summary))
            if getattr(args, "debug_dir", None):
                try:
                    with open(os.path.join(args.debug_dir, "segment_and_mark_debug.json"), "r") as f:
                        dbg = json.load(f)
                    if isinstance(dbg, dict):
                        raw_mask_count = dbg.get("sam1_raw_mask_count")
                        if dbg.get("final_mask_count") is not None:
                            final_mask_count = int(dbg.get("final_mask_count") or final_mask_count)
                except Exception:
                    pass

            meta = {
                "tool": "segment_and_mark",
                "image": args.image,
                "output": args.output,
                "mask_count": int(final_mask_count),
                "raw_mask_count": int(raw_mask_count) if isinstance(raw_mask_count, (int, float)) else None,
                "debug_dir": getattr(args, "debug_dir", None),
                "masks": summary,
            }
            print(json.dumps(meta, ensure_ascii=False, indent=2))
        return

    if args.command == "axis_localizer":
        img = Image.open(args.image)
        tickers = getattr(args, "axis_ticker", None)

        if (tickers is None or not tickers) and getattr(args, "metadata", None):
            try:
                with open(args.metadata, "r") as f:
                    meta_json = json.load(f)
                if isinstance(meta_json, dict):
                    axis_key = {
                        "x": "x_axis_ticker_values",
                        "top_x": None,
                        "y": "y_axis_ticker_values",
                        "right_y": "right_y_axis_ticker_values",
                    }.get(str(args.axis), None)
                    if axis_key and isinstance(meta_json.get(axis_key), list):
                        tickers = meta_json.get(axis_key)
            except Exception:
                tickers = tickers

        axis_values, axis_pixel_positions, axis_bboxes = _axis_localizer_with_boxes(
            img,
            axis=str(args.axis),
            axis_threshold=float(getattr(args, "axis_threshold", 0.2) or 0.2),
            axis_tickers=tickers,
        )

        payload = {
            "tool": "axis_localizer",
            "image": args.image,
            "axis": str(args.axis),
            "axis_threshold": float(getattr(args, "axis_threshold", 0.2) or 0.2),
            "axis_values": [float(v) for v in axis_values],
            "axis_pixel_positions": [int(p) for p in axis_pixel_positions],
            "tickers_used": tickers if tickers is not None else None,
        }

        if getattr(args, "output", None):
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        if getattr(args, "preview", None):
            vis = img.convert("RGB")
            draw = ImageDraw.Draw(vis)
            W, H = vis.size
            thr = max(0.05, min(0.45, float(getattr(args, "axis_threshold", 0.2) or 0.2)))
            if str(args.axis) == "y":
                roi = (0, 0, int(round(thr * W)), H)
            elif str(args.axis) == "right_y":
                roi = (int(round((1.0 - thr) * W)), 0, W, H)
            elif str(args.axis) == "top_x":
                thr_top = max(0.30, min(0.45, float(thr) + 0.15))
                roi = (0, 0, W, int(round(thr_top * H)))
            else:
                roi = (0, int(round((1.0 - thr) * H)), W, H)
            rx0, ry0, rx1, ry1 = [int(v) for v in roi]
            draw.rectangle(list(roi), outline=(150, 150, 150), width=2)
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 18)
            except Exception:
                font = ImageFont.load_default()

            def _fmt(v: float) -> str:
                # Prefer integer display for clean ticks.
                if abs(v - round(v)) <= 1e-6:
                    return str(int(round(v)))
                s = "{:.4g}".format(float(v))
                return s

            for v, p, bb in zip(axis_values, axis_pixel_positions, axis_bboxes):
                label = _fmt(float(v))
                if isinstance(bb, tuple) and len(bb) == 4:
                    x1, y1, x2, y2 = [int(t) for t in bb]
                    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
                    if str(args.axis) == "x":
                        tx = int(x1)
                        ty = int(min(H - 1, y2 + 4))
                        draw.text((tx, ty), label, fill=(0, 0, 255), font=font)
                    elif str(args.axis) == "top_x":
                        tx = int(x1)
                        ty = int(max(0, y1 - 22))
                        draw.text((tx, ty), label, fill=(0, 0, 255), font=font)
                    else:
                        tx = int(max(0, x2 + 6))
                        ty = int(max(0, y1))
                        draw.text((tx, ty), label, fill=(0, 0, 255), font=font)
                else:
                    # No bbox (inferred tick) -> annotate near the axis position.
                    if str(args.axis) in {"x", "top_x"}:
                        draw.text((int(p), max(0, ry0 + 2)), label, fill=(0, 0, 255), font=font)
                    else:
                        draw.text((rx0 + 2, int(p)), label, fill=(0, 0, 255), font=font)
            os.makedirs(os.path.dirname(args.preview) or ".", exist_ok=True)
            vis.save(args.preview)

        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "interpolate_pixel_to_value":
        axis_values = None
        axis_pixel_positions = None

        if getattr(args, "axis_json", None):
            with open(args.axis_json, "r") as f:
                axis_json = json.load(f)
            if not isinstance(axis_json, dict):
                raise ValueError("--axis-json must be a JSON object")
            axis_values = axis_json.get("axis_values")
            axis_pixel_positions = axis_json.get("axis_pixel_positions")

        if axis_values is None or axis_pixel_positions is None:
            axis_values = getattr(args, "axis_value", None)
            axis_pixel_positions = getattr(args, "axis_pixel", None)

        if axis_values is None or axis_pixel_positions is None:
            raise ValueError("Provide either --axis-json or both --axis-value and --axis-pixel.")

        val = interpolate_pixel_to_value(
            float(getattr(args, "pixel")),
            [float(v) for v in list(axis_values)],
            [int(p) for p in list(axis_pixel_positions)],
        )

        payload = {
            "tool": "interpolate_pixel_to_value",
            "pixel": float(getattr(args, "pixel")),
            "value": float(val),
        }
        if getattr(args, "output", None):
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "arithmetic":
        result = arithmetic(
            float(getattr(args, "a")),
            float(getattr(args, "b")),
            operation=str(getattr(args, "operation", "percentage") or "percentage"),
        )
        payload = {
            "tool": "arithmetic",
            "a": float(getattr(args, "a")),
            "b": float(getattr(args, "b")),
            "operation": str(getattr(args, "operation", "percentage") or "percentage"),
            "result": float(result),
        }
        if getattr(args, "output", None):
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "compute_segment_area":
        img = Image.open(args.image)

        masks = None
        if getattr(args, "masks_npz", None):
            import numpy as np

            npz = np.load(args.masks_npz)
            mask_stack = np.asarray(npz.get("masks"))
            ids = npz.get("ids")
            if ids is None:
                ids_arr = np.arange(mask_stack.shape[0], dtype=np.int32) + 1
            else:
                ids_arr = np.asarray(ids).astype(np.int32)

            if mask_stack.ndim != 3:
                raise ValueError("--masks-npz must contain `masks` with shape (N,H,W)")
            if ids_arr.ndim != 1 or ids_arr.shape[0] != mask_stack.shape[0]:
                raise ValueError("--masks-npz `ids` must have shape (N,) matching masks")

            masks = []
            for i in range(int(mask_stack.shape[0])):
                masks.append(
                    {
                        "id": int(ids_arr[i]),
                        "segmentation": np.asarray(mask_stack[i]).astype(bool),
                    }
                )

        if getattr(args, "filter_segment", None) and not getattr(args, "masks_npz", None):
            raise ValueError("--filter-segment requires --masks-npz")

        vis, area = compute_segment_area(
            img,
            tuple(getattr(args, "filter_rgb")) if getattr(args, "filter_rgb", None) else None,
            str(getattr(args, "measure")),
            masks,
            list(getattr(args, "filter_segment")) if getattr(args, "filter_segment", None) else None,
        )

        payload = {
            "tool": "compute_segment_area",
            "image": args.image,
            "measure": str(getattr(args, "measure")),
            "filter_rgb": list(getattr(args, "filter_rgb")) if getattr(args, "filter_rgb", None) else None,
            "masks_npz": getattr(args, "masks_npz", None),
            "filter_segment": list(getattr(args, "filter_segment")) if getattr(args, "filter_segment", None) else None,
            "area": int(area),
        }

        if getattr(args, "vis_output", None):
            os.makedirs(os.path.dirname(args.vis_output) or ".", exist_ok=True)
            vis.save(args.vis_output)
            payload["vis_output"] = args.vis_output

        if getattr(args, "output", None):
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "get_bar":
        img = Image.open(args.image)
        rgb = tuple(getattr(args, "rgb")) if getattr(args, "rgb", None) else None
        bb = get_bar(
            img,
            rgb_of_interest=rgb,
            ticker_label=getattr(args, "ticker_label", None),
            segmentation_model=str(getattr(args, "segmentation_model", "SAM") or "SAM"),
            bar_orientation=str(getattr(args, "bar_orientation", "vertical") or "vertical"),
        )
        payload = {
            "tool": "get_bar",
            "image": args.image,
            "rgb_of_interest": list(rgb) if rgb is not None else None,
            "ticker_label": getattr(args, "ticker_label", None),
            "segmentation_model": str(getattr(args, "segmentation_model", "SAM") or "SAM"),
            "bar_orientation": str(getattr(args, "bar_orientation", "vertical") or "vertical"),
            "bbox_xyxy": [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])],
        }
        if getattr(args, "output", None):
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        if getattr(args, "preview", None):
            vis = img.convert("RGB")
            draw = ImageDraw.Draw(vis)
            draw.rectangle([int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])], outline=(255, 0, 0), width=4)
            os.makedirs(os.path.dirname(args.preview) or ".", exist_ok=True)
            vis.save(args.preview)
            payload["preview"] = args.preview

        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "compute_bar_height":
        img = Image.open(args.image)

        x_tickers = getattr(args, "x_axis_ticker", None)
        y_tickers = getattr(args, "y_axis_ticker", None)

        if getattr(args, "metadata", None):
            try:
                with open(args.metadata, "r") as f:
                    meta_json = json.load(f)
                if isinstance(meta_json, dict):
                    if (x_tickers is None or not x_tickers) and isinstance(
                        meta_json.get("x_axis_ticker_values"), list
                    ):
                        x_tickers = meta_json.get("x_axis_ticker_values")
                    if (y_tickers is None or not y_tickers) and isinstance(
                        meta_json.get("y_axis_ticker_values"), list
                    ):
                        y_tickers = meta_json.get("y_axis_ticker_values")
                    # Best-effort: right-y tickers (if present) overwrite y when bar_orientation asks for right axis.
                    if str(getattr(args, "bar_orientation", "")) == "vertical-right" and isinstance(
                        meta_json.get("right_y_axis_ticker_values"), list
                    ):
                        if meta_json.get("right_y_axis_ticker_values"):
                            y_tickers = meta_json.get("right_y_axis_ticker_values")
            except Exception:
                pass

        bar = tuple(float(v) for v in list(getattr(args, "bar")))
        val = compute_bar_height(
            img,
            bar_of_interest=(bar[0], bar[1], bar[2], bar[3]),
            bar_orientation=str(getattr(args, "bar_orientation", "vertical") or "vertical"),
            axis_threshold=float(getattr(args, "axis_threshold", 0.15) or 0.15),
            x_axis_tickers=x_tickers,
            y_axis_tickers=y_tickers,
            x_axis_title=None,
            y_axis_title=None,
        )

        payload = {
            "tool": "compute_bar_height",
            "image": args.image,
            "bar": [float(v) for v in bar],
            "bar_orientation": str(getattr(args, "bar_orientation", "vertical") or "vertical"),
            "axis_threshold": float(getattr(args, "axis_threshold", 0.15) or 0.15),
            "x_axis_tickers_used": x_tickers if x_tickers is not None else None,
            "y_axis_tickers_used": y_tickers if y_tickers is not None else None,
            "value": float(val),
        }

        if getattr(args, "output", None):
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "get_boxplot":
        import numpy as np

        img = Image.open(args.image)

        npz = np.load(args.masks_npz)
        mask_stack = np.asarray(npz.get("masks"))
        bboxes = np.asarray(npz.get("bboxes")) if npz.get("bboxes") is not None else None
        scores = np.asarray(npz.get("scores")) if npz.get("scores") is not None else None
        ids = npz.get("ids")
        if ids is None:
            ids_arr = np.arange(mask_stack.shape[0], dtype=np.int32) + 1
        else:
            ids_arr = np.asarray(ids).astype(np.int32)

        if mask_stack.ndim != 3:
            raise ValueError("--masks-npz must contain `masks` with shape (N,H,W)")
        if ids_arr.ndim != 1 or ids_arr.shape[0] != mask_stack.shape[0]:
            raise ValueError("--masks-npz `ids` must have shape (N,) matching masks")

        masks: List[dict] = []
        for i in range(int(mask_stack.shape[0])):
            bb = None
            if bboxes is not None and bboxes.ndim == 2 and bboxes.shape[1] == 4:
                bb = tuple(int(v) for v in list(bboxes[i]))
            sc = None
            if scores is not None and scores.ndim == 1 and scores.shape[0] == mask_stack.shape[0]:
                sc = float(scores[i])
            masks.append(
                {
                    "id": int(ids_arr[i]),
                    "bbox_xyxy": bb,
                    "score": sc,
                    "segmentation": np.asarray(mask_stack[i]).astype(bool),
                }
            )

        rgb = tuple(getattr(args, "rgb")) if getattr(args, "rgb", None) else None
        selected = get_boxplot(
            img,
            masks=masks,
            rgb_of_interest=rgb,
            ticker_label=getattr(args, "ticker_label", None),
            box_labels_of_interest=list(getattr(args, "box_label")) if getattr(args, "box_label", None) else None,
            boxplot_orientation=str(getattr(args, "boxplot_orientation", "vertical") or "vertical"),
            axis_threshold=float(getattr(args, "axis_threshold", 0.15) or 0.15),
        )

        # Best-effort: map bboxes back to ids for debug output.
        bbox_to_id = {}
        for m in masks:
            bb = m.get("bbox_xyxy")
            if isinstance(bb, tuple) and len(bb) == 4 and int(m.get("id", 0) or 0) > 0:
                bbox_to_id[tuple(int(v) for v in bb)] = int(m.get("id", 0) or 0)
        selected_ids = [bbox_to_id.get(tuple(bb)) for bb in selected]

        payload = {
            "tool": "get_boxplot",
            "image": args.image,
            "masks_npz": getattr(args, "masks_npz", None),
            "rgb_of_interest": list(rgb) if rgb is not None else None,
            "ticker_label": getattr(args, "ticker_label", None),
            "box_labels_of_interest": list(getattr(args, "box_label")) if getattr(args, "box_label", None) else None,
            "boxplot_orientation": str(getattr(args, "boxplot_orientation", "vertical") or "vertical"),
            "axis_threshold": float(getattr(args, "axis_threshold", 0.15) or 0.15),
            "selected_ids": selected_ids,
            "bboxes_xyxy": [[int(v) for v in bb] for bb in selected],
        }

        if getattr(args, "output", None):
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        if getattr(args, "preview", None):
            vis = img.convert("RGB")
            draw = ImageDraw.Draw(vis)
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 18)
            except Exception:
                font = ImageFont.load_default()
            for i, bb in enumerate(selected, start=1):
                draw.rectangle([int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])], outline=(255, 0, 0), width=3)
                draw.text((int(bb[0]) + 2, int(bb[1]) + 2), str(i), fill=(255, 0, 0), font=font)
            os.makedirs(os.path.dirname(args.preview) or ".", exist_ok=True)
            vis.save(args.preview)
            payload["preview"] = args.preview

        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "compute_boxplot_entity":
        img = Image.open(args.image)

        x_tickers = getattr(args, "x_axis_ticker", None)
        y_tickers = getattr(args, "y_axis_ticker", None)

        if getattr(args, "metadata", None):
            try:
                with open(args.metadata, "r") as f:
                    meta_json = json.load(f)
                if isinstance(meta_json, dict):
                    if (x_tickers is None or not x_tickers) and isinstance(
                        meta_json.get("x_axis_ticker_values"), list
                    ):
                        x_tickers = meta_json.get("x_axis_ticker_values")
                    if (y_tickers is None or not y_tickers) and isinstance(
                        meta_json.get("y_axis_ticker_values"), list
                    ):
                        y_tickers = meta_json.get("y_axis_ticker_values")
            except Exception:
                pass

        bboxes = None
        if getattr(args, "boxplot_json", None):
            with open(args.boxplot_json, "r") as f:
                box_json = json.load(f)
            if not (isinstance(box_json, dict) and isinstance(box_json.get("bboxes_xyxy"), list)):
                raise ValueError("--boxplot-json must contain a JSON dict with key `bboxes_xyxy` (list)")
            bboxes = []
            for bb in box_json.get("bboxes_xyxy"):
                if isinstance(bb, (list, tuple)) and len(bb) == 4:
                    bboxes.append((int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])))

        if bboxes is None:
            raw = getattr(args, "bbox", None)
            if not raw:
                raise ValueError("Provide either --boxplot-json or at least one --bbox")
            bboxes = [(int(v[0]), int(v[1]), int(v[2]), int(v[3])) for v in raw]

        val = compute_boxplot_entity(
            img,
            boxplot_of_interest=bboxes,
            boxplot_orientation=str(getattr(args, "boxplot_orientation", "vertical") or "vertical"),
            entity_of_interest=str(getattr(args, "entity", "median") or "median"),
            axis_threshold=float(getattr(args, "axis_threshold", 0.15) or 0.15),
            x_axis_tickers=x_tickers,
            y_axis_tickers=y_tickers,
        )

        payload = {
            "tool": "compute_boxplot_entity",
            "image": args.image,
            "boxplot_orientation": str(getattr(args, "boxplot_orientation", "vertical") or "vertical"),
            "entity_of_interest": str(getattr(args, "entity", "median") or "median"),
            "axis_threshold": float(getattr(args, "axis_threshold", 0.15) or 0.15),
            "x_axis_tickers_used": x_tickers if x_tickers is not None else None,
            "y_axis_tickers_used": y_tickers if y_tickers is not None else None,
            "boxplot_json": getattr(args, "boxplot_json", None),
            "boxplot_bboxes": [[int(v) for v in bb] for bb in bboxes],
            "value": float(val),
        }

        if getattr(args, "output", None):
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "get_edgepoints":
        import numpy as np

        img = Image.open(args.image)

        masks = None
        if getattr(args, "masks_npz", None):
            npz = np.load(args.masks_npz)
            mask_stack = np.asarray(npz.get("masks"))
            bboxes = np.asarray(npz.get("bboxes")) if npz.get("bboxes") is not None else None
            ids = npz.get("ids")
            if ids is None:
                ids_arr = np.arange(mask_stack.shape[0], dtype=np.int32) + 1
            else:
                ids_arr = np.asarray(ids).astype(np.int32)

            if mask_stack.ndim != 3:
                raise ValueError("--masks-npz must contain `masks` with shape (N,H,W)")
            if ids_arr.ndim != 1 or ids_arr.shape[0] != mask_stack.shape[0]:
                raise ValueError("--masks-npz `ids` must have shape (N,) matching masks")

            masks = []
            for i in range(int(mask_stack.shape[0])):
                bb = None
                if bboxes is not None and bboxes.ndim == 2 and bboxes.shape[1] == 4:
                    bb = tuple(int(v) for v in list(bboxes[i]))
                masks.append(
                    {
                        "id": int(ids_arr[i]),
                        "bbox_xyxy": bb,
                        "segmentation": np.asarray(mask_stack[i]).astype(bool),
                    }
                )

        if getattr(args, "mask_label", None) and not getattr(args, "masks_npz", None):
            raise ValueError("--mask-label requires --masks-npz")

        rgb = tuple(getattr(args, "rgb")) if getattr(args, "rgb", None) else None
        pts = get_edgepoints(
            img,
            masks=masks,
            rgb_of_interest=rgb,
            ticker_label=getattr(args, "ticker_label", None),
            mask_labels_of_interest=list(getattr(args, "mask_label")) if getattr(args, "mask_label", None) else None,
            chart_orientation=str(getattr(args, "chart_orientation", "vertical") or "vertical"),
            lineplot_get_dot=bool(getattr(args, "lineplot_get_dot", False)),
            axis_threshold=float(getattr(args, "axis_threshold", 0.15) or 0.15),
        )

        payload = {
            "tool": "get_edgepoints",
            "image": args.image,
            "masks_npz": getattr(args, "masks_npz", None),
            "rgb_of_interest": list(rgb) if rgb is not None else None,
            "ticker_label": getattr(args, "ticker_label", None),
            "mask_labels_of_interest": list(getattr(args, "mask_label")) if getattr(args, "mask_label", None) else None,
            "chart_orientation": str(getattr(args, "chart_orientation", "vertical") or "vertical"),
            "lineplot_get_dot": bool(getattr(args, "lineplot_get_dot", False)),
            "axis_threshold": float(getattr(args, "axis_threshold", 0.15) or 0.15),
            "edge_points": [[int(p[0]), int(p[1])] for p in pts],
        }

        if getattr(args, "output", None):
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        if getattr(args, "preview", None):
            vis = img.convert("RGB")
            draw = ImageDraw.Draw(vis)
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 18)
            except Exception:
                font = ImageFont.load_default()
            for i, (x, y) in enumerate(pts, start=1):
                r = 6
                draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 0), width=3)
                draw.text((x + 6, y + 2), str(i), fill=(255, 0, 0), font=font)
            os.makedirs(os.path.dirname(args.preview) or ".", exist_ok=True)
            vis.save(args.preview)
            payload["preview"] = args.preview

        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "get_radial":
        img = Image.open(args.image)
        rgb = tuple(getattr(args, "rgb")) if getattr(args, "rgb", None) else None
        bb = get_radial(
            img,
            rgb_of_interest=rgb,
            ticker_label=getattr(args, "ticker_label", None),
            segmentation_model=str(getattr(args, "segmentation_model", "color") or "color"),
        )

        payload = {
            "tool": "get_radial",
            "image": args.image,
            "rgb_of_interest": list(rgb) if rgb is not None else None,
            "ticker_label": getattr(args, "ticker_label", None),
            "segmentation_model": str(getattr(args, "segmentation_model", "color") or "color"),
            "bbox_xyxy": [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])],
        }

        if getattr(args, "output", None):
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        if getattr(args, "preview", None):
            vis = img.convert("RGB")
            draw = ImageDraw.Draw(vis)
            draw.rectangle([int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])], outline=(255, 0, 0), width=4)
            os.makedirs(os.path.dirname(args.preview) or ".", exist_ok=True)
            vis.save(args.preview)
            payload["preview"] = args.preview

        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "analyze_radial_geometry":
        import numpy as np

        try:
            import cv2  # type: ignore
        except Exception as e:
            raise RuntimeError("analyze_radial_geometry requires opencv-python (cv2).") from e

        img = Image.open(args.image).convert("RGB")
        W, H = img.size

        bbox = None
        if getattr(args, "radial_json", None):
            with open(args.radial_json, "r") as f:
                radial_json = json.load(f)
            if isinstance(radial_json, dict):
                bb0 = radial_json.get("bbox_xyxy")
                if isinstance(bb0, (list, tuple)) and len(bb0) == 4:
                    bbox = (int(bb0[0]), int(bb0[1]), int(bb0[2]), int(bb0[3]))

        if bbox is None and getattr(args, "bbox", None) is not None:
            bb1 = list(getattr(args, "bbox"))
            if len(bb1) == 4:
                bbox = (int(bb1[0]), int(bb1[1]), int(bb1[2]), int(bb1[3]))

        if bbox is None:
            raise ValueError("Provide --bbox or --radial-json (with bbox_xyxy).")

        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(x1), W))
        y1 = max(0, min(int(y1), H))
        x2 = max(0, min(int(x2), W))
        y2 = max(0, min(int(y2), H))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid bbox after clipping")

        rgb = tuple(getattr(args, "rgb")) if getattr(args, "rgb", None) else None

        roi = img.crop((x1, y1, x2, y2))
        arr = np.asarray(roi, dtype=np.uint8)
        arr_i16 = arr.astype(np.int16)

        # Background estimate (corner median) within ROI.
        hh, ww = arr.shape[:2]
        k = max(3, min(hh, ww) // 30)
        corners = np.concatenate(
            [
                arr[0:k, 0:k, :].reshape(-1, 3),
                arr[0:k, ww - k : ww, :].reshape(-1, 3),
                arr[hh - k : hh, 0:k, :].reshape(-1, 3),
                arr[hh - k : hh, ww - k : ww, :].reshape(-1, 3),
            ],
            axis=0,
        )
        med = np.median(corners, axis=0).astype(np.int16)

        if rgb is not None:
            tgt = np.asarray([int(rgb[0]), int(rgb[1]), int(rgb[2])], dtype=np.int16)
            tgt = np.clip(tgt, 0, 255)
            diff_t = np.abs(arr_i16 - tgt.reshape(1, 1, 3)).sum(axis=2)
            diff_bg = np.abs(arr_i16 - med.reshape(1, 1, 3)).sum(axis=2)
            tol = 90
            mask = np.logical_and(diff_t <= int(tol), diff_bg > 25)
            if int(mask.sum()) < 30:
                tol = 120
                mask = np.logical_and(diff_t <= int(tol), diff_bg > 25)
        else:
            diff_bg = np.abs(arr_i16 - med.reshape(1, 1, 3)).sum(axis=2)
            mask = diff_bg > 30

        mask_u8 = (mask.astype(np.uint8) * 255)
        kk = max(3, int(round(0.03 * float(max(ww, hh)))))
        kk = min(kk, 9)
        kernel = np.ones((kk, kk), dtype=np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError("Failed to extract a contour within bbox; try providing --rgb.")
        cnt = max(contours, key=cv2.contourArea)
        if cnt is None or cnt.size == 0:
            raise RuntimeError("Empty contour extracted within bbox")
        cnt_full = cnt.copy()
        cnt_full[:, 0, 0] = cnt_full[:, 0, 0] + int(x1)
        cnt_full[:, 0, 1] = cnt_full[:, 0, 1] + int(y1)

        vis_img, cx, cy, r_outer, r_max = analyze_radial_geometry(img, contour_of_interest=cnt_full)

        payload = {
            "tool": "analyze_radial_geometry",
            "image": args.image,
            "radial_json": getattr(args, "radial_json", None),
            "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "rgb_of_interest": list(rgb) if rgb is not None else None,
            "center_x": int(cx),
            "center_y": int(cy),
            "r_outer": float(r_outer),
            "r_max": float(r_max),
        }

        if getattr(args, "output", None):
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        if getattr(args, "preview", None):
            os.makedirs(os.path.dirname(args.preview) or ".", exist_ok=True)
            vis_img.save(args.preview)
            payload["preview"] = args.preview

        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "estimate_radial_value":
        img = Image.open(args.image).convert("RGB")

        cx = getattr(args, "center_x", None)
        cy = getattr(args, "center_y", None)
        r_outer = getattr(args, "r_outer", None)
        r_max = getattr(args, "r_max", None)

        if getattr(args, "geometry_json", None):
            with open(args.geometry_json, "r") as f:
                geo_json = json.load(f)
            if not isinstance(geo_json, dict):
                raise ValueError("--geometry-json must be a JSON dict")
            cx = geo_json.get("center_x", cx)
            cy = geo_json.get("center_y", cy)
            r_outer = geo_json.get("r_outer", r_outer)
            r_max = geo_json.get("r_max", r_max)

        if cx is None or cy is None or r_outer is None or r_max is None:
            raise ValueError(
                "Provide either --geometry-json or all of --center-x --center-y --r-outer --r-max."
            )

        val = estimate_radial_value(
            img,
            center_x=int(cx),
            center_y=int(cy),
            r_outer=int(round(float(r_outer))),
            r_max=int(round(float(r_max))),
            reference_circle_value=float(getattr(args, "reference_circle_value", 100.0)),
        )

        payload = {
            "tool": "estimate_radial_value",
            "image": args.image,
            "geometry_json": getattr(args, "geometry_json", None),
            "center_x": int(cx),
            "center_y": int(cy),
            "r_outer": float(r_outer),
            "r_max": float(r_max),
            "reference_circle_value": float(getattr(args, "reference_circle_value", 100.0)),
            "value": float(val),
        }

        if getattr(args, "output", None):
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    raise SystemExit("Unknown command: {}".format(args.command))


if __name__ == "__main__":
    main()
