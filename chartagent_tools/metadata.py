import base64
import os
from io import BytesIO
from typing import Any, Dict, List, Optional

from PIL import Image

# Optional: load `OPENAI_API_KEY` from a local `.env` if python-dotenv is installed.
try:
    from dotenv import find_dotenv, load_dotenv  # type: ignore

    load_dotenv(find_dotenv())
except Exception:
    pass


CHART_METADATA_PROMPT = """Instruction:
You are a vision-language model tasked with analyzing a data visualization chart image.
Extract and return the following information as a JSON dictionary using the exact keys specified below.

Important rules:
- For all *_axis_ticker_values fields, list ALL tick labels that are explicitly written on the chart (do not abbreviate, subsample, or skip values).
  - Do NOT use ranges (e.g., "0-120") or ellipses (e.g., "0, 20, ..., 120").
  - Keep the original order as it appears on the axis (x: left→right, y: bottom→top).
  - Return tick labels as strings exactly as shown (verbatim). If a particular tick label is unreadable, omit that label (do not guess or infer missing labels).
  - Do NOT normalize or expand tick labels. Do NOT convert abbreviated forms into more explicit forms. Examples:
      - "’23" must stay "’23" (do not convert to "2023")
      - "Q3" must stay "Q3" (do not convert to "Quarter 3")
      - "1.5K" must stay "1.5K" (do not convert to "1500")
      - "0%" must stay "0%" (do not convert to "0")
      - "1,000" must stay "1,000" (do not convert to "1000")

  - If the figure contains multiple panels/insets with separate x-axes, include x-axis ticks from ALL panels.
    Put them into a single `x_axis_ticker_values` list in left→right reading order across the whole figure.
    In `visual_description`, explicitly describe the panel boundary (e.g., a divider line or inset position) and which tick labels belong to each panel.

- chart_type: e.g., pie chart, multi-ring pie chart, bar chart, line chart, box plot, etc.
- title: Exact chart title as shown.
- legend: List of all legend entries (strings). If no legend is present, return an empty list [].
- highlevel_legend_categories and finegrained_legend_subcategories: If the chart shows category hierarchy, list both, even if names overlap.
- legend_embedded: true if legend is within the chart; false if outside.
- x_axis_label / y_axis_label / right_y_axis_label / color_bar_label: Axis labels (strings). May be empty.
- x_axis_ticker_values / y_axis_ticker_values / right_y_axis_ticker_values / color_bar_ticker_values / radial_axis_ticker_values: Tick values (List). May be empty.
- annotation_type: Either "annotated" or "unannotated".
- visual_description: Concise summary of the chart’s visual structure.

Only output the JSON object."""


def empty_chart_metadata() -> Dict[str, Any]:
    return {
        "chart_type": "",
        "title": "",
        "legend": [],
        "highlevel_legend_categories": [],
        "finegrained_legend_subcategories": [],
        "legend_embedded": False,
        "x_axis_label": "",
        "y_axis_label": "",
        "right_y_axis_label": "",
        "color_bar_label": "",
        "x_axis_ticker_values": [],
        "y_axis_ticker_values": [],
        "right_y_axis_ticker_values": [],
        "color_bar_ticker_values": [],
        "radial_axis_ticker_values": [],
        "annotation_type": "unannotated",
        "visual_description": "",
    }


def validate_chart_metadata(meta: Dict[str, Any], strict: bool = True) -> Dict[str, Any]:
    """
    Validates and normalizes chart metadata.

    - strict=True: raises ValueError if keys are missing or types are incompatible.
    - strict=False: fills missing keys with defaults.
    """
    if not isinstance(meta, dict):
        raise TypeError("metadata must be a dict")

    base = empty_chart_metadata()
    out = dict(base)

    missing = []
    for k in base.keys():
        if k not in meta:
            missing.append(k)
        else:
            out[k] = meta[k]

    if strict and missing:
        raise ValueError("metadata missing keys: {}".format(missing))

    # Basic type normalization
    def _as_str(v) -> str:
        return "" if v is None else str(v)

    out["chart_type"] = _as_str(out.get("chart_type"))
    out["title"] = _as_str(out.get("title"))

    legend = out.get("legend")
    if isinstance(legend, dict):
        out["legend"] = legend
    elif isinstance(legend, list):
        out["legend"] = legend
    else:
        if strict:
            raise ValueError("legend must be a list or dict")
        out["legend"] = []

    for k in ["highlevel_legend_categories", "finegrained_legend_subcategories"]:
        v = out.get(k)
        if isinstance(v, list):
            out[k] = v
        else:
            if strict:
                raise ValueError("{} must be a list".format(k))
            out[k] = []

    out["legend_embedded"] = bool(out.get("legend_embedded"))

    for k in ["x_axis_label", "y_axis_label", "right_y_axis_label", "color_bar_label"]:
        out[k] = _as_str(out.get(k))

    for k in [
        "x_axis_ticker_values",
        "y_axis_ticker_values",
        "right_y_axis_ticker_values",
        "color_bar_ticker_values",
        "radial_axis_ticker_values",
    ]:
        v = out.get(k)
        if isinstance(v, list):
            out[k] = v
        else:
            if strict:
                raise ValueError("{} must be a list".format(k))
            out[k] = []

    ann = _as_str(out.get("annotation_type")).lower().strip()
    if ann not in ("annotated", "unannotated"):
        if strict:
            raise ValueError('annotation_type must be "annotated" or "unannotated"')
        ann = "unannotated"
    out["annotation_type"] = ann

    out["visual_description"] = _as_str(out.get("visual_description"))

    return out


def _data_url_from_image(image: Image.Image, image_format: str = "PNG") -> str:
    buf = BytesIO()
    image_format = (image_format or "PNG").upper()
    image.convert("RGB").save(buf, format=image_format)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if image_format == "PNG" else "image/jpeg"
    return "data:{};base64,{}".format(mime, b64)


def _require_openai_client(base_url: Optional[str] = None):
    """
    Lazily import the OpenAI Python SDK (new) and return an initialized client.

    This matches the user-provided API usage:
      from openai import OpenAI
      client = OpenAI()
      client.responses.create / client.responses.parse
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "OpenAI Python SDK is required (pip install --upgrade openai) and Python 3.8+."
        ) from e
    if base_url:
        return OpenAI(base_url=base_url)
    return OpenAI()


def _pydantic_to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()  # pydantic v2
    if hasattr(obj, "dict"):
        return obj.dict()  # pydantic v1
    if isinstance(obj, dict):
        return obj
    raise TypeError("Unsupported parsed object type: {}".format(type(obj)))


def extract_chart_metadata_openai(
    image: Image.Image,
    model: str = "gpt-5-nano",
    base_url: Optional[str] = None,
    strict: bool = True,
    image_format: str = "PNG",
) -> Dict[str, Any]:
    """
    Extract chart metadata using OpenAI Responses API with Structured Outputs.

    Requires:
      - `OPENAI_API_KEY` in env
      - new `openai` SDK (`OpenAI`, `client.responses.parse`)
      - `pydantic` (Structured Outputs `text_format`)
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = _require_openai_client(base_url=base_url)

    try:
        from pydantic import BaseModel  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "pydantic is required for structured outputs (pip install pydantic)."
        ) from e

    class ChartMetadataExtraction(BaseModel):
        chart_type: str
        title: str
        legend: List[str]
        highlevel_legend_categories: List[str]
        finegrained_legend_subcategories: List[str]
        legend_embedded: bool
        x_axis_label: str
        y_axis_label: str
        right_y_axis_label: str
        color_bar_label: str
        x_axis_ticker_values: List[str]
        y_axis_ticker_values: List[str]
        right_y_axis_ticker_values: List[str]
        color_bar_ticker_values: List[str]
        radial_axis_ticker_values: List[str]
        annotation_type: str
        visual_description: str

    img_url = _data_url_from_image(image, image_format=image_format)

    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": CHART_METADATA_PROMPT},
                    {"type": "input_image", "image_url": img_url},
                ],
            }
        ],
        text_format=ChartMetadataExtraction,
    )

    parsed = getattr(response, "output_parsed", None)
    meta = _pydantic_to_dict(parsed)
    return validate_chart_metadata(meta, strict=strict)


def extract_chart_metadata(
    image: Image.Image,
    strict: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Default entrypoint: OpenAI-only (no heuristic fallback).
    """
    return extract_chart_metadata_openai(image=image, strict=strict, **kwargs)
