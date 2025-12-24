import os
from typing import Any, Dict, Optional


def major_chart_type_from_metadata(metadata: Optional[Dict[str, Any]]) -> str:
    """
    Map free-form metadata["chart_type"] into a small set of "major" types used for ICL retrieval.

    Returns one of:
      - "bar", "line", "pie", "dot_donut", "boxplot", "radial", "generic"
    """
    if not isinstance(metadata, dict):
        return "generic"
    ct = metadata.get("chart_type")
    if isinstance(ct, list):
        ct = " ".join([str(x) for x in ct])
    if not isinstance(ct, str):
        return "generic"

    s = ct.strip().lower()
    if not s:
        return "generic"

    if ("dot" in s or "dots" in s) and any(k in s for k in ("donut", "ring", "pie")):
        return "dot_donut"
    if any(k in s for k in ("boxplot", "box plot", "box-plot")):
        return "boxplot"
    if "radial" in s:
        return "radial"
    if any(k in s for k in ("pie", "donut", "ring")):
        return "pie"
    if "bar" in s:
        return "bar"
    if any(k in s for k in ("line", "area", "timeseries", "time series")):
        return "line"
    return "generic"


def load_icl_prompt(major_type: str) -> str:
    """
    Load the plain-text ICL examples for a major chart type.

    Files are stored under `chartagent/prompts/icl/<major_type>.txt`.
    If the file does not exist, falls back to `generic.txt` if present.
    """
    mt = str(major_type or "").strip().lower() or "generic"
    base_dir = os.path.join(os.path.dirname(__file__), "prompts", "icl")
    path = os.path.join(base_dir, "{}.txt".format(mt))
    if not os.path.exists(path):
        path = os.path.join(base_dir, "generic.txt")
        if not os.path.exists(path):
            return ""
    with open(path, "r") as f:
        return f.read().strip()

