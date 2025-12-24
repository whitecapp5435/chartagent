import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set


# Canonical label set (as used in misviz/misviz.json).
MISVIZ_MISLEADER_LABELS: List[str] = [
    "misrepresentation",
    "3d",
    "truncated axis",
    "inappropriate use of pie chart",
    "inconsistent binning size",
    "discretized continuous variable",
    "inconsistent tick intervals",
    "dual axis",
    "inappropriate use of line chart",
    "inappropriate item order",
    "inverted axis",
    "inappropriate axis range",
]

_NORMALIZED_TO_CANONICAL: Dict[str, str] = {
    " ".join(lbl.split()): lbl for lbl in MISVIZ_MISLEADER_LABELS
}

# Small alias set to tolerate minor wording/casing differences in model outputs.
_ALIASES: Dict[str, str] = {
    "3 d": "3d",
    "3d effect": "3d",
    "3d chart": "3d",
    "truncated y axis": "truncated axis",
    "truncated x axis": "truncated axis",
    "truncated y-axis": "truncated axis",
    "truncated x-axis": "truncated axis",
    "dual y axis": "dual axis",
    "dual y-axis": "dual axis",
    "inverted y axis": "inverted axis",
    "inverted x axis": "inverted axis",
    "inverted y-axis": "inverted axis",
    "inverted x-axis": "inverted axis",
    "inconsistent ticks": "inconsistent tick intervals",
    "inconsistent tick interval": "inconsistent tick intervals",
    "inconsistent tick spacing": "inconsistent tick intervals",
    "inconsistent bin size": "inconsistent binning size",
    "inconsistent binning": "inconsistent binning size",
    "discretized continuous": "discretized continuous variable",
    "discretized continuous values": "discretized continuous variable",
    "inappropriate pie chart": "inappropriate use of pie chart",
    "inappropriate use of pie": "inappropriate use of pie chart",
    "inappropriate line chart": "inappropriate use of line chart",
    "inappropriate use of line": "inappropriate use of line chart",
    "inappropriate axis": "inappropriate axis range",
    "axis range inappropriate": "inappropriate axis range",
    "misrep": "misrepresentation",
    "misrepresentation of data": "misrepresentation",
    "inappropriate item ordering": "inappropriate item order",
    "inappropriate order": "inappropriate item order",
}


def _normalize_label_text(text: str) -> str:
    t = str(text or "").strip().lower()
    t = t.replace("_", " ")
    t = t.replace("-", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_misleader_label(label: str) -> Optional[str]:
    """Map an arbitrary label string to a canonical misviz label (or None)."""
    key = _normalize_label_text(label)
    if not key:
        return None
    if key in _NORMALIZED_TO_CANONICAL:
        return _NORMALIZED_TO_CANONICAL[key]
    if key in _ALIASES:
        return _ALIASES[key]
    return None


def normalize_misleader_labels(labels: Sequence[str], *, strict: bool = False) -> List[str]:
    """Normalize and de-duplicate a list of misleader labels.

    If strict=True, raises ValueError on unknown labels.
    """
    out: Set[str] = set()
    unknown: List[str] = []
    for raw in labels:
        canon = normalize_misleader_label(str(raw))
        if canon is None:
            if str(raw).strip():
                unknown.append(str(raw))
            continue
        out.add(canon)
    if strict and unknown:
        raise ValueError(f"Unknown misleader labels: {unknown}")
    return sorted(out)


@dataclass(frozen=True)
class MisvizExample:
    image_path: str
    split: str
    y_true: List[str]
    chart_type: List[str]


def parse_manifest_line(obj: Dict[str, object]) -> MisvizExample:
    image_path = str(obj.get("image_path", "") or "").strip()
    split = str(obj.get("split", "") or "").strip()
    y_true_raw = obj.get("y_true") or []
    chart_type_raw = obj.get("chart_type") or []
    y_true = (
        normalize_misleader_labels([str(x) for x in y_true_raw], strict=False)
        if isinstance(y_true_raw, list)
        else []
    )
    chart_type = (
        [str(x) for x in chart_type_raw if str(x).strip()]
        if isinstance(chart_type_raw, list)
        else []
    )
    if not image_path:
        raise ValueError("manifest line missing image_path")
    if split not in {"dev", "val"}:
        raise ValueError(f"manifest line has invalid split: {split!r}")
    return MisvizExample(
        image_path=image_path,
        split=split,
        y_true=y_true,
        chart_type=chart_type,
    )


def iter_label_sets(rows: Iterable[Sequence[str]]) -> List[Set[str]]:
    return [set(normalize_misleader_labels(list(r), strict=False)) for r in rows]
