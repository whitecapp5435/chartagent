import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class MisvizRecord:
    image_rel_path: str
    split: str
    chart_type: List[str]
    misleader: List[str]


def _load_misviz_records(path: str) -> List[MisvizRecord]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("misviz.json must be a list")

    out: List[MisvizRecord] = []
    for rec in data:
        if not isinstance(rec, dict):
            continue
        image_rel_path = str(rec.get("image_path", "") or "").strip()
        split = str(rec.get("split", "") or "").strip()
        if not image_rel_path or not split:
            continue
        chart_type_raw = rec.get("chart_type")
        misleader_raw = rec.get("misleader")
        chart_type = (
            [str(x) for x in chart_type_raw if str(x).strip()]
            if isinstance(chart_type_raw, list)
            else []
        )
        misleader = (
            [str(x) for x in misleader_raw if str(x).strip()]
            if isinstance(misleader_raw, list)
            else []
        )
        out.append(
            MisvizRecord(
                image_rel_path=image_rel_path,
                split=split,
                chart_type=chart_type,
                misleader=misleader,
            )
        )
    return out


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--misviz-json", default="misviz/misviz.json")
    ap.add_argument("--misviz-dir", default="misviz")
    ap.add_argument(
        "--splits",
        default="dev,val",
        help='Comma-separated splits to include (default: "dev,val")',
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output JSONL manifest. One line per *existing* image record.",
    )
    args = ap.parse_args()

    splits = {s.strip() for s in str(args.splits).split(",") if s.strip()}
    if not splits:
        raise SystemExit("--splits must include at least one split")
    if "test" in splits:
        raise SystemExit('Do not include "test" split for this experiment.')

    recs = _load_misviz_records(args.misviz_json)

    rows: List[Dict[str, Any]] = []
    missing = 0
    for r in recs:
        if r.split not in splits:
            continue
        abs_path = os.path.join(args.misviz_dir, r.image_rel_path)
        if not (os.path.exists(abs_path) and os.path.isfile(abs_path)):
            missing += 1
            continue
        rows.append(
            {
                "image_path": abs_path,
                "image_rel_path": r.image_rel_path,
                "split": r.split,
                "y_true": r.misleader,
                "chart_type": r.chart_type,
            }
        )

    _write_jsonl(args.out, rows)
    print(f"Wrote {len(rows)} rows to {args.out}")
    if missing:
        print(f"Skipped {missing} records with missing image files")


if __name__ == "__main__":
    main()

