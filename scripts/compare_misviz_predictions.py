import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from chartagent.misviz_schema import MISVIZ_MISLEADER_LABELS, normalize_misleader_labels, parse_manifest_line
from chartagent.multilabel_metrics import multilabel_score


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def _load_manifest(path: str) -> Dict[str, Dict[str, Any]]:
    manifest: Dict[str, Dict[str, Any]] = {}
    for obj in _iter_jsonl(path):
        ex = parse_manifest_line(obj)
        manifest[ex.image_path] = {
            "split": ex.split,
            "y_true": ex.y_true,
            "chart_type": ex.chart_type,
        }
    return manifest


def _load_predictions(path: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    preds: Dict[str, List[str]] = {}
    errors: Dict[str, str] = {}
    for obj in _iter_jsonl(path):
        image_path = str(obj.get("image_path", "") or "").strip()
        if not image_path:
            continue
        y_pred_raw = obj.get("y_pred") or []
        if isinstance(y_pred_raw, list):
            y_pred = normalize_misleader_labels([str(x) for x in y_pred_raw], strict=False)
        else:
            y_pred = []
        preds[image_path] = y_pred
        err = obj.get("error")
        if err is not None and str(err).strip():
            errors[image_path] = str(err)
    return preds, errors


def _sets(xs: List[List[str]]) -> List[Set[str]]:
    return [set(x) for x in xs]


def _fmt_metrics(m: Dict[str, Any]) -> str:
    keys = ["accuracy", "precision", "recall", "exact_match", "partial_match", "micro_f1", "macro_f1"]
    parts: List[str] = []
    for k in keys:
        v = m.get(k)
        if isinstance(v, (int, float)):
            parts.append(f"{k}={float(v):.4f}")
    return ", ".join(parts)


def _delta_metrics(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    keys = ["accuracy", "precision", "recall", "exact_match", "partial_match", "micro_f1", "macro_f1"]
    out: Dict[str, float] = {}
    for k in keys:
        va = a.get(k)
        vb = b.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            out[k] = float(vb) - float(va)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pred-a", required=True, help="Predictions JSONL A (e.g., agent).")
    ap.add_argument("--pred-b", required=True, help="Predictions JSONL B (e.g., zeroshot).")
    ap.add_argument("--out", default="", help="Optional JSON output path for the comparison summary.")
    ap.add_argument("--allow-missing", action="store_true", help="Evaluate on intersection only.")
    args = ap.parse_args()

    manifest = _load_manifest(str(args.manifest))
    preds_a, errors_a = _load_predictions(str(args.pred_a))
    preds_b, errors_b = _load_predictions(str(args.pred_b))

    missing_a = [p for p in manifest.keys() if p not in preds_a]
    missing_b = [p for p in manifest.keys() if p not in preds_b]
    if (missing_a or missing_b) and not bool(args.allow_missing):
        raise SystemExit(
            "Missing predictions. A missing: {} / {}, B missing: {} / {}. Re-run with --allow-missing.".format(
                len(missing_a), len(manifest), len(missing_b), len(manifest)
            )
        )

    matched = [p for p in manifest.keys() if p in preds_a and p in preds_b]
    splits = ["dev", "val"]

    y_true_all: List[List[str]] = [manifest[p]["y_true"] for p in matched]
    y_a_all: List[List[str]] = [preds_a[p] for p in matched]
    y_b_all: List[List[str]] = [preds_b[p] for p in matched]

    m_a = multilabel_score(_sets(y_true_all), _sets(y_a_all), MISVIZ_MISLEADER_LABELS) if matched else {}
    m_b = multilabel_score(_sets(y_true_all), _sets(y_b_all), MISVIZ_MISLEADER_LABELS) if matched else {}

    by_split: Dict[str, Dict[str, Any]] = {}
    for s in splits:
        idx = [i for i, p in enumerate(matched) if manifest[p]["split"] == s]
        yt = _sets([y_true_all[i] for i in idx])
        ya = _sets([y_a_all[i] for i in idx])
        yb = _sets([y_b_all[i] for i in idx])
        by_split[s] = {
            "n": len(idx),
            "a": multilabel_score(yt, ya, MISVIZ_MISLEADER_LABELS) if idx else {},
            "b": multilabel_score(yt, yb, MISVIZ_MISLEADER_LABELS) if idx else {},
        }

    # Example-level buckets (exact/partial match)
    def _is_exact(t: Set[str], p: Set[str]) -> bool:
        return set(t) == set(p)

    def _is_partial(t: Set[str], p: Set[str]) -> bool:
        if not t and not p:
            return True
        return bool(set(t).intersection(p))

    buckets: Dict[str, int] = {
        "exact_both": 0,
        "exact_only_a": 0,
        "exact_only_b": 0,
        "exact_neither": 0,
        "partial_both": 0,
        "partial_only_a": 0,
        "partial_only_b": 0,
        "partial_neither": 0,
    }
    yt_sets = _sets(y_true_all)
    ya_sets = _sets(y_a_all)
    yb_sets = _sets(y_b_all)
    for t, pa, pb in zip(yt_sets, ya_sets, yb_sets):
        ea = _is_exact(t, pa)
        eb = _is_exact(t, pb)
        if ea and eb:
            buckets["exact_both"] += 1
        elif ea and not eb:
            buckets["exact_only_a"] += 1
        elif (not ea) and eb:
            buckets["exact_only_b"] += 1
        else:
            buckets["exact_neither"] += 1

        ra = _is_partial(t, pa)
        rb = _is_partial(t, pb)
        if ra and rb:
            buckets["partial_both"] += 1
        elif ra and not rb:
            buckets["partial_only_a"] += 1
        elif (not ra) and rb:
            buckets["partial_only_b"] += 1
        else:
            buckets["partial_neither"] += 1

    summary: Dict[str, Any] = {
        "manifest": str(args.manifest),
        "pred_a": str(args.pred_a),
        "pred_b": str(args.pred_b),
        "n_manifest": len(manifest),
        "n_matched": len(matched),
        "n_missing_a": len(missing_a),
        "n_missing_b": len(missing_b),
        "n_with_error_a": sum(1 for p in matched if p in errors_a),
        "n_with_error_b": sum(1 for p in matched if p in errors_b),
        "metrics_a": m_a,
        "metrics_b": m_b,
        "metrics_delta_b_minus_a": _delta_metrics(m_a, m_b),
        "by_split": by_split,
        "buckets": buckets,
    }

    print("A:", args.pred_a)
    print("B:", args.pred_b)
    print("Matched:", len(matched), "/", len(manifest))
    if m_a and m_b:
        print("A overall:", _fmt_metrics(m_a))
        print("B overall:", _fmt_metrics(m_b))
        delta = summary["metrics_delta_b_minus_a"]
        if isinstance(delta, dict) and delta:
            print("Delta (B-A):", _fmt_metrics(delta))
    for s in splits:
        bs = by_split.get(s, {})
        if not isinstance(bs, dict) or not bs.get("n"):
            continue
        print(f"{s}: n={bs.get('n')}")
        ma = bs.get("a") or {}
        mb = bs.get("b") or {}
        if isinstance(ma, dict) and ma:
            print("  A:", _fmt_metrics(ma))
        if isinstance(mb, dict) and mb:
            print("  B:", _fmt_metrics(mb))

    if str(args.out).strip():
        out_path = str(args.out).strip()
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("Wrote:", out_path)


if __name__ == "__main__":
    main()

