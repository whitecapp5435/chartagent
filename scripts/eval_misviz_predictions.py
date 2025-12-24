import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, cast

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from chartagent.misviz_schema import MISVIZ_MISLEADER_LABELS, normalize_misleader_labels, parse_manifest_line
from chartagent.multilabel_metrics import multilabel_counts, multilabel_score


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Manifest JSONL (dev/val).")
    ap.add_argument("--predictions", required=True, help="Predictions JSONL (image_path -> y_pred).")
    ap.add_argument("--out", required=True, help="Output JSON summary path.")
    ap.add_argument("--allow-missing", action="store_true", help="Evaluate on intersection only.")
    args = ap.parse_args()

    manifest = _load_manifest(args.manifest)
    preds, errors = _load_predictions(args.predictions)

    missing = [p for p in manifest.keys() if p not in preds]
    if missing and not args.allow_missing:
        raise SystemExit(
            "Missing predictions for {} / {} images. Re-run with --allow-missing to evaluate partial runs.".format(
                len(missing), len(manifest)
            )
        )

    matched: List[str] = [p for p in manifest.keys() if p in preds]
    y_true_all: List[List[str]] = [manifest[p]["y_true"] for p in matched]
    y_pred_all: List[List[str]] = [preds[p] for p in matched]

    splits = ["dev", "val"]
    by_split: Dict[str, Dict[str, Any]] = {}
    for s in splits:
        idx = [i for i, p in enumerate(matched) if manifest[p]["split"] == s]
        yt = _sets([y_true_all[i] for i in idx])
        yp = _sets([y_pred_all[i] for i in idx])
        by_split[s] = {
            "n": len(idx),
            "metrics": multilabel_score(yt, yp, MISVIZ_MISLEADER_LABELS) if idx else {},
        }

    counts = multilabel_counts(_sets(y_true_all), _sets(y_pred_all), MISVIZ_MISLEADER_LABELS) if matched else {}
    per_label = {
        k: {"tp": v.tp, "fp": v.fp, "fn": v.fn}
        for k, v in counts.items()
    }

    # Failure summary (best-effort): count error strings emitted by the runner/agent.
    matched_errors: List[str] = [errors[p] for p in matched if p in errors]
    err_counts: Dict[str, int] = {}
    for e in matched_errors:
        err_counts[e] = err_counts.get(e, 0) + 1
    top_errors = sorted(err_counts.items(), key=lambda kv: kv[1], reverse=True)[:20]

    summary: Dict[str, Any] = {
        "manifest": args.manifest,
        "predictions": args.predictions,
        "n_manifest": len(manifest),
        "n_matched": len(matched),
        "n_missing": len(missing),
        "n_with_error": len(matched_errors),
        "top_errors": [{"error": e, "n": n} for (e, n) in top_errors],
        "metrics": multilabel_score(_sets(y_true_all), _sets(y_pred_all), MISVIZ_MISLEADER_LABELS) if matched else {},
        "by_split": by_split,
        "per_label_counts": per_label,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Wrote summary:", args.out)
    print("Matched:", len(matched), "/", len(manifest), "Missing:", len(missing))
    if summary.get("metrics"):
        m = cast(Dict[str, Any], summary["metrics"])
        keys = ["accuracy", "precision", "recall", "exact_match", "partial_match", "micro_f1", "macro_f1"]
        line = ", ".join([f"{k}={m.get(k):.4f}" for k in keys if isinstance(m.get(k), (int, float))])
        if line:
            print("Overall:", line)
        for s in splits:
            sm = by_split.get(s, {}).get("metrics") or {}
            if isinstance(sm, dict) and sm:
                line = ", ".join([f"{k}={sm.get(k):.4f}" for k in keys if isinstance(sm.get(k), (int, float))])
                if line:
                    print(f"{s}:", line)


if __name__ == "__main__":
    main()
