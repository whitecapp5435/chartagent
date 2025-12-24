import argparse
import json
import os
import sys
import time
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Set

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PIL import Image

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


def _data_url_from_image(image: Image.Image) -> str:
    buf = BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    b64 = __import__("base64").b64encode(buf.getvalue()).decode("ascii")
    return "data:image/png;base64,{}".format(b64)


def _resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = float(max_side) / float(m)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return img.resize((nw, nh), resample=Image.BICUBIC)


def _require_openai_client():
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "OpenAI Python SDK (v1+) is required. Install/upgrade with: pip install --upgrade openai"
        ) from e
    return OpenAI()


def _build_prompt() -> str:
    labels = "\n".join(["- {}".format(x) for x in MISVIZ_MISLEADER_LABELS])
    return (
        "You are a vision-language model. Your task is to classify the chart image into zero or more misleader types.\n"
        "Return only the JSON object matching the provided schema.\n"
        "\n"
        "Important rules:\n"
        "- misleader is a LIST of labels (multi-label). It may be empty.\n"
        "- Use ONLY the canonical labels below (exact spelling).\n"
        "- Do not output 'none' as a label; use an empty list instead.\n"
        "- If you are uncertain, return an empty list and explain uncertainty in rationale.\n"
        "\n"
        "Canonical labels:\n"
        "{}\n".format(labels)
    )


def predict_misleader_for_image(*, client, model: str, image: Image.Image, max_side: int) -> Dict[str, Any]:
    try:
        from pydantic import BaseModel  # type: ignore
    except Exception as e:
        raise RuntimeError("pydantic is required (pip install pydantic).") from e

    class MisleaderPrediction(BaseModel):
        misleader: List[str]
        rationale: str

    img_small = _resize_max_side(image, max_side=max_side)
    img_url = _data_url_from_image(img_small)
    prompt = _build_prompt()

    resp = client.responses.parse(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": img_url},
                ],
            }
        ],
        text_format=MisleaderPrediction,
    )
    parsed = getattr(resp, "output_parsed", None)
    if parsed is None:
        raise RuntimeError("No parsed output returned")
    out = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)  # type: ignore[arg-type]
    y_pred_raw = out.get("misleader", []) or []
    y_pred = normalize_misleader_labels([str(x) for x in y_pred_raw], strict=False)
    return {"y_pred": y_pred, "rationale": str(out.get("rationale", "") or "").strip()}


def _set_list(xs: List[List[str]]) -> List[Set[str]]:
    return [set(x) for x in xs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Manifest JSONL (dev/val).")
    ap.add_argument("--out", required=True, help="Predictions JSONL output.")
    ap.add_argument("--summary", required=True, help="Metrics summary JSON output.")
    ap.add_argument("--model", default="gpt-5-nano")
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="Optional limit for debugging (0 = no limit).")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests.")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)

    done: Set[str] = set()
    if args.resume and os.path.exists(args.out):
        for obj in _iter_jsonl(args.out):
            p = str(obj.get("image_path", "") or "").strip()
            if p:
                done.add(p)

    client = _require_openai_client()

    n_total = 0
    n_skipped = 0
    n_err = 0

    with open(args.out, "a") as f_out:
        for obj in _iter_jsonl(args.manifest):
            ex = parse_manifest_line(obj)
            if args.limit and n_total >= int(args.limit):
                break
            n_total += 1
            if ex.image_path in done:
                n_skipped += 1
                continue
            try:
                img = Image.open(ex.image_path)
                pred = predict_misleader_for_image(
                    client=client,
                    model=str(args.model),
                    image=img,
                    max_side=int(args.max_side),
                )
                rec = {
                    "image_path": ex.image_path,
                    "split": ex.split,
                    "y_true": ex.y_true,
                    "y_pred": pred.get("y_pred", []),
                    "rationale": pred.get("rationale", ""),
                    "model": str(args.model),
                    "max_side": int(args.max_side),
                }
            except Exception as e:
                n_err += 1
                rec = {
                    "image_path": ex.image_path,
                    "split": ex.split,
                    "y_true": ex.y_true,
                    "y_pred": [],
                    "error": str(e),
                    "model": str(args.model),
                }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_out.flush()

            if args.sleep:
                time.sleep(float(args.sleep))

    # Summarize using all available predictions (including resumed rows).
    manifest: Dict[str, Dict[str, Any]] = {}
    for obj in _iter_jsonl(args.manifest):
        ex = parse_manifest_line(obj)
        manifest[ex.image_path] = {"split": ex.split, "y_true": ex.y_true}
    preds: Dict[str, List[str]] = {}
    for obj in _iter_jsonl(args.out):
        p = str(obj.get("image_path", "") or "").strip()
        if not p:
            continue
        y_pred_raw = obj.get("y_pred") or []
        y_pred = (
            normalize_misleader_labels([str(x) for x in y_pred_raw], strict=False)
            if isinstance(y_pred_raw, list)
            else []
        )
        preds[p] = y_pred

    matched = [p for p in manifest.keys() if p in preds]
    missing = [p for p in manifest.keys() if p not in preds]
    y_true_all: List[List[str]] = [manifest[p]["y_true"] for p in matched]
    y_pred_all: List[List[str]] = [preds[p] for p in matched]

    summary = {
        "manifest": args.manifest,
        "predictions": args.out,
        "n_manifest": len(manifest),
        "n_matched": len(matched),
        "n_missing": len(missing),
        "n_total_iterated": n_total,
        "n_skipped_resume": n_skipped,
        "n_errors": n_err,
        "metrics": multilabel_score(_set_list(y_true_all), _set_list(y_pred_all), MISVIZ_MISLEADER_LABELS)
        if matched
        else {},
    }
    with open(args.summary, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Wrote predictions:", args.out)
    print("Wrote summary:", args.summary)


if __name__ == "__main__":
    main()
