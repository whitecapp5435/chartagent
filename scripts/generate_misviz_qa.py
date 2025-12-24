import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image


@dataclass(frozen=True)
class MisvizItem:
    image_rel_path: str
    chart_type: List[str]
    misleader: List[str]
    split: str


def _load_misviz_items(misviz_json_path: str) -> List[MisvizItem]:
    with open(misviz_json_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("misviz.json must be a list")

    items: List[MisvizItem] = []
    for rec in data:
        if not isinstance(rec, dict):
            continue
        image_path = str(rec.get("image_path", "") or "").strip()
        if not image_path:
            continue
        chart_type = rec.get("chart_type")
        misleader = rec.get("misleader")
        split = str(rec.get("split", "") or "").strip()
        items.append(
            MisvizItem(
                image_rel_path=image_path,
                chart_type=[str(x) for x in (chart_type or []) if str(x).strip()]
                if isinstance(chart_type, list)
                else [],
                misleader=[str(x) for x in (misleader or []) if str(x).strip()]
                if isinstance(misleader, list)
                else [],
                split=split,
            )
        )
    return items


def _data_url_from_image(image: Image.Image, image_format: str = "PNG") -> str:
    buf = BytesIO()
    image_format = (image_format or "PNG").upper()
    image.convert("RGB").save(buf, format=image_format)
    b64 = __import__("base64").b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if image_format == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


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


def _sha1_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _require_openai_client():
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "OpenAI Python SDK is required. Install/upgrade with: pip install --upgrade openai"
        ) from e
    return OpenAI()


def _build_prompt(k: int, chart_type_hint: Optional[str]) -> str:
    hint = f"\nChart type hint (may be imperfect): {chart_type_hint}\n" if chart_type_hint else ""
    return (
        "You are generating a small synthetic QA dataset for chart question answering.\n"
        "Given a chart image, generate question-answer pairs that are answerable from the image alone.\n"
        "\n"
        "Requirements:\n"
        f"- Generate up to {k} QA items.\n"
        "- Prefer questions that require reading the chart visually (axes/legend/segments) rather than trivial titles.\n"
        "- Prefer numeric questions (at least 2 if possible): exact value reads, comparisons, differences, ratios, percentages.\n"
        "- Avoid subjective/ambiguous questions.\n"
        "- If a value cannot be read confidently, either skip that question or set a reasonable tolerance.\n"
        "- Do not invent values not supported by the chart.\n"
        "\n"
        "Output rules:\n"
        "- Return ONLY the JSON object matching the provided schema.\n"
        "- For numeric answers, fill answer_numeric and (optionally) tolerance.\n"
        "- Keep questions concise.\n"
        f"{hint}"
    )


def generate_qa_for_image(
    *,
    client,
    image: Image.Image,
    image_path: str,
    chart_type_hint: Optional[str],
    model: str,
    questions_per_image: int,
    max_side: int,
    max_retries: int,
) -> Dict[str, Any]:
    try:
        from pydantic import BaseModel  # type: ignore
        from typing import Literal  # noqa: F401
    except Exception as e:
        raise RuntimeError("pydantic is required for structured outputs (pip install pydantic).") from e

    # Define schema for Structured Outputs.
    class QAItem(BaseModel):
        question: str
        answer: str
        answer_type: str  # "numeric"|"categorical"|"freeform"|"multiple_choice"
        answer_numeric: Optional[float] = None
        tolerance: Optional[float] = None
        unit: Optional[str] = None
        choices: Optional[List[str]] = None

    class QASet(BaseModel):
        items: List[QAItem]
        skip_reason: Optional[str] = None

    img_small = _resize_max_side(image, max_side=max_side)
    img_url = _data_url_from_image(img_small, image_format="PNG")

    prompt = _build_prompt(questions_per_image, chart_type_hint)

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
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
                text_format=QASet,
            )
            parsed = getattr(resp, "output_parsed", None)
            if parsed is None:
                raise RuntimeError("No parsed output returned")
            out = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)  # type: ignore[arg-type]
            return {
                "image_path": image_path,
                "chart_type_hint": chart_type_hint,
                "model": model,
                "questions_per_image": int(questions_per_image),
                "items": out.get("items", []),
                "skip_reason": out.get("skip_reason"),
            }
        except Exception as e:
            last_err = e
            # light backoff
            time.sleep(1.5 * (attempt + 1))
            continue
    raise RuntimeError(f"Failed to generate QA after {max_retries} attempts: {last_err}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--misviz-json", default="misviz/misviz.json")
    ap.add_argument("--misviz-dir", default="misviz")
    ap.add_argument("--out", required=True, help="Output JSONL path (one record per image).")
    ap.add_argument("--manifest", default=None, help="Optional manifest JSON path of sampled images.")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="gpt-5-nano")
    ap.add_argument("--questions-per-image", type=int, default=3)
    ap.add_argument("--max-side", type=int, default=1024, help="Resize images to limit cost (0 disables).")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--split", default=None, help='Optional split filter (e.g., "test").')
    ap.add_argument("--resume", action="store_true", help="Skip images already present in --out.")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    items = _load_misviz_items(args.misviz_json)
    if args.split:
        items = [it for it in items if it.split == str(args.split)]

    # Resolve existing images.
    img_paths: List[Tuple[MisvizItem, str]] = []
    for it in items:
        abs_path = os.path.join(args.misviz_dir, it.image_rel_path)
        if os.path.exists(abs_path) and os.path.isfile(abs_path):
            img_paths.append((it, abs_path))

    if len(img_paths) < args.n:
        raise SystemExit(f"Not enough images found: requested n={args.n}, found={len(img_paths)}")

    rng = random.Random(int(args.seed))
    sampled = rng.sample(img_paths, int(args.n))

    # Resume support.
    done: set[str] = set()
    if args.resume and os.path.exists(args.out):
        for rec in _iter_jsonl(args.out):
            p = str(rec.get("image_path", "") or "")
            if p:
                done.add(p)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    client = _require_openai_client()

    manifest_rows: List[Dict[str, Any]] = []
    for it, abs_path in sampled:
        manifest_rows.append(
            {
                "image_path": abs_path,
                "image_rel_path": it.image_rel_path,
                "chart_type": it.chart_type,
                "misleader": it.misleader,
                "split": it.split,
                "sha1": _sha1_file(abs_path),
            }
        )

    if args.manifest:
        os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
        with open(args.manifest, "w") as f:
            json.dump(
                {
                    "n": int(args.n),
                    "seed": int(args.seed),
                    "split": args.split,
                    "rows": manifest_rows,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    with open(args.out, "a") as f_out:
        for i, (it, abs_path) in enumerate(sampled):
            if abs_path in done:
                continue
            img = Image.open(abs_path)
            chart_type_hint = it.chart_type[0] if it.chart_type else None
            rec = generate_qa_for_image(
                client=client,
                image=img,
                image_path=abs_path,
                chart_type_hint=chart_type_hint,
                model=str(args.model),
                questions_per_image=int(args.questions_per_image),
                max_side=int(args.max_side),
                max_retries=int(args.max_retries),
            )
            rec["idx"] = int(i)
            rec["sha1"] = _sha1_file(abs_path)
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_out.flush()
            # small delay to be gentle on rate limits
            time.sleep(0.2)


if __name__ == "__main__":
    main()

