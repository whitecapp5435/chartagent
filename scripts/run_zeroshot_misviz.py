import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Set

from PIL import Image

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from chartagent.llm_client import create_llm_client
from chartagent.misviz_schema import MISVIZ_MISLEADER_LABELS, normalize_misleader_labels, parse_manifest_line

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore[assignment]


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def _load_done(predictions_jsonl: str) -> Set[str]:
    done: Set[str] = set()
    if not os.path.exists(predictions_jsonl):
        return done
    for obj in _iter_jsonl(predictions_jsonl):
        p = str(obj.get("image_path", "") or "").strip()
        if not p:
            continue
        if obj.get("error"):
            continue
        done.add(p)
    return done


def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _build_zeroshot_prompt() -> str:
    labels = "\n".join(["- {}".format(x) for x in MISVIZ_MISLEADER_LABELS])
    return (
        "You are an expert at analyzing chart images and identifying misleading visualization type(s) (multi-label).\n"
        "\n"
        "Task: Given a chart image, predict which misleading visualization type(s) apply.\n"
        "Return only canonical labels from this list:\n"
        "{}\n"
        "\n"
        "Rules:\n"
        "- Choose 0 or more labels from the list.\n"
        "- Do not invent new labels.\n"
        "- Output must be valid JSON.\n"
        "\n"
        "Output format (strict):\n"
        'ANSWER: {{"misleader": [...]}} TERMINATE\n'.format(labels)
    )


def _extract_json_object_after_marker(text: str, marker: str) -> Optional[Dict[str, Any]]:
    idx = str(text).find(marker)
    if idx < 0:
        return None
    start = str(text).find("{", idx)
    if start < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(str(text)[start:])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _parse_answer(model_text: str) -> Optional[Dict[str, Any]]:
    # Preferred: "ANSWER: { ... }"
    ans = _extract_json_object_after_marker(model_text, "ANSWER:")
    if isinstance(ans, dict):
        return ans

    # Fallback: the model might output just a JSON object.
    start = str(model_text).find("{")
    if start >= 0:
        decoder = json.JSONDecoder()
        try:
            obj, _ = decoder.raw_decode(str(model_text)[start:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="out/misviz_devval_manifest.jsonl")
    ap.add_argument("--out", default="out/misviz_devval_predictions_zeroshot.jsonl")
    ap.add_argument("--model", default="gpt-5-nano", help="OpenAI or local VLM model id.")
    ap.add_argument(
        "--prompt-file",
        default="",
        help="Optional path to a custom prompt (overrides the built-in zeroshot prompt).",
    )
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--split", default="dev,val", help='Comma-separated: "dev", "val", or "dev,val".')
    ap.add_argument("--limit", type=int, default=0, help="Optional max number of samples to run (0=all).")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="Skip image_path already present in --out (without error).")
    ap.add_argument("--include-model-output", action="store_true", help="Store raw model output text in each row.")
    ap.add_argument(
        "--progress",
        default="auto",
        choices=["auto", "tqdm", "print", "none"],
        help='Progress display. "auto" uses tqdm if installed (default).',
    )
    args = ap.parse_args()

    splits = [s.strip() for s in str(args.split).split(",") if s.strip()]
    for s in splits:
        if s not in {"dev", "val"}:
            raise SystemExit("Invalid --split value: {!r}".format(s))

    done: Set[str] = _load_done(args.out) if args.resume else set()

    rows = [parse_manifest_line(obj) for obj in _iter_jsonl(args.manifest)]
    rows = [r for r in rows if r.split in set(splits)]
    if args.shuffle:
        random.seed(int(args.seed))
        random.shuffle(rows)
    if int(args.limit) > 0:
        rows = rows[: int(args.limit)]

    llm = create_llm_client(model=str(args.model), max_side=int(args.max_side))
    prompt = _build_zeroshot_prompt()
    if str(args.prompt_file).strip():
        with open(str(args.prompt_file).strip(), "r") as f:
            prompt = f.read()

    n_total = len(rows)
    n_skip = 0
    n_error = 0

    use_tqdm = False
    if str(args.progress) == "tqdm":
        if tqdm is None:
            raise SystemExit('tqdm is not installed. Install it with: pip install tqdm')
        use_tqdm = True
    elif str(args.progress) == "auto":
        use_tqdm = tqdm is not None

    iterator = rows
    pbar = None
    if use_tqdm:
        pbar = tqdm(rows, total=n_total, unit="img")  # type: ignore[misc]
        iterator = pbar  # type: ignore[assignment]

    use_print = (str(args.progress) == "print") or (str(args.progress) == "auto" and not use_tqdm)

    for i, ex in enumerate(iterator):
        if ex.image_path in done:
            n_skip += 1
            if pbar is not None:
                pbar.set_postfix({"skip": n_skip, "err": n_error}, refresh=False)
            continue

        t0 = time.time()
        pred_line: Dict[str, Any] = {
            "image_path": ex.image_path,
            "split": ex.split,
            "y_true": ex.y_true,
            "chart_type": ex.chart_type,
        }

        try:
            img = Image.open(ex.image_path).convert("RGB")
            model_text = llm.generate(prompt=prompt, images=[img])
            ans = _parse_answer(model_text)
            if not isinstance(ans, dict):
                raise RuntimeError("Could not parse model output as ANSWER JSON")
            y_pred_raw = ans.get("misleader") or []
            y_pred = (
                normalize_misleader_labels([str(x) for x in y_pred_raw], strict=False)
                if isinstance(y_pred_raw, list)
                else []
            )
            pred_line["y_pred"] = y_pred
            pred_line["raw_answer"] = ans
            if bool(args.include_model_output):
                pred_line["model_output"] = model_text
        except Exception as e:
            pred_line["y_pred"] = []
            pred_line["error"] = str(e)
            n_error += 1

        pred_line["elapsed_sec"] = float(time.time() - t0)
        _append_jsonl(str(args.out), pred_line)

        if pbar is not None:
            pbar.set_postfix({"skip": n_skip, "err": n_error, "split": ex.split}, refresh=False)
        elif use_print:
            print(
                "[{}/{}] wrote prediction for {} (split={}, elapsed={:.2f}s)".format(
                    i + 1, n_total, os.path.basename(ex.image_path), ex.split, pred_line["elapsed_sec"]
                )
            )

    print("Done. total={}, skipped={}, out={}".format(n_total, n_skip, args.out))


if __name__ == "__main__":
    main()
