import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Set

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from chartagent.agent import AgentConfig, ChartAgent
from chartagent.llm_client import create_llm_client
from chartagent.misviz_schema import parse_manifest_line
from chartagent.run_trace import summarize_run_dir, tool_sequence

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
        # Only mark as done if the prior run did not record an error.
        # This allows `--resume` to retry failed rows after prompt/code fixes.
        if obj.get("error"):
            continue
        done.add(p)
    return done


def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _parse_int_set(s: str) -> Set[int]:
    out: Set[int] = set()
    for part in str(s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def _parse_str_set(values: Optional[List[str]]) -> Set[str]:
    out: Set[str] = set()
    if not values:
        return out
    for v in values:
        for part in str(v or "").split(","):
            part = part.strip()
            if part:
                out.add(part)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="out/misviz_devval_manifest.jsonl")
    ap.add_argument("--out", default="out/misviz_devval_predictions.jsonl")
    ap.add_argument("--runs-root", default="out/chartagent_runs")
    ap.add_argument("--cache-root", default="out/chartagent_cache")
    ap.add_argument("--no-cache", action="store_true", help="Disable metadata cache.")
    ap.add_argument("--model", default="gpt-5-nano", help="Agent LLM (OpenAI or local VLM).")
    ap.add_argument(
        "--metadata-model",
        default=None,
        help='Model for metadata extraction (OpenAI only). Default: uses --model if it is an OpenAI model; else "gpt-5-nano".',
    )
    ap.add_argument("--no-metadata", action="store_true", help="Disable chart metadata extraction.")
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument(
        "--max-attached-images",
        type=int,
        default=4,
        help="Max number of images to attach to the VLM each step (lower helps avoid GPU OOM on local VLMs).",
    )
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--split", default="dev,val", help='Comma-separated: "dev", "val", or "dev,val".')
    ap.add_argument(
        "--only-idx",
        default="",
        help="Optional comma-separated 0-based indices into the filtered manifest (after --split, before --shuffle/--limit).",
    )
    ap.add_argument(
        "--only-image-path",
        action="append",
        default=None,
        help="Optional exact image_path filter(s). Repeatable; each value may also be comma-separated.",
    )
    ap.add_argument("--limit", type=int, default=0, help="Optional max number of samples to run (0=all).")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="Skip image_path already present in --out.")
    ap.add_argument("--include-trace", action="store_true", help="Embed step-by-step trace summary in each output row.")
    ap.add_argument("--trace-max-chars", type=int, default=500, help="Max chars for model_output_excerpt per step.")
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

    only_image_paths = _parse_str_set(getattr(args, "only_image_path", None))
    if only_image_paths:
        rows = [r for r in rows if r.image_path in only_image_paths]

    only_idx_raw = str(getattr(args, "only_idx", "") or "").strip()
    if only_idx_raw:
        idxs = sorted(_parse_int_set(only_idx_raw))
        picked = []
        for j in idxs:
            if j < 0 or j >= len(rows):
                raise SystemExit(f"--only-idx contains out-of-range index: {j} (rows={len(rows)})")
            picked.append(rows[j])
        rows = picked

    if args.shuffle:
        random.seed(int(args.seed))
        random.shuffle(rows)
    if int(args.limit) > 0:
        rows = rows[: int(args.limit)]

    agent_model = str(args.model)
    metadata_model = str(args.metadata_model) if args.metadata_model else agent_model
    if args.metadata_model is None and not agent_model.lower().startswith(("gpt-", "o1", "o3", "o4", "chatgpt-", "openai:")):
        metadata_model = "gpt-5-nano"
    if metadata_model.startswith("openai:"):
        metadata_model = metadata_model[len("openai:") :]

    llm = create_llm_client(model=agent_model, max_side=int(args.max_side))
    cfg = AgentConfig(
        agent_model=agent_model,
        metadata_model=metadata_model,
        max_steps=int(args.max_steps),
        max_side=int(args.max_side),
        max_attached_images=int(args.max_attached_images),
        out_root=str(args.runs_root),
        cache_root=str(args.cache_root),
        use_cache=not bool(args.no_cache),
        extract_metadata=not bool(args.no_metadata),
    )
    agent = ChartAgent(llm=llm, config=cfg)

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
            res = agent.run(image_path=ex.image_path, metadata=None)
            pred_line["y_pred"] = res.get("y_pred", [])
            pred_line["run_dir"] = res.get("run_dir")
            if "error" in res:
                pred_line["error"] = res.get("error")
            pred_line["raw_answer"] = res.get("raw_answer")
        except Exception as e:
            pred_line["y_pred"] = []
            pred_line["error"] = str(e)
            n_error += 1

        if bool(args.include_trace) and pred_line.get("run_dir"):
            try:
                trace = summarize_run_dir(
                    str(pred_line["run_dir"]),
                    model_output_max_chars=int(args.trace_max_chars),
                    include_action_resolved=False,
                )
                pred_line["trace"] = trace
                pred_line["tool_sequence"] = tool_sequence(trace)
            except Exception as e:
                pred_line["trace_error"] = str(e)

        pred_line["elapsed_sec"] = float(time.time() - t0)
        _append_jsonl(args.out, pred_line)

        if pbar is not None:
            pbar.set_postfix({"skip": n_skip, "err": n_error, "split": ex.split}, refresh=False)
        elif use_print:
            # Minimal progress log (one line per completed image).
            print(
                "[{}/{}] wrote prediction for {} (split={}, elapsed={:.2f}s)".format(
                    i + 1, n_total, os.path.basename(ex.image_path), ex.split, pred_line["elapsed_sec"]
                )
            )

    print("Done. total={}, skipped={}, out={}".format(n_total, n_skip, args.out))


if __name__ == "__main__":
    main()
