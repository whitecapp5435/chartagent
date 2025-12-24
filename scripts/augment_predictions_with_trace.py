import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from chartagent.run_trace import summarize_run_dir, tool_sequence


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="Input predictions JSONL (must contain run_dir).")
    ap.add_argument("--out", required=True, help="Output JSONL with added trace/tool_sequence.")
    ap.add_argument("--trace-max-chars", type=int, default=500, help="Max chars for model_output_excerpt per step.")
    ap.add_argument("--include-action-resolved", action="store_true", help="Include action_resolved.json per step.")
    args = ap.parse_args()

    for row in _iter_jsonl(args.predictions):
        run_dir = row.get("run_dir")
        if run_dir and os.path.isdir(str(run_dir)):
            try:
                trace = summarize_run_dir(
                    str(run_dir),
                    model_output_max_chars=int(args.trace_max_chars),
                    include_action_resolved=bool(args.include_action_resolved),
                )
                row["trace"] = trace
                row["tool_sequence"] = tool_sequence(trace)
            except Exception as e:
                row["trace_error"] = str(e)
        else:
            row.setdefault("trace_error", "Missing or invalid run_dir")

        _append_jsonl(args.out, row)

    print("Wrote:", args.out)


if __name__ == "__main__":
    main()

