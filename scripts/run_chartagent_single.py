import argparse
import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from chartagent.agent import AgentConfig, ChartAgent
from chartagent.llm_client import create_llm_client
from chartagent.run_trace import summarize_run_dir, tool_sequence


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
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
    ap.add_argument("--out-root", default="out/chartagent_runs")
    ap.add_argument("--cache-root", default="out/chartagent_cache")
    ap.add_argument("--no-cache", action="store_true", help="Disable metadata cache.")
    ap.add_argument("--output", default=None, help="Optional output JSON path (also saved under run_dir).")
    ap.add_argument("--include-trace", action="store_true", help="Embed step-by-step trace summary in the printed result.")
    ap.add_argument("--trace-max-chars", type=int, default=500, help="Max chars for model_output_excerpt per step.")
    ap.add_argument("--include-action-resolved", action="store_true", help="Include action_resolved.json per step in trace.")
    args = ap.parse_args()

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
        out_root=str(args.out_root),
        cache_root=str(args.cache_root),
        use_cache=not bool(args.no_cache),
        extract_metadata=not bool(args.no_metadata),
    )
    agent = ChartAgent(llm=llm, config=cfg)

    result = agent.run(image_path=str(args.image))

    if bool(args.include_trace) and result.get("run_dir"):
        try:
            trace = summarize_run_dir(
                str(result["run_dir"]),
                model_output_max_chars=int(args.trace_max_chars),
                include_action_resolved=bool(args.include_action_resolved),
            )
            result["trace"] = trace
            result["tool_sequence"] = tool_sequence(trace)
        except Exception as e:
            result["trace_error"] = str(e)

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
