import json
import os
import re
from typing import Any, Dict, List, Optional


def _read_text(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def _read_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _safe_excerpt(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if max_chars <= 0 or len(t) <= max_chars:
        return t
    return t[:max_chars] + "…"


_STEP_DIR_RE = re.compile(r"^step_(\d+)$")


def summarize_run_dir(
    run_dir: str,
    *,
    model_output_max_chars: int = 500,
    include_action_resolved: bool = False,
) -> List[Dict[str, Any]]:
    """
    Summarize a ChartAgent run directory into step-by-step records suitable for JSONL.

    This reads `steps/step_*/{model_output.txt,action.json,action_resolved.json,observation.json}`.
    """
    steps_root = os.path.join(str(run_dir), "steps")
    if not os.path.isdir(steps_root):
        return []

    step_names = []
    for name in os.listdir(steps_root):
        m = _STEP_DIR_RE.match(name)
        if not m:
            continue
        step_names.append((int(m.group(1)), name))
    step_names.sort(key=lambda x: x[0])

    out: List[Dict[str, Any]] = []
    for step_idx, step_name in step_names:
        step_dir = os.path.join(steps_root, step_name)
        entry: Dict[str, Any] = {
            "step_idx": step_idx,
            "step_dir": os.path.join("steps", step_name),
        }

        model_output_path = os.path.join(step_dir, "model_output.txt")
        if os.path.exists(model_output_path):
            try:
                model_output = _read_text(model_output_path)
                entry["model_output_excerpt"] = _safe_excerpt(model_output, int(model_output_max_chars))
                entry["has_action"] = bool(re.search(r"(?im)^\s*ACTION\s*:", model_output))
                entry["has_answer"] = bool(re.search(r"(?im)^\s*ANSWER\s*:", model_output))
            except Exception:
                entry["model_output_excerpt"] = ""
                entry["has_action"] = False
                entry["has_answer"] = False

        action_path = os.path.join(step_dir, "action.json")
        if os.path.exists(action_path):
            try:
                action = _read_json(action_path)
                if isinstance(action, dict):
                    entry["action"] = action
                    tool = action.get("tool")
                    if tool is not None:
                        entry["action_tool"] = str(tool)
            except Exception:
                pass

        if bool(include_action_resolved):
            action_resolved_path = os.path.join(step_dir, "action_resolved.json")
            if os.path.exists(action_resolved_path):
                try:
                    action_resolved = _read_json(action_resolved_path)
                    if isinstance(action_resolved, dict):
                        entry["action_resolved"] = action_resolved
                except Exception:
                    pass

        obs_path = os.path.join(step_dir, "observation.json")
        if os.path.exists(obs_path):
            try:
                obs = _read_json(obs_path)
                if isinstance(obs, dict):
                    entry["observation"] = obs
                    if "error" in obs and str(obs.get("error") or "").strip():
                        entry["error"] = str(obs.get("error"))
            except Exception:
                pass

        out.append(entry)

    return out


def tool_sequence(trace: List[Dict[str, Any]]) -> List[str]:
    seq: List[str] = []
    for step in trace:
        tool = step.get("action_tool")
        if tool is None:
            continue
        tool_s = str(tool).strip()
        if tool_s:
            seq.append(tool_s)
    return seq

