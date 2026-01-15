import json
import os
import re
import time
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from chartagent.icl import load_icl_prompt, major_chart_type_from_metadata
from chartagent.misviz_schema import MISVIZ_MISLEADER_LABELS, normalize_misleader_labels
from chartagent.tool_executor import ToolExecutor


def _read_text(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _sha1_file(path: str, chunk_size: int = 1 << 20) -> str:
    import hashlib

    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _safe_json_dump(path: str, obj: Any) -> None:
    def _jsonable(x: Any) -> Any:
        if x is None:
            return None
        if isinstance(x, (str, int, float, bool)):
            return x
        if isinstance(x, Image.Image):
            return {"_type": "Image", "size": list(x.size)}
        if isinstance(x, tuple):
            return [_jsonable(v) for v in x]
        if isinstance(x, list):
            return [_jsonable(v) for v in x]
        if isinstance(x, dict):
            return {str(k): _jsonable(v) for k, v in x.items()}
        return {"_type": type(x).__name__}

    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(_jsonable(obj), f, ensure_ascii=False, indent=2)


def _extract_json_object_after_marker(text: str, marker: str) -> Optional[Dict[str, Any]]:
    idx = text.find(marker)
    if idx < 0:
        return None
    start = text.find("{", idx)
    if start < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text[start:])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _extract_json_object_after_marker_re(text: str, marker_re: str) -> Optional[Dict[str, Any]]:
    m = re.search(str(marker_re), str(text))
    if not m:
        return None
    start = str(text).find("{", int(m.end()))
    if start < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(str(text)[start:])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _infer_bar_orientation_from_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Best-effort bar orientation hint from free-form metadata text.

    Returns:
      - "horizontal" | "vertical" | None
    """
    if not isinstance(metadata, dict):
        return None

    parts: List[str] = []
    for k in ("chart_type", "visual_description"):
        v = metadata.get(k)
        if isinstance(v, list):
            v = " ".join([str(x) for x in v])
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    text = " ".join(parts).lower()
    if not text:
        return None

    if re.search(r"\bhorizontal[-\s]+bar\b", text):
        return "horizontal"
    if re.search(r"\bvertical[-\s]+bar\b", text) or re.search(r"\bcolumn\s+chart\b", text):
        return "vertical"
    return None


def _axis_quality_signals(result_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lightweight axis evidence signals to help the LLM judge reliability.

    Notes:
      - This does NOT enforce any label decisions; it only reports stats.
      - Keep this compact to avoid bloating the prompt history.
    """
    axis = str(result_summary.get("axis") or "").strip().lower()
    values_raw = result_summary.get("axis_values")
    pixels_raw = result_summary.get("axis_pixel_positions")

    axis_conf: Optional[float]
    try:
        axis_conf = float(result_summary.get("axis_confidence")) if result_summary.get("axis_confidence") is not None else None
    except Exception:
        axis_conf = None

    n_ticks: Optional[int]
    try:
        n_ticks = int(result_summary.get("n_ticks")) if result_summary.get("n_ticks") is not None else None
    except Exception:
        n_ticks = None

    if not isinstance(values_raw, list):
        return {"axis": axis, "axis_confidence": axis_conf, "n_ticks": n_ticks, "n_values": 0}

    values: List[float] = []
    for v in values_raw:
        try:
            fv = float(v)
        except Exception:
            continue
        if not math.isfinite(fv):
            continue
        values.append(fv)

    abs_vals = sorted([abs(v) for v in values])
    median_abs: Optional[float] = None
    max_abs: Optional[float] = None
    if abs_vals:
        n = len(abs_vals)
        if n % 2 == 1:
            median_abs = float(abs_vals[n // 2])
        else:
            median_abs = float((abs_vals[n // 2 - 1] + abs_vals[n // 2]) / 2.0)
        max_abs = float(abs_vals[-1])

    mono_inc = bool(values) and all(values[i] <= values[i + 1] for i in range(len(values) - 1))
    mono_dec = bool(values) and all(values[i] >= values[i + 1] for i in range(len(values) - 1))

    abs_deltas: List[float] = []
    for i in range(len(values) - 1):
        d = float(values[i + 1] - values[i])
        if math.isfinite(d) and abs(d) > 0.0:
            abs_deltas.append(abs(d))
    delta_min: Optional[float] = float(min(abs_deltas)) if abs_deltas else None
    delta_max: Optional[float] = float(max(abs_deltas)) if abs_deltas else None

    out: Dict[str, Any] = {
        "axis": axis,
        "axis_present": bool(result_summary.get("axis_present")),
        "axis_confidence": axis_conf,
        "n_ticks": n_ticks,
        "n_values": len(values),
        "values_monotonic_increasing": mono_inc,
        "values_monotonic_decreasing": mono_dec,
        "median_abs_value": median_abs,
        "max_abs_value": max_abs,
        "delta_abs_min": delta_min,
        "delta_abs_max": delta_max,
    }
    if median_abs and max_abs is not None:
        try:
            out["max_over_median_abs"] = float(max_abs / median_abs) if median_abs != 0 else None
        except Exception:
            out["max_over_median_abs"] = None
    if delta_min and delta_max is not None:
        try:
            out["delta_max_over_min"] = float(delta_max / delta_min) if delta_min != 0 else None
        except Exception:
            out["delta_max_over_min"] = None

    if isinstance(pixels_raw, list):
        out["n_pixel_positions"] = len(pixels_raw)
    return out


@dataclass
class AgentConfig:
    agent_model: str = "gpt-5-nano"
    metadata_model: str = "gpt-5-nano"
    max_steps: int = 6
    max_side: int = 1024
    out_root: str = "out/chartagent_runs"
    extract_metadata: bool = True
    max_attached_images: int = 4
    cache_root: str = "out/chartagent_cache"
    use_cache: bool = True


class ChartAgent(object):
    def __init__(self, *, llm, tool_executor: Optional[ToolExecutor] = None, config: Optional[AgentConfig] = None) -> None:
        self.llm = llm
        self.tools = tool_executor or ToolExecutor()
        self.config = config or AgentConfig()

        self._system_prompt = _read_text(os.path.join("chartagent", "prompts", "system.txt"))
        self._tools_prompt = _read_text(os.path.join("chartagent", "prompts", "tools.txt"))

    def _build_task_prompt(self, *, metadata: Optional[Dict[str, Any]], step_history: str, refs_hint: str) -> str:
        labels = "\n".join(["- {}".format(x) for x in MISVIZ_MISLEADER_LABELS])
        meta_block = json.dumps(metadata, ensure_ascii=False, indent=2) if metadata else "{}"

        major_type = major_chart_type_from_metadata(metadata)
        icl_text = load_icl_prompt(major_type)
        icl_block = ""
        if icl_text:
            icl_block = (
                "In-context examples (chart type: {}):\n".format(major_type)
                + icl_text.strip()
                + "\n\n"
            )
        return (
            self._system_prompt
            + "\n"
            + self._tools_prompt
            + "\n"
            + icl_block
            + "Task: Predict misleader type(s) for this chart image.\n"
            + "Return only canonical labels from this list:\n"
            + labels
            + "\n\n"
            + "Chart metadata (may be imperfect):\n"
            + meta_block
            + "\n\n"
            + "Runtime refs available (use {'$ref': '<id>'} in ACTION args):\n"
            + (refs_hint or "(none)")
            + "\n\n"
            + "Conversation so far:\n"
            + (step_history or "(none)")
            + "\n\n"
            + "Now produce the next THOUGHT + (ACTION or ANSWER).\n"
        )

    def _parse_action(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        obj = _extract_json_object_after_marker_re(text, r"(?im)^\s*ACTION\s*:")
        if obj is None:
            return None
        tool = str(obj.get("tool", "") or "").strip()
        args = obj.get("args") or {}
        if not isinstance(args, dict):
            raise ValueError("ACTION args must be an object")
        return tool, args

    def _parse_answer(self, text: str) -> Optional[Dict[str, Any]]:
        obj = _extract_json_object_after_marker_re(text, r"(?im)^\s*ANSWER\s*:")
        if obj is not None:
            return obj

        # Fallback: some local VLMs incorrectly emit the final answer as an ACTION tool call:
        # ACTION: {"tool": "ANSWER", "args": {...}}
        act = _extract_json_object_after_marker_re(text, r"(?im)^\s*ACTION\s*:")
        if isinstance(act, dict):
            tool = str(act.get("tool", "") or "").strip().lower()
            if tool in {"answer", "final"}:
                args = act.get("args")
                if isinstance(args, dict):
                    return args
        return None

    def _resolve_refs(self, obj: Any, memory: Dict[str, Any]) -> Any:
        if isinstance(obj, dict) and set(obj.keys()) == {"$ref"}:
            ref_id = str(obj.get("$ref") or "").strip()
            base = ref_id
            bracket = ""
            m = re.match(r"^([^\[]+)(\[.+\])$", ref_id)
            if m:
                base = m.group(1)
                bracket = m.group(2)

            if base not in memory:
                raise KeyError("Unknown $ref: {}".format(ref_id))
            cur: Any = memory[base]

            if bracket:
                parts = re.findall(r"\[([^\]]+)\]", bracket)
                try:
                    for part in parts:
                        p = str(part).strip()
                        if re.fullmatch(r"-?\d+", p):
                            idx = int(p)
                            cur = cur[idx]  # type: ignore[index]
                            continue
                        if (p.startswith("'") and p.endswith("'")) or (p.startswith('"') and p.endswith('"')):
                            key = p[1:-1]
                        else:
                            key = p
                        cur = cur[key]  # type: ignore[index]
                except Exception:
                    raise KeyError("Unknown $ref: {} (indexing failed)".format(ref_id))
            return cur
        if isinstance(obj, dict):
            return {k: self._resolve_refs(v, memory) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._resolve_refs(v, memory) for v in obj]
        return obj

    def run(self, *, image_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg = self.config
        img = Image.open(image_path).convert("RGB")

        img_sha1 = _sha1_file(image_path)
        run_id = "{}_{}".format(_now_id(), img_sha1[:10])
        run_dir = os.path.join(cfg.out_root, run_id)
        steps_dir = os.path.join(run_dir, "steps")
        _ensure_dir(steps_dir)

        metadata_err: Optional[str] = None
        metadata_source: str = "provided" if metadata is not None else "none"
        metadata_cache_path: Optional[str] = None
        if metadata is None and bool(cfg.extract_metadata):
            try:
                # 1) metadata cache (image hash based)
                if bool(cfg.use_cache):
                    metadata_cache_path = os.path.join(str(cfg.cache_root), "metadata", "{}.json".format(img_sha1))
                    if os.path.exists(metadata_cache_path):
                        with open(metadata_cache_path, "r") as f:
                            cached = json.load(f)
                        if isinstance(cached, dict):
                            metadata = cached
                            metadata_source = "cache"

                # 2) fresh extraction
                if metadata is None:
                    from chartagent_tools.metadata import extract_chart_metadata  # local import

                    metadata = extract_chart_metadata(img, model=str(cfg.metadata_model), strict=False)
                    metadata_source = "fresh"

                    if bool(cfg.use_cache) and metadata_cache_path and isinstance(metadata, dict):
                        _ensure_dir(os.path.dirname(metadata_cache_path) or ".")
                        with open(metadata_cache_path, "w") as f:
                            json.dump(metadata, f, ensure_ascii=False, indent=2)
            except Exception as e:
                metadata_err = str(e)
                metadata = None
                metadata_source = "error"

        # Save original image + metadata.
        img_path_out = os.path.join(run_dir, "input.png")
        img.save(img_path_out)
        if metadata is not None:
            _safe_json_dump(os.path.join(run_dir, "metadata.json"), metadata)
        _safe_json_dump(
            os.path.join(run_dir, "metadata_info.json"),
            {
                "source": metadata_source,
                "cache_path": metadata_cache_path,
                "metadata_model": str(cfg.metadata_model),
                "agent_model": str(cfg.agent_model),
            },
        )
        if metadata_err:
            _safe_json_dump(os.path.join(run_dir, "metadata_error.json"), {"error": metadata_err})

        memory: Dict[str, Any] = {}
        # Seed refs.
        memory["input.image"] = img
        current_image: Image.Image = img
        memory["state.current_image"] = current_image

        refs_hint = "- input.image (PIL image, original)\n- state.current_image (PIL image, default tool input)\n"
        if metadata is not None:
            memory["input.metadata"] = metadata
            refs_hint += "- input.metadata (dict)\n"

        history_lines: List[str] = []
        visual_history: List[Tuple[str, str]] = []  # (description, ref_id)
        axis_status: Dict[str, Dict[str, Any]] = {}
        tools_used: set[str] = set()
        last_action_sig: Optional[str] = None

        bar_orientation_hint = _infer_bar_orientation_from_metadata(metadata)

        for step_idx in range(int(cfg.max_steps)):
            step_dir = os.path.join(steps_dir, "step_{:03d}".format(step_idx))
            _ensure_dir(step_dir)

            images_for_llm: List[Image.Image] = []
            images_desc: List[str] = []

            def _attach(desc: str, im: Image.Image) -> None:
                if len(images_for_llm) >= int(cfg.max_attached_images):
                    return
                images_for_llm.append(im)
                images_desc.append(desc)

            _attach("1) input.image (original chart)", img)
            if current_image is not img:
                _attach("2) state.current_image (working image)", current_image)

            # Attach most recent tool output images (for visual self-verification).
            for desc, ref_id in reversed(visual_history):
                if len(images_for_llm) >= int(cfg.max_attached_images):
                    break
                im = memory.get(ref_id)
                if isinstance(im, Image.Image) and im is not img and im is not current_image:
                    _attach("{} ({})".format(desc, ref_id), im)

            prompt = self._build_task_prompt(
                metadata=metadata,
                step_history="\n".join(history_lines),
                refs_hint=refs_hint,
            )
            steps_left = int(cfg.max_steps) - int(step_idx)
            prompt += "\nRuntime note: {} step(s) remaining. Avoid repeating tools; answer as soon as you have enough evidence.\n".format(
                steps_left
            )
            if images_desc:
                prompt += "\nAttached images (in order):\n" + "\n".join(images_desc) + "\n"
            with open(os.path.join(step_dir, "prompt.txt"), "w") as f:
                f.write(prompt)

            # Always attach the main chart image; attach additional debug images later (MVP-2).
            model_text = self.llm.generate(prompt=prompt, images=images_for_llm or [img])
            with open(os.path.join(step_dir, "model_output.txt"), "w") as f:
                f.write(model_text)

            ans = self._parse_answer(model_text)
            if ans is not None:
                y_pred_raw = ans.get("misleader") or []
                y_pred = (
                    normalize_misleader_labels([str(x) for x in y_pred_raw], strict=False)
                    if isinstance(y_pred_raw, list)
                    else []
                )

                major_type = major_chart_type_from_metadata(metadata)
                annotation_type = ""
                if isinstance(metadata, dict):
                    annotation_type = str(metadata.get("annotation_type", "") or "").strip().lower()
                answer_warnings: List[Dict[str, Any]] = []

                # Require at least one tool call before answering (evidence-first), especially for misviz-style tasks.
                # This keeps the agent from making unsupported guesses or returning empty labels prematurely.
                if int(step_idx) == 0 and not tools_used:
                    suggested = None
                    if major_type == "bar" and annotation_type == "annotated":
                        orient = "vertical"
                        if bar_orientation_hint in {"horizontal", "vertical"}:
                            orient = str(bar_orientation_hint)

                        axis_guess = "x" if orient == "horizontal" else "y"
                        tick_key = "x_axis_ticker_values" if axis_guess == "x" else "y_axis_ticker_values"
                        tick_list = metadata.get(tick_key) if isinstance(metadata, dict) else None
                        has_axis_tickers = isinstance(tick_list, list) and len(tick_list) >= 3

                        # Prefer axis evidence when tick labels are available; use bar_value_consistency as a fallback
                        # when ticks are missing/unreliable.
                        if has_axis_tickers:
                            suggested = {
                                "tool": "axis_localizer",
                                "args": {
                                    "image": {"$ref": "input.image"},
                                    "axis": axis_guess,
                                    "axis_threshold": 0.2,
                                    "axis_tickers": {"$ref": "input.metadata[{}]".format(tick_key)},
                                },
                            }
                        else:
                            suggested = {"tool": "bar_value_consistency", "args": {"image": {"$ref": "input.image"}, "bar_orientation": orient}}
                    elif major_type in {"bar", "line"}:
                        suggested = {"tool": "axis_localizer", "args": {"image": {"$ref": "input.image"}, "axis": "y", "axis_threshold": 0.2}}
                    elif major_type in {"pie", "donut", "radial"}:
                        suggested = {"tool": "segment_and_mark", "args": {"image": {"$ref": "input.image"}, "segmentation_model": "SAM", "min_area": 5000}}
                    else:
                        suggested = {"tool": "axis_localizer", "args": {"image": {"$ref": "input.image"}, "axis": "y", "axis_threshold": 0.2}}

                    obs = {
                        "validation_error": "Evidence-first policy: you must run at least one relevant tool before answering.",
                        "suggested_next_action": suggested,
                    }
                    _safe_json_dump(os.path.join(step_dir, "observation.json"), obs)
                    history_lines.append(model_text.strip())
                    history_lines.append("OBSERVATION: {}".format(json.dumps(obs, ensure_ascii=False)))
                    continue

                # Guardrail: axis-based labels require reliable axis evidence.
                axis_labels = {
                    "truncated axis",
                    "inconsistent tick intervals",
                    "dual axis",
                    "inverted axis",
                    "inappropriate axis range",
                }
                if any(lbl in axis_labels for lbl in y_pred):
                    def _axis_present(s: object) -> bool:
                        if not isinstance(s, dict):
                            return False
                        return bool(s.get("axis_present")) and int(s.get("n_ticks") or 0) >= 3

                    y_ok = _axis_present(axis_status.get("y"))
                    ry_ok = _axis_present(axis_status.get("right_y"))
                    x_ok = _axis_present(axis_status.get("x"))
                    tx_ok = _axis_present(axis_status.get("top_x"))

                    ok = True
                    if "dual axis" in y_pred:
                        # Accept either dual-y (left + right) or dual-x (bottom + top).
                        ok = bool((y_ok and ry_ok) or (x_ok and tx_ok))
                    else:
                        ok = bool(y_ok or ry_ok or x_ok or tx_ok)

                    if not ok:
                        # Non-blocking: surface the missing/weak evidence so the trace stays debuggable,
                        # but leave the final decision to the LLM (no hard rule-based filtering).
                        warning: Dict[str, Any] = {
                            "validation_warning": "Axis-based label(s) were predicted without strong axis_localizer evidence "
                            "(need axis_present=true and typically >=3 ticks). Treat axis evidence as potentially unreliable.",
                            "axis_status": axis_status,
                        }
                        if "dual axis" in y_pred:
                            # Helpful suggestion: gather the missing second-axis evidence.
                            if x_ok and not tx_ok:
                                warning["suggested_next_action"] = {
                                    "tool": "axis_localizer",
                                    "args": {"image": {"$ref": "input.image"}, "axis": "top_x", "axis_threshold": 0.2},
                                }
                            elif tx_ok and not x_ok:
                                warning["suggested_next_action"] = {
                                    "tool": "axis_localizer",
                                    "args": {"image": {"$ref": "input.image"}, "axis": "x", "axis_threshold": 0.2},
                                }
                            elif y_ok and not ry_ok:
                                warning["suggested_next_action"] = {
                                    "tool": "axis_localizer",
                                    "args": {"image": {"$ref": "input.image"}, "axis": "right_y", "axis_threshold": 0.2},
                                }
                            elif ry_ok and not y_ok:
                                warning["suggested_next_action"] = {
                                    "tool": "axis_localizer",
                                    "args": {"image": {"$ref": "input.image"}, "axis": "y", "axis_threshold": 0.2},
                                }
                        _safe_json_dump(os.path.join(step_dir, "observation.json"), warning)
                        history_lines.append(model_text.strip())
                        history_lines.append("OBSERVATION: {}".format(json.dumps(warning, ensure_ascii=False)))
                        answer_warnings.append(warning)

                # If the model is about to output an empty list for a bar chart after an unreliable axis check,
                # run the bar-value consistency tool once before finalizing.
                if (
                    major_type == "bar"
                    and not y_pred
                    and "bar_value_consistency" not in tools_used
                    and (
                        (isinstance(axis_status.get("y"), dict) and axis_status.get("y", {}).get("axis_present") is False)
                        or (isinstance(axis_status.get("right_y"), dict) and axis_status.get("right_y", {}).get("axis_present") is False)
                    )
                ):
                    steps_left = int(cfg.max_steps) - int(step_idx) - 1
                    if steps_left >= 2:
                        warning: Dict[str, Any] = {
                            "validation_warning": "Axis detection appears unreliable for a bar chart (axis_present=false). "
                            "An empty label set may be under-supported; consider running bar_value_consistency "
                            "to check whether printed values agree with bar sizes.",
                            "axis_status": axis_status,
                        }
                        if annotation_type == "annotated":
                            orient = "vertical"
                            if bar_orientation_hint in {"horizontal", "vertical"}:
                                orient = str(bar_orientation_hint)
                            warning["suggested_next_action"] = {
                                "tool": "bar_value_consistency",
                                "args": {"image": {"$ref": "input.image"}, "bar_orientation": orient},
                            }
                        _safe_json_dump(os.path.join(step_dir, "observation.json"), warning)
                        history_lines.append(model_text.strip())
                        history_lines.append("OBSERVATION: {}".format(json.dumps(warning, ensure_ascii=False)))
                        answer_warnings.append(warning)

                out = {
                    "image_path": image_path,
                    "y_pred": y_pred,
                    "raw_answer": ans,
                    "run_dir": run_dir,
                }
                if answer_warnings:
                    out["validation_warnings"] = answer_warnings
                _safe_json_dump(os.path.join(run_dir, "answer.json"), out)
                return out

            act = self._parse_action(model_text)
            if act is None:
                # Local VLMs sometimes violate the format; treat as a retryable validation error.
                obs = {
                    "validation_error": "Model output contained neither ACTION nor ANSWER. "
                    "Reformat your response to include exactly one `ACTION:` or `ANSWER:` line.",
                }
                _safe_json_dump(os.path.join(step_dir, "observation.json"), obs)
                history_lines.append(model_text.strip())
                history_lines.append("OBSERVATION: {}".format(json.dumps(obs, ensure_ascii=False)))
                continue

            tool_name, tool_args = act
            _safe_json_dump(os.path.join(step_dir, "action.json"), {"tool": tool_name, "args": tool_args})

            # Guardrail: skip repeated identical tool calls (common local-VLM failure mode).
            try:
                action_sig = json.dumps({"tool": tool_name, "args": tool_args}, ensure_ascii=False, sort_keys=True)
            except Exception:
                action_sig = None
            if action_sig is not None and action_sig == last_action_sig:
                obs = {
                    "validation_error": "Repeated identical ACTION. Do not repeat the same tool call; "
                    "use the previous observation and output ANSWER.",
                    "skipped_execution": True,
                }
                _safe_json_dump(os.path.join(step_dir, "observation.json"), obs)
                history_lines.append(model_text.strip())
                history_lines.append("OBSERVATION: {}".format(json.dumps(obs, ensure_ascii=False)))
                continue
            last_action_sig = action_sig

            tool_args = self._resolve_refs(tool_args, memory)
            # Provide image automatically if missing.
            if "image" not in tool_args:
                tool_args["image"] = current_image

            pre_tool_warnings: List[Dict[str, Any]] = []

            # Guardrail: avoid anchoring on misrepresentation when axis tick labels are available.
            # For annotated bar charts, prefer screening axis-based misleaders (e.g., dual axis, truncated axis)
            # before running bar_value_consistency.
            major_type = major_chart_type_from_metadata(metadata)
            annotation_type = ""
            if isinstance(metadata, dict):
                annotation_type = str(metadata.get("annotation_type", "") or "").strip().lower()
            if tool_name == "bar_value_consistency" and major_type == "bar" and annotation_type == "annotated":
                orient = str(tool_args.get("bar_orientation", "") or "").strip().lower()
                if bar_orientation_hint in {"horizontal", "vertical"}:
                    orient = str(bar_orientation_hint)
                axis_guess = "x" if orient == "horizontal" else "y"
                tick_key = "x_axis_ticker_values" if axis_guess == "x" else "y_axis_ticker_values"
                tick_list = metadata.get(tick_key) if isinstance(metadata, dict) else None
                has_axis_tickers = isinstance(tick_list, list) and len(tick_list) >= 3
                if has_axis_tickers and axis_guess not in axis_status:
                    pre_tool_warnings.append(
                        {
                            "validation_warning": "Metadata provides tick values for this bar chart axis. "
                            "Consider screening axis-based misleaders with axis_localizer before relying on bar_value_consistency.",
                            "suggested_next_action": {
                                "tool": "axis_localizer",
                                "args": {
                                    "image": {"$ref": "input.image"},
                                    "axis": axis_guess,
                                    "axis_threshold": 0.2,
                                    "axis_tickers": {"$ref": "input.metadata[{}]".format(tick_key)},
                                },
                            },
                        }
                    )

            arg_corrections: Dict[str, Any] = {}
            if tool_name == "bar_value_consistency" and bar_orientation_hint in {"horizontal", "vertical"}:
                want = str(bar_orientation_hint)
                got = str(tool_args.get("bar_orientation", "") or "").strip().lower()
                if want == "horizontal" and got in {"", "vertical", "vertical-right"}:
                    tool_args["bar_orientation"] = "horizontal"
                    arg_corrections["bar_orientation"] = "horizontal"
                elif want == "vertical" and got == "horizontal":
                    tool_args["bar_orientation"] = "vertical"
                    arg_corrections["bar_orientation"] = "vertical"

            # If axis tickers are available in metadata, use them to guide OCR by default (x/y only).
            if tool_name == "axis_localizer" and isinstance(metadata, dict):
                axis = str(tool_args.get("axis") or "").strip().lower()
                if axis in {"x", "y"} and "axis_tickers" not in tool_args:
                    tick_key = "x_axis_ticker_values" if axis == "x" else "y_axis_ticker_values"
                    ticks = metadata.get(tick_key)
                    if isinstance(ticks, list) and len(ticks) >= 3:
                        tool_args["axis_tickers"] = ticks
                        arg_corrections["axis_tickers"] = {"source": "metadata", "key": tick_key}
            _safe_json_dump(os.path.join(step_dir, "action_resolved.json"), {"tool": tool_name, "args": tool_args})

            try:
                tool_res = self.tools.execute(
                    tool=tool_name,
                    args=tool_args,
                    memory=memory,
                    run_step_dir=step_dir,
                    step_idx=step_idx,
                    cache_root=str(cfg.cache_root),
                    use_cache=bool(cfg.use_cache),
                )
                obs = {
                    "tool": tool_res.tool,
                    "result_summary": tool_res.result_summary,
                    "refs": tool_res.refs,
                    "artifacts": tool_res.artifacts,
                }
            except Exception as e:
                obs = {
                    "tool": tool_name,
                    "error": str(e),
                }

            # Non-blocking: include quality signals to help the LLM judge axis evidence reliability.
            if tool_name == "axis_localizer" and isinstance(obs, dict):
                rs = obs.get("result_summary")
                if isinstance(rs, dict):
                    obs["quality_signals"] = _axis_quality_signals(rs)

            if pre_tool_warnings and isinstance(obs, dict):
                obs.setdefault("validation_warnings", [])
                if isinstance(obs["validation_warnings"], list):
                    obs["validation_warnings"].extend(pre_tool_warnings)
                else:
                    obs["validation_warnings"] = pre_tool_warnings

            if arg_corrections:
                obs["arg_corrections"] = arg_corrections
            _safe_json_dump(os.path.join(step_dir, "observation.json"), obs)

            history_lines.append(model_text.strip())
            history_lines.append("OBSERVATION: {}".format(json.dumps(obs, ensure_ascii=False)))

            tools_used.add(str(tool_name))

            # Track axis reliability info for answer-time validation.
            if tool_name == "axis_localizer" and isinstance(obs, dict):
                rs = obs.get("result_summary")
                if isinstance(rs, dict):
                    ax = str(rs.get("axis") or "").strip().lower()
                    if ax in {"x", "top_x", "y", "right_y"}:
                        axis_status[ax] = rs

            # Update refs hint with new refs.
            if isinstance(obs, dict) and isinstance(obs.get("refs"), dict):
                for k, v in obs["refs"].items():
                    refs_hint += "- {} -> {}\n".format(k, v)
                    # Track visual outputs for later self-verification.
                    if isinstance(v, str) and isinstance(memory.get(v), Image.Image):
                        visual_history.append(("tool_image.{}".format(k), v))

            # If we cleaned the chart, adopt it as the default working image.
            if tool_name == "clean_chart_image" and isinstance(obs, dict) and isinstance(obs.get("refs"), dict):
                cleaned_ref = obs["refs"].get("cleaned_image")
                cleaned = memory.get(cleaned_ref) if isinstance(cleaned_ref, str) else None
                if isinstance(cleaned, Image.Image):
                    current_image = cleaned
                    memory["state.current_image"] = current_image
                    refs_hint += "- state.current_image updated -> {}\n".format(str(cleaned_ref))

        # Last-chance: force a final answer-only step (helps local VLMs that keep calling tools).
        final_step_idx = int(cfg.max_steps)
        final_step_dir = os.path.join(steps_dir, "step_{:03d}".format(final_step_idx))
        _ensure_dir(final_step_dir)

        images_for_llm: List[Image.Image] = []
        images_desc: List[str] = []

        def _attach_final(desc: str, im: Image.Image) -> None:
            if len(images_for_llm) >= int(cfg.max_attached_images):
                return
            images_for_llm.append(im)
            images_desc.append(desc)

        _attach_final("1) input.image (original chart)", img)
        if current_image is not img:
            _attach_final("2) state.current_image (working image)", current_image)
        for desc, ref_id in reversed(visual_history):
            if len(images_for_llm) >= int(cfg.max_attached_images):
                break
            im = memory.get(ref_id)
            if isinstance(im, Image.Image) and im is not img and im is not current_image:
                _attach_final("{} ({})".format(desc, ref_id), im)

        prompt = self._build_task_prompt(
            metadata=metadata,
            step_history="\n".join(history_lines),
            refs_hint=refs_hint,
        )
        prompt += (
            "\nFINAL STEP: Do NOT call tools.\n"
            "Output only:\n"
            "ANSWER: {\"misleader\": [...], \"confidence\": {...}, \"rationale\": \"...\"} TERMINATE\n"
        )
        if images_desc:
            prompt += "\nAttached images (in order):\n" + "\n".join(images_desc) + "\n"
        with open(os.path.join(final_step_dir, "prompt.txt"), "w") as f:
            f.write(prompt)

        model_text = self.llm.generate(prompt=prompt, images=images_for_llm or [img])
        with open(os.path.join(final_step_dir, "model_output.txt"), "w") as f:
            f.write(model_text)

        ans = self._parse_answer(model_text)
        if isinstance(ans, dict):
            y_pred_raw = ans.get("misleader") or []
            y_pred = (
                normalize_misleader_labels([str(x) for x in y_pred_raw], strict=False)
                if isinstance(y_pred_raw, list)
                else []
            )
            out = {
                "image_path": image_path,
                "y_pred": y_pred,
                "raw_answer": ans,
                "run_dir": run_dir,
            }
            _safe_json_dump(os.path.join(run_dir, "answer.json"), out)
            return out

        out = {
            "image_path": image_path,
            "y_pred": [],
            "error": "Max steps reached without ANSWER",
            "model_output": model_text,
            "run_dir": run_dir,
        }
        _safe_json_dump(os.path.join(run_dir, "answer.json"), out)
        return out
