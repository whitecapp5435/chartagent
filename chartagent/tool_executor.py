import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from PIL import ImageDraw

TOOL_RESULT_CACHE_VERSION = 1
_CACHE_IGNORE_TOPLEVEL_ARG_KEYS = {"debug_dir"}


def _jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, tuple):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, list):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    # PIL images and other objects are represented by their type string.
    return {"_type": type(obj).__name__}


def _save_image(path: str, image: Image.Image) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    image.convert("RGB").save(path)


def _safe_write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _sha1_hex(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _sha1_image_pixels(img: Image.Image) -> str:
    im = img.convert("RGBA")
    payload = im.tobytes() + str(tuple(im.size)).encode("utf-8")
    return _sha1_hex(payload)


def _to_jsonable_cache(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, tuple):
        return [_to_jsonable_cache(x) for x in obj]
    if isinstance(obj, list):
        return [_to_jsonable_cache(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable_cache(v) for k, v in obj.items()}
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return {"_type": type(obj).__name__}


def _normalize_for_cache_key(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Image.Image):
        return {"_type": "Image", "sha1": _sha1_image_pixels(obj), "size": list(obj.size)}
    if isinstance(obj, tuple):
        return [_normalize_for_cache_key(x) for x in obj]
    if isinstance(obj, list):
        return [_normalize_for_cache_key(x) for x in obj]
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k in sorted(obj.keys(), key=lambda x: str(x)):
            out[str(k)] = _normalize_for_cache_key(obj[k])
        return out
    raise TypeError("Unsupported type for tool cache key: {}".format(type(obj).__name__))


def _build_tool_cache_identity(tool: str, args: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    try:
        norm_args: Dict[str, Any] = {}
        for k, v in args.items():
            if str(k) in _CACHE_IGNORE_TOPLEVEL_ARG_KEYS:
                continue
            norm_args[str(k)] = _normalize_for_cache_key(v)
        payload = {"version": TOOL_RESULT_CACHE_VERSION, "tool": str(tool), "args": norm_args}
        s = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return _sha1_hex(s.encode("utf-8")), norm_args
    except Exception:
        return None


def _looks_like_masks_with_arrays(obj: Any) -> bool:
    if not isinstance(obj, list):
        return False
    try:
        import numpy as np  # type: ignore
    except Exception:
        return False
    for item in obj:
        if isinstance(item, dict) and isinstance(item.get("segmentation"), np.ndarray):
            return True
    return False


def _save_masks(values_dir: str, masks: List[object]) -> Tuple[str, str]:
    import numpy as np  # type: ignore

    meta: List[Dict[str, Any]] = []
    seg_arrays: Dict[str, np.ndarray] = {}
    for i, item in enumerate(masks):
        if not isinstance(item, dict):
            continue
        seg = item.get("segmentation")
        seg_key: Optional[str] = None
        if isinstance(seg, np.ndarray):
            seg_key = "seg_{}".format(i)
            seg_arrays[seg_key] = seg.astype(bool, copy=False)
        m: Dict[str, Any] = {}
        for k, v in item.items():
            if str(k) == "segmentation":
                continue
            m[str(k)] = _to_jsonable_cache(v)
        m["_segmentation_key"] = seg_key
        meta.append(m)

    meta_path = os.path.join(values_dir, "masks_meta.json")
    npz_path = os.path.join(values_dir, "masks_seg.npz")
    _safe_write_json(meta_path, meta)
    # Only write NPZ if we actually captured arrays; otherwise leave it absent.
    if seg_arrays:
        np.savez_compressed(npz_path, **seg_arrays)
    return meta_path, npz_path


def _load_masks(meta_path: str, npz_path: str) -> List[Dict[str, Any]]:
    import numpy as np  # type: ignore

    with open(meta_path, "r") as f:
        meta = json.load(f)
    if not isinstance(meta, list):
        raise TypeError("masks_meta.json must contain a list")
    arrays: Dict[str, np.ndarray] = {}
    if os.path.exists(npz_path):
        with np.load(npz_path) as data:
            arrays = {k: data[k] for k in data.files}
    out: List[Dict[str, Any]] = []
    for entry in meta:
        if not isinstance(entry, dict):
            continue
        seg_key = entry.pop("_segmentation_key", None)
        item = dict(entry)
        if isinstance(seg_key, str) and seg_key in arrays:
            item["segmentation"] = arrays[seg_key]
        out.append(item)
    return out


def _cache_paths(cache_root: str, tool: str, cache_key: str) -> Tuple[str, str, str, str]:
    cache_dir = os.path.join(cache_root, "tool_results", tool, cache_key)
    values_dir = os.path.join(cache_dir, "values")
    artifacts_dir = os.path.join(cache_dir, "artifacts")
    manifest_path = os.path.join(cache_dir, "manifest.json")
    return cache_dir, values_dir, artifacts_dir, manifest_path


def _load_from_cache(
    *,
    cache_dir: str,
    run_step_dir: str,
    store: Any,
    artifacts_out: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    manifest_path = os.path.join(cache_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        return None
    if int(manifest.get("version", -1)) != TOOL_RESULT_CACHE_VERSION:
        return None

    outputs = manifest.get("outputs") or {}
    if not isinstance(outputs, dict):
        return None

    # Restore refs/memory.
    for out_key, desc in outputs.items():
        if not isinstance(out_key, str) or not isinstance(desc, dict):
            continue
        kind = str(desc.get("kind") or "")
        if kind == "image":
            rel = desc.get("file")
            if not isinstance(rel, str):
                continue
            p = os.path.join(cache_dir, rel)
            with Image.open(p) as im:
                store(out_key, im.convert("RGB"))
            continue
        if kind == "masks":
            meta_rel = desc.get("meta_file")
            npz_rel = desc.get("npz_file")
            if not (isinstance(meta_rel, str) and isinstance(npz_rel, str)):
                continue
            meta_p = os.path.join(cache_dir, meta_rel)
            npz_p = os.path.join(cache_dir, npz_rel)
            store(out_key, _load_masks(meta_p, npz_p))
            continue
        # Default: json.
        rel = desc.get("file")
        if not isinstance(rel, str):
            continue
        p = os.path.join(cache_dir, rel)
        with open(p, "r") as f:
            store(out_key, json.load(f))

    # Restore artifacts (copy into run_step_dir for self-contained logs).
    artifacts = manifest.get("artifacts") or {}
    if isinstance(artifacts, dict):
        for art_key, rel in artifacts.items():
            if not (isinstance(art_key, str) and isinstance(rel, str)):
                continue
            src = os.path.join(cache_dir, rel)
            if not os.path.exists(src):
                continue
            dst = os.path.join(run_step_dir, rel)
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            shutil.copy2(src, dst)
            artifacts_out[art_key] = dst

    rs = manifest.get("result_summary") or {}
    return rs if isinstance(rs, dict) else {}


def _save_to_cache(
    *,
    cache_root: str,
    tool: str,
    cache_key: str,
    norm_args: Dict[str, Any],
    outputs: Dict[str, Any],
    result: "ToolExecutionResult",
    run_step_dir: str,
) -> None:
    cache_dir, values_dir, _artifacts_dir, manifest_path = _cache_paths(cache_root, tool, cache_key)
    if os.path.exists(manifest_path):
        return

    os.makedirs(values_dir, exist_ok=True)

    outputs_desc: Dict[str, Dict[str, Any]] = {}
    for out_key, obj in outputs.items():
        out_key_s = str(out_key)
        if isinstance(obj, Image.Image):
            fn = "{}.png".format(out_key_s)
            rel = os.path.join("values", fn)
            _save_image(os.path.join(cache_dir, rel), obj)
            outputs_desc[out_key_s] = {"kind": "image", "file": rel}
            continue
        if out_key_s == "masks" and _looks_like_masks_with_arrays(obj):
            meta_path, npz_path = _save_masks(os.path.join(cache_dir, "values"), obj)
            outputs_desc[out_key_s] = {
                "kind": "masks",
                "meta_file": os.path.relpath(meta_path, cache_dir),
                "npz_file": os.path.relpath(npz_path, cache_dir),
            }
            continue
        fn = "{}.json".format(out_key_s)
        rel = os.path.join("values", fn)
        _safe_write_json(os.path.join(cache_dir, rel), _to_jsonable_cache(obj))
        outputs_desc[out_key_s] = {"kind": "json", "file": rel}

    artifacts_desc: Dict[str, str] = {}
    for art_key, p in result.artifacts.items():
        if not (isinstance(art_key, str) and isinstance(p, str)):
            continue
        if not os.path.exists(p):
            continue
        rel = os.path.relpath(p, run_step_dir)
        if rel.startswith(".."):
            continue
        rel = rel.replace("\\", "/")
        dest = os.path.join(cache_dir, rel)
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        shutil.copy2(p, dest)
        artifacts_desc[art_key] = rel

    manifest = {
        "version": TOOL_RESULT_CACHE_VERSION,
        "tool": str(tool),
        "cache_key": str(cache_key),
        "args": norm_args,
        "result_summary": result.result_summary,
        "outputs": outputs_desc,
        "artifacts": artifacts_desc,
    }
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    tmp = manifest_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_to_jsonable_cache(manifest), f, ensure_ascii=False, indent=2)
    os.replace(tmp, manifest_path)


@dataclass
class ToolExecutionResult:
    tool: str
    result_summary: Dict[str, Any]
    refs: Dict[str, str]
    artifacts: Dict[str, str]


class ToolExecutor(object):
    def __init__(self) -> None:
        from chartagent_tools import tools as tool_mod  # local import to keep agent import light

        self._tools = tool_mod

    def execute(
        self,
        *,
        tool: str,
        args: Dict[str, Any],
        memory: Dict[str, Any],
        run_step_dir: str,
        step_idx: int,
        cache_root: Optional[str] = None,
        use_cache: bool = True,
    ) -> ToolExecutionResult:
        name = str(tool or "").strip()
        if not name:
            raise ValueError("tool name is required")

        os.makedirs(run_step_dir, exist_ok=True)

        refs: Dict[str, str] = {}
        artifacts: Dict[str, str] = {}
        outputs: Dict[str, Any] = {}

        cache_id: Optional[Tuple[str, Dict[str, Any]]] = None
        if use_cache and cache_root:
            cache_id = _build_tool_cache_identity(name, args)

        def _store(key: str, obj: Any) -> str:
            ref_id = "step_{:03d}.{}".format(int(step_idx), key)
            memory[ref_id] = obj
            outputs[str(key)] = obj
            refs[key] = ref_id
            return ref_id

        # Cache hit: materialize cached outputs + artifacts into this run dir.
        if cache_id is not None and cache_root:
            cache_key, _norm_args = cache_id
            cache_dir, _values_dir, _artifacts_dir, manifest_path = _cache_paths(cache_root, name, cache_key)
            if os.path.exists(manifest_path):
                try:
                    rs = _load_from_cache(
                        cache_dir=cache_dir,
                        run_step_dir=run_step_dir,
                        store=_store,
                        artifacts_out=artifacts,
                    )
                    if rs is not None:
                        rs_out = dict(rs)
                        rs_out["cache_hit"] = True
                        # Backfill newer fields on cache hits for selected tools.
                        if name == "axis_localizer" and "axis_present" not in rs_out:
                            try:
                                img0 = args.get("image")
                                axis0 = str(args.get("axis", "") or "").strip()
                                thr0 = float(args.get("axis_threshold", 0.2))
                                if isinstance(img0, Image.Image):
                                    axis_values0 = memory.get(refs.get("axis_values", ""))
                                    axis_pos0 = memory.get(refs.get("axis_pixel_positions", ""))
                                    tick_bboxes0 = memory.get(refs.get("axis_tick_bboxes", ""))
                                    if isinstance(axis_values0, list) and isinstance(axis_pos0, list):
                                        W, H = img0.size
                                        ax_s = str(axis0 or "").strip().lower()
                                        is_y = ax_s in ("y", "right_y")
                                        n = int(len(axis_values0))
                                        if n >= 2:
                                            dim = float(H if is_y else W)
                                            try:
                                                pos_range = float(max(axis_pos0) - min(axis_pos0))
                                            except Exception:
                                                pos_range = 0.0
                                            span_score = float(max(0.0, min(1.0, pos_range / float(max(1.0, 0.55 * dim)))))
                                            n_score = float(max(0.0, min(1.0, (float(n) - 1.0) / 4.0)))
                                            corr_score = 0.0
                                            if n >= 3:
                                                import math

                                                pairs = []
                                                for p, v in zip(axis_pos0, axis_values0):
                                                    try:
                                                        pairs.append((float(p), float(v)))
                                                    except Exception:
                                                        continue
                                                pairs.sort(key=lambda t: t[0])
                                                if len(pairs) >= 3:
                                                    ps = [p for p, _ in pairs]
                                                    vs = [v for _, v in pairs]
                                                    mp = sum(ps) / float(len(ps))
                                                    mv = sum(vs) / float(len(vs))
                                                    cov = sum((p - mp) * (v - mv) for p, v in zip(ps, vs))
                                                    vp = sum((p - mp) ** 2 for p in ps)
                                                    vv = sum((v - mv) ** 2 for v in vs)
                                                    if vp > 1e-9 and vv > 1e-9:
                                                        corr = float(cov / float(math.sqrt(vp * vv)))
                                                        sign_ok = corr < 0.0 if is_y else corr > 0.0
                                                        corr_score = float(max(0.0, min(1.0, abs(corr)))) if sign_ok else 0.0

                                            edge_score = 0.5
                                            thr_clamped = max(0.05, min(0.45, float(thr0)))
                                            if ax_s == "y":
                                                roi = (0, 0, int(round(thr_clamped * W)), H)
                                            elif ax_s == "right_y":
                                                roi = (int(round((1.0 - thr_clamped) * W)), 0, W, H)
                                            else:
                                                roi = (0, int(round((1.0 - thr_clamped) * H)), W, H)
                                            x0, y0, x1, y1 = [int(v) for v in roi]
                                            rw = max(1, int(x1 - x0))
                                            rh = max(1, int(y1 - y0))
                                            if isinstance(tick_bboxes0, list) and tick_bboxes0:
                                                xs = []
                                                ys = []
                                                for bb in tick_bboxes0:
                                                    if not bb:
                                                        continue
                                                    try:
                                                        bx1, by1, bx2, by2 = [int(v) for v in bb]
                                                    except Exception:
                                                        continue
                                                    xs.append(0.5 * float(bx1 + bx2))
                                                    ys.append(0.5 * float(by1 + by2))
                                                if xs:
                                                    xs_sorted = sorted(xs)
                                                    mid = len(xs_sorted) // 2
                                                    med_x = float(xs_sorted[mid]) if len(xs_sorted) % 2 == 1 else 0.5 * float(xs_sorted[mid - 1] + xs_sorted[mid])
                                                    if ax_s == "y":
                                                        rel = (med_x - float(x0)) / float(rw)
                                                        edge_score = 1.0 if rel <= 0.82 else 0.0
                                                    elif ax_s == "right_y":
                                                        rel = (med_x - float(x0)) / float(rw)
                                                        edge_score = 1.0 if rel >= 0.18 else 0.0
                                                    else:
                                                        ys_sorted = sorted(ys)
                                                        midy = len(ys_sorted) // 2
                                                        med_y = float(ys_sorted[midy]) if len(ys_sorted) % 2 == 1 else 0.5 * float(ys_sorted[midy - 1] + ys_sorted[midy])
                                                        rel = (med_y - float(y0)) / float(rh)
                                                        edge_score = 1.0 if rel >= 0.25 else 0.0

                                            conf = (0.35 * n_score) + (0.25 * span_score) + (0.20 * corr_score) + (0.20 * edge_score)
                                            if n < 3:
                                                conf *= 0.30
                                            conf = float(max(0.0, min(1.0, conf)))
                                            axis_present = bool(conf >= 0.55 and n >= 3)
                                            rs_out.update(
                                                {
                                                    "axis_present": axis_present,
                                                    "axis_confidence": conf,
                                                    "roi_bbox_xyxy": [x0, y0, x1, y1],
                                                    "warning": "too_few_ticks" if (not axis_present and n < 3) else "low_confidence",
                                                }
                                            )
                            except Exception:
                                pass
                        # Hide misleading tick arrays for absent/unreliable axes (kept in refs for debugging).
                        if name == "axis_localizer" and rs_out.get("axis_present") is False:
                            rs_out["axis_values"] = []
                            rs_out["axis_pixel_positions"] = []
                        return ToolExecutionResult(tool=name, result_summary=rs_out, refs=refs, artifacts=artifacts)
                except Exception:
                    # Best-effort: fall through to normal execution if cache read fails.
                    pass

        def _maybe_get_image(v: Any) -> Image.Image:
            if isinstance(v, Image.Image):
                return v
            raise TypeError("Expected PIL.Image.Image for image argument")

        def _finalize(res: ToolExecutionResult) -> ToolExecutionResult:
            if cache_id is None or not cache_root or not use_cache:
                return res
            try:
                cache_key, norm_args = cache_id
                _save_to_cache(
                    cache_root=str(cache_root),
                    tool=name,
                    cache_key=cache_key,
                    norm_args=norm_args,
                    outputs=outputs,
                    result=res,
                    run_step_dir=run_step_dir,
                )
            except Exception:
                pass
            return res

        # ---- Tools ----
        if name == "clean_chart_image":
            img = _maybe_get_image(args.get("image"))
            title = args.get("title", self._tools._AUTO)  # type: ignore[attr-defined]
            legend = args.get("legend", self._tools._AUTO)  # type: ignore[attr-defined]
            out = self._tools.clean_chart_image(img, title=title, legend=legend)
            _store("cleaned_image", out)
            p = os.path.join(run_step_dir, "cleaned.png")
            _save_image(p, out)
            artifacts["cleaned_image"] = p
            return _finalize(ToolExecutionResult(
                tool=name,
                result_summary={"image": "stored", "size": list(out.size)},
                refs=refs,
                artifacts=artifacts,
            ))

        if name == "annotate_legend":
            img = _maybe_get_image(args.get("image"))
            legend = args.get("legend")
            debug_dir = args.get("debug_dir")
            if debug_dir is None:
                debug_dir = os.path.join(run_step_dir, "annot_debug")
            legend_img, labeled, mapping = self._tools.annotate_legend(img, legend, debug_dir=str(debug_dir))
            _store("legend_image", legend_img)
            _store("labeled_legend", labeled)
            _store("bbox_mapping", mapping)
            p1 = os.path.join(run_step_dir, "legend_crop.png")
            p2 = os.path.join(run_step_dir, "labeled_legend.png")
            _save_image(p1, legend_img)
            _save_image(p2, labeled)
            artifacts["legend_image"] = p1
            artifacts["labeled_legend"] = p2
            # mapping can be large; store file too.
            pj = os.path.join(run_step_dir, "bbox_mapping.json")
            _safe_write_json(pj, _jsonable(mapping))
            artifacts["bbox_mapping_json"] = pj
            return _finalize(ToolExecutionResult(
                tool=name,
                result_summary={
                    "legend_crop_size": list(legend_img.size),
                    "n_mapping": len(mapping) if isinstance(mapping, dict) else None,
                    "debug_dir": str(debug_dir),
                },
                refs=refs,
                artifacts=artifacts,
            ))

        if name == "get_marker_rgb":
            img = _maybe_get_image(args.get("image"))
            bbox_mapping = args.get("bbox_mapping")
            text = args.get("text_of_interest")
            label = args.get("label_of_interest")
            dist = args.get("distance_between_text_and_marker", 5)
            rgb = self._tools.get_marker_rgb(
                img,
                bbox_mapping,
                text_of_interest=text,
                label_of_interest=label,
                distance_between_text_and_marker=int(dist),
            )
            _store("rgb", tuple(rgb))
            return _finalize(ToolExecutionResult(
                tool=name,
                result_summary={"rgb": list(rgb)},
                refs=refs,
                artifacts=artifacts,
            ))

        if name == "segment_and_mark":
            img = _maybe_get_image(args.get("image"))
            debug_dir = args.get("debug_dir")
            if debug_dir is None:
                debug_dir = os.path.join(run_step_dir, "seg_debug")
            labeled, masks = self._tools.segment_and_mark(
                img,
                segmentation_model=str(args.get("segmentation_model", "SAM")),
                min_area=int(args.get("min_area", 5000)),
                iou_thresh_unique=float(args.get("iou_thresh_unique", 0.9)),
                iou_thresh_composite=float(args.get("iou_thresh_composite", 0.98)),
                white_ratio_thresh=float(args.get("white_ratio_thresh", 0.95)),
                remove_background_color=bool(args.get("remove_background_color", False)),
                debug_dir=str(debug_dir),
                metadata=args.get("metadata"),
            )
            _store("labeled_image", labeled)
            _store("masks", masks)
            p = os.path.join(run_step_dir, "seg_labeled.png")
            _save_image(p, labeled)
            artifacts["labeled_image"] = p
            pj = os.path.join(run_step_dir, "masks.json")
            # Store only jsonable version; raw masks may be heavy.
            _safe_write_json(pj, _jsonable(masks))
            artifacts["masks_json"] = pj
            return _finalize(ToolExecutionResult(
                tool=name,
                result_summary={
                    "n_masks": len(masks) if isinstance(masks, list) else None,
                    "debug_dir": str(debug_dir),
                },
                refs=refs,
                artifacts=artifacts,
            ))

        if name == "axis_localizer":
            img = _maybe_get_image(args.get("image"))
            axis = str(args.get("axis", "") or "").strip()
            thr = float(args.get("axis_threshold", 0.2))
            axis_tickers = args.get("axis_tickers")
            axis_tickers_source = "none"
            # Security/robustness: treat axis_tickers as metadata-derived; ignore ad-hoc tickers when metadata lacks them.
            meta = memory.get("input.metadata")
            if isinstance(meta, dict):
                key = None
                ax_s = str(axis or "").strip().lower()
                if ax_s == "x":
                    key = "x_axis_ticker_values"
                elif ax_s == "y":
                    key = "y_axis_ticker_values"
                elif ax_s == "right_y":
                    key = "right_y_axis_ticker_values"
                meta_ticks = meta.get(key) if key else None
                if isinstance(meta_ticks, list) and any(str(t).strip() for t in meta_ticks):
                    axis_tickers = meta_ticks
                    axis_tickers_source = "metadata"
                elif axis_tickers is not None:
                    # If metadata has no tick list, ignore user/model-provided tickers to prevent hallucinated axes.
                    axis_tickers = None
                    axis_tickers_source = "ignored_no_metadata"
            elif axis_tickers is not None:
                axis_tickers_source = "args"
            tick_bboxes = None
            # Prefer the internal helper that also returns per-tick bboxes so we can generate a preview.
            if hasattr(self._tools, "_axis_localizer_with_boxes"):
                axis_values, axis_pixel_positions, tick_bboxes = self._tools._axis_localizer_with_boxes(  # type: ignore[attr-defined]
                    img,
                    axis=axis,
                    axis_threshold=thr,
                    axis_tickers=axis_tickers,
                )
            else:
                axis_values, axis_pixel_positions = self._tools.axis_localizer(
                    img,
                    axis=axis,
                    axis_threshold=thr,
                    axis_tickers=axis_tickers,
                )
            _store("axis_values", list(axis_values))
            _store("axis_pixel_positions", list(axis_pixel_positions))
            if isinstance(tick_bboxes, list) and tick_bboxes:
                _store("axis_tick_bboxes", tick_bboxes)
                # Draw a preview for visual self-verification.
                preview = img.convert("RGB").copy()
                draw = ImageDraw.Draw(preview)
                for i, bb in enumerate(tick_bboxes):
                    if not bb:
                        continue
                    try:
                        x1, y1, x2, y2 = [int(v) for v in bb]
                    except Exception:
                        continue
                    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
                    label = str(axis_values[i]) if i < len(axis_values) else str(i)
                    draw.text((x1, max(0, y1 - 12)), label, fill=(0, 0, 255))
                _store("axis_preview", preview)
                p = os.path.join(run_step_dir, "axis_{}_preview.png".format(axis or "axis"))
                _save_image(p, preview)
                artifacts["axis_preview"] = p

            # Heuristic confidence to avoid mistaking in-plot annotations for axis ticks.
            def _median(xs: list[float]) -> float:
                if not xs:
                    return 0.0
                ys = sorted(xs)
                m = len(ys) // 2
                if len(ys) % 2 == 1:
                    return float(ys[m])
                return 0.5 * float(ys[m - 1] + ys[m])

            def _axis_confidence() -> dict[str, object]:
                W, H = img.size
                n = int(len(axis_values))
                if n < 2:
                    return {"axis_present": False, "axis_confidence": 0.0, "warning": "too_few_ticks"}

                axis_s = str(axis or "").strip().lower()
                is_y = axis_s in ("y", "right_y")
                dim = float(H if is_y else W)
                try:
                    pos_range = float(max(axis_pixel_positions) - min(axis_pixel_positions))
                except Exception:
                    pos_range = 0.0
                span_score = float(max(0.0, min(1.0, pos_range / float(max(1.0, 0.55 * dim)))))
                n_score = float(max(0.0, min(1.0, (float(n) - 1.0) / 4.0)))

                # Correlation/monotonicity check.
                pairs = []
                for p, v in zip(axis_pixel_positions, axis_values):
                    try:
                        pairs.append((float(p), float(v)))
                    except Exception:
                        continue
                pairs.sort(key=lambda t: t[0])
                corr_score = 0.0
                if len(pairs) >= 3:
                    import math

                    ps = [p for p, _ in pairs]
                    vs = [v for _, v in pairs]
                    mp = sum(ps) / float(len(ps))
                    mv = sum(vs) / float(len(vs))
                    cov = sum((p - mp) * (v - mv) for p, v in zip(ps, vs))
                    vp = sum((p - mp) ** 2 for p in ps)
                    vv = sum((v - mv) ** 2 for v in vs)
                    if vp > 1e-9 and vv > 1e-9:
                        corr = float(cov / float(math.sqrt(vp * vv)))
                        sign_ok = corr < 0.0 if is_y else corr > 0.0
                        corr_score = float(max(0.0, min(1.0, abs(corr)))) if sign_ok else 0.0

                edge_score = 0.5
                # Compute ROI like tools.axis_localizer: edge heuristics depend on where OCR boxes fall within the ROI.
                try:
                    thr0 = max(0.05, min(0.45, float(thr)))
                except Exception:
                    thr0 = 0.2
                if axis_s == "y":
                    roi = (0, 0, int(round(thr0 * W)), H)
                elif axis_s == "right_y":
                    roi = (int(round((1.0 - thr0) * W)), 0, W, H)
                else:
                    roi = (0, int(round((1.0 - thr0) * H)), W, H)
                x0, y0, x1, y1 = [int(v) for v in roi]
                rw = max(1, int(x1 - x0))
                rh = max(1, int(y1 - y0))
                if isinstance(tick_bboxes, list) and tick_bboxes:
                    xs = []
                    ys = []
                    for bb in tick_bboxes:
                        if not bb:
                            continue
                        try:
                            bx1, by1, bx2, by2 = [int(v) for v in bb]
                        except Exception:
                            continue
                        xs.append(0.5 * float(bx1 + bx2))
                        ys.append(0.5 * float(by1 + by2))
                    if xs:
                        if axis_s == "y":
                            rel = (_median(xs) - float(x0)) / float(rw)
                            # If all "ticks" are deep inside the ROI (near the plot), it's often bar labels / annotations.
                            edge_score = 1.0 if rel <= 0.82 else 0.0
                        elif axis_s == "right_y":
                            rel = (_median(xs) - float(x0)) / float(rw)
                            # Right-y tick labels should not be glued to the ROI's left edge (inside the plot).
                            edge_score = 1.0 if rel >= 0.18 else 0.0
                        else:
                            rel = (_median(ys) - float(y0)) / float(rh)
                            edge_score = 1.0 if rel >= 0.25 else 0.0

                # Combine scores. Cap confidence when we have <3 ticks (too weak to diagnose axis-based misleaders).
                conf = (0.35 * n_score) + (0.25 * span_score) + (0.20 * corr_score) + (0.20 * edge_score)
                if n < 3:
                    conf *= 0.30
                conf = float(max(0.0, min(1.0, conf)))
                axis_present = bool(conf >= 0.55 and n >= 3)

                warn = ""
                if not axis_present:
                    if n < 3:
                        warn = "too_few_ticks"
                    elif edge_score <= 0.1:
                        warn = "ticks_look_in_plot"
                    elif corr_score <= 0.1 and n >= 3:
                        warn = "non_monotonic_or_nonlinear"
                    else:
                        warn = "low_confidence"

                return {
                    "axis_present": axis_present,
                    "axis_confidence": conf,
                    "roi_bbox_xyxy": [x0, y0, x1, y1],
                    "warning": warn or None,
                }

            conf_info = _axis_confidence()
            # If the axis is deemed absent/unreliable, hide the (often misleading) raw tick lists from the LLM-facing
            # summary. The raw arrays are still available via refs for debugging.
            axis_values_out = axis_values[:10]
            axis_pixel_positions_out = axis_pixel_positions[:10]
            if isinstance(conf_info, dict) and conf_info.get("axis_present") is False:
                axis_values_out = []
                axis_pixel_positions_out = []
            return _finalize(ToolExecutionResult(
                tool=name,
                result_summary={
                    "axis": axis,
                    "n_ticks": len(axis_values),
                    "axis_values": axis_values_out,
                    "axis_pixel_positions": axis_pixel_positions_out,
                    "has_preview": bool(artifacts.get("axis_preview")),
                    "axis_tickers_source": axis_tickers_source,
                    **conf_info,
                },
                refs=refs,
                artifacts=artifacts,
            ))

        if name == "interpolate_pixel_to_value":
            pixel = float(args.get("pixel"))
            axis_values = args.get("axis_values")
            axis_pixel_positions = args.get("axis_pixel_positions")
            val = self._tools.interpolate_pixel_to_value(pixel, axis_values, axis_pixel_positions)
            _store("value", float(val))
            return _finalize(ToolExecutionResult(
                tool=name,
                result_summary={"value": float(val)},
                refs=refs,
                artifacts=artifacts,
            ))

        if name == "arithmetic":
            a = float(args.get("a"))
            b = float(args.get("b"))
            op = str(args.get("operation"))
            out = self._tools.arithmetic(a=a, b=b, operation=op)
            _store("result", out)
            return _finalize(ToolExecutionResult(
                tool=name,
                result_summary={"result": out},
                refs=refs,
                artifacts=artifacts,
            ))

        if name == "compute_segment_area":
            img = _maybe_get_image(args.get("image"))
            measure = str(args.get("measure") or "").strip()
            rgb = args.get("rgb_of_interest")
            masks = args.get("masks")
            labels = args.get("mask_labels_of_interest")

            rgb_t: Optional[Tuple[int, int, int]] = None
            if rgb is not None:
                if isinstance(rgb, (list, tuple)) and len(rgb) == 3:
                    rgb_t = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
                else:
                    raise TypeError("rgb_of_interest must be a 3-item list/tuple or null")

            vis, area = self._tools.compute_segment_area(
                img,
                filter_rgb=rgb_t,
                measure=measure,
                masks=masks,
                filter_segment=labels,
            )
            _store("visualization_image", vis)
            _store("area", int(area))

            p = os.path.join(run_step_dir, "segment_area_preview.png")
            _save_image(p, vis)
            artifacts["area_preview"] = p

            pj = os.path.join(run_step_dir, "segment_area.json")
            _safe_write_json(
                pj,
                {
                    "area": int(area),
                    "measure": measure,
                    "rgb_of_interest": list(rgb_t) if rgb_t is not None else None,
                    "mask_labels_of_interest": labels,
                },
            )
            artifacts["area_json"] = pj
            return _finalize(ToolExecutionResult(
                tool=name,
                result_summary={
                    "area": int(area),
                    "measure": measure,
                    "used_rgb": list(rgb_t) if rgb_t is not None else None,
                },
                refs=refs,
                artifacts=artifacts,
            ))

        if name == "bar_value_consistency":
            img = _maybe_get_image(args.get("image"))
            orient = str(args.get("bar_orientation", "vertical"))
            debug_dir = args.get("debug_dir")
            preview, result = self._tools.bar_value_consistency(
                img,
                bar_orientation=orient,
                debug_dir=str(debug_dir) if isinstance(debug_dir, str) and debug_dir.strip() else None,
            )
            _store("preview_image", preview)
            _store("result", result)

            p = os.path.join(run_step_dir, "bar_value_consistency_preview.png")
            _save_image(p, preview)
            artifacts["preview"] = p

            pj = os.path.join(run_step_dir, "bar_value_consistency.json")
            _safe_write_json(pj, result)
            artifacts["result_json"] = pj

            rs = dict(result) if isinstance(result, dict) else {}
            return _finalize(ToolExecutionResult(
                tool=name,
                result_summary={
                    "bar_orientation": orient,
                    "n_value_labels": int(rs.get("n_value_labels", 0) or 0),
                    "n_pairs": int(rs.get("n_pairs", 0) or 0),
                    "kendall_tau": rs.get("kendall_tau"),
                    "discordant_ratio": rs.get("discordant_ratio"),
                    "is_mismatch": bool(rs.get("is_mismatch", False)),
                    "has_preview": True,
                },
                refs=refs,
                artifacts=artifacts,
            ))

        if name == "get_bar":
            img = _maybe_get_image(args.get("image"))
            rgb = args.get("rgb_of_interest")
            ticker = args.get("ticker_label")
            out = self._tools.get_bar(
                img,
                rgb_of_interest=rgb,
                ticker_label=ticker,
                segmentation_model=str(args.get("segmentation_model", "SAM")),
                bar_orientation=str(args.get("bar_orientation", "vertical")),
            )
            _store("bar_bbox_xyxy", tuple(out))
            # Preview for self-verification: draw the selected bar bbox.
            try:
                x1, y1, x2, y2 = [int(v) for v in out]
                preview = img.convert("RGB").copy()
                draw = ImageDraw.Draw(preview)
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
                _store("bar_preview", preview)
                p = os.path.join(run_step_dir, "bar_preview.png")
                _save_image(p, preview)
                artifacts["bar_preview"] = p
            except Exception:
                pass
            return _finalize(ToolExecutionResult(
                tool=name,
                result_summary={"bar_bbox_xyxy": list(out), "has_preview": bool(artifacts.get("bar_preview"))},
                refs=refs,
                artifacts=artifacts,
            ))

        if name == "compute_bar_height":
            img = _maybe_get_image(args.get("image"))
            bar = args.get("bar_of_interest")
            if not (isinstance(bar, (list, tuple)) and len(bar) == 4):
                raise TypeError("bar_of_interest must be a 4-item list/tuple (xyxy or xywh)")

            orient = str(args.get("bar_orientation", "vertical"))
            axis_threshold = float(args.get("axis_threshold", 0.15))
            x_axis_tickers = args.get("x_axis_tickers")
            y_axis_tickers = args.get("y_axis_tickers")
            x_axis_title = args.get("x_axis_title")
            y_axis_title = args.get("y_axis_title")

            val = self._tools.compute_bar_height(
                img,
                bar_of_interest=tuple(int(v) for v in bar),
                bar_orientation=orient,
                axis_threshold=axis_threshold,
                x_axis_tickers=x_axis_tickers,
                y_axis_tickers=y_axis_tickers,
                x_axis_title=x_axis_title,
                y_axis_title=y_axis_title,
            )
            _store("bar_value", float(val))
            pj = os.path.join(run_step_dir, "bar_value.json")
            _safe_write_json(pj, {"bar_value": float(val)})
            artifacts["bar_value_json"] = pj
            return _finalize(ToolExecutionResult(
                tool=name,
                result_summary={"bar_value": float(val)},
                refs=refs,
                artifacts=artifacts,
            ))

        if name == "get_boxplot":
            img = _maybe_get_image(args.get("image"))
            masks = args.get("masks")
            if not isinstance(masks, list):
                raise TypeError("get_boxplot requires `masks` (list) from segment_and_mark")

            out = self._tools.get_boxplot(
                img,
                masks=masks,
                rgb_of_interest=args.get("rgb_of_interest"),
                ticker_label=args.get("ticker_label"),
                box_labels_of_interest=args.get("box_labels_of_interest"),
                boxplot_orientation=str(args.get("boxplot_orientation", "vertical")),
                axis_threshold=float(args.get("axis_threshold", 0.15)),
            )
            _store("boxplot_of_interest", out)
            pj = os.path.join(run_step_dir, "boxplot.json")
            _safe_write_json(pj, _jsonable(out))
            artifacts["boxplot_json"] = pj

            # Optional preview: draw all returned bboxes.
            try:
                preview = img.convert("RGB").copy()
                draw = ImageDraw.Draw(preview)
                for i, bb in enumerate(out, start=1):
                    x1, y1, x2, y2 = [int(v) for v in bb]
                    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
                    draw.text((x1, max(0, y1 - 12)), str(i), fill=(0, 0, 255))
                _store("boxplot_preview", preview)
                p = os.path.join(run_step_dir, "boxplot_preview.png")
                _save_image(p, preview)
                artifacts["boxplot_preview"] = p
            except Exception:
                pass
            return _finalize(ToolExecutionResult(
                tool=name,
                result_summary={"n_bboxes": len(out) if isinstance(out, list) else None},
                refs=refs,
                artifacts=artifacts,
            ))

        if name == "compute_boxplot_entity":
            img = _maybe_get_image(args.get("image"))
            boxplot = args.get("boxplot_of_interest")
            if not (isinstance(boxplot, list) and boxplot):
                raise TypeError("boxplot_of_interest must be a non-empty list of bboxes")

            orient = str(args.get("boxplot_orientation", "vertical"))
            entity = str(args.get("entity_of_interest", "median"))
            axis_threshold = float(args.get("axis_threshold", 0.15))
            x_axis_tickers = args.get("x_axis_tickers")
            y_axis_tickers = args.get("y_axis_tickers")

            val = self._tools.compute_boxplot_entity(
                img,
                boxplot_of_interest=boxplot,
                boxplot_orientation=orient,
                entity_of_interest=entity,
                axis_threshold=axis_threshold,
                x_axis_tickers=x_axis_tickers,
                y_axis_tickers=y_axis_tickers,
            )
            _store("boxplot_entity_value", float(val))
            pj = os.path.join(run_step_dir, "boxplot_entity_value.json")
            _safe_write_json(pj, {"boxplot_entity_value": float(val), "entity_of_interest": entity})
            artifacts["boxplot_entity_value_json"] = pj
            return _finalize(ToolExecutionResult(
                tool=name,
                result_summary={"boxplot_entity_value": float(val), "entity_of_interest": entity},
                refs=refs,
                artifacts=artifacts,
            ))

        if name == "get_edgepoints":
            img = _maybe_get_image(args.get("image"))
            out = self._tools.get_edgepoints(
                img,
                masks=args.get("masks"),
                rgb_of_interest=args.get("rgb_of_interest"),
                ticker_label=args.get("ticker_label"),
                mask_labels_of_interest=args.get("mask_labels_of_interest"),
                chart_orientation=str(args.get("chart_orientation", "vertical")),
                lineplot_get_dot=bool(args.get("lineplot_get_dot", False)),
                axis_threshold=float(args.get("axis_threshold", 0.15)),
            )
            _store("edgepoints", out)
            pj = os.path.join(run_step_dir, "edgepoints.json")
            _safe_write_json(pj, _jsonable(out))
            artifacts["edgepoints_json"] = pj

            # Optional preview: draw points.
            try:
                preview = img.convert("RGB").copy()
                draw = ImageDraw.Draw(preview)
                for (x, y) in out:
                    r = 4
                    draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 0), width=2)
                _store("edgepoints_preview", preview)
                p = os.path.join(run_step_dir, "edgepoints_preview.png")
                _save_image(p, preview)
                artifacts["edgepoints_preview"] = p
            except Exception:
                pass
            return _finalize(ToolExecutionResult(
                tool=name,
                result_summary={"edgepoints": _jsonable(out)},
                refs=refs,
                artifacts=artifacts,
            ))

        if name == "get_radial":
            img = _maybe_get_image(args.get("image"))
            rgb = args.get("rgb_of_interest")
            ticker = args.get("ticker_label")
            seg = str(args.get("segmentation_model", "color"))
            out = self._tools.get_radial(
                img,
                rgb_of_interest=rgb,
                ticker_label=ticker,
                segmentation_model=seg,
            )
            _store("radial_bbox_xyxy", tuple(out))
            pj = os.path.join(run_step_dir, "radial.json")
            _safe_write_json(
                pj,
                {
                    "bbox_xyxy": [int(v) for v in out],
                    "rgb_of_interest": rgb,
                    "ticker_label": ticker,
                    "segmentation_model": seg,
                },
            )
            artifacts["radial_json"] = pj
            return _finalize(ToolExecutionResult(
                tool=name,
                result_summary={"radial_bbox_xyxy": list(out)},
                refs=refs,
                artifacts=artifacts,
            ))

        if name == "analyze_radial_geometry":
            img = _maybe_get_image(args.get("image"))
            bbox = args.get("radial_bbox_xyxy")
            rgb = args.get("rgb_of_interest")

            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                raise TypeError("radial_bbox_xyxy must be a 4-item list/tuple (x1,y1,x2,y2)")
            if not (isinstance(rgb, (list, tuple)) and len(rgb) == 3):
                raise TypeError("rgb_of_interest must be a 3-item list/tuple (R,G,B)")

            try:
                import cv2  # type: ignore
                import numpy as np  # type: ignore
            except Exception as e:
                raise RuntimeError("analyze_radial_geometry requires opencv-python (cv2) and numpy.") from e

            x1, y1, x2, y2 = [int(v) for v in bbox]
            W, H = img.size
            x1 = max(0, min(x1, W))
            x2 = max(0, min(x2, W))
            y1 = max(0, min(y1, H))
            y2 = max(0, min(y2, H))
            if x2 <= x1 or y2 <= y1:
                raise ValueError("radial_bbox_xyxy is empty after clipping")

            roi = img.crop((x1, y1, x2, y2)).convert("RGB")
            arr = np.asarray(roi, dtype=np.int16)
            rgb_np = np.asarray([int(rgb[0]), int(rgb[1]), int(rgb[2])], dtype=np.int16).reshape(1, 1, 3)
            diff = np.abs(arr - rgb_np).sum(axis=2)
            tol = int(args.get("rgb_tol_l1", 120))
            mask = (diff <= tol).astype("uint8") * 255
            if int(mask.sum()) == 0:
                raise RuntimeError("No pixels matched rgb_of_interest inside radial_bbox_xyxy")

            contours, _hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                raise RuntimeError("Failed to extract contour for radial segment")
            contour = max(contours, key=lambda c: float(cv2.contourArea(c)))
            contour = contour.copy()
            contour[:, 0, 0] += int(x1)
            contour[:, 0, 1] += int(y1)

            vis, cx, cy, r_outer, r_max = self._tools.analyze_radial_geometry(img, contour_of_interest=contour)
            geometry = {
                "center_x": int(cx),
                "center_y": int(cy),
                "r_outer": float(r_outer),
                "r_max": float(r_max),
            }
            _store("radial_geometry", geometry)
            _store("radial_geometry_preview", vis)

            p = os.path.join(run_step_dir, "radial_geometry_preview.png")
            _save_image(p, vis)
            artifacts["radial_geometry_preview"] = p

            pj = os.path.join(run_step_dir, "radial_geometry.json")
            _safe_write_json(pj, _jsonable(geometry))
            artifacts["radial_geometry_json"] = pj
            return _finalize(ToolExecutionResult(
                tool=name,
                result_summary=geometry,
                refs=refs,
                artifacts=artifacts,
            ))

        if name == "estimate_radial_value":
            img = _maybe_get_image(args.get("image"))
            geometry = args.get("radial_geometry")
            if geometry is None:
                geometry = {
                    "center_x": args.get("center_x"),
                    "center_y": args.get("center_y"),
                    "r_outer": args.get("r_outer"),
                    "r_max": args.get("r_max"),
                }
            if not isinstance(geometry, dict):
                raise TypeError("radial_geometry must be a dict (or pass center_x/center_y/r_outer/r_max)")

            try:
                cx = int(geometry.get("center_x"))  # type: ignore[arg-type]
                cy = int(geometry.get("center_y"))  # type: ignore[arg-type]
                r_outer = int(float(geometry.get("r_outer")))  # type: ignore[arg-type]
                r_max = int(float(geometry.get("r_max")))  # type: ignore[arg-type]
            except Exception as e:
                raise TypeError("radial_geometry must contain center_x, center_y, r_outer, r_max") from e

            ref_val = float(args.get("reference_circle_value", 100.0))
            val = self._tools.estimate_radial_value(
                img,
                center_x=cx,
                center_y=cy,
                r_outer=r_outer,
                r_max=r_max,
                reference_circle_value=float(ref_val),
            )
            _store("radial_value", float(val))
            pj = os.path.join(run_step_dir, "radial_value.json")
            _safe_write_json(
                pj,
                {
                    "radial_value": float(val),
                    "reference_circle_value": float(ref_val),
                    "center_x": cx,
                    "center_y": cy,
                    "r_outer": r_outer,
                    "r_max": r_max,
                },
            )
            artifacts["radial_value_json"] = pj
            return _finalize(ToolExecutionResult(
                tool=name,
                result_summary={"radial_value": float(val)},
                refs=refs,
                artifacts=artifacts,
            ))

        raise ValueError("Unknown tool: {}".format(name))
