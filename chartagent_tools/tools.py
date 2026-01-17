from bisect import bisect_left
import json
import math
import os
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


RgbTuple = Tuple[int, int, int]
BboxXyxy = Tuple[int, int, int, int]

_AUTO = object()


_SAM1_CACHE: Dict[str, object] = {
    "model_key": None,
    "model": None,
    "predictor": None,
    "mask_generator_key": None,
    "mask_generator": None,
}
_EASYOCR_CACHE: Dict[str, object] = {"key": None, "reader": None}


def _resolve_sam1_checkpoint_path(checkpoint_path: Optional[str]) -> str:
    if isinstance(checkpoint_path, str) and checkpoint_path.strip():
        return checkpoint_path
    env = os.environ.get("SAM1_CHECKPOINT_PATH") or os.environ.get("SAM_CHECKPOINT_PATH")
    if isinstance(env, str) and env.strip():
        return env
    # Common default (repo root) used by this project.
    default_names = ["sam_vit_h_4b8939.pth"]
    candidates: List[str] = []
    for name in default_names:
        candidates.append(os.path.abspath(name))
        candidates.append(
            os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, name))
        )
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise RuntimeError(
        "SAM1 checkpoint path is required. Set `SAM1_CHECKPOINT_PATH` (or `SAM_CHECKPOINT_PATH`), "
        "or place `sam_vit_h_4b8939.pth` in the repo root (default)."
    )


def _get_sam1_model(
    checkpoint_path: Optional[str] = None,
    model_type: str = "vit_h",
    device: Optional[str] = None,
):
    """
    Lazy-load SAM1 (Segment Anything) model.

    Uses the official `segment_anything` package and requires a checkpoint file
    such as `sam_vit_h_4b8939.pth`.
    """
    global _SAM1_CACHE

    ckpt = _resolve_sam1_checkpoint_path(checkpoint_path)
    mt = str(model_type or "vit_h").strip() or "vit_h"

    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = str(device).strip() or "cpu"

    cache_key = (ckpt, mt, dev)
    if _SAM1_CACHE.get("model_key") == cache_key and _SAM1_CACHE.get("model") is not None:
        return _SAM1_CACHE["model"], dev

    from segment_anything import sam_model_registry  # type: ignore

    if mt not in sam_model_registry:
        raise ValueError(
            "Unknown SAM1 model_type: {} (expected one of: {})".format(
                mt, sorted(list(sam_model_registry.keys()))
            )
        )

    sam = sam_model_registry[mt](checkpoint=ckpt)
    sam.to(device=dev)

    _SAM1_CACHE["model_key"] = cache_key
    _SAM1_CACHE["model"] = sam
    _SAM1_CACHE["predictor"] = None
    _SAM1_CACHE["mask_generator_key"] = None
    _SAM1_CACHE["mask_generator"] = None
    return sam, dev


def _get_sam1_predictor(
    checkpoint_path: Optional[str] = None,
    model_type: str = "vit_h",
    device: Optional[str] = None,
):
    global _SAM1_CACHE
    sam, _ = _get_sam1_model(checkpoint_path=checkpoint_path, model_type=model_type, device=device)
    if _SAM1_CACHE.get("predictor") is not None:
        return _SAM1_CACHE["predictor"]
    from segment_anything import SamPredictor  # type: ignore

    predictor = SamPredictor(sam)
    _SAM1_CACHE["predictor"] = predictor
    return predictor


def _get_sam1_mask_generator(
    checkpoint_path: Optional[str] = None,
    model_type: str = "vit_h",
    device: Optional[str] = None,
    *,
    points_per_side: int = 64,
    crop_n_layers: int = 1,
    pred_iou_thresh: float = 0.80,
    stability_score_thresh: float = 0.85,
    box_nms_thresh: float = 0.90,
    min_mask_region_area: int = 0,
    output_mode: str = "coco_rle",
):
    """
    Create (and cache) a SAM1 `SamAutomaticMaskGenerator` configured for chart instance masks.
    """
    global _SAM1_CACHE
    sam, _ = _get_sam1_model(checkpoint_path=checkpoint_path, model_type=model_type, device=device)

    key = (
        _SAM1_CACHE.get("model_key"),
        int(points_per_side),
        int(crop_n_layers),
        float(pred_iou_thresh),
        float(stability_score_thresh),
        float(box_nms_thresh),
        int(min_mask_region_area),
        str(output_mode),
    )
    if _SAM1_CACHE.get("mask_generator_key") == key and _SAM1_CACHE.get("mask_generator") is not None:
        return _SAM1_CACHE["mask_generator"]

    from segment_anything import SamAutomaticMaskGenerator  # type: ignore

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=int(points_per_side),
        crop_n_layers=int(crop_n_layers),
        box_nms_thresh=float(box_nms_thresh),
        pred_iou_thresh=float(pred_iou_thresh),
        stability_score_thresh=float(stability_score_thresh),
        min_mask_region_area=int(min_mask_region_area),
        output_mode=str(output_mode),
    )
    _SAM1_CACHE["mask_generator_key"] = key
    _SAM1_CACHE["mask_generator"] = mask_generator
    return mask_generator


def _easyocr_available() -> bool:
    try:
        import easyocr  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def _get_easyocr_reader(langs: Optional[Sequence[str]] = None, gpu: Optional[bool] = None):
    """
    Lazy-load an EasyOCR Reader.

    Notes:
    - EasyOCR will download its model weights on first use if not cached.
    - `gpu` defaults to torch.cuda.is_available().
    """
    global _EASYOCR_CACHE
    langs_t = tuple([str(x) for x in (langs or ("en",))])
    if gpu is None:
        try:
            import torch

            gpu = bool(torch.cuda.is_available())
        except Exception:
            gpu = False
    key = (langs_t, bool(gpu))
    if _EASYOCR_CACHE.get("key") == key and _EASYOCR_CACHE.get("reader") is not None:
        return _EASYOCR_CACHE["reader"]

    import easyocr  # type: ignore

    reader = easyocr.Reader(list(langs_t), gpu=bool(gpu))
    _EASYOCR_CACHE = {"key": key, "reader": reader}
    return reader


def _easyocr_words(image: Image.Image) -> List[Dict[str, object]]:
    """
    Returns a list of OCR word/phrase dicts:
      {text: str, conf: float, bbox_xyxy: (x1,y1,x2,y2)}
    """
    reader = _get_easyocr_reader()
    arr = np.asarray(image.convert("RGB"))
    out: List[Dict[str, object]] = []
    try:
        det = reader.readtext(arr, detail=1, paragraph=False)
    except Exception:
        det = []
    for item in det:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        quad, text, conf = item[0], item[1], item[2]
        text_s = str(text or "").strip()
        if not text_s:
            continue
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.0
        # quad: 4 points [[x,y], ...]
        xs: List[float] = []
        ys: List[float] = []
        try:
            for p in quad:
                xs.append(float(p[0]))
                ys.append(float(p[1]))
        except Exception:
            continue
        if not xs or not ys:
            continue
        x1 = int(max(0, min(xs)))
        y1 = int(max(0, min(ys)))
        x2 = int(max(xs))
        y2 = int(max(ys))
        if x2 <= x1 or y2 <= y1:
            continue
        out.append({"text": text_s, "conf": conf_f, "bbox_xyxy": (x1, y1, x2, y2)})
    return out


def _easyocr_lines(image: Image.Image) -> List[Dict[str, object]]:
    """
    Convert EasyOCR word/phrase detections into approximate line boxes.

    Returns:
      {text: str, bbox_xyxy: (x1,y1,x2,y2), mean_conf: float (0..100)}
    """
    words = _easyocr_words(image)
    if not words:
        return []

    items = []
    heights: List[int] = []
    for w in words:
        bb = w.get("bbox_xyxy")
        if not isinstance(bb, tuple) or len(bb) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bb]
        if x2 <= x1 or y2 <= y1:
            continue
        text = str(w.get("text", "") or "").strip()
        if not text:
            continue
        conf = float(w.get("conf", 0.0) or 0.0)
        cy = 0.5 * float(y1 + y2)
        items.append((x1, y1, x2, y2, cy, text, conf))
        heights.append(max(1, y2 - y1))

    if not items:
        return []

    heights_sorted = sorted(heights)
    med_h = heights_sorted[len(heights_sorted) // 2] if heights_sorted else 12
    y_thresh = max(10.0, 0.75 * float(med_h))
    # Split long "lines" into smaller segments when there are large horizontal gaps.
    # This prevents unrelated text at the same y-level (e.g., left-axis ticks + right-side legend)
    # from being merged into a single giant bbox.
    gap_thresh = max(18.0, 2.5 * float(med_h))

    items.sort(key=lambda t: (t[4], t[0]))  # by cy, then x1
    groups: List[List[Tuple[int, int, int, int, float, str, float]]] = []
    cur: List[Tuple[int, int, int, int, float, str, float]] = []
    cur_cy: Optional[float] = None
    for it in items:
        if cur_cy is None:
            cur = [it]
            cur_cy = float(it[4])
            continue
        if abs(float(it[4]) - float(cur_cy)) <= float(y_thresh):
            cur.append(it)
            # incremental mean
            cur_cy = float(cur_cy) + (float(it[4]) - float(cur_cy)) / float(len(cur))
        else:
            groups.append(cur)
            cur = [it]
            cur_cy = float(it[4])
    if cur:
        groups.append(cur)

    out_lines: List[Dict[str, object]] = []
    for g in groups:
        g_sorted = sorted(g, key=lambda t: t[0])
        if not g_sorted:
            continue

        segs: List[List[Tuple[int, int, int, int, float, str, float]]] = []
        cur_seg: List[Tuple[int, int, int, int, float, str, float]] = [g_sorted[0]]
        cur_x2 = int(g_sorted[0][2])
        for it in g_sorted[1:]:
            gap = int(it[0]) - int(cur_x2)
            if gap > gap_thresh:
                segs.append(cur_seg)
                cur_seg = [it]
                cur_x2 = int(it[2])
            else:
                cur_seg.append(it)
                cur_x2 = max(int(cur_x2), int(it[2]))
        if cur_seg:
            segs.append(cur_seg)

        for seg in segs:
            x1 = min(t[0] for t in seg)
            y1 = min(t[1] for t in seg)
            x2 = max(t[2] for t in seg)
            y2 = max(t[3] for t in seg)
            text = " ".join([t[5] for t in seg]).strip()
            if not text:
                continue
            mean_conf = float(sum([t[6] for t in seg]) / float(max(1, len(seg))))
            out_lines.append(
                {
                    "text": text,
                    "bbox_xyxy": (int(x1), int(y1), int(x2), int(y2)),
                    "mean_conf": float(mean_conf * 100.0),
                }
            )
    return out_lines


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[\s]+", " ", text)
    text = re.sub(r"[^0-9a-z%$.,:/()\-+ ]+", "", text)
    return text.strip()


def _similarity(a: str, b: str) -> float:
    a_n = _normalize_text(a)
    b_n = _normalize_text(b)
    if not a_n or not b_n:
        return 0.0
    if a_n == b_n:
        return 1.0
    # NOTE: naive substring matching can create catastrophic false positives
    # (e.g., matching legend "Social Platform A" to a single character "o" in an axis label).
    # Only treat substring as a strong match when the shorter string is long enough.
    short, long = (a_n, b_n) if len(a_n) <= len(b_n) else (b_n, a_n)
    if len(short) >= 4 and short in long:
        return 0.95
    return SequenceMatcher(None, a_n, b_n).ratio()


def _clip_bbox_xyxy(bbox: BboxXyxy, width: int, height: int) -> BboxXyxy:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), width))
    y1 = max(0, min(int(y1), height))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def _expand_bbox_xyxy(
    bbox: BboxXyxy, margin: int, width: int, height: int
) -> BboxXyxy:
    x1, y1, x2, y2 = bbox
    return _clip_bbox_xyxy((x1 - margin, y1 - margin, x2 + margin, y2 + margin), width, height)


def _union_bboxes_xyxy(bboxes: Sequence[BboxXyxy]) -> Optional[BboxXyxy]:
    if not bboxes:
        return None
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    return (x1, y1, x2, y2)


def _estimate_background_rgb(image: Image.Image) -> RgbTuple:
    img = image.convert("RGB")
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    k = max(5, min(h, w) // 50)
    corners = np.concatenate(
        [
            arr[0:k, 0:k, :].reshape(-1, 3),
            arr[0:k, w - k : w, :].reshape(-1, 3),
            arr[h - k : h, 0:k, :].reshape(-1, 3),
            arr[h - k : h, w - k : w, :].reshape(-1, 3),
        ],
        axis=0,
    )
    med = np.median(corners, axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


def remove_regions(
    image: Image.Image, regions_xyxy: Sequence[BboxXyxy], fill_rgb: Optional[RgbTuple] = None
) -> Image.Image:
    """
    Utility used by tools: remove rectangular regions by painting them with a background-like color.
    """
    img = image.convert("RGB")
    width, height = img.size
    regions_xyxy = [_clip_bbox_xyxy(b, width, height) for b in regions_xyxy if b is not None]
    if not regions_xyxy:
        return img
    if fill_rgb is None:
        fill_rgb = _estimate_background_rgb(img)
    draw = ImageDraw.Draw(img)
    for x1, y1, x2, y2 in regions_xyxy:
        if x2 <= x1 or y2 <= y1:
            continue
        draw.rectangle([x1, y1, x2, y2], fill=fill_rgb)
    return img


def _tesseract_available() -> bool:
    try:
        import pytesseract  # type: ignore

        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False
    except Exception:
        return False


def _tesseract_lines(
    image: Image.Image, psm: int = 6
) -> List[Dict[str, object]]:
    """
    Returns a list of line dicts:
      {text: str, bbox_xyxy: (x1,y1,x2,y2), mean_conf: float}
    """
    import pytesseract  # type: ignore

    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT,
        config="--oem 3 --psm {}".format(int(psm)),
    )
    n = len(data.get("text", []))
    words = []
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0
        if conf < 0:
            continue
        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        words.append(
            {
                "text": text,
                "conf": conf,
                "bbox_xyxy": (x, y, x + w, y + h),
                "block": int(data.get("block_num", [0])[i]),
                "par": int(data.get("par_num", [0])[i]),
                "line": int(data.get("line_num", [0])[i]),
            }
        )

    lines_map = {}
    for w in words:
        key = (w["block"], w["par"], w["line"])
        lines_map.setdefault(key, []).append(w)

    lines = []
    for _, line_words in lines_map.items():
        line_words = sorted(line_words, key=lambda d: int(d["bbox_xyxy"][0]))
        text = " ".join([w["text"] for w in line_words])
        bboxes = [w["bbox_xyxy"] for w in line_words]
        bbox = _union_bboxes_xyxy(bboxes)
        if bbox is None:
            continue
        mean_conf = float(sum([w["conf"] for w in line_words]) / max(1, len(line_words)))
        lines.append({"text": text, "bbox_xyxy": bbox, "mean_conf": mean_conf})
    return lines


def _is_probably_numeric(text: str) -> bool:
    t = _normalize_text(text)
    if not t:
        return False
    # Remove common numeric symbols and separators
    t = re.sub(r"[\s,$%:/\-+()]", "", t)
    t = re.sub(r"[.,]", "", t)
    return bool(t) and all(ch.isdigit() for ch in t)


def _find_title_bbox_auto_ocr(image: Image.Image) -> Optional[BboxXyxy]:
    if not _tesseract_available():
        return None
    img = image.convert("RGB")
    w, h = img.size
    top_h = max(1, int(0.30 * h))
    roi = img.crop((0, 0, w, top_h))
    roi_up = roi.resize((w * 2, top_h * 2), resample=Image.BICUBIC)
    lines = _tesseract_lines(roi_up, psm=6)
    if not lines:
        return None
    # Downscale bbox back to original ROI coordinates.
    for ln in lines:
        x1, y1, x2, y2 = ln["bbox_xyxy"]
        ln["bbox_xyxy"] = (int(x1 / 2), int(y1 / 2), int(x2 / 2), int(y2 / 2))

    best_bbox = None
    best_score = -1e9
    for ln in lines:
        text = str(ln.get("text", "") or "")
        bbox = ln.get("bbox_xyxy")
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        if bh < max(12, int(0.03 * top_h)):
            continue
        if y1 > int(0.22 * top_h):
            continue
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        center_penalty = abs(cx - (w / 2.0)) / float(w)
        # Titles are rarely purely numeric; downweight numeric-only lines.
        numeric_penalty = 0.25 if _is_probably_numeric(text) else 0.0
        conf = float(ln.get("mean_conf", 0.0) or 0.0)

        # Score: larger font/width, near top & centered, decent confidence.
        score = (
            (2.2 * bh)
            + (0.6 * bw)
            + (0.1 * len(text))
            + (0.02 * conf)
            - (220.0 * center_penalty)
            - (30.0 * (cy / float(top_h)))
            - (80.0 * numeric_penalty)
        )
        if score > best_score:
            best_score = score
            best_bbox = (int(x1), int(y1), int(x2), int(y2))

    if best_bbox is None:
        return None
    best_bbox = _expand_bbox_xyxy(best_bbox, margin=10, width=w, height=top_h)
    return best_bbox


def _find_title_bbox_auto_easyocr(image: Image.Image) -> Optional[BboxXyxy]:
    if not _easyocr_available():
        return None
    img = image.convert("RGB")
    w, h = img.size
    top_h = max(1, int(0.30 * h))
    roi = img.crop((0, 0, w, top_h))
    roi_up = roi.resize((w * 2, top_h * 2), resample=Image.BICUBIC)
    lines = _easyocr_lines(roi_up)
    if not lines:
        return None
    for ln in lines:
        x1, y1, x2, y2 = ln["bbox_xyxy"]
        ln["bbox_xyxy"] = (int(x1 / 2), int(y1 / 2), int(x2 / 2), int(y2 / 2))

    best_bbox = None
    best_score = -1e9
    for ln in lines:
        text = str(ln.get("text", "") or "")
        bbox = ln.get("bbox_xyxy")
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        if bh < max(12, int(0.03 * top_h)):
            continue
        if y1 > int(0.22 * top_h):
            continue
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        center_penalty = abs(cx - (w / 2.0)) / float(w)
        numeric_penalty = 0.25 if _is_probably_numeric(text) else 0.0
        conf = float(ln.get("mean_conf", 0.0) or 0.0)
        score = (
            (2.2 * bh)
            + (0.6 * bw)
            + (0.1 * len(text))
            + (0.02 * conf)
            - (220.0 * center_penalty)
            - (30.0 * (cy / float(top_h)))
            - (80.0 * numeric_penalty)
        )
        if score > best_score:
            best_score = score
            best_bbox = (int(x1), int(y1), int(x2), int(y2))

    if best_bbox is None:
        return None
    best_bbox = _expand_bbox_xyxy(best_bbox, margin=10, width=w, height=top_h)
    return best_bbox


def _cluster_bboxes_by_center(
    bboxes: Sequence[BboxXyxy], x_thresh: float, y_thresh: float
) -> List[List[BboxXyxy]]:
    if not bboxes:
        return []
    centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in bboxes]
    remaining = set(range(len(bboxes)))
    clusters: List[List[BboxXyxy]] = []
    while remaining:
        seed = next(iter(remaining))
        remaining.remove(seed)
        cluster_idx = {seed}
        queue = [seed]
        while queue:
            i = queue.pop()
            cx_i, cy_i = centers[i]
            close = []
            for j in list(remaining):
                cx_j, cy_j = centers[j]
                if abs(cx_i - cx_j) <= x_thresh and abs(cy_i - cy_j) <= y_thresh:
                    close.append(j)
            for j in close:
                remaining.remove(j)
                cluster_idx.add(j)
                queue.append(j)
        clusters.append([bboxes[i] for i in cluster_idx])
    return clusters


def _refine_legend_bbox_minimal(
    image: Image.Image,
    *,
    text_bbox: BboxXyxy,
    median_text_height: int,
    base_margin: int = 6,
) -> BboxXyxy:
    """
    Given a legend *text* bbox (union of OCR boxes), expand only as needed to include the marker
    immediately to the left of the text, and keep the overall bbox as tight as possible.

    This prevents over-aggressive legend removal that can wipe part of the plot.
    """
    img = image.convert("RGB")
    w, h = img.size
    tx1, ty1, tx2, ty2 = _clip_bbox_xyxy(text_bbox, w, h)
    if tx2 <= tx1 or ty2 <= ty1:
        return (tx1, ty1, tx2, ty2)
    x1, y1, x2, y2 = int(tx1), int(ty1), int(tx2), int(ty2)

    med_h = int(max(8, median_text_height))
    # Limit how far we can reach left to find markers; avoids wiping the chart.
    max_left = int(min(0.25 * w, max(24, 6 * med_h)))
    search_x1 = int(max(0, x1 - max_left))

    # Background-aware non-bg detection.
    arr = np.asarray(img, dtype=np.uint8)
    bg = np.asarray(_estimate_background_rgb(img), dtype=np.int16)
    diff = np.abs(arr.astype(np.int16) - bg.reshape(1, 1, 3)).sum(axis=2)
    non_bg = diff > 24

    # Search for marker pixels left of the text within the legend vertical span.
    roi = non_bg[y1:y2, search_x1:x1]
    if roi.size > 0:
        col_sum = roi.sum(axis=0).astype(np.int32)
        # Marker squares/circles are dense (many pixels per column), while chart lines/gridlines are sparse.
        # Use a threshold tied to text height to avoid expanding into the plot when legend is inside the axes.
        col_thresh = max(6, int(round(0.80 * float(med_h))))
        cols = np.where(col_sum >= int(col_thresh))[0]
        if cols.size > 0:
            # Choose the right-most contiguous run (closest to the text), not the left-most pixel.
            cols_sorted = np.sort(cols)
            runs: List[Tuple[int, int]] = []
            start = int(cols_sorted[0])
            prev = int(cols_sorted[0])
            for c in cols_sorted[1:]:
                ci = int(c)
                if ci == prev + 1:
                    prev = ci
                    continue
                runs.append((start, prev))
                start = ci
                prev = ci
            runs.append((start, prev))

            run_start, _run_end = max(runs, key=lambda r: int(r[1]))
            left = int(search_x1 + int(run_start))

            # small padding so marker border isn't cut
            pad = int(max(2, round(0.25 * med_h)))
            x1_new = int(max(0, left - pad))

            # Clamp how far we expand left: markers should be close to the text.
            max_marker_reach = int(max(24, round(4.0 * float(med_h))))
            x1 = int(max(x1_new, int(tx1) - max_marker_reach))
        else:
            # Conservative fallback: small margin only.
            x1 = int(max(0, int(tx1) - int(max(12, round(2.0 * med_h)))))

    # Tighten vertical bounds to actual non-bg content inside the combined (marker+text) bbox.
    bbox_x1 = x1
    bbox_x2 = x2
    bbox_y1 = y1
    bbox_y2 = y2
    roi2 = non_bg[bbox_y1:bbox_y2, bbox_x1:bbox_x2]
    if roi2.size > 0:
        row_sum = roi2.sum(axis=1).astype(np.int32)
        row_thresh = max(2, int(0.01 * max(1, bbox_x2 - bbox_x1)))
        rows = np.where(row_sum >= row_thresh)[0]
        if rows.size > 0:
            top = int(bbox_y1 + int(rows.min()))
            bot = int(bbox_y1 + int(rows.max() + 1))
            pad_y = int(max(2, round(0.25 * med_h)))
            bbox_y1 = int(max(0, top - pad_y))
            bbox_y2 = int(min(h, bot + pad_y))

    # Final small expansion for safety.
    margin = int(max(2, base_margin))
    return _expand_bbox_xyxy((bbox_x1, bbox_y1, bbox_x2, bbox_y2), margin=margin, width=w, height=h)


def _find_legend_bbox_auto_ocr(image: Image.Image) -> Optional[BboxXyxy]:
    if not _tesseract_available():
        return None
    img = image.convert("RGB")
    w, h = img.size
    img_up = img.resize((w * 2, h * 2), resample=Image.BICUBIC)
    lines = _tesseract_lines(img_up, psm=6)
    if not lines:
        return None
    for ln in lines:
        x1, y1, x2, y2 = ln["bbox_xyxy"]
        ln["bbox_xyxy"] = (int(x1 / 2), int(y1 / 2), int(x2 / 2), int(y2 / 2))

    # Candidate lines: decent confidence, not too small.
    cand: List[BboxXyxy] = []
    cand_heights: List[int] = []
    for ln in lines:
        bbox = ln.get("bbox_xyxy")
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            continue
        conf = float(ln.get("mean_conf", 0.0) or 0.0)
        if conf < 40.0:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        if bh < 10 or bw < 10:
            continue
        # Skip title-like lines (very wide and near top center).
        cx = (x1 + x2) / 2.0
        if y2 < int(0.25 * h) and bw > int(0.50 * w) and abs(cx - (w / 2.0)) < 0.15 * w:
            continue
        cand.append((x1, y1, x2, y2))
        cand_heights.append(bh)

    if len(cand) < 2:
        return None

    x_thresh = 0.18 * w
    y_thresh = 0.18 * h
    clusters = _cluster_bboxes_by_center(cand, x_thresh=x_thresh, y_thresh=y_thresh)

    img_area = float(w * h)
    best_bbox = None
    best_score = -1e9
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        u = _union_bboxes_xyxy(cluster)
        if u is None:
            continue
        bw = max(1, u[2] - u[0])
        bh = max(1, u[3] - u[1])
        area = float(bw * bh)
        if area < 0.003 * img_area or area > 0.35 * img_area:
            continue
        ar = bw / float(bh)
        if ar > 8.0 or ar < 0.15:
            continue
        # Prefer near-border.
        border_dist = min(u[0], u[1], w - u[2], h - u[3])
        border_score = -border_dist / float(min(w, h) + 1e-6)
        score = (3.0 * len(cluster)) + (2.0 * border_score) - (3.0 * (area / img_area))
        if score > best_score:
            best_score = score
            best_bbox = u

    if best_bbox is None:
        return None

    heights = [max(1, b[3] - b[1]) for b in cand]
    med_h = sorted(heights)[len(heights) // 2] if heights else 12
    best_bbox = _refine_legend_bbox_minimal(img, text_bbox=best_bbox, median_text_height=int(med_h))
    if (best_bbox[2] - best_bbox[0]) * (best_bbox[3] - best_bbox[1]) > 0.45 * img_area:
        return None
    return best_bbox


def _find_legend_bbox_auto_easyocr(image: Image.Image) -> Optional[BboxXyxy]:
    if not _easyocr_available():
        return None
    img = image.convert("RGB")
    w, h = img.size
    img_up = img.resize((w * 2, h * 2), resample=Image.BICUBIC)
    lines = _easyocr_lines(img_up)
    if not lines:
        return None
    for ln in lines:
        x1, y1, x2, y2 = ln["bbox_xyxy"]
        ln["bbox_xyxy"] = (int(x1 / 2), int(y1 / 2), int(x2 / 2), int(y2 / 2))

    cand: List[BboxXyxy] = []
    cand_heights: List[int] = []
    for ln in lines:
        bbox = ln.get("bbox_xyxy")
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            continue
        conf = float(ln.get("mean_conf", 0.0) or 0.0)
        if conf < 35.0:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        if bh < 10 or bw < 10:
            continue
        cx = (x1 + x2) / 2.0
        if y2 < int(0.25 * h) and bw > int(0.50 * w) and abs(cx - (w / 2.0)) < 0.15 * w:
            continue
        cand.append((x1, y1, x2, y2))
        cand_heights.append(bh)

    if len(cand) < 2:
        return None

    x_thresh = 0.18 * w
    y_thresh = 0.18 * h
    clusters = _cluster_bboxes_by_center(cand, x_thresh=x_thresh, y_thresh=y_thresh)

    img_area = float(w * h)
    best_bbox = None
    best_score = -1e9
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        u = _union_bboxes_xyxy(cluster)
        if u is None:
            continue
        bw = max(1, u[2] - u[0])
        bh = max(1, u[3] - u[1])
        area = float(bw * bh)
        if area < 0.003 * img_area or area > 0.35 * img_area:
            continue
        ar = bw / float(bh)
        if ar > 8.0 or ar < 0.15:
            continue
        border_dist = min(u[0], u[1], w - u[2], h - u[3])
        border_score = -border_dist / float(min(w, h) + 1e-6)
        score = (3.0 * len(cluster)) + (2.0 * border_score) - (3.0 * (area / img_area))
        if score > best_score:
            best_score = score
            best_bbox = u

    if best_bbox is None:
        return None

    heights = [max(1, b[3] - b[1]) for b in cand]
    med_h = sorted(heights)[len(heights) // 2] if heights else 12
    best_bbox = _refine_legend_bbox_minimal(img, text_bbox=best_bbox, median_text_height=int(med_h))
    if (best_bbox[2] - best_bbox[0]) * (best_bbox[3] - best_bbox[1]) > 0.45 * img_area:
        return None
    return best_bbox


def _extract_legend_texts(legend: Optional[Dict]) -> List[str]:
    if not legend:
        return []
    if isinstance(legend, dict):
        return [str(k) for k in legend.keys()]
    if isinstance(legend, (list, set, tuple)):
        return [str(x) for x in legend]
    raise TypeError("legend must be a dict[str], list[str], set[str], tuple[str], or None")


def _find_best_line_bbox(
    lines: Sequence[Dict[str, object]],
    target_text: str,
    min_similarity: float,
) -> Optional[BboxXyxy]:
    best = None
    best_score = -1.0
    for ln in lines:
        text = str(ln.get("text", "") or "")
        score = _similarity(target_text, text)
        if score > best_score:
            best_score = score
            best = ln
    if best is None or best_score < min_similarity:
        return None
    bbox = best.get("bbox_xyxy")
    if not isinstance(bbox, tuple) or len(bbox) != 4:
        return None
    return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))


def _find_title_bbox(image: Image.Image, title: str) -> Optional[BboxXyxy]:
    if not title:
        return None
    if not _tesseract_available():
        return None
    img = image.convert("RGB")
    w, h = img.size
    top_h = max(1, int(0.30 * h))
    roi = img.crop((0, 0, w, top_h))
    roi_up = roi.resize((w * 2, top_h * 2), resample=Image.BICUBIC)
    lines = _tesseract_lines(roi_up, psm=6)
    # Downscale bbox back to original ROI coordinates.
    for ln in lines:
        x1, y1, x2, y2 = ln["bbox_xyxy"]
        ln["bbox_xyxy"] = (int(x1 / 2), int(y1 / 2), int(x2 / 2), int(y2 / 2))
    bbox = _find_best_line_bbox(lines, title, min_similarity=0.60)
    if bbox is None:
        return None
    bbox = _expand_bbox_xyxy(bbox, margin=8, width=w, height=top_h)
    # convert from ROI coords to image coords (ROI starts at 0,0 so no offset)
    return (bbox[0], bbox[1], bbox[2], bbox[3])


def _find_title_bbox_easyocr(image: Image.Image, title: str) -> Optional[BboxXyxy]:
    if not title:
        return None
    if not _easyocr_available():
        return None
    img = image.convert("RGB")
    w, h = img.size
    top_h = max(1, int(0.30 * h))
    roi = img.crop((0, 0, w, top_h))
    roi_up = roi.resize((w * 2, top_h * 2), resample=Image.BICUBIC)
    lines = _easyocr_lines(roi_up)
    for ln in lines:
        x1, y1, x2, y2 = ln["bbox_xyxy"]
        ln["bbox_xyxy"] = (int(x1 / 2), int(y1 / 2), int(x2 / 2), int(y2 / 2))
    bbox = _find_best_line_bbox(lines, title, min_similarity=0.60)
    if bbox is None:
        return None
    bbox = _expand_bbox_xyxy(bbox, margin=8, width=w, height=top_h)
    return (bbox[0], bbox[1], bbox[2], bbox[3])


def _group_bboxes_vertically(bboxes: Sequence[BboxXyxy]) -> List[List[BboxXyxy]]:
    if not bboxes:
        return []
    b_sorted = sorted(bboxes, key=lambda b: (b[1], b[0]))
    heights = [max(1, b[3] - b[1]) for b in b_sorted]
    med_h = sorted(heights)[len(heights) // 2]
    gap_thresh = int(max(10, 3 * med_h))

    groups: List[List[BboxXyxy]] = []
    cur: List[BboxXyxy] = [b_sorted[0]]
    cur_y2 = b_sorted[0][3]
    for b in b_sorted[1:]:
        if b[1] <= cur_y2 + gap_thresh:
            cur.append(b)
            cur_y2 = max(cur_y2, b[3])
        else:
            groups.append(cur)
            cur = [b]
            cur_y2 = b[3]
    groups.append(cur)
    return groups


def _select_legend_text_group(
    bboxes: Sequence[BboxXyxy],
    *,
    width: int,
    height: int,
    median_text_height: int,
) -> List[BboxXyxy]:
    """
    Select the most likely legend *text* group from OCR matches.

    Legend entries often also appear elsewhere in the chart (e.g., wedge labels like "Pop: 35").
    A y-only grouping can accidentally merge these and yield an over-wide bbox that wipes the plot.

    Strategy:
    - bin by x-center (legend lists are usually column-aligned)
    - pick the densest bin (with a small neighbor merge) with best compactness
    """
    if not bboxes:
        return []
    w = int(width)
    h = int(height)
    med_h = int(max(8, median_text_height))

    # Wider bins tolerate slight x jitter across OCR boxes, but should not merge the whole chart.
    bin_w = int(max(50, min(0.18 * w, 7 * med_h)))
    if bin_w <= 0:
        bin_w = 50

    bins: Dict[int, List[BboxXyxy]] = {}
    for b in bboxes:
        cx = 0.5 * float(b[0] + b[2])
        k = int(cx // float(bin_w))
        bins.setdefault(k, []).append(b)

    best_group: List[BboxXyxy] = []
    best_key = None
    for k in list(bins.keys()):
        g: List[BboxXyxy] = []
        for kk in (k - 1, k, k + 1):
            g.extend(bins.get(kk, []))
        if len(g) < 2:
            continue
        u = _union_bboxes_xyxy(g)
        if u is None:
            continue
        ux1, uy1, ux2, uy2 = _clip_bbox_xyxy(u, w, h)
        area = max(1, (ux2 - ux1) * (uy2 - uy1))
        edge_margin = int(min(ux1, w - ux2))
        # Prefer: many matches, compact area, near an edge.
        key = (len(g), -area, -edge_margin)
        if best_key is None or key > best_key:
            best_key = key
            best_group = g

    if best_group:
        return best_group

    # Fallback: prior behavior (y-groups), in case we only have a single match.
    groups = _group_bboxes_vertically(bboxes)
    if not groups:
        return []
    return list(max(groups, key=lambda g: len(g)))


def _find_legend_bbox(image: Image.Image, legend_texts: Sequence[str]) -> Optional[BboxXyxy]:
    if not legend_texts:
        return None
    if not _tesseract_available():
        return None

    img = image.convert("RGB")
    w, h = img.size
    img_up = img.resize((w * 2, h * 2), resample=Image.BICUBIC)
    lines = _tesseract_lines(img_up, psm=6)
    for ln in lines:
        x1, y1, x2, y2 = ln["bbox_xyxy"]
        ln["bbox_xyxy"] = (int(x1 / 2), int(y1 / 2), int(x2 / 2), int(y2 / 2))

    matched_bboxes: List[BboxXyxy] = []
    matched_heights: List[int] = []
    for ln in lines:
        line_text = str(ln.get("text", "") or "")
        bbox = ln.get("bbox_xyxy")
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            continue
        best_score = -1.0
        for item in legend_texts:
            score = _similarity(item, line_text)
            if score > best_score:
                best_score = score
        if best_score >= 0.70:
            b = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            matched_bboxes.append(b)
            matched_heights.append(max(1, b[3] - b[1]))

    if not matched_bboxes:
        return None

    med_h = sorted(matched_heights)[len(matched_heights) // 2] if matched_heights else 12
    best_group = _select_legend_text_group(
        matched_bboxes, width=w, height=h, median_text_height=int(med_h)
    )
    if not best_group:
        return None
    bbox = _union_bboxes_xyxy(best_group)
    if bbox is None:
        return None

    bbox = _refine_legend_bbox_minimal(img, text_bbox=bbox, median_text_height=int(med_h))

    # Guardrail: if bbox is too large, return None (avoid wiping the plot).
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    if area > 0.45 * (w * h):
        return None
    return bbox


def _find_legend_bbox_easyocr(image: Image.Image, legend_texts: Sequence[str]) -> Optional[BboxXyxy]:
    """
    EasyOCR-based legend bbox localization.
    """
    if not legend_texts:
        return None
    if not _easyocr_available():
        return None

    img = image.convert("RGB")
    w, h = img.size
    # Prefer line-based (upscaled) matching. This is more robust for multi-word legend entries
    # and cases where EasyOCR splits words unevenly.
    matched_bboxes: List[BboxXyxy] = []
    matched_heights: List[int] = []
    scale = 2
    img_up = img.resize((max(2, w * scale), max(2, h * scale)), resample=Image.BICUBIC)
    lines = _easyocr_lines(img_up)
    for ln in lines:
        bbox = ln.get("bbox_xyxy")
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        b0 = (int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale))
        text = str(ln.get("text", "") or "")
        best_score = -1.0
        for item in legend_texts:
            best_score = max(best_score, _similarity(str(item), text))
        if best_score >= 0.70:
            matched_bboxes.append(b0)
            matched_heights.append(max(1, b0[3] - b0[1]))

    # Fallback: word-level matching (original behavior), in case line grouping fails.
    if not matched_bboxes:
        words = _easyocr_words(img)
        for wd in words:
            text = str(wd.get("text", "") or "")
            bbox = wd.get("bbox_xyxy")
            if not isinstance(bbox, tuple) or len(bbox) != 4:
                continue
            try:
                conf = float(wd.get("conf", 0.0) or 0.0)
            except Exception:
                conf = 0.0
            if conf < 0.15:
                continue
            best_score = -1.0
            for item in legend_texts:
                best_score = max(best_score, _similarity(str(item), text))
            if best_score >= 0.70:
                b = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                matched_bboxes.append(b)
                matched_heights.append(max(1, b[3] - b[1]))

    if not matched_bboxes:
        return None

    med_h = sorted(matched_heights)[len(matched_heights) // 2] if matched_heights else 12
    best_group = _select_legend_text_group(
        matched_bboxes, width=w, height=h, median_text_height=int(med_h)
    )
    if not best_group:
        return None
    bbox = _union_bboxes_xyxy(best_group)
    if bbox is None:
        return None

    bbox = _refine_legend_bbox_minimal(img, text_bbox=bbox, median_text_height=int(med_h))

    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    if area > 0.45 * (w * h):
        return None
    return bbox


def debug_clean_chart_ocr(
    image: Image.Image,
    title: Optional[str] = None,
    legend: Optional[Sequence[str]] = None,
    max_candidates: int = 120,
) -> Dict[str, object]:
    """
    Debug helper for `clean_chart_image`.

    Returns JSON-serializable diagnostics about:
      - raw OCR line outputs (text/bbox/conf)
      - similarity scores vs the provided title/legend strings
      - selected bboxes from the current OCR-based localizers
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")

    img = image.convert("RGB")
    w, h = img.size

    use_easyocr = _easyocr_available()
    use_tesseract = _tesseract_available()
    if not use_easyocr and not use_tesseract:
        raise RuntimeError(
            "debug_clean_chart_ocr requires an OCR backend. Install EasyOCR (`pip install easyocr`) "
            "or install `tesseract` + `pytesseract`."
        )
    ocr_engine = "easyocr" if use_easyocr else "tesseract"

    title_text = (title or "").strip()
    legend_entries = [str(x).strip() for x in (legend or []) if str(x).strip()]

    out: Dict[str, object] = {
        "image_size": [int(w), int(h)],
        "ocr_engine": ocr_engine,
        "tesseract": {"psm": 6, "upscale": 2} if ocr_engine == "tesseract" else None,
        "easyocr": {"upscale": 2} if ocr_engine == "easyocr" else None,
        "title_query": title_text,
        "legend_queries": legend_entries,
        "title_selected_bbox_xyxy": None,
        "legend_selected_bbox_xyxy": None,
        "legend_intermediate": {},
        "title_candidates": [],
        "legend_candidates": [],
    }

    def _bbox_to_list(b: Optional[BboxXyxy]) -> Optional[List[int]]:
        if b is None:
            return None
        return [int(b[0]), int(b[1]), int(b[2]), int(b[3])]

    def _ocr_lines(image_: Image.Image) -> List[Dict[str, object]]:
        if ocr_engine == "easyocr":
            return _easyocr_lines(image_)
        return _tesseract_lines(image_, psm=6)

    # Title candidates (top band).
    if title_text:
        top_h = max(1, int(0.30 * h))
        roi = img.crop((0, 0, w, top_h))
        roi_up = roi.resize((w * 2, top_h * 2), resample=Image.BICUBIC)
        lines = _ocr_lines(roi_up)
        cand: List[Dict[str, object]] = []
        for ln in lines:
            bbox = ln.get("bbox_xyxy")
            if not isinstance(bbox, tuple) or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            bbox0 = (int(x1 / 2), int(y1 / 2), int(x2 / 2), int(y2 / 2))
            text = str(ln.get("text", "") or "")
            cand.append(
                {
                    "text": text,
                    "bbox_xyxy": [int(bbox0[0]), int(bbox0[1]), int(bbox0[2]), int(bbox0[3])],
                    "mean_conf": float(ln.get("mean_conf", 0.0) or 0.0),
                    "score": float(_similarity(title_text, text)),
                }
            )
        cand.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)
        out["title_candidates"] = cand[: int(max_candidates)]
        if ocr_engine == "easyocr":
            out["title_selected_bbox_xyxy"] = _bbox_to_list(_find_title_bbox_easyocr(img, title_text))
        else:
            out["title_selected_bbox_xyxy"] = _bbox_to_list(_find_title_bbox(img, title_text))

    # Legend candidates (whole image).
    if legend_entries:
        img_up = img.resize((w * 2, h * 2), resample=Image.BICUBIC)
        lines = _ocr_lines(img_up)
        cand2: List[Dict[str, object]] = []
        matched_bboxes: List[BboxXyxy] = []
        matched_heights: List[int] = []
        for ln in lines:
            bbox = ln.get("bbox_xyxy")
            if not isinstance(bbox, tuple) or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            bbox0 = (int(x1 / 2), int(y1 / 2), int(x2 / 2), int(y2 / 2))
            text = str(ln.get("text", "") or "")
            best_item = None
            best_score = -1.0
            top_scores: List[Tuple[float, str]] = []
            for item in legend_entries:
                s = _similarity(item, text)
                top_scores.append((float(s), item))
                if s > best_score:
                    best_score = float(s)
                    best_item = item
            top_scores.sort(key=lambda t: t[0], reverse=True)
            cand2.append(
                {
                    "text": text,
                    "bbox_xyxy": [int(bbox0[0]), int(bbox0[1]), int(bbox0[2]), int(bbox0[3])],
                    "mean_conf": float(ln.get("mean_conf", 0.0) or 0.0),
                    "best_match": best_item,
                    "best_score": float(best_score),
                    "top3": [{"legend": it, "score": sc} for (sc, it) in top_scores[:3]],
                }
            )
            if best_score >= 0.70:
                matched_bboxes.append(bbox0)
                matched_heights.append(max(1, int(bbox0[3] - bbox0[1])))
        cand2.sort(key=lambda d: float(d.get("best_score", 0.0)), reverse=True)
        out["legend_candidates"] = cand2[: int(max_candidates)]
        if ocr_engine == "easyocr":
            out["legend_selected_bbox_xyxy"] = _bbox_to_list(_find_legend_bbox_easyocr(img, legend_entries))
        else:
            out["legend_selected_bbox_xyxy"] = _bbox_to_list(_find_legend_bbox(img, legend_entries))

        # Intermediate legend geometry: union/group/marker margin.
        legend_intermediate: Dict[str, object] = {}
        if matched_bboxes:
            groups = _group_bboxes_vertically(matched_bboxes)
            best_group = None
            best_key = None
            for g in groups:
                u = _union_bboxes_xyxy(g)
                if u is None:
                    continue
                area = max(1, (u[2] - u[0]) * (u[3] - u[1]))
                key = (len(g), -area)
                if best_key is None or key > best_key:
                    best_key = key
                    best_group = g

            if best_group:
                union_bbox = _union_bboxes_xyxy(best_group)
                legend_intermediate["matched_count"] = len(matched_bboxes)
                legend_intermediate["best_group_count"] = len(best_group)
                legend_intermediate["union_bbox_xyxy"] = _bbox_to_list(union_bbox)

                med_h = (
                    sorted(matched_heights)[len(matched_heights) // 2]
                    if matched_heights
                    else 12
                )
                legend_intermediate["median_text_height"] = int(med_h)
                if union_bbox is not None:
                    legend_intermediate["refined_bbox_xyxy"] = _bbox_to_list(
                        _refine_legend_bbox_minimal(
                            img, text_bbox=union_bbox, median_text_height=int(med_h)
                        )
                    )

        out["legend_intermediate"] = legend_intermediate

    return out


def detect_clean_chart_regions(
    image: Image.Image, title=_AUTO, legend=_AUTO
) -> Dict[str, Optional[BboxXyxy]]:
    """
    OCR-only region detector used by `clean_chart_image`.

    Returns a dict with:
      - title_bbox_xyxy (or None)
      - legend_bbox_xyxy (or None)
      - ocr_engine: "easyocr" | "tesseract"
      - used_tesseract (bool)
      - title_bbox_source: "<engine>" | "<engine>-auto" | None
      - legend_bbox_source: "<engine>" | "<engine>-auto" | None
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")

    img = image.convert("RGB")
    use_easyocr = _easyocr_available()
    use_tesseract = _tesseract_available()
    if not use_easyocr and not use_tesseract:
        raise RuntimeError(
            "clean_chart_image requires an OCR backend. Install EasyOCR (`pip install easyocr`) "
            "or install `tesseract` + `pytesseract`."
        )
    ocr_engine = "easyocr" if use_easyocr else "tesseract"
    used_tesseract = bool(ocr_engine == "tesseract")

    # Semantics (tri-state):
    # - title=_AUTO (default): auto-detect title (best-effort)
    # - title=str: remove the title matching this text (OCR-based)
    # - title=None: skip title removal
    # - legend=_AUTO (default): auto-detect legend (best-effort)
    # - legend=dict/list/set: legend strings used for OCR matching
    # - legend=None: skip legend removal

    # Treat blank inputs as "unknown" -> auto-detect.
    title_is_auto = title is _AUTO or (isinstance(title, str) and not title.strip())
    legend_is_auto = legend is _AUTO or (
        isinstance(legend, (dict, list, set, tuple)) and not legend
    )

    title_text = None
    if (not title_is_auto) and title is not None and str(title).strip():
        title_text = str(title).strip()

    legend_texts: List[str] = []
    if (not legend_is_auto) and legend is not None:
        legend_texts = _extract_legend_texts(legend)

    title_bbox = None
    title_bbox_source = None
    if title_is_auto:
        if ocr_engine == "easyocr":
            title_bbox = _find_title_bbox_auto_easyocr(img)
        else:
            title_bbox = _find_title_bbox_auto_ocr(img)
        if title_bbox is not None:
            title_bbox_source = "{}-auto".format(ocr_engine)
    elif title is None:
        title_bbox = None
        title_bbox_source = None
    elif title_text:
        if ocr_engine == "easyocr":
            title_bbox = _find_title_bbox_easyocr(img, title_text)
        else:
            title_bbox = _find_title_bbox(img, title_text)
        if title_bbox is None:
            raise RuntimeError('Failed to localize title via OCR: "{}"'.format(title_text))
        title_bbox_source = ocr_engine

    legend_bbox = None
    legend_bbox_source = None
    if legend_is_auto:
        if ocr_engine == "easyocr":
            legend_bbox = _find_legend_bbox_auto_easyocr(img)
        else:
            legend_bbox = _find_legend_bbox_auto_ocr(img)
        if legend_bbox is not None:
            legend_bbox_source = "{}-auto".format(ocr_engine)
    elif legend is None:
        legend_bbox = None
        legend_bbox_source = None
    elif legend_texts:
        if ocr_engine == "easyocr":
            legend_bbox = _find_legend_bbox_easyocr(img, legend_texts)
        else:
            legend_bbox = _find_legend_bbox(img, legend_texts)
        if legend_bbox is None:
            raise RuntimeError(
                "Failed to localize legend via OCR for entries: {}".format(legend_texts)
            )
        legend_bbox_source = ocr_engine

    return {
        "title_bbox_xyxy": title_bbox,
        "legend_bbox_xyxy": legend_bbox,
        "ocr_engine": ocr_engine,
        "used_tesseract": used_tesseract,
        "title_bbox_source": title_bbox_source,
        "legend_bbox_source": legend_bbox_source,
    }


def clean_chart_image(image: Image.Image, title=_AUTO, legend=_AUTO) -> Image.Image:
    """
    Cleans a chart image by removing title and legend if provided.

    Args:
      image: Input PIL image of the chart.
      title:
        - (default) omitted: auto-detect and remove title (best-effort)
        - str: remove the title matching this text (OCR-based)
        - None: skip title removal
      legend:
        - (default) omitted: auto-detect and remove legend (best-effort)
        - dict/list/set: legend strings (keys/entries) used for OCR matching
        - None: skip legend removal

    Returns:
      cleaned_image: Cleaned chart image.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")

    img = image.convert("RGB")
    regions = detect_clean_chart_regions(img, title=title, legend=legend)

    bboxes_to_remove: List[BboxXyxy] = []

    if regions.get("title_bbox_xyxy") is not None:
        bboxes_to_remove.append(regions["title_bbox_xyxy"])

    if regions.get("legend_bbox_xyxy") is not None:
        bboxes_to_remove.append(regions["legend_bbox_xyxy"])

    return remove_regions(img, bboxes_to_remove)


def _tesseract_words(image: Image.Image, psm: int = 6) -> List[Dict[str, object]]:
    """
    Returns a list of word dicts:
      {text: str, bbox_xyxy: (x1,y1,x2,y2), conf: float, block: int, par: int, line: int, word: int}
    """
    import pytesseract  # type: ignore

    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT,
        config="--oem 3 --psm {}".format(int(psm)),
    )
    n = len(data.get("text", []))
    out: List[Dict[str, object]] = []
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0
        if conf < 0:
            continue
        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        out.append(
            {
                "text": text,
                "conf": conf,
                "bbox_xyxy": (x, y, x + w, y + h),
                "block": int(data.get("block_num", [0])[i]),
                "par": int(data.get("par_num", [0])[i]),
                "line": int(data.get("line_num", [0])[i]),
                "word": int(data.get("word_num", [0])[i]),
            }
        )
    return out


def annotate_legend(
    image: Image.Image,
    legend: Dict,
    debug_dir: Optional[str] = None,
) -> Tuple[Image.Image, Image.Image, Dict[int, Dict[str, object]]]:
    """
    Detect legend region, crop it, and overlay numeric labels for each legend entry.

    Uses OCR (EasyOCR preferred; Tesseract fallback) to localize legend text,
    and SAM1 (Segment Anything) to segment the corresponding legend markers for stable marker bboxes.

    Returns:
      - legend_image: cropped legend image
      - labeled_legend: legend image with boxes + numeric labels
      - bbox_mapping:
          - marker labels: 1, 3, 5, ... (one per legend entry)
          - text labels: 2, 4, 6, ... (one per legend entry)
        Each label maps to:
          {
            "legend_text": str,
            "component": "marker"|"text",
            "bbox": (x1,y1,x2,y2),
            "text_bbox": (x1,y1,x2,y2),
            "marker_bbox": (x1,y1,x2,y2),
            "paired_label": int,
          }
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")

    use_easyocr = _easyocr_available()
    if not use_easyocr and not _tesseract_available():
        raise RuntimeError(
            "annotate_legend requires an OCR backend. Install EasyOCR (`pip install easyocr`) "
            "or install `tesseract` + `pytesseract`."
        )

    def _safe_name(s: str, max_len: int = 48) -> str:
        s = re.sub(r"[^0-9a-zA-Z._-]+", "_", str(s)).strip("_")
        return (s[:max_len] or "item").lower()

    legend_texts = _extract_legend_texts(legend)
    if not legend_texts:
        raise ValueError("legend must contain at least one entry")

    img = image.convert("RGB")
    legend_bbox = _find_legend_bbox_easyocr(img, legend_texts) if use_easyocr else _find_legend_bbox(img, legend_texts)
    if legend_bbox is None:
        raise RuntimeError("Failed to localize legend via OCR for entries: {}".format(legend_texts))

    legend_image = img.crop(legend_bbox)
    lw, lh = legend_image.size

    debug: Optional[Dict[str, object]] = None
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        debug = {
            "legend_texts": list(legend_texts),
            "legend_bbox_xyxy": [int(v) for v in legend_bbox],
            "legend_crop_size": [int(lw), int(lh)],
            "ocr_backend": "easyocr" if use_easyocr else "tesseract",
            "entries": [],
        }
        try:
            legend_image.save(os.path.join(debug_dir, "legend_crop.png"))
        except Exception:
            pass
        try:
            preview = img.copy()
            pd = ImageDraw.Draw(preview)
            x1, y1, x2, y2 = [int(v) for v in legend_bbox]
            pd.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=4)
            preview.save(os.path.join(debug_dir, "legend_bbox_preview.png"))
        except Exception:
            pass

    # OCR inside the legend crop to get per-entry text boxes.
    legend_up = legend_image.resize((lw * 2, lh * 2), resample=Image.BICUBIC)
    if use_easyocr:
        words = _easyocr_words(legend_up)
        for w in words:
            x1, y1, x2, y2 = w["bbox_xyxy"]
            w["bbox_xyxy"] = (int(x1 / 2), int(y1 / 2), int(x2 / 2), int(y2 / 2))
    else:
        words = _tesseract_words(legend_up, psm=6)
        for w in words:
            x1, y1, x2, y2 = w["bbox_xyxy"]
            w["bbox_xyxy"] = (int(x1 / 2), int(y1 / 2), int(x2 / 2), int(y2 / 2))

    if not words:
        raise RuntimeError("No OCR words detected inside the legend crop")

    # Group words into line-based text segments, split by large x-gaps.
    heights = [max(1, int(w["bbox_xyxy"][3] - w["bbox_xyxy"][1])) for w in words]
    heights_sorted = sorted(heights)
    med_h = heights_sorted[len(heights_sorted) // 2] if heights_sorted else 12
    gap_thresh = int(max(18, 2.5 * med_h))

    segments: List[Dict[str, object]] = []
    if use_easyocr:
        # EasyOCR doesn't provide block/line ids; cluster by y-center.
        ys = [((int(w["bbox_xyxy"][1]) + int(w["bbox_xyxy"][3])) / 2.0) for w in words]
        hs = [max(1, int(w["bbox_xyxy"][3] - w["bbox_xyxy"][1])) for w in words]
        hs_sorted = sorted(hs)
        med_h2 = hs_sorted[len(hs_sorted) // 2] if hs_sorted else 12
        y_thresh = float(max(8.0, 0.65 * float(med_h2)))
        order = sorted(range(len(words)), key=lambda i: ys[i])
        lines: List[List[Dict[str, object]]] = []
        line_centers: List[float] = []
        for i in order:
            wdict = words[i]
            cy = ys[i]
            placed = False
            for li, lc in enumerate(line_centers):
                if abs(cy - lc) <= y_thresh:
                    lines[li].append(wdict)
                    # update center (running average)
                    line_centers[li] = (lc * (len(lines[li]) - 1) + cy) / float(len(lines[li]))
                    placed = True
                    break
            if not placed:
                lines.append([wdict])
                line_centers.append(cy)

        for line_words in lines:
            line_words = sorted(line_words, key=lambda d: int(d["bbox_xyxy"][0]))
            cur: List[Dict[str, object]] = []
            for ww in line_words:
                if not cur:
                    cur = [ww]
                    continue
                prev = cur[-1]
                gap = int(ww["bbox_xyxy"][0]) - int(prev["bbox_xyxy"][2])
                if gap > gap_thresh:
                    segments.append({"words": cur})
                    cur = [ww]
                else:
                    cur.append(ww)
            if cur:
                segments.append({"words": cur})
    else:
        lines_map: Dict[Tuple[int, int, int], List[Dict[str, object]]] = {}
        for w in words:
            key = (int(w.get("block", 0)), int(w.get("par", 0)), int(w.get("line", 0)))
            lines_map.setdefault(key, []).append(w)

        for _, line_words in lines_map.items():
            line_words = sorted(line_words, key=lambda d: int(d["bbox_xyxy"][0]))
            cur: List[Dict[str, object]] = []
            for ww in line_words:
                if not cur:
                    cur = [ww]
                    continue
                prev = cur[-1]
                gap = int(ww["bbox_xyxy"][0]) - int(prev["bbox_xyxy"][2])
                if gap > gap_thresh:
                    segments.append({"words": cur})
                    cur = [ww]
                else:
                    cur.append(ww)
            if cur:
                segments.append({"words": cur})

    def _segment_text(seg: Dict[str, object]) -> str:
        ws = seg.get("words") or []
        return " ".join([str(w.get("text", "") or "") for w in ws]).strip()

    def _segment_bbox(seg: Dict[str, object]) -> Optional[BboxXyxy]:
        ws = seg.get("words") or []
        bboxes = [w.get("bbox_xyxy") for w in ws if isinstance(w.get("bbox_xyxy"), tuple)]
        bboxes = [b for b in bboxes if isinstance(b, tuple) and len(b) == 4]
        return _union_bboxes_xyxy(bboxes) if bboxes else None

    # Precompute segment texts/bboxes.
    seg_texts: List[str] = []
    seg_bboxes: List[BboxXyxy] = []
    seg_words: List[List[Dict[str, object]]] = []
    for seg in segments:
        txt = _segment_text(seg)
        bb = _segment_bbox(seg)
        if not txt or bb is None:
            continue
        seg_texts.append(txt)
        seg_bboxes.append(bb)
        seg_words.append([w for w in (seg.get("words") or []) if isinstance(w, dict)])

    if not seg_texts:
        raise RuntimeError("No OCR text segments detected inside the legend crop")

    # Match legend entries -> segments (one-to-one).
    pairs: List[Tuple[float, int, int]] = []
    for li, item in enumerate(legend_texts):
        for si, txt in enumerate(seg_texts):
            pairs.append((_similarity(item, txt), li, si))
    pairs.sort(key=lambda t: t[0], reverse=True)

    used_segments = set()
    assigned: Dict[int, int] = {}
    for score, li, si in pairs:
        if score < 0.65:
            break
        if li in assigned or si in used_segments:
            continue
        assigned[li] = si
        used_segments.add(si)
        if len(assigned) == len(legend_texts):
            break

    if len(assigned) != len(legend_texts):
        missing = [legend_texts[i] for i in range(len(legend_texts)) if i not in assigned]
        raise RuntimeError(
            "Failed to match all legend entries in legend crop via OCR. Missing: {}".format(missing)
        )

    matched_items: List[Tuple[str, List[Dict[str, object]], BboxXyxy]] = []
    for li, si in assigned.items():
        matched_items.append((legend_texts[li], list(seg_words[si]), seg_bboxes[si]))

    def _order_items_reading_order(
        items: List[Tuple[str, List[Dict[str, object]], BboxXyxy]],
    ) -> List[Tuple[str, List[Dict[str, object]], BboxXyxy]]:
        if not items:
            return items
        heights = [max(1, int(bb[3] - bb[1])) for _, _, bb in items]
        heights_sorted = sorted(heights)
        med_h = heights_sorted[len(heights_sorted) // 2] if heights_sorted else 12
        y_thresh = float(max(4.0, 0.45 * float(med_h)))

        ordered = sorted(items, key=lambda t: (((t[2][1] + t[2][3]) / 2.0), int(t[2][0])))
        lines: List[List[Tuple[str, List[Dict[str, object]], BboxXyxy]]] = []
        line_centers: List[float] = []
        for it in ordered:
            bb = it[2]
            cy = (float(bb[1]) + float(bb[3])) / 2.0
            placed = False
            for li, lc in enumerate(line_centers):
                if abs(cy - lc) <= y_thresh:
                    lines[li].append(it)
                    line_centers[li] = (lc * (len(lines[li]) - 1) + cy) / float(len(lines[li]))
                    placed = True
                    break
            if not placed:
                lines.append([it])
                line_centers.append(cy)

        out: List[Tuple[str, List[Dict[str, object]], BboxXyxy]] = []
        for li in sorted(range(len(lines)), key=lambda i: line_centers[i]):
            out.extend(sorted(lines[li], key=lambda t: int(t[2][0])))
        return out

    # Order labels in reading order, tolerant to minor y-jitter within a single legend row.
    matched_items = _order_items_reading_order(matched_items)

    # Build bbox mapping and draw labels.
    bbox_mapping: Dict[int, Dict[str, object]] = {}

    try:
        from PIL import ImageFont  # type: ignore

        # Keep labels compact; we draw small numbers near the bboxes.
        font_size = max(8, int(round(0.08 * max(1, lh))))
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=font_size)
        except Exception:
            font = ImageFont.load_default()
    except Exception:
        font = None  # type: ignore

    def _font_measure(text: str) -> Tuple[int, int]:
        if font is not None:
            try:
                x1, y1, x2, y2 = font.getbbox(text)
                return int(x2 - x1), int(y2 - y1)
            except Exception:
                pass
            try:
                tw, th = font.getsize(text)  # type: ignore[attr-defined]
                return int(tw), int(th)
            except Exception:
                pass
        return (max(1, int(len(text) * 6)), 11)

    labeled = legend_image.copy()
    draw = ImageDraw.Draw(labeled)
    canvas_w, canvas_h = int(lw), int(lh)

    # SAM1 is used to localize legend markers robustly (OCR is used only for text matching/bboxes).
    try:
        sam1_predictor = _get_sam1_predictor()
        sam1_predictor.set_image(np.asarray(legend_image.convert("RGB"), dtype=np.uint8))
    except Exception as e:
        raise RuntimeError(
            "annotate_legend requires SAM1 (segment_anything) for marker localization. "
            "Set `SAM1_CHECKPOINT_PATH` (or `SAM_CHECKPOINT_PATH`) to a local SAM checkpoint such as "
            "`sam_vit_h_4b8939.pth`.\n"
            f"Original error: {e}"
        ) from e

    def _tokenize_target(text: str) -> List[str]:
        toks = [t for t in re.findall(r"\w+", str(text).lower(), flags=re.UNICODE) if t]
        return toks if toks else [str(text).lower().strip()]

    def _max_token_sim(word_text: str, tokens: List[str]) -> float:
        wt = str(word_text).lower().strip()
        if not wt:
            return 0.0
        best = 0.0
        for t in tokens:
            # Prefer the repo-wide normalizer when it preserves content; otherwise fall back to raw similarity
            # (e.g., for non-ASCII legend labels).
            wt_n = _normalize_text(wt)
            t_n = _normalize_text(t)
            if wt_n and t_n:
                best = max(best, _similarity(wt, t))
            else:
                best = max(best, SequenceMatcher(None, wt, str(t).lower().strip()).ratio())
        return best

    def _refine_text_bbox(
        words_in_seg: List[Dict[str, object]],
        target_text: str,
        seg_bbox: BboxXyxy,
    ) -> Tuple[BboxXyxy, Optional[BboxXyxy]]:
        toks = _tokenize_target(target_text)
        seg_x1, seg_y1, seg_x2, seg_y2 = seg_bbox
        seg_w = max(1, seg_x2 - seg_x1)
        seg_h = max(1, seg_y2 - seg_y1)
        left_edge_cut = seg_x1 + int(0.30 * seg_w)

        matched_bboxes: List[BboxXyxy] = []
        fallback_bboxes: List[BboxXyxy] = []
        noise_bboxes: List[BboxXyxy] = []

        for wdict in words_in_seg:
            wt = str(wdict.get("text", "") or "").strip()
            bb = wdict.get("bbox_xyxy")
            if not wt or not isinstance(bb, tuple) or len(bb) != 4:
                continue
            bb_i = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))

            if not any(ch.isalnum() for ch in wt):
                continue

            fallback_bboxes.append(bb_i)

            ww = max(1, bb_i[2] - bb_i[0])
            wh = max(1, bb_i[3] - bb_i[1])
            near_left = bb_i[0] <= left_edge_cut
            squareish = ww <= int(1.6 * wh)
            short = len(wt) <= 2
            token_sim = _max_token_sim(wt, toks)

            # Strong match to any legend token → treat as text.
            if token_sim >= 0.45:
                matched_bboxes.append(bb_i)
                continue

            # Weak/no match on the left side → likely marker/noise OCR.
            if near_left and (short or squareish) and token_sim < 0.25:
                noise_bboxes.append(bb_i)

        # Prefer bboxes that match the legend tokens; fallback to any alnum words.
        text_bbox = _union_bboxes_xyxy(matched_bboxes) or _union_bboxes_xyxy(fallback_bboxes) or seg_bbox
        text_bbox = _clip_bbox_xyxy(text_bbox, lw, lh)
        noise_bbox = _union_bboxes_xyxy(noise_bboxes)
        if noise_bbox is not None:
            noise_bbox = _clip_bbox_xyxy(noise_bbox, lw, lh)
        return text_bbox, noise_bbox

    def _detect_marker_bbox(
        img_legend: Image.Image,
        text_bbox: BboxXyxy,
        approx_marker_bbox: Optional[BboxXyxy],
        prev_text_bbox: Optional[BboxXyxy] = None,
        debug_rec: Optional[Dict[str, object]] = None,
    ) -> Optional[BboxXyxy]:
        x1, y1, x2, y2 = text_bbox
        text_h = max(1, y2 - y1)
        span = int(max(24, 4.0 * text_h))
        pad_y = int(max(2, 0.45 * text_h))
        # Marker ROI:
        # - default (clamped-at-left or noisy x1): keep a wider ROI (more forgiving)
        # - otherwise: end just left of the text bbox to reduce interference from legend text
        roi_x1_default = max(0, x1 - span)
        roi_x2_default = min(lw, x1 + int(0.25 * text_h))
        roi_x2_default = min(lw, max(roi_x2_default, roi_x1_default + span + int(0.25 * text_h)))

        if roi_x1_default == 0 and (x1 - span) < 0:
            # Clamped: x1 may be unreliable (marker/noise). Use the forgiving ROI.
            roi_x1, roi_x2 = roi_x1_default, roi_x2_default
        else:
            # Not clamped: prefer an ROI that ends just left of the text bbox.
            roi_x2 = int(max(0, x1 - max(2, int(0.10 * text_h))))
            roi_x2 = min(lw, max(roi_x2, 1))
            roi_x1 = max(0, roi_x2 - span)
        roi_y1 = max(0, y1 - pad_y)
        roi_y2 = min(lh, y2 + pad_y)
        if prev_text_bbox is not None:
            px1, py1, px2, py2 = prev_text_bbox
            prev_h = max(1, py2 - py1)
            overlap = max(0, min(py2, y2) - max(py1, y1))
            if overlap >= int(0.35 * min(prev_h, text_h)):
                # Only clamp against the previous text when it is actually to the LEFT of the current
                # entry (same-row horizontal legends). If items are processed out-of-order (e.g. due
                # to tiny y-jitter), clamping with a right-side previous box can invalidate the ROI.
                if int(px2) <= int(x1):
                    cand_x1 = max(int(roi_x1), int(px2 + 2))
                    if cand_x1 < int(roi_x2):
                        roi_x1 = cand_x1
                    elif debug_rec is not None:
                        debug_rec["roi_adjustment_skipped"] = {
                            "reason": "prev_text_bbox clamp would invalidate ROI",
                            "roi_before_xyxy": [int(roi_x1), int(roi_y1), int(roi_x2), int(roi_y2)],
                            "prev_text_bbox_xyxy": [int(px1), int(py1), int(px2), int(py2)],
                        }
        if debug_rec is not None:
            debug_rec.setdefault("marker_roi_precheck_xyxy", [int(roi_x1), int(roi_y1), int(roi_x2), int(roi_y2)])
        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            if debug_rec is not None:
                debug_rec["roi_invalid"] = True
                debug_rec["marker_roi_xyxy"] = [int(roi_x1), int(roi_y1), int(roi_x2), int(roi_y2)]
            return None

        roi = img_legend.crop((roi_x1, roi_y1, roi_x2, roi_y2)).convert("RGB")
        arr = np.asarray(roi, dtype=np.uint8)
        if arr.size == 0:
            return None

        # Expected marker center (ROI coordinates).
        roi_h, roi_w = arr.shape[:2]
        if approx_marker_bbox is not None:
            amx1, amy1, amx2, amy2 = approx_marker_bbox
            exp_x = ((amx1 + amx2) / 2.0) - float(roi_x1)
            exp_y = ((amy1 + amy2) / 2.0) - float(roi_y1)
        else:
            # Marker is typically adjacent to the legend text (to its immediate left).
            if roi_x1 == 0 and (x1 - span) < 0:
                exp_x = min(0.40 * float(roi_w), 1.4 * float(text_h))
            else:
                exp_x_abs = float(x1) - (1.8 * float(text_h))
                exp_x = exp_x_abs - float(roi_x1)
            exp_y = float(roi_h) / 2.0
        exp_x = float(max(0.0, min(exp_x, float(max(0, roi_w - 1)))))
        exp_y = float(max(0.0, min(exp_y, float(max(0, roi_h - 1)))))

        # Pick a robust positive point for SAM1 inside the marker region.
        border = np.concatenate(
            [
                arr[0, :, :].reshape(-1, 3),
                arr[-1, :, :].reshape(-1, 3),
                arr[:, 0, :].reshape(-1, 3),
                arr[:, -1, :].reshape(-1, 3),
            ],
            axis=0,
        )
        bg = np.median(border, axis=0).astype(np.float32)
        diff = np.abs(arr.astype(np.float32) - bg).sum(axis=2)
        sat = (arr.max(axis=2).astype(np.float32) - arr.min(axis=2).astype(np.float32))
        # Suppress black legend text: it has high diff but near-zero saturation.
        score = diff * (0.05 + (sat / 255.0)) + (0.50 * sat)
        yy, xx = np.indices(score.shape)
        dist2 = (xx.astype(np.float32) - float(exp_x)) ** 2 + (yy.astype(np.float32) - float(exp_y)) ** 2
        alpha = 0.02
        score2 = score - (alpha * dist2)
        # Avoid selecting points in very dark pixels (typical legend text).
        gray = (
            0.299 * arr[:, :, 0].astype(np.float32)
            + 0.587 * arr[:, :, 1].astype(np.float32)
            + 0.114 * arr[:, :, 2].astype(np.float32)
        )
        score2 = np.where(gray > 15.0, score2, -1e9)
        # When legend entries are stacked vertically, the ROI can include adjacent-row markers.
        # Restrict the positive-point search to a tight vertical band around the expected center
        # to avoid selecting the wrong row's marker.
        exp_y_i = int(round(float(exp_y)))
        band_px = int(max(4, round(0.55 * float(text_h))))
        y_lo = int(max(0, exp_y_i - band_px))
        y_hi = int(min(int(roi_h), exp_y_i + band_px + 1))
        score2 = np.where((yy >= y_lo) & (yy < y_hi), score2, -1e9)
        if debug_rec is not None:
            debug_rec["pos_point_search_y_band"] = [int(y_lo), int(y_hi)]

        flat_i = int(np.argmax(score2.reshape(-1)))
        py = int(flat_i // roi_w)
        px = int(flat_i % roi_w)
        max_score = float(score2[py, px]) if score2.size else -1e9
        if max_score < -1e8:
            # Fallback: expected-center (clipped) if everything got filtered out.
            px = int(round(exp_x))
            py = int(round(exp_y))

        pos_x = int(roi_x1 + px)
        pos_y = int(roi_y1 + py)
        mid_y = int(min(lh - 1, max(0, int((roi_y1 + roi_y2) / 2))))
        bg_y = int(min(lh - 1, max(0, roi_y1 + 2)))
        # Negative points:
        #  - one near the left edge to prevent the mask from "flooding" into blank padding
        #  - one inside/near the text region to avoid capturing legend text.
        neg_pts: List[Tuple[int, int]] = []
        neg_pts.append((int(min(lw - 1, max(0, roi_x1 + 2))), int(bg_y)))
        neg_pts.append((int(min(lw - 1, max(0, roi_x2 - 2))), int(bg_y)))
        if not (roi_x1 == 0 and (x1 - span) < 0):
            neg_pts.append((int(min(lw - 1, max(0, x1 + max(2, int(0.35 * text_h))))), int(mid_y)))
        # Deduplicate.
        neg_pts_u = []
        seen = set()
        for nx, ny in neg_pts:
            key = (int(nx), int(ny))
            if key in seen or key == (int(pos_x), int(pos_y)):
                continue
            seen.add(key)
            neg_pts_u.append(key)

        # Run SAM1 with a box prompt + one positive and negative points.
        box = np.array([roi_x1, roi_y1, roi_x2, roi_y2], dtype=np.float32)
        point_coords = np.array([[pos_x, pos_y]] + [[nx, ny] for nx, ny in neg_pts_u], dtype=np.float32)
        point_labels = np.array([1] + [0] * len(neg_pts_u), dtype=np.int32)

        masks, scores, _ = sam1_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )

        # Select the best mask that (a) includes the positive point and (b) mostly lies within ROI.
        roi_mask = np.zeros((lh, lw), dtype=bool)
        roi_mask[roi_y1:roi_y2, roi_x1:roi_x2] = True

        area_min = int(max(6, 0.08 * text_h * text_h))
        area_max = int(max(area_min + 1, 50.0 * text_h * text_h))
        roi_area = int(max(1, (roi_x2 - roi_x1) * (roi_y2 - roi_y1)))
        area_max = int(min(area_max, 0.70 * float(roi_area)))

        best = None
        best_score = -1e9
        cand_info: List[Dict[str, object]] = []

        masks_arr = np.asarray(masks).astype(bool)
        scores_arr = np.asarray(scores).astype(np.float32, copy=False)
        for mi in range(int(masks_arr.shape[0] or 0)):
            m = masks_arr[mi]
            m_bin = m.astype(bool)
            if pos_y < 0 or pos_y >= lh or pos_x < 0 or pos_x >= lw:
                continue
            if not bool(m_bin[pos_y, pos_x]):
                continue
            # Skip masks that include any negative point.
            bad = False
            for nx, ny in neg_pts_u:
                if 0 <= ny < lh and 0 <= nx < lw and bool(m_bin[ny, nx]):
                    bad = True
                    break
            if bad:
                continue
            m_roi = m_bin & roi_mask
            if not bool(m_roi[pos_y, pos_x]):
                continue

            # Use the connected component containing the positive point within the ROI.
            sub = m_roi[roi_y1:roi_y2, roi_x1:roi_x2]
            sy = int(pos_y - roi_y1)
            sx = int(pos_x - roi_x1)
            if sy < 0 or sy >= sub.shape[0] or sx < 0 or sx >= sub.shape[1]:
                continue
            if not bool(sub[sy, sx]):
                continue
            stack = [(sy, sx)]
            sub_vis = np.zeros_like(sub, dtype=bool)
            sub_vis[sy, sx] = True
            minx = maxx = sx
            miny = maxy = sy
            comp_area = 0
            while stack:
                cy, cx = stack.pop()
                comp_area += 1
                if cx < minx:
                    minx = cx
                if cx > maxx:
                    maxx = cx
                if cy < miny:
                    miny = cy
                if cy > maxy:
                    maxy = cy
                for ny in (cy - 1, cy, cy + 1):
                    if ny < 0 or ny >= sub.shape[0]:
                        continue
                    for nx in (cx - 1, cx, cx + 1):
                        if nx < 0 or nx >= sub.shape[1]:
                            continue
                        if sub[ny, nx] and not sub_vis[ny, nx]:
                            sub_vis[ny, nx] = True
                            stack.append((ny, nx))

            area = int(comp_area)
            if area <= 0:
                continue
            if area < area_min or area > area_max:
                continue
            bx1 = int(roi_x1 + minx)
            bx2 = int(roi_x1 + maxx + 1)
            by1 = int(roi_y1 + miny)
            by2 = int(roi_y1 + maxy + 1)
            bw = max(1, bx2 - bx1)
            bh = max(1, by2 - by1)
            ar = bw / float(bh)
            if ar < 0.10 or ar > 12.0:
                continue
            touches = 0
            if bx1 <= int(roi_x1 + 1):
                touches += 1
            if bx2 >= int(roi_x2 - 1):
                touches += 1
            if by1 <= int(roi_y1 + 1):
                touches += 1
            if by2 >= int(roi_y2 - 1):
                touches += 1
            if touches >= 3:
                continue
            s = float(scores_arr[mi]) if scores_arr.size else 0.0
            # Prefer higher SAM1 score and slightly prefer compact masks.
            sel_score = (1000.0 * s) - (0.001 * float(area))
            cand_info.append(
                {
                    "mask_index": int(mi),
                    "score": float(s),
                    "area": int(area),
                    "bbox_xyxy": [int(bx1), int(by1), int(bx2), int(by2)],
                    "ar": float(ar),
                    "touches": int(touches),
                    "score": float(sel_score),
                }
            )
            if sel_score > best_score:
                best_score = sel_score
                best = (bx1, by1, bx2, by2, int(mi), float(s), int(area))

        if debug_rec is not None:
            debug_rec["marker_roi_xyxy"] = [int(roi_x1), int(roi_y1), int(roi_x2), int(roi_y2)]
            debug_rec["marker_bg_rgb"] = [float(bg[0]), float(bg[1]), float(bg[2])]
            debug_rec["sam1_prompt"] = {
                "box_xyxy": [int(roi_x1), int(roi_y1), int(roi_x2), int(roi_y2)],
                "point_coords_xy": [[int(pos_x), int(pos_y)]] + [[int(nx), int(ny)] for nx, ny in neg_pts_u],
                "point_labels": [1] + [0] * len(neg_pts_u),
                "expected_center_xy": [float(exp_x), float(exp_y)],
                "area_min": int(area_min),
                "area_max": int(area_max),
            }
            debug_rec["sam1_mask_candidates"] = cand_info[:10]
            try:
                debug_rec["_marker_roi_image"] = roi
            except Exception:
                pass

        if best is None:
            return None

        bx1, by1, bx2, by2, sel_i, sel_score, sel_area = best
        mb = (int(bx1), int(by1), int(bx2), int(by2))
        mb = _expand_bbox_xyxy(mb, margin=1, width=lw, height=lh)

        if debug_rec is not None:
            debug_rec["sam1_selected"] = {
                "mask_index": int(sel_i),
                "score": float(sel_score),
                "area": int(sel_area),
            }

        if debug_dir and debug_rec is not None:
            try:
                from PIL import Image as _PILImage

                sel_mask = masks_arr[int(sel_i)] & roi_mask
                mask_img = _PILImage.fromarray((sel_mask.astype(np.uint8) * 255))
                safe = _safe_name(str(debug_rec.get("text", "legend")))
                idx_str = str(int(debug_rec.get("idx", sel_i)))
                mask_img.save(os.path.join(debug_dir, f"sam1_marker_mask_{idx_str}_{safe}.png"))
            except Exception:
                pass
        return mb

    def _shrink_text_bbox_after_marker(
        words_in_seg: List[Dict[str, object]],
        text_bbox: BboxXyxy,
        marker_bbox: BboxXyxy,
    ) -> BboxXyxy:
        tx1, ty1, tx2, ty2 = text_bbox
        _, _, mx2, _ = marker_bbox
        text_h = max(1, ty2 - ty1)
        gap = int(max(2, 0.12 * text_h))
        min_x = int(min(lw, max(0, mx2 + gap)))

        adj: List[BboxXyxy] = []
        for wdict in words_in_seg:
            wt = str(wdict.get("text", "") or "").strip()
            bb = wdict.get("bbox_xyxy")
            if not wt or not isinstance(bb, tuple) or len(bb) != 4:
                continue
            if not any(ch.isalnum() for ch in wt):
                continue
            x1, y1, x2, y2 = [int(v) for v in bb]
            x1 = max(x1, min_x)
            if x2 <= x1:
                continue
            adj.append((x1, y1, x2, y2))

        u = _union_bboxes_xyxy(adj) if adj else None
        if u is not None:
            return _clip_bbox_xyxy(u, lw, lh)

        # Fallback: shift the original text bbox's left edge to after the marker.
        return _clip_bbox_xyxy((max(tx1, min_x), ty1, tx2, ty2), lw, lh)

    def _measure_text(text: str) -> Tuple[int, int]:
        # Try draw-based measurement first (accounts for font rendering), then fallback.
        try:
            if hasattr(draw, "textbbox"):
                if font is None:
                    x1, y1, x2, y2 = draw.textbbox((0, 0), text)  # type: ignore[arg-type]
                else:
                    x1, y1, x2, y2 = draw.textbbox((0, 0), text, font=font)  # type: ignore[arg-type]
                return int(x2 - x1), int(y2 - y1)
        except Exception:
            pass
        return _font_measure(text)

    def _draw_text_with_outline(
        xy: Tuple[int, int],
        text: str,
        fill: Tuple[int, int, int],
        outline: Tuple[int, int, int] = (0, 0, 0),
        outline_width: int = 2,
    ) -> None:
        x, y = int(xy[0]), int(xy[1])
        try:
            draw.text(
                (x, y),
                text,
                fill=fill,
                font=font,
                stroke_width=int(outline_width),
                stroke_fill=outline,
            )
            return
        except TypeError:
            pass

        ow = int(max(0, outline_width))
        if ow > 0:
            for dy in range(-ow, ow + 1):
                for dx in range(-ow, ow + 1):
                    if dx == 0 and dy == 0:
                        continue
                    if (dx * dx + dy * dy) > (ow * ow):
                        continue
                    draw.text((x + dx, y + dy), text, fill=outline, font=font)
        draw.text((x, y), text, fill=fill, font=font)

    def _place_label_near_bbox(
        label_text: str,
        bbox: BboxXyxy,
        color: Tuple[int, int, int],
        prefer: str,
    ) -> None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        tw, th = _measure_text(label_text)
        pad = 2

        if prefer == "marker":
            candidates = [
                (x1 - tw - pad, y1),
                (x1, y1 - th - pad),
                (x1, y2 + pad),
                (x1 + pad, y1 + pad),
            ]
        else:
            candidates = [
                (x1, y1 - th - pad),
                (x2 + pad, y1),
                (x1, y2 + pad),
                (x1 + pad, y1 + pad),
            ]

        chosen = None
        for cx, cy in candidates:
            cx_i, cy_i = int(round(cx)), int(round(cy))
            if 0 <= cx_i <= (canvas_w - tw) and 0 <= cy_i <= (canvas_h - th):
                chosen = (cx_i, cy_i)
                break

        if chosen is None:
            cx_i, cy_i = int(round(candidates[0][0])), int(round(candidates[0][1]))
            cx_i = int(min(max(0, cx_i), max(0, canvas_w - tw)))
            cy_i = int(min(max(0, cy_i), max(0, canvas_h - th)))
            chosen = (cx_i, cy_i)

        _draw_text_with_outline(chosen, label_text, fill=color, outline=(0, 0, 0), outline_width=2)

    prev_text_bb: Optional[BboxXyxy] = None
    for idx, (text, words_in_seg, seg_bb) in enumerate(matched_items, start=1):
        entry_debug: Optional[Dict[str, object]] = None
        if debug is not None:
            entry_debug = {
                "idx": int(idx),
                "text": str(text),
                "seg_bbox_xyxy": [int(v) for v in seg_bb],
            }
        text_bb, approx_marker_bb = _refine_text_bbox(words_in_seg, str(text), seg_bb)
        if entry_debug is not None:
            entry_debug["text_bbox_pre_marker_xyxy"] = [int(v) for v in text_bb]
            entry_debug["approx_marker_bbox_xyxy"] = (
                [int(v) for v in approx_marker_bb] if approx_marker_bb is not None else None
            )
        marker_bb = _detect_marker_bbox(
            legend_image, text_bb, approx_marker_bb, prev_text_bbox=prev_text_bb, debug_rec=entry_debug
        )
        if marker_bb is None:
            if debug is not None and entry_debug is not None:
                debug["entries"].append(entry_debug)
                try:
                    safe = _safe_name(str(text))
                    roi_img = entry_debug.get("_marker_roi_image")
                    if isinstance(roi_img, Image.Image):
                        roi_img.save(os.path.join(debug_dir or ".", f"fail_marker_roi_{idx}_{safe}.png"))
                    mask_img = entry_debug.get("_marker_mask_image")
                    if isinstance(mask_img, Image.Image):
                        mask_img.save(os.path.join(debug_dir or ".", f"fail_marker_mask_{idx}_{safe}.png"))
                except Exception:
                    pass
                try:
                    # Remove non-serializable helpers before saving JSON.
                    for e in debug.get("entries", []):
                        if isinstance(e, dict):
                            e.pop("_marker_roi_image", None)
                            e.pop("_marker_mask_image", None)
                    with open(os.path.join(debug_dir or ".", "annotate_legend_debug.json"), "w") as f:
                        import json

                        json.dump(debug, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
            raise RuntimeError("Failed to localize legend marker for: {}".format(text))

        # Refine the text bbox to exclude the marker region (OCR bboxes may include it).
        text_bb = _shrink_text_bbox_after_marker(words_in_seg, text_bb, marker_bb)
        if entry_debug is not None:
            entry_debug["text_bbox_xyxy"] = [int(v) for v in text_bb]

        tx1, ty1, tx2, ty2 = [int(v) for v in text_bb]
        mx1, my1, mx2, my2 = [int(v) for v in marker_bb]
        marker_label_i = int(2 * idx - 1)
        text_label_i = int(2 * idx)
        bbox_mapping[marker_label_i] = {
            "legend_text": str(text),
            "component": "marker",
            "bbox": (mx1, my1, mx2, my2),
            "marker_bbox": (mx1, my1, mx2, my2),
            "text_bbox": (tx1, ty1, tx2, ty2),
            "paired_label": text_label_i,
        }
        bbox_mapping[text_label_i] = {
            "legend_text": str(text),
            "component": "text",
            "bbox": (tx1, ty1, tx2, ty2),
            "marker_bbox": (mx1, my1, mx2, my2),
            "text_bbox": (tx1, ty1, tx2, ty2),
            "paired_label": marker_label_i,
        }
        prev_text_bb = text_bb

        marker_label = str(marker_label_i)
        text_label = str(text_label_i)

        # Draw text and marker boxes.
        draw.rectangle([tx1, ty1, tx2, ty2], outline=(255, 0, 0), width=2)
        draw.rectangle([mx1, my1, mx2, my2], outline=(0, 0, 255), width=2)

        # Draw compact numeric labels near each bbox (no background box).
        _place_label_near_bbox(marker_label, (mx1, my1, mx2, my2), (0, 0, 255), prefer="marker")
        _place_label_near_bbox(text_label, (tx1, ty1, tx2, ty2), (255, 0, 0), prefer="text")

        if debug is not None and entry_debug is not None:
            entry_debug["marker_bbox_xyxy"] = [int(v) for v in marker_bb]
            entry_debug.pop("_marker_roi_image", None)
            entry_debug.pop("_marker_mask_image", None)
            debug["entries"].append(entry_debug)

    if debug is not None:
        try:
            with open(os.path.join(debug_dir or ".", "annotate_legend_debug.json"), "w") as f:
                import json

                json.dump(debug, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    return legend_image, labeled, bbox_mapping


def get_marker_rgb(
    image: Image.Image,
    bbox_mapping: Dict,  # json-loaded mappings may have str keys
    text_of_interest: Optional[str] = None,
    label_of_interest: Optional[int] = None,
    distance_between_text_and_marker: int = 5,
) -> RgbTuple:
    """
    Retrieves the dominant RGB color of a legend marker, either by label (from an annotated legend image)
    or by associated text.

    Notes:
      - `image` should be the legend crop returned by `annotate_legend` (not the labeled overlay).
      - `bbox_mapping` follows the `annotate_legend` schema. If a legacy mapping is passed, best-effort
        fallbacks are applied (marker bbox from `marker_bbox`, or derived from `text_bbox` + distance).
    """

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    if not isinstance(bbox_mapping, dict):
        raise TypeError("bbox_mapping must be a dict")

    img = image.convert("RGBA")
    w, h = img.size

    # Normalize mapping keys to int (JSON loads keys as strings).
    mapping: Dict[int, Dict[str, object]] = {}
    for k, v in bbox_mapping.items():
        try:
            ik = int(k)
        except Exception:
            continue
        if isinstance(v, dict):
            mapping[ik] = v

    if not mapping:
        raise ValueError("bbox_mapping is empty or has no valid entries")

    def _parse_bbox(v: object) -> Optional[BboxXyxy]:
        if isinstance(v, tuple) and len(v) == 4:
            try:
                x1, y1, x2, y2 = [int(round(float(x))) for x in v]
                return (x1, y1, x2, y2)
            except Exception:
                return None
        if isinstance(v, list) and len(v) == 4:
            try:
                x1, y1, x2, y2 = [int(round(float(x))) for x in v]
                return (x1, y1, x2, y2)
            except Exception:
                return None
        return None

    def _get_text(entry: Dict[str, object]) -> str:
        t = entry.get("legend_text")
        if isinstance(t, str) and t.strip():
            return t.strip()
        # Legacy key (older docs/output)
        t2 = entry.get("text")
        if isinstance(t2, str) and t2.strip():
            return t2.strip()
        return ""

    def _get_marker_bbox(entry: Dict[str, object]) -> Optional[BboxXyxy]:
        # Preferred: explicit marker bbox.
        mb = _parse_bbox(entry.get("marker_bbox"))
        if mb is not None:
            return mb

        # New schema: if this entry is the marker component, `bbox` is the marker bbox.
        comp = entry.get("component")
        if isinstance(comp, str) and comp.lower() == "marker":
            mb2 = _parse_bbox(entry.get("bbox"))
            if mb2 is not None:
                return mb2

        # If this is the text component, try its paired label.
        paired = entry.get("paired_label")
        if paired is not None:
            try:
                paired_i = int(paired)
            except Exception:
                paired_i = None
            if paired_i is not None and paired_i in mapping:
                mb3 = _parse_bbox(mapping[paired_i].get("bbox"))
                if mb3 is not None:
                    return mb3

        # Legacy: derive marker bbox from text bbox + distance.
        tb = _parse_bbox(entry.get("text_bbox"))
        if tb is None:
            tb = _parse_bbox(entry.get("bbox")) if (isinstance(comp, str) and comp.lower() == "text") else None
        if tb is None:
            return None
        tx1, ty1, tx2, ty2 = tb
        th = max(1, ty2 - ty1)
        mw = th
        mx2 = int(tx1 - max(0, int(distance_between_text_and_marker)))
        mx1 = int(mx2 - mw)
        return (mx1, ty1, mx2, ty2)

    # Select entry.
    selected_label: Optional[int] = None
    selected_entry: Optional[Dict[str, object]] = None

    if label_of_interest is not None:
        try:
            label_i = int(label_of_interest)
        except Exception as e:
            raise TypeError("label_of_interest must be an int") from e
        entry = mapping.get(label_i)
        if entry is None:
            raise KeyError(f"label_of_interest={label_i} not found in bbox_mapping")
        selected_label = label_i
        selected_entry = entry
    else:
        if not isinstance(text_of_interest, str) or not text_of_interest.strip():
            raise ValueError("Provide either label_of_interest or a non-empty text_of_interest")
        target = str(text_of_interest).strip()

        best = None
        best_score = -1.0
        best_is_marker = False
        for lab, entry in mapping.items():
            txt = _get_text(entry)
            if not txt:
                continue
            score = _similarity(target, txt)
            is_marker = str(entry.get("component") or "").lower() == "marker" or (lab % 2 == 1)
            if (score > best_score) or (score == best_score and is_marker and not best_is_marker):
                best = (lab, entry)
                best_score = float(score)
                best_is_marker = bool(is_marker)

        if best is None or best_score < 0.55:
            raise RuntimeError(f'No legend entry matched text_of_interest="{target}" (best_score={best_score:.2f})')

        selected_label, selected_entry = best[0], best[1]

    assert selected_entry is not None
    marker_bbox = _get_marker_bbox(selected_entry)
    if marker_bbox is None:
        raise RuntimeError("Failed to determine marker bbox from bbox_mapping for the selected entry")

    marker_bbox = _clip_bbox_xyxy(marker_bbox, w, h)
    mx1, my1, mx2, my2 = marker_bbox
    if mx2 <= mx1 or my2 <= my1:
        raise RuntimeError("Marker bbox is empty after clipping: {}".format(marker_bbox))

    roi = img.crop(marker_bbox)
    arr = np.asarray(roi, dtype=np.uint8)
    if arr.size == 0:
        raise RuntimeError("Marker ROI is empty")

    # If the ROI is large, ignore a small border to reduce outline/background effects.
    rh, rw = arr.shape[:2]
    if rw >= 6 and rh >= 6:
        inner = arr[1:-1, 1:-1, :]
        if inner.size > 0:
            arr = inner
            rh, rw = arr.shape[:2]

    # Flatten pixels; drop fully transparent.
    pix = arr.reshape(-1, 4)
    if pix.shape[0] == 0:
        raise RuntimeError("Marker ROI has no pixels")
    alpha = pix[:, 3].astype(np.uint8)
    pix_rgb = pix[:, :3].astype(np.uint8)
    pix_rgb = pix_rgb[alpha > 0]
    if pix_rgb.shape[0] == 0:
        raise RuntimeError("Marker ROI has no visible pixels (all alpha=0)")

    # Estimate a background-ish color from the ROI border (median).
    arr_rgb = pix_rgb.astype(np.float32)
    arr_full = np.asarray(roi.convert("RGB"), dtype=np.uint8)
    bh, bw = arr_full.shape[:2]
    border = np.concatenate(
        [
            arr_full[0, :, :].reshape(-1, 3),
            arr_full[-1, :, :].reshape(-1, 3),
            arr_full[:, 0, :].reshape(-1, 3),
            arr_full[:, -1, :].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.float32)
    bg = np.median(border, axis=0)

    # Score pixels by "marker-likeness": favor saturation and being far from the ROI border median.
    diff = np.abs(arr_rgb - bg).sum(axis=1)
    sat = arr_rgb.max(axis=1) - arr_rgb.min(axis=1)
    score = sat + (0.25 * diff)

    n = int(arr_rgb.shape[0])
    k = int(min(n, max(30, n // 10)))
    if k <= 0:
        k = n
    top_idx = np.argpartition(score, -k)[-k:]
    sel = arr_rgb[top_idx]

    # Prefer pixels not too close to the estimated background.
    sel_diff = np.abs(sel - bg).sum(axis=1)
    thr = float(max(20.0, np.percentile(sel_diff, 35)))
    sel2 = sel[sel_diff >= thr]
    if sel2.shape[0] < 5:
        sel2 = sel

    # Quantize and take the mode bin, then refine by taking median around that bin.
    q = (np.clip(sel2, 0, 255).astype(np.uint8) // 8) * 8
    vals, counts = np.unique(q, axis=0, return_counts=True)
    mode = vals[int(np.argmax(counts))].astype(np.float32)
    d = np.abs(sel2 - mode).sum(axis=1)
    d_thr = float(np.percentile(d, 60)) if d.size else 0.0
    sel3 = sel2[d <= d_thr] if sel2.shape[0] >= 5 else sel2
    rgb = np.median(sel3, axis=0) if sel3.size else np.median(sel2, axis=0)
    rgb = np.clip(rgb, 0, 255)
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _segment_and_mark_legacy(
    image: Image.Image,
    segmentation_model: str = "SAM",
    min_area: int = 5000,
    iou_thresh_unique: float = 0.9,
    iou_thresh_composite: float = 0.98,
    white_ratio_thresh: float = 0.95,
    remove_background_color: bool = False,
    max_points: int = 256,
    text_prompt: Optional[str] = None,
    metadata: Optional[Dict[str, object]] = None,
    debug_dir: Optional[str] = None,
) -> Tuple[Image.Image, List[Dict[str, object]]]:
    """
    Segment an image (SAM1) and apply post-processing to clean masks:
      - remove small masks
      - remove near-duplicate masks by IoU
      - remove composite (union-of-others) masks
      - remove background-dominated masks (near background color)

    Returns:
      - labeled_image: RGB image with mask boundaries + numeric ids
      - cleaned_masks: list of dicts with at least:
          {"id": int, "bbox_xyxy": (x1,y1,x2,y2), "area": int, "segmentation": np.ndarray[bool], "score": float}
    """

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    if str(segmentation_model or "").upper() != "SAM":
        raise ValueError('segmentation_model must be "SAM" (SAM1 backend).')

    img = image.convert("RGB")

    def _extract_chart_type(meta: Optional[Dict[str, object]]) -> Optional[str]:
        if not isinstance(meta, dict):
            return None
        ct = meta.get("chart_type")
        return str(ct).strip() if isinstance(ct, str) and str(ct).strip() else None

    def _extract_title_legend(meta: Optional[Dict[str, object]]) -> Tuple[Optional[str], Optional[object]]:
        if not isinstance(meta, dict):
            return None, None
        title = meta.get("title")
        title_s = str(title).strip() if isinstance(title, str) and str(title).strip() else None
        legend = meta.get("legend")
        if isinstance(legend, (list, dict)) and legend:
            return title_s, legend
        return title_s, None

    def _prompt_candidates_from_chart_type(chart_type: Optional[str]) -> List[str]:
        ct = (chart_type or "").lower()
        if not ct:
            return [
                "pie slice",
                "wedge",
                "bar",
                "line",
                "area",
                "scatter point",
                "box",
                "segment",
            ]
        if any(k in ct for k in ("pie", "donut", "ring", "radial")):
            # In practice SAM3 often responds better to generic prompts like "segment"/"circle"
            # on synthetic charts than to "pie slice"/"wedge".
            return [
                "colored dot",
                "dots",
                "dot",
                "point",
                "small circle",
                "segment",
                "circle",
                "colored region",
                "pie chart",
                "pie slice",
                "wedge",
                "donut chart",
                "ring segment",
                "sector",
            ]
        if "bar" in ct:
            return ["bar", "bar chart", "stacked bar", "bar segment"]
        if any(k in ct for k in ("line", "timeseries", "time series")):
            return ["line", "line chart", "curve", "area"]
        if any(k in ct for k in ("box", "boxplot", "box plot")):
            return ["box", "box plot", "boxplot"]
        if any(k in ct for k in ("scatter", "bubble")):
            return ["scatter point", "bubble", "dot"]
        return ["segment", "object", "region"]

    def _expected_segments_from_metadata(meta: Optional[Dict[str, object]]) -> Optional[int]:
        if not isinstance(meta, dict):
            return None
        legend = meta.get("legend")
        if isinstance(legend, list):
            n = len([x for x in legend if str(x).strip()])
            return n if n > 0 else None
        if isinstance(legend, dict):
            n = len([k for k in legend.keys() if str(k).strip()])
            return n if n > 0 else None
        return None

    def _is_pie_like(chart_type: Optional[str]) -> bool:
        ct = (chart_type or "").lower()
        return any(k in ct for k in ("pie", "donut", "ring", "radial", "wedge", "sector"))

    chart_type = _extract_chart_type(metadata)
    expected_k = _expected_segments_from_metadata(metadata)

    debug: Dict[str, object] = {}
    debug["chart_type"] = chart_type
    debug["expected_segments_from_legend"] = expected_k
    debug["text_prompt_override"] = str(text_prompt) if isinstance(text_prompt, str) and text_prompt.strip() else None

    # Pre-clean (best-effort): removing title/legend often improves segmentation quality.
    seg_img = img
    preclean_info: Dict[str, object] = {"attempted": False, "success": False, "error": None, "used_metadata": False}
    try:
        title_s, legend_obj = _extract_title_legend(metadata)
        preclean_info["attempted"] = True
        preclean_info["used_metadata"] = bool(title_s is not None or legend_obj is not None)
        if title_s is not None or legend_obj is not None:
            seg_img = clean_chart_image(img, title=title_s if title_s is not None else _AUTO, legend=legend_obj if legend_obj is not None else _AUTO)
        else:
            seg_img = clean_chart_image(img)
        preclean_info["success"] = True
    except Exception as e:
        seg_img = img
        preclean_info["success"] = False
        preclean_info["error"] = str(e)

    debug["preclean"] = preclean_info
    if isinstance(debug_dir, str) and debug_dir.strip():
        import json
        import os

        os.makedirs(debug_dir, exist_ok=True)
        try:
            seg_img.save(os.path.join(debug_dir, "seg_input.png"))
        except Exception:
            pass
        try:
            img.save(os.path.join(debug_dir, "original_input.png"))
        except Exception:
            pass
        try:
            with open(os.path.join(debug_dir, "segment_and_mark_debug.json"), "w") as f:
                json.dump(debug, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    w, h = seg_img.size
    if w <= 1 or h <= 1:
        return seg_img, []

    arr = np.asarray(seg_img, dtype=np.uint8)
    bg = np.asarray(_estimate_background_rgb(seg_img), dtype=np.int16)
    diff_bg = np.abs(arr.astype(np.int16) - bg.reshape(1, 1, 3)).sum(axis=2)
    bg_like = diff_bg <= 24
    fg_like = np.logical_not(bg_like)

    def _bbox_from_mask(mask: np.ndarray) -> Optional[BboxXyxy]:
        ys, xs = np.where(mask)
        if xs.size == 0 or ys.size == 0:
            return None
        x1 = int(xs.min())
        x2 = int(xs.max()) + 1
        y1 = int(ys.min())
        y2 = int(ys.max()) + 1
        return (x1, y1, x2, y2)

    def _bbox_iou(a: BboxXyxy, b: BboxXyxy) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        union = area_a + area_b - inter
        return float(inter) / float(max(1, union))

    def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
        inter = int(np.logical_and(a, b).sum())
        if inter <= 0:
            return 0.0
        union = int(np.logical_or(a, b).sum())
        return float(inter) / float(max(1, union))

    def _bg_ratio(mask: np.ndarray, area: int) -> float:
        if area <= 0:
            return 1.0
        bg_in = int(np.logical_and(mask, bg_like).sum())
        return float(bg_in) / float(max(1, area))

    def _mask_boundary_ratio(mask: np.ndarray, area: int) -> float:
        if area <= 0:
            return 1.0
        m = mask.astype(bool)
        pad = np.pad(m, 1, mode="constant", constant_values=False)
        er = pad[1 : h + 1, 1 : w + 1].copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                er &= pad[1 + dy : h + 1 + dy, 1 + dx : w + 1 + dx]
        boundary = np.logical_and(m, np.logical_not(er))
        return float(int(boundary.sum())) / float(max(1, area))

    def _select_expected_k(
        candidates: List[Dict[str, object]],
        expected_k: int,
        *,
        prefer_disjoint_iou: float = 0.35,
    ) -> List[Dict[str, object]]:
        if expected_k <= 0 or not candidates:
            return candidates
        fg_total = int(fg_like.sum())
        if fg_total <= 0:
            return candidates

        k = int(expected_k)
        min_frac = max(0.0015, 0.15 / float(max(1, k)))
        max_frac = min(0.92, 4.0 / float(max(1, k)))

        scored: List[Tuple[float, Dict[str, object]]] = []
        for c in candidates:
            m = c.get("mask")
            if not isinstance(m, np.ndarray):
                continue
            area = int(c.get("area", 0) or 0)
            if area <= 0:
                continue

            area_frac = float(area) / float(max(1, fg_total))
            if area_frac < min_frac or area_frac > max_frac:
                continue

            fg_in = int(np.logical_and(m, fg_like).sum())
            fg_ratio = float(fg_in) / float(max(1, area))
            br = _mask_boundary_ratio(m, area)
            s = float(c.get("score", 0.0) or 0.0)

            q = (1.2 * fg_ratio) + (0.35 * math.log1p(float(area))) + (0.25 * s) - (2.0 * br)
            scored.append((q, c))

        if not scored:
            return candidates

        scored.sort(key=lambda t: t[0], reverse=True)
        pool = [c for _, c in scored[: max(k * 6, k + 10)]]

        selected: List[Dict[str, object]] = []
        selected_ids: set = set()
        for cand in pool:
            if len(selected) >= k:
                break
            cm = cand.get("mask")
            if not isinstance(cm, np.ndarray):
                continue
            if any(
                isinstance(kept.get("mask"), np.ndarray)
                and _mask_iou(cm, kept["mask"]) >= float(prefer_disjoint_iou)  # type: ignore[index]
                for kept in selected
            ):
                continue
            selected.append(cand)
            selected_ids.add(id(cand))

        if len(selected) < k:
            for cand in pool:
                if len(selected) >= k:
                    break
                if id(cand) in selected_ids:
                    continue
                selected.append(cand)
                selected_ids.add(id(cand))

        return selected

    def _color_cluster_masks(expected_k: int) -> List[Dict[str, object]]:
        """
        Fallback for dot/point-based charts where SAM3 may return a single "donut" mask.

        Clusters foreground pixels by color into K groups and returns K masks.
        """
        if expected_k <= 0:
            return []

        arr_u8 = np.asarray(seg_img.convert("RGB"), dtype=np.uint8)
        diff = diff_bg  # precomputed (sum mean abs diff to bg)
        sat = arr_u8.max(axis=2).astype(np.int16) - arr_u8.min(axis=2).astype(np.int16)
        bright = arr_u8.max(axis=2).astype(np.int16)

        # Keep likely "marker" pixels: non-background, somewhat saturated, not black text.
        keep = np.logical_and(fg_like, sat >= 10)
        keep = np.logical_and(keep, bright >= 90)
        keep = np.logical_and(keep, diff >= 30)
        ys, xs = np.where(keep)
        if xs.size < expected_k * 200:
            # Relax saturation/brightness if the chart uses very light colors.
            keep = np.logical_and(fg_like, diff >= 26)
            keep = np.logical_and(keep, bright >= 70)
            ys, xs = np.where(keep)
        if xs.size < expected_k * 100:
            return []

        # Sample pixels for k-means.
        rng = np.random.RandomState(0)
        n = int(xs.size)
        max_samples = int(min(40000, n))
        idx = rng.choice(n, size=max_samples, replace=False) if n > max_samples else np.arange(n)
        samp = arr_u8[ys[idx], xs[idx], :].astype(np.float32)
        if samp.shape[0] < expected_k * 50:
            return []

        # Init centers by farthest-point sampling for stability.
        centers = np.zeros((expected_k, 3), dtype=np.float32)
        centers[0] = samp[int(rng.randint(0, samp.shape[0]))]
        d2 = np.sum((samp - centers[0]) ** 2, axis=1)
        for i in range(1, expected_k):
            j = int(np.argmax(d2))
            centers[i] = samp[j]
            d2 = np.minimum(d2, np.sum((samp - centers[i]) ** 2, axis=1))

        # Lloyd iterations.
        for _ in range(20):
            # (S,K)
            dist2 = ((samp[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            lab = dist2.argmin(axis=1).astype(np.int32)
            new_centers = centers.copy()
            for k in range(expected_k):
                sel = samp[lab == k]
                if sel.size == 0:
                    # re-seed empty cluster
                    new_centers[k] = samp[int(rng.randint(0, samp.shape[0]))]
                else:
                    new_centers[k] = sel.mean(axis=0)
            if float(np.max(np.abs(new_centers - centers))) < 1.5:
                centers = new_centers
                break
            centers = new_centers

        # Assign all kept pixels.
        pix = arr_u8[ys, xs, :].astype(np.float32)
        dist2_all = ((pix[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        lab_all = dist2_all.argmin(axis=1).astype(np.int32)

        out: List[Dict[str, object]] = []
        for k in range(expected_k):
            m = np.zeros((h, w), dtype=bool)
            sel = (lab_all == k)
            if int(sel.sum()) <= 0:
                continue
            m[ys[sel], xs[sel]] = True
            area = int(m.sum())
            if area < int(min_area):
                continue
            bb = _bbox_from_mask(m)
            if bb is None:
                continue
            # Score: compactness in color space (lower intra-cluster distance is better).
            d = dist2_all[sel, k]
            score = 1.0 / float(1.0 + float(np.mean(d)) / 400.0)
            out.append({"mask": m, "area": int(area), "score": float(score), "bbox": bb, "source": "color_cluster"})
        return out

    def _looks_dotty(meta: Optional[Dict[str, object]]) -> bool:
        if not isinstance(meta, dict):
            return False
        vd = meta.get("visual_description")
        if not isinstance(vd, str):
            return False
        t = vd.lower()
        return ("dot" in t) or ("dots" in t) or ("point" in t) or ("points" in t)

    def _dotty_overlap_suppress(
        masks: List[Dict[str, object]],
        *,
        overlap_small_frac: float = 0.92,
        overlap_candidate_frac: float = 0.35,
    ) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
        """
        SAM1 sometimes produces multiple overlapping/nested masks per dot (e.g., dot interior shards).
        For dot-based charts we want (approximately) one mask per dot. We suppress masks that overlap
        an already-kept mask by a large fraction of the *smaller* mask area.
        """

        if not masks:
            return masks, {"enabled": False, "reason": "empty"}

        # Estimate typical dot size to make a stable scoring function.
        areas = np.asarray([int(m.get("area", 0) or 0) for m in masks], dtype=np.float32)
        med_area = float(np.median(areas[areas > 0])) if float(np.sum(areas > 0)) > 0 else 0.0

        widths: List[int] = []
        heights: List[int] = []
        for m in masks:
            bb = m.get("bbox")
            if isinstance(bb, tuple) and len(bb) == 4:
                widths.append(int(bb[2]) - int(bb[0]))
                heights.append(int(bb[3]) - int(bb[1]))
        med_w = float(np.median(np.asarray(widths, dtype=np.float32))) if widths else 0.0
        med_h = float(np.median(np.asarray(heights, dtype=np.float32))) if heights else 0.0
        # Area-based diameter estimate (circle).
        med_diam = 0.0
        if med_area > 0:
            med_diam = 2.0 * math.sqrt(float(med_area) / float(math.pi))
        dot_diam = float(max(med_diam, med_w, med_h, 8.0))
        cell_size = int(max(8, round(1.5 * dot_diam)))

        def _quality(m: Dict[str, object]) -> float:
            a = float(m.get("area", 0) or 0.0)
            s = float(m.get("score", 0.0) or 0.0)
            bb = m.get("bbox")
            if not (isinstance(bb, tuple) and len(bb) == 4):
                return s - 10.0
            bw = float(int(bb[2]) - int(bb[0]))
            bh = float(int(bb[3]) - int(bb[1]))
            # Prefer masks close to typical dot area and near-square bboxes.
            area_dev = 0.0
            if med_area > 0 and a > 0:
                area_dev = abs(math.log(max(1e-6, a / med_area)))
            aspect_dev = abs(math.log(max(1e-6, bw / max(1e-6, bh))))
            return float(s) - (1.35 * float(area_dev)) - (0.75 * float(aspect_dev))

        # Sort by quality so we keep "single-dot-like" masks before bigger composite/merged masks.
        scored: List[Tuple[float, Dict[str, object]]] = [(_quality(m), m) for m in masks]
        scored.sort(key=lambda t: t[0], reverse=True)

        kept: List[Dict[str, object]] = []
        grid: Dict[Tuple[int, int], List[int]] = {}
        removed = 0

        for q, cand in scored:
            cm = cand.get("mask")
            cb = cand.get("bbox")
            ca = int(cand.get("area", 0) or 0)
            if not isinstance(cm, np.ndarray) or not (isinstance(cb, tuple) and len(cb) == 4) or ca <= 0:
                continue
            cx = 0.5 * (float(cb[0]) + float(cb[2]))
            cy = 0.5 * (float(cb[1]) + float(cb[3]))
            gx = int(cx // float(cell_size))
            gy = int(cy // float(cell_size))

            drop = False
            for ny in (gy - 1, gy, gy + 1):
                for nx in (gx - 1, gx, gx + 1):
                    for kept_idx in grid.get((nx, ny), []):
                        km = kept[kept_idx]
                        kmask = km.get("mask")
                        kb = km.get("bbox")
                        ka = int(km.get("area", 0) or 0)
                        if not isinstance(kmask, np.ndarray) or not (isinstance(kb, tuple) and len(kb) == 4) or ka <= 0:
                            continue
                        ix1 = max(int(cb[0]), int(kb[0]))
                        iy1 = max(int(cb[1]), int(kb[1]))
                        ix2 = min(int(cb[2]), int(kb[2]))
                        iy2 = min(int(cb[3]), int(kb[3]))
                        if ix2 <= ix1 or iy2 <= iy1:
                            continue
                        inter = int(
                            np.logical_and(
                                cm[iy1:iy2, ix1:ix2],
                                kmask[iy1:iy2, ix1:ix2],
                            ).sum()
                        )
                        if inter <= 0:
                            continue
                        # 1) Almost-contained (nested) masks.
                        if inter >= int(float(overlap_small_frac) * float(min(ca, ka))):
                            drop = True
                            break
                        # 2) General overlap suppression: dot instances should be (nearly) disjoint.
                        if inter >= int(float(overlap_candidate_frac) * float(ca)):
                            drop = True
                            break
                    if drop:
                        break
                if drop:
                    break

            if drop:
                removed += 1
                continue

            kept.append(cand)
            grid.setdefault((gx, gy), []).append(len(kept) - 1)

        dbg: Dict[str, object] = {
            "enabled": True,
            "before": int(len(masks)),
            "after": int(len(kept)),
            "removed": int(removed),
            "overlap_small_frac": float(overlap_small_frac),
            "overlap_candidate_frac": float(overlap_candidate_frac),
            "median_area": float(med_area),
            "dot_diameter_est": float(dot_diam),
            "cell_size": int(cell_size),
        }
        return kept, dbg

    # -------- 1) Initialize SAM1 backend --------
    # NOTE: `text_prompt`/`max_points` were used for SAM3; SAM1 uses an automatic mask generator.
    dotty = bool(_looks_dotty(metadata))
    min_area_eff = int(min_area)
    # Dot-based donut charts need a much smaller min area than the legacy default (5000).
    if dotty and min_area_eff >= 1000:
        min_area_eff = int(max(5, round(0.00001 * float(w * h))))

    debug["sam_backend"] = "sam1"
    debug["dotty"] = bool(dotty)
    debug["min_area_effective"] = int(min_area_eff)
    debug["ignored_text_prompt"] = str(text_prompt) if isinstance(text_prompt, str) and str(text_prompt).strip() else None
    debug["ignored_max_points"] = int(max_points)

    def _apply_basic_filters(candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for c in candidates:
            m = c.get("mask")
            if not isinstance(m, np.ndarray):
                continue
            area = int(c.get("area", 0) or 0)
            if area < int(min_area_eff):
                continue
            if _bg_ratio(m, area) >= float(white_ratio_thresh):
                continue
            if bool(remove_background_color):
                m2 = np.logical_and(m, np.logical_not(bg_like))
                area2 = int(m2.sum())
                if area2 < int(min_area_eff):
                    continue
                c = dict(c)
                c["mask"] = m2
                c["area"] = int(area2)
                m = m2
                area = area2
            bb = c.get("bbox")
            if not (isinstance(bb, tuple) and len(bb) == 4):
                bb = _bbox_from_mask(m)
            if bb is None:
                continue
            bb_i = _clip_bbox_xyxy(bb, w, h)
            x1, y1, x2, y2 = bb_i
            if x2 <= x1 or y2 <= y1:
                continue
            # Drop full-frame-ish masks early.
            if (x2 - x1) >= int(0.98 * w) and (y2 - y1) >= int(0.98 * h):
                continue
            out.append(
                {
                    "mask": m.astype(bool),
                    "area": int(area),
                    "score": float(c.get("score", 0.0) or 0.0),
                    "bbox": bb_i,
                }
            )
        return out

    def _run_text_prompt(prompt: str) -> List[Dict[str, object]]:
        if state_text is None or sam3_text_processor is None:
            return []
        try:
            sam3_text_processor.reset_all_prompts(state_text)
        except Exception:
            pass
        try:
            out = sam3_text_processor.set_text_prompt(state=state_text, prompt=str(prompt))
        except Exception:
            return []

        if not isinstance(out, dict):
            out = {}
        masks_t = out.get("masks") if out.get("masks") is not None else state_text.get("masks")
        boxes_t = out.get("boxes") if out.get("boxes") is not None else state_text.get("boxes")
        scores_t = out.get("scores") if out.get("scores") is not None else state_text.get("scores")
        import torch

        def _to_cpu_numpy(x: object, *, force_float32: bool = False) -> np.ndarray:
            if isinstance(x, torch.Tensor):
                t = x.detach().to("cpu")
                if force_float32 and t.dtype in (torch.bfloat16, torch.float16):
                    t = t.to(torch.float32)
                return t.numpy()
            return np.asarray(x)

        masks_np = _to_cpu_numpy(masks_t, force_float32=True)
        # SAM3 returns masks as either (N,H,W) or (N,1,H,W).
        if masks_np.ndim == 4 and int(masks_np.shape[1]) == 1:
            masks_np = masks_np[:, 0, :, :]
        if masks_np.ndim == 3:
            # Convert to boolean mask stack.
            if masks_np.dtype != np.bool_:
                masks_np = masks_np > 0.5
        boxes_np = _to_cpu_numpy(boxes_t, force_float32=True)
        scores_np = _to_cpu_numpy(scores_t, force_float32=True).astype(np.float32, copy=False)

        if masks_np.ndim != 3 or masks_np.shape[1] != h or masks_np.shape[2] != w:
            return []
        n = int(masks_np.shape[0])
        if n <= 0:
            return []

        # Prefer top-scoring masks to keep runtime manageable.
        order = np.argsort(scores_np.reshape(-1)) if scores_np.size else np.arange(n)
        order = order[::-1][: min(n, 250)]
        out: List[Dict[str, object]] = []
        for i in order:
            m = masks_np[int(i)]
            area = int(m.sum())
            bb = None
            if boxes_np is not None and boxes_np.size and int(i) < int(boxes_np.shape[0]):
                b = boxes_np[int(i)]
                try:
                    x1, y1, x2, y2 = [int(round(float(v))) for v in list(b)]
                    bb = (x1, y1, x2, y2)
                except Exception:
                    bb = None
            if bb is None:
                bb = _bbox_from_mask(m)
            out.append({"mask": m, "area": int(area), "score": float(scores_np[int(i)] if scores_np.size else 0.0), "bbox": bb})
        return _apply_basic_filters(out)

    def _prompt_metrics(cands: List[Dict[str, object]]) -> Dict[str, float]:
        if not cands:
            return {"coverage": 0.0, "union_ratio": 0.0, "mean_score": 0.0, "k": 0.0}
        # Downsample for coverage scoring.
        ds = int(max(1, round(max(w, h) / 512.0)))
        fg_ds = fg_like[::ds, ::ds]
        fg_total = int(fg_ds.sum())
        union = np.zeros_like(fg_ds, dtype=bool)
        for c in cands[:50]:
            m = c["mask"]
            union |= m[::ds, ::ds]
        cov = float(np.logical_and(union, fg_ds).sum()) / float(max(1, fg_total))
        union_ratio = float(union.sum()) / float(max(1, union.size))
        mean_score = float(np.mean([float(c.get("score", 0.0) or 0.0) for c in cands[:50]]))
        k = float(len(cands))
        return {"coverage": float(cov), "union_ratio": float(union_ratio), "mean_score": float(mean_score), "k": k}

    def _score_prompt(cands: List[Dict[str, object]]) -> float:
        if not cands:
            return -1e9
        m = _prompt_metrics(cands)
        cov = float(m["coverage"])
        union_ratio = float(m["union_ratio"])
        mean_score = float(m["mean_score"])
        k = int(m["k"])
        # Prefer prompt outputs that cover foreground but don't flood the image; prefer moderate counts.
        score = (1.2 * cov) - (0.6 * union_ratio) + (0.08 * math.log1p(float(k))) + (0.10 * mean_score) - (0.02 * max(0.0, float(k - 40)))

        # If metadata provides an expected segment count (legend size) and the chart is pie/donut-like,
        # prefer prompts whose mask count is close to that expected count.
        if expected_k is not None and expected_k > 0 and _is_pie_like(chart_type):
            dev = abs(float(k) - float(expected_k)) / float(expected_k)
            score -= 0.35 * dev
            if k < int(expected_k):
                score -= 0.60
            if k == int(expected_k):
                score += 0.12

        return float(score)

    # -------- 1b) Run SAM1 automatic mask generator --------
    try:
        mask_generator = _get_sam1_mask_generator(output_mode="binary_mask")
    except Exception as e:
        raise RuntimeError(
            "segment_and_mark requires SAM1 (Segment Anything v1). "
            "Provide a checkpoint via `SAM1_CHECKPOINT_PATH`/`SAM_CHECKPOINT_PATH`, "
            "or place `sam_vit_h_4b8939.pth` in the repo root.\n"
            f"Original error: {e}"
        ) from e

    img_rgb = np.asarray(seg_img.convert("RGB"), dtype=np.uint8)
    sam_masks = mask_generator.generate(img_rgb)
    debug["sam1_raw_mask_count"] = int(len(sam_masks))
    raw_masks_for_debug: Optional[List[Dict[str, object]]] = None
    cleaned_pre_overlap: Optional[List[Dict[str, object]]] = None

    raw_candidates: List[Dict[str, object]] = []
    for sm in sam_masks:
        if not isinstance(sm, dict):
            continue
        mask = sm.get("segmentation")
        if not isinstance(mask, np.ndarray) or mask.shape != (h, w):
            continue
        m = mask.astype(bool)
        area = int(sm.get("area", int(m.sum())) or 0)
        if area <= 0:
            continue

        bb = None
        b = sm.get("bbox")
        if isinstance(b, (list, tuple)) and len(b) == 4:
            try:
                x, y, bw, bh = [float(v) for v in b]
                bb = (int(round(x)), int(round(y)), int(round(x + bw)), int(round(y + bh)))
            except Exception:
                bb = None
        if bb is None:
            bb = _bbox_from_mask(m)

        score = sm.get("predicted_iou", None)
        if score is None:
            score = sm.get("stability_score", 0.0)
        try:
            score_f = float(score or 0.0)
        except Exception:
            score_f = 0.0

        raw_candidates.append({"mask": m, "area": int(area), "score": float(score_f), "bbox": bb})

    raw_masks = _apply_basic_filters(raw_candidates)
    debug["sam1_after_basic_filters_count"] = int(len(raw_masks))
    raw_masks_for_debug = list(raw_masks)
    if not raw_masks:
        return seg_img, []

    # -------- 2) Remove near-duplicate masks --------
    raw_masks.sort(key=lambda d: (float(d.get("score", 0.0)), int(d.get("area", 0))), reverse=True)
    uniq: List[Dict[str, object]] = []
    for cand in raw_masks:
        cb = cand.get("bbox")
        if not isinstance(cb, tuple) or len(cb) != 4:
            continue
        keep = True
        for kept in uniq:
            kb = kept.get("bbox")
            if not isinstance(kb, tuple) or len(kb) != 4:
                continue
            if _bbox_iou(cb, kb) < 0.10:
                continue
            iou = _mask_iou(cand["mask"], kept["mask"])  # type: ignore[index]
            if iou >= float(iou_thresh_unique):
                keep = False
                break
        if keep:
            uniq.append(cand)
    debug["after_dedup_count"] = int(len(uniq))

    # -------- 3) Remove composite masks (union of contained masks) --------
    uniq.sort(key=lambda d: int(d.get("area", 0)), reverse=True)
    remove_idx = set()
    for i, big in enumerate(uniq):
        if i in remove_idx:
            continue
        big_mask = big.get("mask")
        if not isinstance(big_mask, np.ndarray):
            continue
        big_area = int(big.get("area", 0))
        if big_area <= 0:
            continue

        subs: List[np.ndarray] = []
        for j in range(i + 1, len(uniq)):
            small = uniq[j]
            small_mask = small.get("mask")
            if not isinstance(small_mask, np.ndarray):
                continue
            small_area = int(small.get("area", 0))
            if small_area <= 0 or small_area >= int(0.95 * big_area):
                continue
            # small mostly inside big?
            inter = int(np.logical_and(big_mask, small_mask).sum())
            if inter >= int(0.90 * small_area):
                subs.append(small_mask)

        if len(subs) < 2:
            continue
        union_small = np.logical_or.reduce(subs)
        comp_iou = _mask_iou(big_mask, union_small)
        if comp_iou >= float(iou_thresh_composite):
            remove_idx.add(i)

    cleaned = [m for i, m in enumerate(uniq) if i not in remove_idx]
    debug["after_composite_count"] = int(len(cleaned))
    if not cleaned:
        return seg_img, []

    # Dot-based donut charts: drop very large "global" masks (e.g., the whole ring) so we keep per-dot instances.
    if dotty and len(cleaned) >= 50:
        try:
            areas = np.asarray([int(x.get("area", 0) or 0) for x in cleaned], dtype=np.int64)
            med = float(np.median(areas)) if areas.size else 0.0
            # Keep anything that's within a reasonable factor of the typical dot size.
            # The global ring mask is orders of magnitude larger than the median dot.
            max_keep = int(max(5000.0, 25.0 * med))
            before = int(len(cleaned))
            cleaned = [m for m in cleaned if int(m.get("area", 0) or 0) <= max_keep]
            debug["dotty_global_mask_filter"] = {
                "enabled": True,
                "before": before,
                "after": int(len(cleaned)),
                "median_area": float(med),
                "max_keep_area": int(max_keep),
            }
        except Exception as e:
            debug["dotty_global_mask_filter"] = {"enabled": True, "error": str(e)}

    # Dot-based charts: suppress nested/overlapping masks so we keep ~one mask per dot.
    if dotty and len(cleaned) >= 50:
        try:
            if isinstance(debug_dir, str) and debug_dir.strip():
                cleaned_pre_overlap = list(cleaned)
            before = int(len(cleaned))
            cleaned, dbg_supp = _dotty_overlap_suppress(cleaned)
            dbg_supp = dict(dbg_supp)
            dbg_supp["before"] = before
            dbg_supp["after"] = int(len(cleaned))
            debug["dotty_overlap_suppression"] = dbg_supp
        except Exception as e:
            debug["dotty_overlap_suppression"] = {"enabled": True, "error": str(e)}

    # -------- 3.5) Optional downselect by legend size (pie/donut-like charts) --------
    # For dot-based donut charts we keep per-dot instances; do NOT downselect to legend length.
    if expected_k is not None and expected_k > 0 and _is_pie_like(chart_type) and (not dotty):
        if len(cleaned) > int(expected_k):
            cleaned = _select_expected_k(cleaned, int(expected_k))
            debug["selected_expected_k"] = int(expected_k)

    debug["final_mask_count"] = int(len(cleaned))
    if isinstance(debug_dir, str) and debug_dir.strip():
        import json
        import os

        try:
            with open(os.path.join(debug_dir, "segment_and_mark_debug.json"), "w") as f:
                json.dump(debug, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # -------- 4) Draw boundaries + ids --------
    out_arr = np.asarray(seg_img.convert("RGB"), dtype=np.uint8).copy()

    def _binary_erode_3x3(mask: np.ndarray) -> np.ndarray:
        m = mask.astype(bool)
        pad = np.pad(m, 1, mode="constant", constant_values=False)
        er = pad[1 : h + 1, 1 : w + 1].copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                er &= pad[1 + dy : h + 1 + dy, 1 + dx : w + 1 + dx]
        return er

    def _binary_dilate_3x3(mask: np.ndarray) -> np.ndarray:
        m = mask.astype(bool)
        pad = np.pad(m, 1, mode="constant", constant_values=False)
        di = pad[1 : h + 1, 1 : w + 1].copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                di |= pad[1 + dy : h + 1 + dy, 1 + dx : w + 1 + dx]
        return di

    def _id_color(i: int) -> Tuple[int, int, int]:
        # Deterministic vivid-ish palette.
        palette = [
            (230, 25, 75),
            (60, 180, 75),
            (255, 225, 25),
            (0, 130, 200),
            (245, 130, 48),
            (145, 30, 180),
            (70, 240, 240),
            (240, 50, 230),
            (210, 245, 60),
            (250, 190, 212),
            (0, 128, 128),
            (220, 190, 255),
            (170, 110, 40),
            (255, 250, 200),
            (128, 0, 0),
            (170, 255, 195),
            (128, 128, 0),
            (255, 215, 180),
            (0, 0, 128),
            (128, 128, 128),
        ]
        return palette[(i - 1) % len(palette)]

    try:
        from PIL import ImageFont  # type: ignore

        # Keep labels compact so they don't obscure segments.
        font_size = int(round(0.012 * max(w, h)))
        font_size = max(8, min(font_size, 18))
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=font_size)
        except Exception:
            font = ImageFont.load_default()
    except Exception:
        font = None  # type: ignore
        font_size = 12  # type: ignore[assignment]

    stroke_w = int(max(1, round(float(font_size) / 9.0)))

    def _draw_label(x: int, y: int, text: str) -> None:
        x_i = int(max(0, min(x, w - 1)))
        y_i = int(max(0, min(y, h - 1)))
        # Center the label at the chosen point when possible.
        try:
            if font is not None and hasattr(draw, "textbbox"):
                bx1, by1, bx2, by2 = draw.textbbox(
                    (0, 0), text, font=font, stroke_width=int(stroke_w)
                )
                tw = int(max(1, bx2 - bx1))
                th = int(max(1, by2 - by1))
                x_i = int(max(0, min(x_i - (tw // 2), w - tw)))
                y_i = int(max(0, min(y_i - (th // 2), h - th)))
        except Exception:
            pass
        try:
            draw.text(
                (x_i, y_i),
                text,
                fill=(0, 0, 0),
                font=font,
                stroke_width=int(stroke_w),
                stroke_fill=(255, 255, 255),
            )
        except TypeError:
            draw.text((x_i, y_i), text, fill=(0, 0, 0), font=font)

    def _choose_label_point(mask: np.ndarray) -> Tuple[int, int]:
        m = mask.astype(bool)
        # Try to pick a point comfortably inside the mask (reduce overlap with the boundary).
        inner = m
        for _ in range(6):
            er = _binary_erode_3x3(inner)
            if int(er.sum()) < 25:
                break
            inner = er

        ys, xs = np.where(inner)
        if xs.size == 0 or ys.size == 0:
            ys, xs = np.where(m)
        if xs.size == 0 or ys.size == 0:
            return 0, 0
        # Median is stable and usually lands inside the component.
        return int(np.median(xs)), int(np.median(ys))

    cleaned_masks: List[Dict[str, object]] = []
    # For dot charts, thick boundaries can make dots look "filled" and obscure the original color.
    if dotty:
        boundary_thickness = 1
    else:
        boundary_thickness = int(max(2, round(0.003 * max(w, h))))
    for mid, m in enumerate(cleaned, start=1):
        mask = m.get("mask")
        bb = m.get("bbox")
        if not isinstance(mask, np.ndarray) or not isinstance(bb, tuple) or len(bb) != 4:
            continue
        bb_i = _clip_bbox_xyxy(bb, w, h)
        area = int(mask.sum())
        if area <= 0:
            continue

        # Boundary overlay.
        er = _binary_erode_3x3(mask)
        boundary = np.logical_and(mask, np.logical_not(er))
        for _ in range(max(0, boundary_thickness - 1)):
            boundary = _binary_dilate_3x3(boundary)
        col = _id_color(mid)
        out_arr[boundary] = np.asarray(col, dtype=np.uint8)

        cleaned_masks.append(
            {
                "id": int(mid),
                "bbox_xyxy": (int(bb_i[0]), int(bb_i[1]), int(bb_i[2]), int(bb_i[3])),
                "area": int(area),
                "score": float(m.get("score", 0.0) or 0.0),
                "segmentation": mask.astype(bool),
            }
        )

    labeled_img = Image.fromarray(out_arr)
    draw = ImageDraw.Draw(labeled_img)
    # If there are too many instances (e.g., dot charts), labels become unreadable.
    # In that case we only draw boundaries.
    if len(cleaned_masks) <= 80:
        for item in cleaned_masks:
            seg = item.get("segmentation")
            if not isinstance(seg, np.ndarray):
                continue
            px, py = _choose_label_point(seg.astype(bool))
            _draw_label(px, py, str(int(item.get("id", 1))))

    # Optional debug outputs (mask overlays + counts).
    if isinstance(debug_dir, str) and debug_dir.strip():
        import json
        import os

        os.makedirs(debug_dir, exist_ok=True)

        def _render_overlay(
            base: Image.Image,
            masks_list: List[Dict[str, object]],
            *,
            thickness: int = 1,
            max_masks: Optional[int] = None,
        ) -> Image.Image:
            base_arr = np.asarray(base.convert("RGB"), dtype=np.uint8).copy()
            it = masks_list
            if isinstance(max_masks, int) and max_masks > 0 and len(it) > int(max_masks):
                it = it[: int(max_masks)]
            for i, m in enumerate(it, start=1):
                mm = m.get("mask")
                if not isinstance(mm, np.ndarray) or mm.shape != (h, w):
                    continue
                er = _binary_erode_3x3(mm)
                boundary = np.logical_and(mm, np.logical_not(er))
                for _ in range(max(0, int(thickness) - 1)):
                    boundary = _binary_dilate_3x3(boundary)
                base_arr[boundary] = np.asarray(_id_color(i), dtype=np.uint8)
            return Image.fromarray(base_arr)

        debug_outputs: Dict[str, str] = {}
        # Final overlay (same as tool output) for quick access inside debug dir.
        try:
            out_path = os.path.join(debug_dir, "final_masks_overlay.png")
            labeled_img.save(out_path)
            debug_outputs["final_masks_overlay"] = out_path
        except Exception:
            pass

        # Raw SAM1 mask overlay (after basic filters) – helps inspect noisy/nested masks.
        if isinstance(raw_masks_for_debug, list) and raw_masks_for_debug:
            try:
                out_path = os.path.join(debug_dir, "sam1_raw_masks_overlay.png")
                # Sort so higher-score masks draw first (lower-score overwrite later),
                # making suspicious shards easier to spot.
                raw_sorted = sorted(
                    raw_masks_for_debug,
                    key=lambda d: (float(d.get("score", 0.0) or 0.0), int(d.get("area", 0) or 0)),
                    reverse=True,
                )
                _render_overlay(seg_img, raw_sorted, thickness=1).save(out_path)
                debug_outputs["sam1_raw_masks_overlay"] = out_path
            except Exception:
                pass

        # Dotty charts: visualize masks before overlap suppression as well.
        if isinstance(cleaned_pre_overlap, list) and cleaned_pre_overlap:
            try:
                out_path = os.path.join(debug_dir, "pre_overlap_masks_overlay.png")
                pre_sorted = sorted(
                    cleaned_pre_overlap,
                    key=lambda d: (float(d.get("score", 0.0) or 0.0), int(d.get("area", 0) or 0)),
                    reverse=True,
                )
                _render_overlay(seg_img, pre_sorted, thickness=1).save(out_path)
                debug_outputs["pre_overlap_masks_overlay"] = out_path
            except Exception:
                pass

        if debug_outputs:
            debug["debug_outputs"] = debug_outputs
            try:
                with open(os.path.join(debug_dir, "segment_and_mark_debug.json"), "w") as f:
                    json.dump(debug, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

    return labeled_img, cleaned_masks


def segment_and_mark(
    image: Image.Image,
    segmentation_model: str = "SAM",
    min_area: int = 5000,
    iou_thresh_unique: float = 0.9,
    iou_thresh_composite: float = 0.98,
    white_ratio_thresh: float = 0.95,
    remove_background_color: bool = False,
    max_points: int = 256,
    text_prompt: Optional[str] = None,
    metadata: Optional[Dict[str, object]] = None,
    debug_dir: Optional[str] = None,
) -> Tuple[Image.Image, List[Dict[str, object]]]:
    """
    Chart segmentation (SAM1) implementation transplanted from `gpt_segment.py`.

    Pipeline:
      1) (best-effort) pre-clean title/legend via `clean_chart_image`
      2) SAM1 automatic mask generation
      3) Mask cleaning pipeline (small / duplicate / composite / background / text / axis / grid / legend)
      4) Draw contours + ids and return masks

    Notes:
      - `text_prompt` / `max_points` are legacy (SAM3) and ignored.
      - Debug outputs (when `debug_dir` is set):
          - `sam1_raw_masks_overlay.png`, `final_masks_overlay.png`, `segment_and_mark_debug.json`
    """

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    if str(segmentation_model or "").upper() != "SAM":
        raise ValueError('segmentation_model must be "SAM" (SAM1 backend).')

    import json
    import os

    import cv2  # type: ignore

    # NOTE: Some Misviz charts are extremely high-resolution (e.g., 7k+ pixels wide).
    # SAM1 mask generation on full-res images can OOM due to many full-size binary masks.
    # We downscale by default for segmentation-only use cases; override via env var if needed.
    img_full = image.convert("RGB")
    img = img_full

    def _looks_dotty(meta: Optional[Dict[str, object]]) -> bool:
        if not isinstance(meta, dict):
            return False
        vd = meta.get("visual_description")
        if not isinstance(vd, str):
            return False
        t = vd.lower()
        return ("dot" in t) or ("dots" in t) or ("point" in t) or ("points" in t)

    def _extract_title_legend(meta: Optional[Dict[str, object]]) -> Tuple[Optional[str], Optional[object]]:
        if not isinstance(meta, dict):
            return None, None
        title = meta.get("title")
        title_s = str(title).strip() if isinstance(title, str) and str(title).strip() else None
        legend = meta.get("legend")
        if isinstance(legend, (list, dict)) and legend:
            return title_s, legend
        return title_s, None

    debug: Dict[str, object] = {
        "sam_backend": "sam1",
        "ignored_text_prompt": str(text_prompt) if isinstance(text_prompt, str) and text_prompt.strip() else None,
        "ignored_max_points": int(max_points),
        "dotty": bool(_looks_dotty(metadata)),
        "input_size": list(img.size),
    }
    dotty = bool(debug.get("dotty"))
    cleaned_pre_overlap: Optional[List[Dict[str, object]]] = None
    chart_type = None
    if isinstance(metadata, dict) and isinstance(metadata.get("chart_type"), str):
        chart_type = str(metadata.get("chart_type") or "").strip()
    pie_like = False
    if isinstance(chart_type, str) and chart_type:
        ct = chart_type.lower()
        pie_like = any(k in ct for k in ("pie", "donut", "ring", "radial", "wedge", "sector"))
    debug["chart_type"] = chart_type
    debug["pie_like"] = bool(pie_like)

    # 0) Optional downscale for SAM stability/memory.
    sam_max_side = 2048
    try:
        env_v = os.environ.get("CHARTAGENT_SAM_MAX_SIDE")
        if isinstance(env_v, str) and env_v.strip():
            sam_max_side = int(env_v.strip())
    except Exception:
        sam_max_side = 2048
    debug["sam_max_side"] = int(sam_max_side)

    if int(sam_max_side) > 0:
        w0, h0 = img.size
        m0 = max(int(w0), int(h0))
        if m0 > int(sam_max_side):
            scale = float(int(sam_max_side)) / float(m0)
            nw = max(2, int(round(float(w0) * scale)))
            nh = max(2, int(round(float(h0) * scale)))
            img = img.resize((nw, nh), resample=Image.BICUBIC)
            debug["sam_input_resized"] = {
                "enabled": True,
                "orig_size": [int(w0), int(h0)],
                "new_size": [int(nw), int(nh)],
                "scale": float(scale),
            }
        else:
            debug["sam_input_resized"] = {"enabled": False, "reason": "below_threshold"}
    else:
        debug["sam_input_resized"] = {"enabled": False, "reason": "disabled"}

    # 0) Pre-clean: removing title/legend improves segmentation stability on many charts.
    seg_img = img
    preclean_info: Dict[str, object] = {"attempted": False, "success": False, "error": None, "used_metadata": False}
    try:
        title_s, legend_obj = _extract_title_legend(metadata)
        preclean_info["attempted"] = True
        preclean_info["used_metadata"] = bool(isinstance(metadata, dict))
        # Be conservative about legend removal: auto-detection can accidentally erase axis tick labels
        # on charts without a legend (e.g., boxplots with categorical tick labels "A/B/C").
        title_arg = title_s if title_s is not None else _AUTO
        if isinstance(metadata, dict):
            legend_arg = legend_obj if legend_obj is not None else None
        else:
            legend_arg = None
        seg_img = clean_chart_image(img, title=title_arg, legend=legend_arg)
        preclean_info["success"] = True
    except Exception as e:
        seg_img = img
        preclean_info["success"] = False
        preclean_info["error"] = str(e)
    debug["preclean"] = preclean_info

    # 0.5) Stronger pre-clean for pie-like charts: remove OCR text pixels (labels/annotations)
    # to prevent SAM from producing tiny text/leader-line masks that split wedges.
    preclean_text_info: Dict[str, object] = {"attempted": False, "removed_boxes": 0, "min_conf": 0.30}
    try:
        if (not bool(debug.get("dotty"))) and pie_like and _easyocr_available():
            preclean_text_info["attempted"] = True
            # Use word-level detections to get tight bboxes.
            words = _easyocr_words(seg_img)
            boxes: List[BboxXyxy] = []
            for w0 in words:
                bb = w0.get("bbox_xyxy")
                if not isinstance(bb, tuple) or len(bb) != 4:
                    continue
                try:
                    conf_f = float(w0.get("conf", 0.0) or 0.0)
                except Exception:
                    conf_f = 0.0
                if conf_f < float(preclean_text_info["min_conf"]):
                    continue
                x1, y1, x2, y2 = [int(v) for v in bb]
                if x2 <= x1 or y2 <= y1:
                    continue
                w_box = x2 - x1
                h_box = y2 - y1
                expand_x = int(0.15 * float(w_box))
                expand_y = int(0.30 * float(h_box))
                boxes.append((x1 - expand_x, y1 - expand_y, x2 + expand_x, y2 + expand_y))
            if boxes:
                seg_img = remove_regions(seg_img, boxes)
                preclean_text_info["removed_boxes"] = int(len(boxes))
    except Exception:
        pass
    debug["preclean_text"] = preclean_text_info

    if isinstance(debug_dir, str) and debug_dir.strip():
        os.makedirs(debug_dir, exist_ok=True)
        try:
            img_full.save(os.path.join(debug_dir, "original_input.png"))
        except Exception:
            pass
        try:
            seg_img.save(os.path.join(debug_dir, "seg_input.png"))
        except Exception:
            pass

    # Convert to OpenCV images.
    image_rgb = np.asarray(seg_img.convert("RGB"), dtype=np.uint8)
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise RuntimeError("Invalid image array shape for segmentation.")
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    H, W = image_bgr.shape[:2]
    img_area = int(H * W)

    # ---------------------------------------------------------
    # gpt_segment.py helpers (ported)
    # ---------------------------------------------------------
    def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        mask_a = mask_a.astype(bool)
        mask_b = mask_b.astype(bool)
        inter = int(np.logical_and(mask_a, mask_b).sum())
        if inter <= 0:
            return 0.0
        union = int(np.logical_or(mask_a, mask_b).sum())
        return float(inter) / float(max(1, union))

    def is_background_dominated(
        mask: np.ndarray,
        brightness_thresh: int = 220,
        edge_ratio_thresh: float = 0.001,
    ) -> bool:
        if int(mask.sum()) == 0:
            return True
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        region_vals = gray[mask.astype(bool)]
        mean_val = float(region_vals.mean()) if region_vals.size else 0.0
        if mean_val < float(brightness_thresh):
            return False
        edges = cv2.Canny(gray, 100, 200)
        edge_in_mask = edges[mask.astype(bool)] > 0
        edge_ratio = float(int(edge_in_mask.sum())) / float(max(1, int(mask.sum())))
        return edge_ratio < float(edge_ratio_thresh)

    def build_text_mask(
        ocr_boxes: List[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        text_mask = np.zeros((H, W), dtype=bool)
        for (x_min, y_min, x_max, y_max) in ocr_boxes:
            w_box = int(x_max - x_min)
            h_box = int(y_max - y_min)
            expand_x = int(0.15 * float(w_box))
            expand_y = int(0.3 * float(h_box))
            x1 = max(0, int(x_min) - expand_x)
            y1 = max(0, int(y_min) - expand_y)
            x2 = min(W, int(x_max) + expand_x)
            y2 = min(H, int(y_max) + expand_y)
            if x2 <= x1 or y2 <= y1:
                continue
            text_mask[y1:y2, x1:x2] = True
        return text_mask

    def is_axis_like(
        mask: np.ndarray,
        bbox_xywh: Sequence[int],
        aspect_ratio_thresh: float = 10.0,
        margin_ratio: float = 0.05,
        max_area_ratio: float = 0.2,
    ) -> bool:
        if len(bbox_xywh) != 4:
            return False
        x, y, bw, bh = [int(v) for v in bbox_xywh]
        if bw <= 0 or bh <= 0:
            return False
        area = int(mask.sum())
        if area <= 0:
            return False
        if area > float(max_area_ratio) * float(img_area):
            return False
        aspect = max(float(bw) / float(max(1, bh)), float(bh) / float(max(1, bw)))
        if aspect < float(aspect_ratio_thresh):
            return False
        near_left = x < float(margin_ratio) * float(W)
        near_right = (x + bw) > (1.0 - float(margin_ratio)) * float(W)
        near_bottom = (y + bh) > (1.0 - float(margin_ratio)) * float(H)
        near_top = y < float(margin_ratio) * float(H)
        return bool(near_left or near_right or near_bottom or near_top)

    def is_gridline_like(
        mask: np.ndarray,
        bbox_xywh: Sequence[int],
        min_length_ratio: float = 0.5,
        max_thickness_px: int = 4,
    ) -> bool:
        if len(bbox_xywh) != 4:
            return False
        x, y, bw, bh = [int(v) for v in bbox_xywh]
        if bw <= 0 or bh <= 0:
            return False
        area = int(mask.sum())
        if area <= 0:
            return False

        horiz = bool(bw >= bh)
        length = bw if horiz else bh
        thickness = bh if horiz else bw
        length_ratio = float(length) / float(W if horiz else H)
        if length_ratio < float(min_length_ratio):
            return False
        if thickness > int(max_thickness_px):
            return False

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        s = hsv[..., 1][mask.astype(bool)]
        v = hsv[..., 2][mask.astype(bool)]
        if s.size == 0 or v.size == 0:
            return False
        mean_s = float(s.mean())
        mean_v = float(v.mean())
        if mean_v < 180:
            return False
        if mean_s > 60:
            return False
        return True

    def run_ocr_to_boxes(min_confidence: float = 0.4) -> List[Tuple[int, int, int, int]]:
        if not _easyocr_available():
            return []
        try:
            reader = _get_easyocr_reader()
        except Exception:
            return []
        try:
            # Upscale to improve recall on small tick labels (e.g., single letters "A/B/C").
            scale = 2 if max(int(W), int(H)) >= 900 else 3
            up_w = max(2, int(W) * int(scale))
            up_h = max(2, int(H) * int(scale))
            up = cv2.resize(image_rgb, (up_w, up_h), interpolation=cv2.INTER_CUBIC)
            results = reader.readtext(up, detail=1, paragraph=False)
        except Exception:
            return []
        boxes: List[Tuple[int, int, int, int]] = []
        for item in results:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            quad, _text, conf = item[0], item[1], item[2]
            try:
                conf_f = float(conf)
            except Exception:
                conf_f = 0.0
            if conf_f < float(min_confidence):
                continue
            xs: List[float] = []
            ys: List[float] = []
            try:
                for p in quad:
                    xs.append(float(p[0]))
                    ys.append(float(p[1]))
            except Exception:
                continue
            if not xs or not ys:
                continue
            x_min = int(min(xs) / float(scale))
            x_max = int(max(xs) / float(scale))
            y_min = int(min(ys) / float(scale))
            y_max = int(max(ys) / float(scale))
            x_min = max(0, min(int(W), x_min))
            x_max = max(0, min(int(W), x_max))
            y_min = max(0, min(int(H), y_min))
            y_max = max(0, min(int(H), y_max))
            if x_max <= x_min or y_max <= y_min:
                continue
            boxes.append((x_min, y_min, x_max, y_max))
        return boxes

    def filter_sam_masks(
        sam_masks: List[Dict[str, object]],
        *,
        min_area_ratio: float,
        iou_dup_thresh: float,
        composite_iou_thresh: float,
        text_mask: Optional[np.ndarray],
        text_overlap_thresh: float,
        drop_axis_like: bool,
        ocr_boxes: Optional[List[Tuple[int, int, int, int]]],
        drop_legend_like: bool,
    ) -> List[Dict[str, object]]:
        # For non-dot charts, match the original `gpt_segment.py` behavior as closely as possible
        # (full-mask IoU / containment checks). For dot charts, keep the optimized version.

        if not dotty:
            # Step 1: size filter
            large_enough: List[Dict[str, object]] = []
            for m in sam_masks:
                area_v = int(m.get("area", 0) or 0)
                if area_v < float(min_area_ratio) * float(img_area):
                    continue
                seg = m.get("segmentation")
                if not isinstance(seg, np.ndarray):
                    continue
                mm = dict(m)
                mm["_area"] = int(area_v)
                large_enough.append(mm)
            if not large_enough:
                return []

            large_enough.sort(key=lambda m: int(m.get("_area", m.get("area", 0) or 0) or 0), reverse=True)
            areas_for_median = np.asarray([int(m.get("_area", 0) or 0) for m in large_enough], dtype=np.float32)
            median_area = float(np.median(areas_for_median)) if areas_for_median.size else 0.0

            # Step 2: near-duplicate filter (full mask IoU)
            kept: List[Dict[str, object]] = []
            for m in large_enough:
                seg = m.get("segmentation")
                if not isinstance(seg, np.ndarray):
                    continue
                seg_b = seg.astype(bool)
                is_duplicate = False
                for km in kept:
                    kseg = km.get("segmentation")
                    if not isinstance(kseg, np.ndarray):
                        continue
                    iou = mask_iou(seg_b, np.asarray(kseg).astype(bool))
                    if iou > float(iou_dup_thresh):
                        is_duplicate = True
                        break
                if is_duplicate:
                    continue
                kept.append(m)

            # Step 3: composite/global-mask filter (contain >=2 smaller masks)
            non_composite: List[Dict[str, object]] = []
            for i, m in enumerate(kept):
                seg = m.get("segmentation")
                if not isinstance(seg, np.ndarray):
                    continue
                seg_b = seg.astype(bool)
                area_m = int(seg_b.sum())
                if area_m <= 0:
                    continue
                contain_count = 0
                for j, other in enumerate(kept):
                    if i == j:
                        continue
                    other_seg = other.get("segmentation")
                    if not isinstance(other_seg, np.ndarray):
                        continue
                    other_b = np.asarray(other_seg).astype(bool)
                    area_o = int(other_b.sum())
                    if area_o == 0 or area_o >= area_m:
                        continue
                    inter = int(np.logical_and(seg_b, other_b).sum())
                    if float(inter) / float(max(1, area_o)) > 0.98:
                        contain_count += 1
                        if contain_count >= 2:
                            break
                if contain_count >= 2:
                    continue
                non_composite.append(m)

            # Step 4~7: background / text / axis / legend / grid / border-text
            final_masks: List[Dict[str, object]] = []
            for m in non_composite:
                seg = m.get("segmentation")
                if not isinstance(seg, np.ndarray):
                    continue
                seg_b = seg.astype(bool)
                area = int(seg_b.sum())
                if area == 0:
                    continue

                # Keep only largest connected component
                seg_uint8 = seg_b.astype(np.uint8)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seg_uint8, connectivity=8)
                if num_labels > 2:
                    comp_areas = stats[1:, cv2.CC_STAT_AREA]
                    largest_label = 1 + int(np.argmax(comp_areas))
                    seg_b = labels == largest_label
                    area = int(seg_b.sum())
                    if area == 0:
                        continue
                    m = dict(m)
                    m["segmentation"] = seg_b.astype(np.uint8)

                if is_background_dominated(seg_b):
                    continue

                if text_mask is not None:
                    overlap_ratio = float(int(np.logical_and(seg_b, text_mask).sum())) / float(max(1, area))
                    if overlap_ratio > float(text_overlap_thresh):
                        continue

                bbox = m.get("bbox")
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    x, y, bw, bh = [int(v) for v in bbox]
                    if drop_axis_like and is_axis_like(seg_b, [x, y, bw, bh]):
                        continue
                    if is_gridline_like(seg_b, [x, y, bw, bh]):
                        continue

                    near_left = x < 0.06 * W
                    near_right = (x + bw) > 0.94 * W
                    near_bottom = (y + bh) > 0.94 * H
                    near_top = y < 0.06 * H
                    if median_area > 0 and area < 0.2 * median_area and (near_left or near_right or near_bottom or near_top):
                        continue

                if (
                    drop_legend_like
                    and ocr_boxes is not None
                    and isinstance(m.get("bbox"), (list, tuple))
                    and len(m.get("bbox")) == 4
                ):
                    x, y, bw, bh = [int(v) for v in m.get("bbox")]  # type: ignore[misc]
                    cx = x + 0.5 * float(bw)
                    cy = y + 0.5 * float(bh)
                    if cx > 0.8 * W:
                        is_legend = False
                        for (tx1, ty1, tx2, ty2) in ocr_boxes:
                            tx_min, tx_max = min(tx1, tx2), max(tx1, tx2)
                            ty_min, ty_max = min(ty1, ty2), max(ty1, ty2)
                            text_cy = 0.5 * float(ty_min + ty_max)
                            if abs(cy - text_cy) < 1.5 * float(bh) and (tx_min - 4 * bh) <= cx <= (tx_min + 1.5 * bh):
                                is_legend = True
                                break
                        if is_legend:
                            continue

                final_masks.append(m)

            if not final_masks:
                return []

            # Step 8: remove tiny leftovers relative to final median
            areas_final = np.asarray(
                [int(np.asarray(m["segmentation"]).astype(bool).sum()) for m in final_masks],
                dtype=np.float32,
            )
            median_area_final = float(np.median(areas_final)) if areas_final.size else 0.0
            min_keep_area = max(5.0, 0.05 * median_area_final) if median_area_final > 0 else 5.0
            filtered_final: List[Dict[str, object]] = []
            for m in final_masks:
                a = int(np.asarray(m["segmentation"]).astype(bool).sum())
                if float(a) >= float(min_keep_area):
                    filtered_final.append(m)
            return filtered_final

        # dotty charts:
        # Performance note: for dot charts we may have hundreds of masks.
        # Use bbox-gated intersection to avoid full-image IoU checks.

        def _bbox_xyxy_from_xywh(b: Sequence[int]) -> Tuple[int, int, int, int]:
            x, y, bw, bh = [int(v) for v in b]
            return (x, y, x + bw, y + bh)

        def _bbox_iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
            area_b = max(1, (bx2 - bx1) * (by2 - by1))
            return float(inter) / float(max(1, area_a + area_b - inter))

        def _mask_intersection(
            ma: np.ndarray,
            ba: Tuple[int, int, int, int],
            mb: np.ndarray,
            bb: Tuple[int, int, int, int],
        ) -> int:
            ix1 = max(int(ba[0]), int(bb[0]))
            iy1 = max(int(ba[1]), int(bb[1]))
            ix2 = min(int(ba[2]), int(bb[2]))
            iy2 = min(int(ba[3]), int(bb[3]))
            if ix2 <= ix1 or iy2 <= iy1:
                return 0
            return int(
                np.logical_and(
                    ma[iy1:iy2, ix1:ix2],
                    mb[iy1:iy2, ix1:ix2],
                ).sum()
            )

        # Step 1: size filter
        large_enough: List[Dict[str, object]] = []
        for m in sam_masks:
            area_v = int(m.get("area", 0) or 0)
            if area_v < float(min_area_ratio) * float(img_area):
                continue
            seg = m.get("segmentation")
            if not isinstance(seg, np.ndarray):
                continue
            bbox = m.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                bb_xyxy = _bbox_xyxy_from_xywh([int(v) for v in bbox])
            else:
                ys, xs = np.where(seg.astype(bool))
                if xs.size == 0 or ys.size == 0:
                    continue
                bb_xyxy = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
            mm = dict(m)
            mm["_bbox_xyxy"] = bb_xyxy
            mm["_area"] = int(area_v)
            large_enough.append(mm)
        if not large_enough:
            return []
        large_enough.sort(key=lambda m: int(m.get("area", 0) or 0), reverse=True)

        areas_for_median = np.asarray([int(m.get("_area", m.get("area", 0) or 0) or 0) for m in large_enough], dtype=np.float32)
        median_area = float(np.median(areas_for_median)) if areas_for_median.size else 0.0

        kept: List[Dict[str, object]] = []
        # Step 2: near-duplicate filter
        for m in large_enough:
            seg = m.get("segmentation")
            if not isinstance(seg, np.ndarray):
                continue
            seg_b = seg.astype(bool)
            bb_xyxy = m.get("_bbox_xyxy")
            if not (isinstance(bb_xyxy, tuple) and len(bb_xyxy) == 4):
                continue
            area_m = int(m.get("_area", m.get("area", 0) or int(seg_b.sum())) or 0)
            is_duplicate = False
            for km in kept:
                kseg = km.get("segmentation")
                if not isinstance(kseg, np.ndarray):
                    continue
                kbb = km.get("_bbox_xyxy")
                if not (isinstance(kbb, tuple) and len(kbb) == 4):
                    continue
                if _bbox_iou_xyxy(bb_xyxy, kbb) < 0.10:
                    continue
                area_k = int(km.get("_area", km.get("area", 0) or int(np.asarray(kseg).astype(bool).sum())) or 0)
                inter = _mask_intersection(seg_b, bb_xyxy, np.asarray(kseg).astype(bool), kbb)
                if inter <= 0:
                    continue
                iou = float(inter) / float(max(1, area_m + area_k - inter))
                if iou > float(iou_dup_thresh):
                    is_duplicate = True
                    break
            if is_duplicate:
                continue
            kept.append(m)

        # Step 3: composite/global filter (contain >=2 smaller masks)
        non_composite: List[Dict[str, object]] = []
        for i, m in enumerate(kept):
            seg = m.get("segmentation")
            if not isinstance(seg, np.ndarray):
                continue
            seg_b = seg.astype(bool)
            area_m = int(seg_b.sum())
            if area_m == 0:
                continue
            bb_xyxy = m.get("_bbox_xyxy")
            if not (isinstance(bb_xyxy, tuple) and len(bb_xyxy) == 4):
                continue
            contain_count = 0
            for j, other in enumerate(kept):
                if i == j:
                    continue
                other_seg = other.get("segmentation")
                if not isinstance(other_seg, np.ndarray):
                    continue
                other_b = other_seg.astype(bool)
                area_o = int(other_b.sum())
                if area_o == 0 or area_o >= area_m:
                    continue
                obb = other.get("_bbox_xyxy")
                if not (isinstance(obb, tuple) and len(obb) == 4):
                    continue
                # Only require some overlap in bbox space; full containment is too strict and
                # can keep global masks alive on charts with anti-aliased boundaries/labels.
                if _bbox_iou_xyxy(bb_xyxy, obb) < 0.01:
                    continue
                inter = _mask_intersection(seg_b, bb_xyxy, other_b, obb)
                if float(inter) / float(max(1, area_o)) > 0.98:
                    contain_count += 1
                    if contain_count >= 2:
                        break
            if contain_count >= 2:
                continue
            non_composite.append(m)

        final_masks: List[Dict[str, object]] = []
        for m in non_composite:
            seg = m.get("segmentation")
            if not isinstance(seg, np.ndarray):
                continue
            seg_b = seg.astype(bool)
            area = int(seg_b.sum())
            if area == 0:
                continue

            # Keep only largest connected component (reduce speckle noise).
            # This is expensive on dot charts; skip there.
            if not dotty:
                seg_uint8 = seg_b.astype(np.uint8)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seg_uint8, connectivity=8)
                if num_labels > 2:
                    comp_areas = stats[1:, cv2.CC_STAT_AREA]
                    largest_label = 1 + int(np.argmax(comp_areas))
                    seg_b = labels == largest_label
                    area = int(seg_b.sum())
                    if area == 0:
                        continue
                    m = dict(m)
                    m["segmentation"] = seg_b.astype(np.uint8)

            # 4) background-dominated
            if is_background_dominated(seg_b):
                continue

            # 5) text overlap filter
            if text_mask is not None:
                overlap_ratio = float(int(np.logical_and(seg_b, text_mask).sum())) / float(max(1, area))
                if overlap_ratio > float(text_overlap_thresh):
                    continue

            # 6) axis/grid filters
            bbox = m.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x, y, bw, bh = [int(v) for v in bbox]
                if drop_axis_like and is_axis_like(seg_b, [x, y, bw, bh]):
                    continue
                if is_gridline_like(seg_b, [x, y, bw, bh]):
                    continue

                # 6-1) border small-text filter
                near_left = x < 0.06 * W
                near_right = (x + bw) > 0.94 * W
                near_bottom = (y + bh) > 0.94 * H
                near_top = y < 0.06 * H
                if median_area > 0 and area < 0.2 * median_area and (near_left or near_right or near_bottom or near_top):
                    continue

            # 7) legend-like filter (markers near OCR text on right)
            if drop_legend_like and ocr_boxes is not None and isinstance(m.get("bbox"), (list, tuple)) and len(m.get("bbox")) == 4:
                x, y, bw, bh = [int(v) for v in m.get("bbox")]  # type: ignore[misc]
                cx = x + 0.5 * float(bw)
                cy = y + 0.5 * float(bh)
                if cx > 0.8 * W:
                    is_legend = False
                    for (tx1, ty1, tx2, ty2) in ocr_boxes:
                        tx_min, tx_max = min(tx1, tx2), max(tx1, tx2)
                        ty_min, ty_max = min(ty1, ty2), max(ty1, ty2)
                        text_cy = 0.5 * float(ty_min + ty_max)
                        if abs(cy - text_cy) < 1.5 * float(bh) and (tx_min - 4 * bh) <= cx <= (tx_min + 1.5 * bh):
                            is_legend = True
                            break
                    if is_legend:
                        continue

            final_masks.append(m)

        if not final_masks:
            return []

        areas_final = np.asarray([int(np.asarray(m["segmentation"]).astype(bool).sum()) for m in final_masks], dtype=np.float32)
        median_area_final = float(np.median(areas_final)) if areas_final.size else 0.0
        min_keep_area = max(5.0, 0.05 * median_area_final) if median_area_final > 0 else 5.0
        filtered_final: List[Dict[str, object]] = []
        for m in final_masks:
            a = int(np.asarray(m["segmentation"]).astype(bool).sum())
            if float(a) >= float(min_keep_area):
                filtered_final.append(m)
        return filtered_final

    # ---------------------------------------------------------
    # 1) SAM1 segmentation
    # ---------------------------------------------------------
    # Adapt SAM params to reduce mask explosion on large non-dot charts.
    sam_points_per_side = 64
    sam_crop_n_layers = 1
    try:
        w_s, h_s = seg_img.size
        max_side_s = max(int(w_s), int(h_s))
        if not dotty:
            if max_side_s >= 1600:
                sam_points_per_side = 32
                sam_crop_n_layers = 0
            elif max_side_s >= 1200:
                sam_points_per_side = 48
                sam_crop_n_layers = 0
    except Exception:
        pass
    debug["sam_params"] = {"points_per_side": int(sam_points_per_side), "crop_n_layers": int(sam_crop_n_layers)}
    try:
        mask_generator = _get_sam1_mask_generator(
            output_mode="binary_mask",
            points_per_side=int(sam_points_per_side),
            crop_n_layers=int(sam_crop_n_layers),
            box_nms_thresh=0.9,
            pred_iou_thresh=0.80,
            stability_score_thresh=0.85,
            min_mask_region_area=0,
        )
    except Exception as e:
        raise RuntimeError(
            "segment_and_mark requires SAM1 (Segment Anything v1). "
            "Provide a checkpoint via `SAM1_CHECKPOINT_PATH`/`SAM_CHECKPOINT_PATH`, "
            "or place `sam_vit_h_4b8939.pth` in the repo root.\n"
            f"Original error: {e}"
        ) from e

    raw_masks = mask_generator.generate(image_rgb)
    debug["sam1_raw_mask_count"] = int(len(raw_masks))

    # 2) OCR -> text_mask (skip for dotty charts; pre-clean already removes title/legend)
    ocr_boxes: Optional[List[Tuple[int, int, int, int]]] = None
    text_mask: Optional[np.ndarray] = None
    if (not dotty) and _easyocr_available():
        try:
            ocr_boxes = run_ocr_to_boxes(min_confidence=0.30)
            text_mask = build_text_mask(ocr_boxes)
        except Exception:
            ocr_boxes = None
            text_mask = None
    debug["ocr_boxes_count"] = int(len(ocr_boxes)) if ocr_boxes is not None else 0

    # 3) Filter masks (gpt_segment defaults)
    # The legacy CLI default (`min_area=5000`) is not compatible with dot charts.
    # Keep gpt_segment behavior by default.
    if dotty or int(min_area) == 5000:
        min_area_ratio = 0.00005
    else:
        # Interpret `min_area` as absolute pixels.
        min_area_ratio = float(max(1, int(min_area))) / float(max(1, img_area))

    # gpt_segment uses 0.95 as default composite threshold.
    composite_thresh = 0.95 if float(iou_thresh_composite) == 0.98 else float(iou_thresh_composite)

    cleaned_mask_dicts = filter_sam_masks(
        raw_masks,
        min_area_ratio=float(min_area_ratio),
        iou_dup_thresh=float(iou_thresh_unique),
        composite_iou_thresh=float(composite_thresh),
        text_mask=text_mask,
        text_overlap_thresh=0.3,
        drop_axis_like=True,
        ocr_boxes=ocr_boxes,
        drop_legend_like=True,
    )

    # Fallback (less aggressive) if nothing remains.
    if not cleaned_mask_dicts and raw_masks:
        cleaned_mask_dicts = filter_sam_masks(
            raw_masks,
            min_area_ratio=max(0.00002, float(min_area_ratio) * 0.4),
            iou_dup_thresh=float(iou_thresh_unique),
            composite_iou_thresh=float(composite_thresh),
            text_mask=None,
            text_overlap_thresh=0.9,
            drop_axis_like=False,
            ocr_boxes=None,
            drop_legend_like=False,
        )

    def _dotty_overlap_suppress(
        masks: List[Dict[str, object]],
        *,
        overlap_small_frac: float = 0.92,
        overlap_candidate_frac: float = 0.35,
    ) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
        if not masks:
            return masks, {"enabled": False, "reason": "empty"}

        # Estimate typical dot size.
        areas = np.asarray(
            [int(np.asarray(m.get("segmentation")).astype(bool).sum()) for m in masks],
            dtype=np.float32,
        )
        med_area = float(np.median(areas[areas > 0])) if float(np.sum(areas > 0)) > 0 else 0.0

        widths: List[int] = []
        heights: List[int] = []
        bbs_xyxy: List[Tuple[int, int, int, int]] = []
        for m in masks:
            bb = m.get("bbox")
            if isinstance(bb, (list, tuple)) and len(bb) == 4:
                x, y, bw, bh = [int(v) for v in bb]
                bb_xyxy = (x, y, x + bw, y + bh)
            else:
                seg = m.get("segmentation")
                if not isinstance(seg, np.ndarray):
                    bb_xyxy = (0, 0, 0, 0)
                else:
                    ys, xs = np.where(np.asarray(seg).astype(bool))
                    if xs.size == 0 or ys.size == 0:
                        bb_xyxy = (0, 0, 0, 0)
                    else:
                        bb_xyxy = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
            bbs_xyxy.append(bb_xyxy)
            widths.append(max(1, int(bb_xyxy[2]) - int(bb_xyxy[0])))
            heights.append(max(1, int(bb_xyxy[3]) - int(bb_xyxy[1])))

        med_w = float(np.median(np.asarray(widths, dtype=np.float32))) if widths else 0.0
        med_h = float(np.median(np.asarray(heights, dtype=np.float32))) if heights else 0.0
        med_diam = 0.0
        if med_area > 0:
            med_diam = 2.0 * math.sqrt(float(med_area) / float(math.pi))
        dot_diam = float(max(med_diam, med_w, med_h, 8.0))
        cell_size = int(max(8, round(1.5 * dot_diam)))

        def _score(m: Dict[str, object], bb_xyxy: Tuple[int, int, int, int], area: float) -> float:
            s = m.get("predicted_iou", None)
            if s is None:
                s = m.get("stability_score", 0.0)
            try:
                s_f = float(s or 0.0)
            except Exception:
                s_f = 0.0
            bw = float(max(1, int(bb_xyxy[2]) - int(bb_xyxy[0])))
            bh = float(max(1, int(bb_xyxy[3]) - int(bb_xyxy[1])))
            area_dev = 0.0
            if med_area > 0 and area > 0:
                area_dev = abs(math.log(max(1e-6, float(area) / float(med_area))))
            aspect_dev = abs(math.log(max(1e-6, bw / max(1e-6, bh))))
            return float(s_f) - (1.35 * float(area_dev)) - (0.75 * float(aspect_dev))

        # Sort best dot-like masks first.
        scored = []
        for m, bb_xyxy, a in zip(masks, bbs_xyxy, areas):
            scored.append((_score(m, bb_xyxy, float(a)), m, bb_xyxy, int(a)))
        scored.sort(key=lambda t: t[0], reverse=True)

        kept: List[Dict[str, object]] = []
        kept_meta: List[Tuple[np.ndarray, Tuple[int, int, int, int], int]] = []
        grid: Dict[Tuple[int, int], List[int]] = {}
        removed = 0

        for _q, cand, cbb, ca in scored:
            seg = cand.get("segmentation")
            if not isinstance(seg, np.ndarray) or ca <= 0:
                continue
            cm = np.asarray(seg).astype(bool)
            cx = 0.5 * (float(cbb[0]) + float(cbb[2]))
            cy = 0.5 * (float(cbb[1]) + float(cbb[3]))
            gx = int(cx // float(cell_size))
            gy = int(cy // float(cell_size))

            drop = False
            for ny in (gy - 1, gy, gy + 1):
                for nx in (gx - 1, gx, gx + 1):
                    for idx_k in grid.get((nx, ny), []):
                        km, kbb, ka = kept_meta[idx_k]
                        ix1 = max(int(cbb[0]), int(kbb[0]))
                        iy1 = max(int(cbb[1]), int(kbb[1]))
                        ix2 = min(int(cbb[2]), int(kbb[2]))
                        iy2 = min(int(cbb[3]), int(kbb[3]))
                        if ix2 <= ix1 or iy2 <= iy1:
                            continue
                        inter = int(np.logical_and(cm[iy1:iy2, ix1:ix2], km[iy1:iy2, ix1:ix2]).sum())
                        if inter <= 0:
                            continue
                        if inter >= int(float(overlap_small_frac) * float(min(ca, ka))):
                            drop = True
                            break
                        if inter >= int(float(overlap_candidate_frac) * float(ca)):
                            drop = True
                            break
                    if drop:
                        break
                if drop:
                    break

            if drop:
                removed += 1
                continue

            kept.append(cand)
            kept_meta.append((cm, cbb, ca))
            grid.setdefault((gx, gy), []).append(len(kept) - 1)

        dbg = {
            "enabled": True,
            "before": int(len(masks)),
            "after": int(len(kept)),
            "removed": int(removed),
            "overlap_small_frac": float(overlap_small_frac),
            "overlap_candidate_frac": float(overlap_candidate_frac),
            "median_area": float(med_area),
            "dot_diameter_est": float(dot_diam),
            "cell_size": int(cell_size),
        }
        return kept, dbg

    # Dot-based donut charts: drop very large "global" masks (e.g., the whole ring) so we keep per-dot instances.
    if dotty and len(cleaned_mask_dicts) >= 50:
        try:
            areas = np.asarray([int(np.asarray(m.get("segmentation")).astype(bool).sum()) for m in cleaned_mask_dicts], dtype=np.int64)
            med = float(np.median(areas)) if areas.size else 0.0
            max_keep = int(max(5000.0, 25.0 * med))
            before = int(len(cleaned_mask_dicts))
            filtered: List[Dict[str, object]] = []
            for m in cleaned_mask_dicts:
                a = int(np.asarray(m.get("segmentation")).astype(bool).sum())
                if a <= max_keep:
                    filtered.append(m)
            cleaned_mask_dicts = filtered
            debug["dotty_global_mask_filter"] = {
                "enabled": True,
                "before": before,
                "after": int(len(cleaned_mask_dicts)),
                "median_area": float(med),
                "max_keep_area": int(max_keep),
            }
        except Exception as e:
            debug["dotty_global_mask_filter"] = {"enabled": True, "error": str(e)}

    # Dot-based donut charts: suppress overlapping/nested masks (dot interior shards).
    if dotty and len(cleaned_mask_dicts) >= 50:
        try:
            if isinstance(debug_dir, str) and debug_dir.strip():
                cleaned_pre_overlap = list(cleaned_mask_dicts)
            cleaned_mask_dicts, dbg_supp = _dotty_overlap_suppress(cleaned_mask_dicts)
            debug["dotty_overlap_suppression"] = dbg_supp
        except Exception as e:
            debug["dotty_overlap_suppression"] = {"enabled": True, "error": str(e)}

    debug["final_mask_count"] = int(len(cleaned_mask_dicts))

    # 4) Draw contours + labels (opencv) and build return payload.
    vis = image_bgr.copy()
    cleaned_out: List[Dict[str, object]] = []

    # Labeling: avoid clutter when there are too many instances.
    draw_labels = bool(len(cleaned_mask_dicts) <= 80)
    contour_thickness = 2
    font_scale = 0.45 if max(H, W) <= 1600 else 0.55
    font_thickness = 1

    for idx, m in enumerate(cleaned_mask_dicts, start=1):
        seg = m.get("segmentation")
        if not isinstance(seg, np.ndarray):
            continue
        mask = seg.astype(bool)
        area = int(mask.sum())
        if area <= 0:
            continue
        mask_uint8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        color = ((37 * idx) % 255, (97 * idx) % 255, (173 * idx) % 255)
        cv2.drawContours(vis, contours, -1, color, contour_thickness)

        if draw_labels:
            largest_cnt = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_cnt)
            if float(M.get("m00", 0.0)) != 0.0:
                cx = int(float(M["m10"]) / float(M["m00"]))
                cy = int(float(M["m01"]) / float(M["m00"]))
                cv2.putText(
                    vis,
                    str(idx),
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    font_thickness,
                    lineType=cv2.LINE_AA,
                )

        # Convert bbox xywh -> xyxy (fallback to mask bbox).
        bb_xyxy: Tuple[int, int, int, int]
        bb = m.get("bbox")
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            x, y, bw, bh = [int(v) for v in bb]
            bb_xyxy = (x, y, x + bw, y + bh)
        else:
            ys, xs = np.where(mask)
            if xs.size and ys.size:
                bb_xyxy = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
            else:
                bb_xyxy = (0, 0, 0, 0)

        score = m.get("predicted_iou", None)
        if score is None:
            score = m.get("stability_score", 0.0)
        try:
            score_f = float(score or 0.0)
        except Exception:
            score_f = 0.0

        cleaned_out.append(
            {
                "id": int(idx),
                "bbox_xyxy": (int(bb_xyxy[0]), int(bb_xyxy[1]), int(bb_xyxy[2]), int(bb_xyxy[3])),
                "area": int(area),
                "score": float(score_f),
                "segmentation": mask.astype(bool),
            }
        )

    labeled_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    labeled_img = Image.fromarray(labeled_rgb)

    # 5) Debug outputs: overlays + counts (+ small gallery to inspect masks).
    if isinstance(debug_dir, str) and debug_dir.strip():
        os.makedirs(debug_dir, exist_ok=True)

        def _overlay_boundaries(base_rgb: np.ndarray, masks_list: List[Dict[str, object]], *, thickness: int = 1) -> np.ndarray:
            out = base_rgb.copy()
            for i, mm in enumerate(masks_list, start=1):
                seg = mm.get("segmentation")
                if not isinstance(seg, np.ndarray):
                    continue
                msk = seg.astype(bool)
                er = cv2.erode(msk.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
                boundary = np.logical_and(msk, np.logical_not(er))
                if thickness > 1:
                    boundary = cv2.dilate(boundary.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=int(thickness - 1)) > 0
                col = ((37 * i) % 255, (97 * i) % 255, (173 * i) % 255)
                out[boundary] = np.asarray(col, dtype=np.uint8)
            return out

        def _mask_gallery(base: np.ndarray, masks_list: List[Dict[str, object]], out_path: str, *, max_n: int = 64) -> None:
            from PIL import ImageFont

            n = min(int(max_n), len(masks_list))
            if n <= 0:
                return
            cols = 8
            rows = int(math.ceil(float(n) / float(cols)))
            cell = 96
            pad = 6
            canvas = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
            draw = ImageDraw.Draw(canvas)
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 14)
            except Exception:
                font = ImageFont.load_default()

            for i in range(n):
                m = masks_list[i].get("segmentation")
                if not isinstance(m, np.ndarray):
                    continue
                msk = m.astype(bool)
                ys, xs = np.where(msk)
                if xs.size == 0 or ys.size == 0:
                    continue
                x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
                # crop around mask bbox with small margin
                mx = int(0.15 * max(1, x2 - x1))
                my = int(0.15 * max(1, y2 - y1))
                x1 = max(0, x1 - mx)
                y1 = max(0, y1 - my)
                x2 = min(W, x2 + mx)
                y2 = min(H, y2 + my)
                crop = base[y1:y2, x1:x2]
                # overlay mask region
                overlay = crop.copy()
                local = msk[y1:y2, x1:x2]
                overlay[local] = (0.5 * overlay[local] + 0.5 * np.asarray([255, 255, 0], dtype=np.uint8)).astype(np.uint8)
                tile = Image.fromarray(overlay).resize((cell - 2 * pad, cell - 2 * pad), resample=Image.BILINEAR)
                r = i // cols
                c = i % cols
                ox = c * cell + pad
                oy = r * cell + pad
                canvas.paste(tile, (ox, oy))
                draw.text((c * cell + 4, r * cell + 2), str(i + 1), fill=(0, 0, 0), font=font)

            canvas.save(out_path)

        try:
            raw_overlay = _overlay_boundaries(image_rgb, raw_masks, thickness=1)
            Image.fromarray(raw_overlay).save(os.path.join(debug_dir, "sam1_raw_masks_overlay.png"))
        except Exception:
            pass
        if isinstance(cleaned_pre_overlap, list) and cleaned_pre_overlap:
            try:
                pre_overlay = _overlay_boundaries(image_rgb, cleaned_pre_overlap, thickness=1)
                Image.fromarray(pre_overlay).save(os.path.join(debug_dir, "pre_overlap_masks_overlay.png"))
            except Exception:
                pass
        try:
            final_overlay = _overlay_boundaries(image_rgb, cleaned_mask_dicts, thickness=1)
            Image.fromarray(final_overlay).save(os.path.join(debug_dir, "final_masks_overlay.png"))
        except Exception:
            pass
        try:
            _mask_gallery(image_rgb, cleaned_mask_dicts, os.path.join(debug_dir, "final_masks_gallery.png"), max_n=64)
        except Exception:
            pass

        # Update debug JSON
        try:
            with open(os.path.join(debug_dir, "segment_and_mark_debug.json"), "w") as f:
                json.dump(debug, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    return labeled_img, cleaned_out


def _parse_axis_tick_value(text: str) -> Optional[float]:
    """
    Parse a numeric axis tick label into a float.

    Intended for axis tickers only (rejects alphanumeric labels like "Pop: 35").
    Supports:
      - thousands separators: 1,000
      - sign: -10, −10
      - percent: 40%
      - suffix multipliers: 1.2k, 3M, 4.5B
      - scientific notation: 1e3, 2.5E-2
      - currency symbols are ignored ($, € etc.)
    """
    if not isinstance(text, str):
        return None
    s0 = text.strip()
    if not s0:
        return None

    # Normalize unicode minus/dash.
    s0 = s0.replace("−", "-").replace("–", "-").replace("—", "-")

    # Reject if any letters/symbols other than allowed numeric tokens are present.
    # This prevents parsing things like "Country: 12" or "Pop: 35".
    disallowed = re.sub(r"[0-9\s,.\-+()$€£¥%kKmMbBtTeE]", "", s0)
    if disallowed:
        return None

    s = s0.replace(" ", "")

    neg = False
    if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        neg = True
        s = s[1:-1]

    # Strip leading currency symbols (common on y-axes).
    s = re.sub(r"^[\\$€£¥]+", "", s)

    # Percent sign just indicates unit; value stays in "percent points".
    if s.endswith("%"):
        s = s[:-1]

    # Common OCR confusion: 'O'/'o' for zero in numeric contexts.
    s = re.sub(r"(?<=\\d)[oO](?=\\d)", "0", s)
    s = re.sub(r"^[oO](?=\\d)", "0", s)
    s = re.sub(r"(?<=\\d)[oO]$", "0", s)

    mult = 1.0
    m = re.match(r"^(.*?)([kKmMbBtT])$", s)
    if m:
        s = m.group(1)
        suf = m.group(2).lower()
        mult = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}.get(suf, 1.0)

    s = s.replace(",", "")
    if not s:
        return None

    # Full-string numeric parse preferred; fallback to numeric token.
    try:
        v = float(s)
    except Exception:
        m2 = re.search(r"[-+]?(?:\\d+\\.?\\d*|\\d*\\.\\d+)(?:[eE][-+]?\\d+)?", s)
        if not m2:
            return None
        token = m2.group(0)
        # If there is any other junk besides the token, reject.
        if token != s:
            return None
        try:
            v = float(token)
        except Exception:
            return None

    v = float(v) * float(mult)
    if neg:
        v = -v
    return float(v)


def axis_localizer(
    pil_image: Image.Image,
    axis: str,
    axis_threshold: float = 0.2,
    axis_tickers: Optional[List] = None,
) -> Tuple[List[float], List[int]]:
    """
    Localize a chosen axis (x, left y, right y), detect its tick labels, and map them to pixel positions.

    Returns:
      - axis_values: list[float]
      - axis_pixel_positions: list[int] (x for x-axis, y for y-axes)
    """
    axis_values, axis_pixel_positions, _, _, _, _ = _axis_localizer_with_boxes(
        pil_image, axis=axis, axis_threshold=axis_threshold, axis_tickers=axis_tickers
    )
    return axis_values, axis_pixel_positions


def _axis_localizer_with_boxes(
    pil_image: Image.Image,
    *,
    axis: str,
    axis_threshold: float = 0.2,
    axis_tickers: Optional[List] = None,
) -> Tuple[
    List[float],
    List[int],
    List[Optional[BboxXyxy]],
    List[str],
    List[int],
    List[Optional[BboxXyxy]],
]:
    """
    Internal helper for CLI previews: like `axis_localizer`, but also returns per-tick bboxes (xyxy)
    and OCR tick texts (including non-numeric tick labels when available).

    For inferred ticks (filled via linear fit), bbox is None.
    """
    if not isinstance(pil_image, Image.Image):
        raise TypeError("pil_image must be a PIL.Image.Image")
    axis_s = str(axis or "").strip().lower()
    if axis_s not in ("x", "top_x", "y", "right_y"):
        raise ValueError("axis must be one of: 'x', 'top_x', 'y', 'right_y'")

    img = pil_image.convert("RGB")
    W, H = img.size
    if W <= 1 or H <= 1:
        raise ValueError("Invalid image size for axis_localizer")

    thr = float(axis_threshold)
    thr = max(0.05, min(0.45, thr))

    if axis_s == "y":
        roi = (0, 0, int(round(thr * W)), H)
    elif axis_s == "right_y":
        roi = (int(round((1.0 - thr) * W)), 0, W, H)
    elif axis_s == "top_x":
        # Top x-axis tick labels are often below the title/legend; use a slightly larger ROI
        # than the bottom-x default to avoid missing them.
        thr_top = max(0.30, min(0.45, thr + 0.15))
        roi = (0, 0, W, int(round(thr_top * H)))
    else:
        # Bottom x-axis tick labels can sit above legends/footnotes; use a slightly larger ROI
        # than the default threshold to avoid missing them.
        thr_bottom = max(0.25, min(0.45, thr + 0.10))
        roi = (0, int(round((1.0 - thr_bottom) * H)), W, H)

    x0, y0, x1, y1 = [int(v) for v in roi]
    x0 = max(0, min(x0, W))
    x1 = max(0, min(x1, W))
    y0 = max(0, min(y0, H))
    y1 = max(0, min(y1, H))
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError("axis_localizer produced an empty ROI")

    roi_img = img.crop((x0, y0, x1, y1))
    rw, rh = roi_img.size

    # OCR backend selection (prefer EasyOCR for rotation robustness).
    used_engine = None
    ocr_words_numeric: List[Dict[str, object]] = []
    ocr_words_text: List[Dict[str, object]] = []

    def _append_word(
        target: List[Dict[str, object]],
        text: str,
        conf: float,
        bb_xyxy: Tuple[int, int, int, int],
        *,
        tile_off: Tuple[int, int],
    ) -> None:
        if not text:
            return
        tx, ty = tile_off
        x1_, y1_, x2_, y2_ = bb_xyxy
        target.append(
            {
                "text": str(text),
                "conf": float(conf),
                "bbox_xyxy": (int(x1_ + tx), int(y1_ + ty), int(x2_ + tx), int(y2_ + ty)),
            }
        )

    if _easyocr_available():
        used_engine = "easyocr"
        reader = _get_easyocr_reader()
        allow = "0123456789.-+%kKmMbBtTeE,()$€£¥"
        allow_text = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789%.,:/\\-+_()'’"

        def _easyocr_words_allow(im: Image.Image) -> List[Dict[str, object]]:
            arr = np.asarray(im.convert("RGB"))
            try:
                det = reader.readtext(arr, detail=1, paragraph=False, allowlist=allow)
            except Exception:
                det = []
            out: List[Dict[str, object]] = []
            for item in det:
                if not isinstance(item, (list, tuple)) or len(item) < 3:
                    continue
                quad, text, conf = item[0], item[1], item[2]
                text_s = str(text or "").strip()
                if not text_s:
                    continue
                try:
                    conf_f = float(conf)
                except Exception:
                    conf_f = 0.0
                xs: List[float] = []
                ys: List[float] = []
                try:
                    for p in quad:
                        xs.append(float(p[0]))
                        ys.append(float(p[1]))
                except Exception:
                    continue
                if not xs or not ys:
                    continue
                x1_ = int(max(0, min(xs)))
                y1_ = int(max(0, min(ys)))
                x2_ = int(max(xs))
                y2_ = int(max(ys))
                if x2_ <= x1_ or y2_ <= y1_:
                    continue
                out.append({"text": text_s, "conf": conf_f, "bbox_xyxy": (x1_, y1_, x2_, y2_)})
            return out

        def _easyocr_words_any(im: Image.Image) -> List[Dict[str, object]]:
            arr = np.asarray(im.convert("RGB"))
            try:
                det = reader.readtext(arr, detail=1, paragraph=False)
            except Exception:
                det = []
            out: List[Dict[str, object]] = []
            for item in det:
                if not isinstance(item, (list, tuple)) or len(item) < 3:
                    continue
                quad, text, conf = item[0], item[1], item[2]
                text_s = str(text or "").strip()
                if not text_s:
                    continue
                try:
                    conf_f = float(conf)
                except Exception:
                    conf_f = 0.0
                xs: List[float] = []
                ys: List[float] = []
                try:
                    for p in quad:
                        xs.append(float(p[0]))
                        ys.append(float(p[1]))
                except Exception:
                    continue
                if not xs or not ys:
                    continue
                x1_ = int(max(0, min(xs)))
                y1_ = int(max(0, min(ys)))
                x2_ = int(max(xs))
                y2_ = int(max(ys))
                if x2_ <= x1_ or y2_ <= y1_:
                    continue
                out.append({"text": text_s, "conf": conf_f, "bbox_xyxy": (x1_, y1_, x2_, y2_)})
            return out

        def _easyocr_words_text_allow(im: Image.Image) -> List[Dict[str, object]]:
            arr = np.asarray(im.convert("RGB"))
            try:
                det = reader.readtext(arr, detail=1, paragraph=False, allowlist=allow_text)
            except Exception:
                det = []
            out: List[Dict[str, object]] = []
            for item in det:
                if not isinstance(item, (list, tuple)) or len(item) < 3:
                    continue
                quad, text, conf = item[0], item[1], item[2]
                text_s = str(text or "").strip()
                if not text_s:
                    continue
                try:
                    conf_f = float(conf)
                except Exception:
                    conf_f = 0.0
                xs: List[float] = []
                ys: List[float] = []
                try:
                    for p in quad:
                        xs.append(float(p[0]))
                        ys.append(float(p[1]))
                except Exception:
                    continue
                if not xs or not ys:
                    continue
                x1_ = int(max(0, min(xs)))
                y1_ = int(max(0, min(ys)))
                x2_ = int(max(xs))
                y2_ = int(max(ys))
                if x2_ <= x1_ or y2_ <= y1_:
                    continue
                out.append({"text": text_s, "conf": conf_f, "bbox_xyxy": (x1_, y1_, x2_, y2_)})
            return out

        def _prep_text_ocr(im: Image.Image) -> Image.Image:
            try:
                from PIL import ImageEnhance, ImageFilter, ImageOps  # type: ignore

                g = im.convert("L")
                g = ImageOps.autocontrast(g)
                g = ImageEnhance.Contrast(g).enhance(2.0)
                g = g.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
                return g.convert("RGB")
            except Exception:
                return im

        # For very wide ROIs (typical x-axis), tile OCR improves recall vs running on a huge strip.
        if axis_s == "x" and rw >= 900:
            n_tiles = int(max(1, math.ceil(float(rw) / 900.0)))
            overlap = int(round(0.10 * float(rw) / float(max(1, n_tiles))))
            tile_w = int(math.ceil(float(rw) / float(n_tiles)))
            scale = 2
            for i in range(n_tiles):
                tx1 = max(0, i * tile_w - overlap)
                tx2 = min(rw, (i + 1) * tile_w + overlap)
                if tx2 <= tx1:
                    continue
                tile = roi_img.crop((tx1, 0, tx2, rh))
                tile_up = tile.resize((max(2, (tx2 - tx1) * scale), max(2, rh * scale)), resample=Image.BICUBIC)
                det = _easyocr_words_allow(tile_up)
                for w0 in det:
                    bb = w0.get("bbox_xyxy")
                    if not isinstance(bb, tuple) or len(bb) != 4:
                        continue
                    bx1, by1, bx2, by2 = [int(round(float(v) / float(scale))) for v in bb]
                    _append_word(
                        ocr_words_numeric,
                        str(w0.get("text", "") or ""),
                        float(w0.get("conf", 0.0) or 0.0),
                        (bx1, by1, bx2, by2),
                        tile_off=(x0 + tx1, y0),
                    )
        else:
            scale = 2 if max(rw, rh) >= 1400 else 3
            roi_up = roi_img.resize((max(2, rw * scale), max(2, rh * scale)), resample=Image.BICUBIC)
            det = _easyocr_words_allow(roi_up)
            for w0 in det:
                bb = w0.get("bbox_xyxy")
                if not isinstance(bb, tuple) or len(bb) != 4:
                    continue
                bx1, by1, bx2, by2 = [int(round(float(v) / float(scale))) for v in bb]
                _append_word(
                    ocr_words_numeric,
                    str(w0.get("text", "") or ""),
                    float(w0.get("conf", 0.0) or 0.0),
                    (bx1, by1, bx2, by2),
                    tile_off=(x0, y0),
                )

        # Second pass (x-axes only): capture non-numeric tick labels (e.g., months) without an allowlist.
        if axis_s in ("x", "top_x"):
            if axis_s == "x" and rw >= 900:
                n_tiles = int(max(1, math.ceil(float(rw) / 900.0)))
                overlap = int(round(0.10 * float(rw) / float(max(1, n_tiles))))
                tile_w = int(math.ceil(float(rw) / float(n_tiles)))
                scale = 2
                for i in range(n_tiles):
                    tx1 = max(0, i * tile_w - overlap)
                    tx2 = min(rw, (i + 1) * tile_w + overlap)
                    if tx2 <= tx1:
                        continue
                    tile = roi_img.crop((tx1, 0, tx2, rh))
                    tile_up = tile.resize(
                        (max(2, (tx2 - tx1) * scale), max(2, rh * scale)),
                        resample=Image.BICUBIC,
                    )
                    tile_up = _prep_text_ocr(tile_up)
                    det = _easyocr_words_text_allow(tile_up)
                    if not det:
                        det = _easyocr_words_any(tile_up)
                    for w0 in det:
                        bb = w0.get("bbox_xyxy")
                        if not isinstance(bb, tuple) or len(bb) != 4:
                            continue
                        bx1, by1, bx2, by2 = [int(round(float(v) / float(scale))) for v in bb]
                        _append_word(
                            ocr_words_text,
                            str(w0.get("text", "") or ""),
                            float(w0.get("conf", 0.0) or 0.0),
                            (bx1, by1, bx2, by2),
                            tile_off=(x0 + tx1, y0),
                        )
            else:
                scale = 2 if max(rw, rh) >= 1400 else 3
                roi_up = roi_img.resize((max(2, rw * scale), max(2, rh * scale)), resample=Image.BICUBIC)
                roi_up = _prep_text_ocr(roi_up)
                det = _easyocr_words_text_allow(roi_up)
                if not det:
                    det = _easyocr_words_any(roi_up)
                for w0 in det:
                    bb = w0.get("bbox_xyxy")
                    if not isinstance(bb, tuple) or len(bb) != 4:
                        continue
                    bx1, by1, bx2, by2 = [int(round(float(v) / float(scale))) for v in bb]
                    _append_word(
                        ocr_words_text,
                        str(w0.get("text", "") or ""),
                        float(w0.get("conf", 0.0) or 0.0),
                        (bx1, by1, bx2, by2),
                        tile_off=(x0, y0),
                    )

    elif _tesseract_available():
        used_engine = "tesseract"
        scale = 2
        roi_up = roi_img.resize((max(2, rw * scale), max(2, rh * scale)), resample=Image.BICUBIC)
        det = _tesseract_words(roi_up, psm=6)
        for w0 in det:
            bb = w0.get("bbox_xyxy")
            if not isinstance(bb, tuple) or len(bb) != 4:
                continue
            bx1, by1, bx2, by2 = [int(round(float(v) / float(scale))) for v in bb]
            _append_word(
                ocr_words_numeric,
                str(w0.get("text", "") or ""),
                float(w0.get("conf", 0.0) or 0.0) / 100.0,
                (bx1, by1, bx2, by2),
                tile_off=(x0, y0),
            )
            if axis_s in ("x", "top_x"):
                _append_word(
                    ocr_words_text,
                    str(w0.get("text", "") or ""),
                    float(w0.get("conf", 0.0) or 0.0) / 100.0,
                    (bx1, by1, bx2, by2),
                    tile_off=(x0, y0),
                )
    else:
        raise RuntimeError(
            "axis_localizer requires an OCR backend. Install EasyOCR (`pip install easyocr`) "
            "or install `tesseract` + `pytesseract`."
        )

    # Non-numeric tick labels are most important for x-axes (e.g., months/quarters/categories).
    # For y/right_y, exposing text ticks tends to add OCR noise without helping downstream reasoning.
    ocr_words_text_eff = ocr_words_text if axis_s in ("x", "top_x") else []

    def _filter_reasonable(words: List[Dict[str, object]]) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for w in words:
            text = str(w.get("text", "") or "").strip()
            bbox = w.get("bbox_xyxy")
            if not text or not isinstance(bbox, tuple) or len(bbox) != 4:
                continue
            try:
                conf = float(w.get("conf", 0.0) or 0.0)
            except Exception:
                conf = 0.0
            fx1, fy1, fx2, fy2 = [int(v) for v in bbox]
            if fx2 <= fx1 or fy2 <= fy1:
                continue
            bw = fx2 - fx1
            bh = fy2 - fy1
            if bw <= 0 or bh <= 0:
                continue
            # Reject extremely large boxes (likely not tick labels).
            if bw > int(0.85 * W) or bh > int(0.60 * H):
                continue
            out.append({"text": text, "conf": float(conf), "bbox_xyxy": (fx1, fy1, fx2, fy2)})
        return out

    def _select_densest_band(words: List[Dict[str, object]]) -> List[Dict[str, object]]:
        # For x-axes: keep the densest horizontal band (by y-center).
        # For y-axes: keep the densest vertical band (by x-center).
        if len(words) < 3:
            return words
        use_y = axis_s in ("x", "top_x")
        centers: List[float] = []
        sizes: List[int] = []
        for w in words:
            bb = w.get("bbox_xyxy")
            if not isinstance(bb, tuple) or len(bb) != 4:
                continue
            x1_, y1_, x2_, y2_ = [int(v) for v in bb]
            if use_y:
                centers.append(0.5 * float(y1_ + y2_))
                sizes.append(max(1, int(y2_ - y1_)))
            else:
                centers.append(0.5 * float(x1_ + x2_))
                sizes.append(max(1, int(x2_ - x1_)))
        if len(centers) < 3 or not sizes:
            return words
        med_sz = float(np.median(np.asarray(sizes, dtype=np.float64)))
        tol = max(10.0, 1.8 * med_sz)
        centers_sorted = sorted(centers)
        best_i = 0
        best_j = 0
        j = 0
        for i in range(len(centers_sorted)):
            while j < len(centers_sorted) and centers_sorted[j] <= centers_sorted[i] + tol:
                j += 1
            if (j - i) > (best_j - best_i):
                best_i, best_j = i, j
        lo = centers_sorted[best_i]
        hi = centers_sorted[best_j - 1] if best_j > best_i else centers_sorted[best_i]

        band: List[Dict[str, object]] = []
        for w in words:
            bb = w.get("bbox_xyxy")
            if not isinstance(bb, tuple) or len(bb) != 4:
                continue
            x1_, y1_, x2_, y2_ = [int(v) for v in bb]
            c = 0.5 * float(y1_ + y2_) if use_y else 0.5 * float(x1_ + x2_)
            if (lo - 1e-6) <= c <= (hi + 1e-6):
                band.append(w)
        return band if len(band) >= 3 else words

    words_numeric = _select_densest_band(_filter_reasonable(ocr_words_numeric))
    words_text_all = _filter_reasonable(ocr_words_text_eff)

    # Collect numeric candidates (tick values).
    candidates: List[Dict[str, object]] = []
    for w in words_numeric:
        text = str(w.get("text", "") or "").strip()
        bb = w.get("bbox_xyxy")
        if not text or not isinstance(bb, tuple) or len(bb) != 4:
            continue
        val = _parse_axis_tick_value(text)
        if val is None:
            continue
        fx1, fy1, fx2, fy2 = [int(v) for v in bb]
        cx = int(round(0.5 * float(fx1 + fx2)))
        cy = int(round(0.5 * float(fy1 + fy2)))
        pos = cy if axis_s in ("y", "right_y") else cx
        candidates.append(
            {
                "text": text,
                "value": float(val),
                "pos": int(pos),
                "conf": float(w.get("conf", 0.0) or 0.0),
                "bbox_xyxy": (int(fx1), int(fy1), int(fx2), int(fy2)),
            }
        )

    def _looks_like_tick_text(text: str) -> bool:
        s = str(text or "").strip()
        if not s:
            return False
        if len(s) > 16:
            return False
        # Exclude multi-word legend-like strings; tick labels are usually short tokens.
        if any(ch.isspace() for ch in s):
            return False
        return True

    # Filter to tick-like text tokens first to avoid the legend (often multi-word) dominating the band selection.
    words_text_tickish = [
        w
        for w in words_text_all
        if _looks_like_tick_text(str(w.get("text", "") or "")) and float(w.get("conf", 0.0) or 0.0) >= 0.20
    ]
    words_text = _select_densest_band(words_text_tickish)

    # Collect text ticks (evidence for non-numeric tick labels).
    text_candidates: List[Dict[str, object]] = []
    for w in words_text:
        text = str(w.get("text", "") or "").strip()
        bb = w.get("bbox_xyxy")
        if not text or not isinstance(bb, tuple) or len(bb) != 4:
            continue
        fx1, fy1, fx2, fy2 = [int(v) for v in bb]
        cx = int(round(0.5 * float(fx1 + fx2)))
        cy = int(round(0.5 * float(fy1 + fy2)))
        pos = cy if axis_s in ("y", "right_y") else cx
        text_candidates.append(
            {
                "text": text,
                "pos": int(pos),
                "conf": float(w.get("conf", 0.0) or 0.0),
                "bbox_xyxy": (int(fx1), int(fy1), int(fx2), int(fy2)),
            }
        )

    # Deduplicate text ticks by proximity along the axis direction.
    tick_texts: List[str] = []
    tick_positions: List[int] = []
    tick_bboxes: List[Optional[BboxXyxy]] = []
    if text_candidates:
        sizes = []
        for c in text_candidates:
            bb = c.get("bbox_xyxy")
            if isinstance(bb, tuple) and len(bb) == 4:
                sizes.append(max(1, int(bb[3] - bb[1])) if axis_s in ("y", "right_y") else max(1, int(bb[2] - bb[0])))
        med_sz = int(np.median(np.asarray(sizes, dtype=np.int32))) if sizes else 12
        merge_dist = int(max(6, round(0.7 * float(med_sz))))
        text_sorted = sorted(text_candidates, key=lambda d: int(d.get("pos", 0)))
        merged_text: List[Dict[str, object]] = []
        for c in text_sorted:
            if not merged_text:
                merged_text.append(c)
                continue
            if abs(int(c.get("pos", 0)) - int(merged_text[-1].get("pos", 0))) <= merge_dist:
                c_conf = float(c.get("conf", 0.0) or 0.0)
                m_conf = float(merged_text[-1].get("conf", 0.0) or 0.0)
                if (c_conf, len(str(c.get("text", "")))) > (m_conf, len(str(merged_text[-1].get("text", "")))):
                    merged_text[-1] = c
            else:
                merged_text.append(c)
        for c in merged_text:
            tick_texts.append(str(c.get("text", "")))
            tick_positions.append(int(c.get("pos", 0)))
            bb = c.get("bbox_xyxy")
            if isinstance(bb, tuple) and len(bb) == 4:
                tick_bboxes.append((int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])))
            else:
                tick_bboxes.append(None)

    # If tickers are supplied, pick best matches per expected ticker.
    selected: List[Dict[str, object]]
    expected_texts: List[str] = []
    if axis_tickers is not None:
        try:
            expected_texts = [str(t).strip() for t in list(axis_tickers) if str(t).strip()]
        except Exception:
            expected_texts = []

    if expected_texts:
        used_idx: set[int] = set()
        picked: List[Dict[str, object]] = []
        for exp in expected_texts:
            exp_val = _parse_axis_tick_value(exp)
            best_i = None
            best_score = -1e9
            for i, cand in enumerate(candidates):
                if i in used_idx:
                    continue
                sim = _similarity(exp, str(cand.get("text", "") or ""))
                num_score = 0.0
                if exp_val is not None:
                    try:
                        dv = abs(float(cand.get("value")) - float(exp_val))
                    except Exception:
                        dv = 1e9
                    denom = max(1.0, abs(float(exp_val)))
                    # Allow some absolute slack for small tick values.
                    tol = 0.02 * denom + 0.5
                    num_score = max(0.0, 1.0 - (dv / float(max(1e-6, 4.0 * tol))))
                conf = float(cand.get("conf", 0.0) or 0.0)
                score = (0.65 * float(sim)) + (0.25 * float(num_score)) + (0.10 * float(conf))
                if score > best_score:
                    best_score = score
                    best_i = i
            if best_i is not None and best_score >= 0.45:
                used_idx.add(best_i)
                picked.append(candidates[best_i])
        selected = picked if len(picked) >= 2 else candidates
    else:
        selected = candidates

    # Deduplicate by proximity along the axis direction.
    sizes = []
    for c in selected:
        bb = c.get("bbox_xyxy")
        if isinstance(bb, tuple) and len(bb) == 4:
            sizes.append(max(1, int(bb[3] - bb[1])) if axis_s in ("y", "right_y") else max(1, int(bb[2] - bb[0])))
    med_sz = int(np.median(np.asarray(sizes, dtype=np.int32))) if sizes else 12
    merge_dist = int(max(6, round(0.7 * float(med_sz))))

    selected_sorted = sorted(selected, key=lambda d: int(d.get("pos", 0)))
    merged: List[Dict[str, object]] = []
    for c in selected_sorted:
        if not merged:
            merged.append(c)
            continue
        if abs(int(c.get("pos", 0)) - int(merged[-1].get("pos", 0))) <= merge_dist:
            # Keep the one with higher confidence, tie-break by longer text.
            c_conf = float(c.get("conf", 0.0) or 0.0)
            m_conf = float(merged[-1].get("conf", 0.0) or 0.0)
            if (c_conf, len(str(c.get("text", "")))) > (m_conf, len(str(merged[-1].get("text", "")))):
                merged[-1] = c
        else:
            merged.append(c)

    # If tickers are supplied and OCR missed a few, fill them via a linear fit (best-effort).
    # This improves robustness for dense axes where OCR may miss occasional labels.
    if expected_texts:
        exp_vals: List[float] = []
        for t in expected_texts:
            v = _parse_axis_tick_value(t)
            if v is None:
                continue
            exp_vals.append(float(v))
        if len(exp_vals) >= 2 and len(merged) >= 2:
            cur_vals = np.asarray([float(m.get("value")) for m in merged], dtype=np.float64)
            cur_pos = np.asarray([float(m.get("pos")) for m in merged], dtype=np.float64)
            # Fit pos = a*value + b
            try:
                a, b = np.polyfit(cur_vals, cur_pos, deg=1)
                pred = a * cur_vals + b
                resid = np.abs(pred - cur_pos)
                resid_med = float(np.median(resid)) if resid.size else 1e9
                pos_sorted = np.sort(cur_pos)
                spac = np.diff(pos_sorted)
                med_sp = float(np.median(spac[spac > 0])) if spac.size else 0.0
                tol = max(8.0, 0.35 * med_sp) if med_sp > 0 else 14.0
                if resid_med <= tol:
                    # Add missing values.
                    cur_set: List[float] = list(cur_vals.tolist())

                    def _has_value(v: float) -> bool:
                        for vv in cur_set:
                            if abs(float(vv) - float(v)) <= max(1e-6, 1e-3 * max(1.0, abs(float(v)))):
                                return True
                        return False

                    filled = 0
                    for v in exp_vals:
                        if _has_value(v):
                            continue
                        p = float(a * float(v) + float(b))
                        if axis_s == "x":
                            p = float(max(0.0, min(float(W - 1), p)))
                        else:
                            p = float(max(0.0, min(float(H - 1), p)))
                        merged.append(
                            {
                                "text": str(v),
                                "value": float(v),
                                "pos": int(round(p)),
                                "conf": 0.0,
                                "bbox_xyxy": None,
                            }
                        )
                        cur_set.append(float(v))
                        filled += 1
                    if filled:
                        merged = sorted(merged, key=lambda d: int(d.get("pos", 0)))
            except Exception:
                pass

    axis_values = [float(m.get("value")) for m in merged]
    axis_pixel_positions = [int(m.get("pos")) for m in merged]
    axis_bboxes: List[Optional[BboxXyxy]] = []
    for m in merged:
        bb = m.get("bbox_xyxy")
        if isinstance(bb, tuple) and len(bb) == 4:
            axis_bboxes.append((int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])))
        else:
            axis_bboxes.append(None)
    return axis_values, axis_pixel_positions, axis_bboxes, tick_texts, tick_positions, tick_bboxes


def interpolate_pixel_to_value(
    pixel: float,
    axis_values: List[float],
    axis_pixel_positions: List[int],
) -> float:
    """
    Maps a pixel coordinate to its corresponding axis value using linear interpolation between known axis ticks
    and their pixel positions.

    Args:
      pixel: Pixel coordinate to map.
      axis_values: Numeric axis values (e.g., [0, 200, 400, 600]).
      axis_pixel_positions: Pixel positions corresponding to axis_values (e.g., [950, 850, 750, 650]).

    Returns:
      Interpolated axis value corresponding to the given pixel.

    Notes:
      - The input ticks are sorted by pixel position internally.
      - If `pixel` lies outside the tick range, this function performs linear extrapolation using the nearest two ticks.
    """
    if axis_values is None or axis_pixel_positions is None:
        raise ValueError("axis_values and axis_pixel_positions are required")
    if len(axis_values) != len(axis_pixel_positions):
        raise ValueError("axis_values and axis_pixel_positions must have the same length")
    if len(axis_values) < 2:
        raise ValueError("Need at least 2 ticks to interpolate")

    try:
        px = float(pixel)
    except Exception as e:
        raise ValueError("pixel must be a number") from e

    pairs: List[Tuple[float, float]] = []
    for v, p in zip(axis_values, axis_pixel_positions):
        try:
            fv = float(v)
            fp = float(p)
        except Exception:
            continue
        if not (np.isfinite(fv) and np.isfinite(fp)):
            continue
        pairs.append((fp, fv))

    if len(pairs) < 2:
        raise ValueError("Not enough valid tick pairs after parsing inputs")

    pairs.sort(key=lambda t: t[0])

    # Deduplicate by pixel position (keep average value for identical positions).
    dedup: List[Tuple[float, float]] = []
    cur_pos: Optional[float] = None
    acc_val = 0.0
    acc_n = 0
    for pos, val in pairs:
        if cur_pos is None:
            cur_pos = pos
            acc_val = float(val)
            acc_n = 1
            continue
        if pos == cur_pos:
            acc_val += float(val)
            acc_n += 1
            continue
        dedup.append((cur_pos, acc_val / float(max(1, acc_n))))
        cur_pos = pos
        acc_val = float(val)
        acc_n = 1
    if cur_pos is not None:
        dedup.append((cur_pos, acc_val / float(max(1, acc_n))))

    if len(dedup) < 2:
        raise ValueError("Need at least 2 unique tick positions to interpolate")

    pos_list = [p for p, _ in dedup]
    val_list = [v for _, v in dedup]

    # Exact hit.
    i = bisect_left(pos_list, px)
    if i < len(pos_list) and pos_list[i] == px:
        return float(val_list[i])

    # Clamp to nearest segment endpoints for extrapolation.
    if i <= 0:
        i0, i1 = 0, 1
    elif i >= len(pos_list):
        i0, i1 = len(pos_list) - 2, len(pos_list) - 1
    else:
        i0, i1 = i - 1, i

    p0, v0 = float(pos_list[i0]), float(val_list[i0])
    p1, v1 = float(pos_list[i1]), float(val_list[i1])
    if p1 == p0:
        return float(v0)
    t = (px - p0) / (p1 - p0)
    return float(v0 + t * (v1 - v0))


def arithmetic(a: float, b: float, operation: str = "percentage") -> float:
    """
    Performs a specified arithmetic operation between two numeric inputs.

    Supported operations:
      - add
      - subtract (alias: sub)
      - multiply (alias: mul)
      - divide (alias: div)
      - percentage (a / b * 100)
      - ratio (a / b)

    Raises:
      ValueError: for unsupported operations or division by zero.
    """

    def _to_float(x: object, name: str) -> float:
        try:
            v = float(x)  # type: ignore[arg-type]
        except Exception as e:
            raise ValueError(f"{name} must be a number") from e
        if not math.isfinite(v):
            raise ValueError(f"{name} must be finite")
        return v

    av = _to_float(a, "a")
    bv = _to_float(b, "b")

    op = str(operation or "percentage").strip().lower()

    if op in ("add", "+", "sum"):
        return float(av + bv)
    if op in ("subtract", "sub", "-", "minus"):
        return float(av - bv)
    if op in ("multiply", "mul", "*", "times"):
        return float(av * bv)
    if op in ("divide", "div", "/"):
        if bv == 0.0:
            raise ValueError("division by zero")
        return float(av / bv)
    if op in ("percentage", "percent", "%"):
        if bv == 0.0:
            raise ValueError("division by zero")
        return float((av / bv) * 100.0)
    if op in ("ratio",):
        if bv == 0.0:
            raise ValueError("division by zero")
        return float(av / bv)

    raise ValueError(
        'Unsupported operation: {} (supported: "add", "subtract", "multiply", "divide", "percentage", "ratio")'.format(
            operation
        )
    )


def compute_segment_area(
    image: Image.Image,
    filter_rgb: Optional[RgbTuple],
    measure: str,
    masks: Optional[List],
    filter_segment: Optional[List],
) -> Tuple[Image.Image, int]:
    """
    Computes the area of a chart segment by:
      (1) counting discrete visual elements of a specified color (measure="discrete-dots"),
      (2) counting pixels of a specified color (measure="pixels"), or
      (3) counting pixels within a segment identified by mask label IDs (measure="pixels" + masks/filter_segment).

    Returns:
      - visualization: image with the measured region highlighted
      - area: int (pixels or discrete-dot count depending on `measure`)
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    measure_s = str(measure or "").strip().lower()
    if measure_s not in ("pixels", "discrete-dots"):
        raise ValueError('measure must be one of: "pixels", "discrete-dots"')

    img = image.convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    H, W = arr.shape[:2]
    if H <= 1 or W <= 1:
        raise ValueError("Invalid image size")

    # Resolve ROI mask from segmentation masks + filter_segment.
    roi_mask = np.ones((H, W), dtype=bool)
    selected_mask_count = 0

    def _mask_from_item(item: object) -> Optional[np.ndarray]:
        if isinstance(item, np.ndarray):
            m = item.astype(bool)
            return m if m.shape == (H, W) else None
        if isinstance(item, dict):
            seg = item.get("segmentation")
            if isinstance(seg, np.ndarray):
                m = seg.astype(bool)
                return m if m.shape == (H, W) else None
        return None

    def _id_from_item(item: object, idx1: int) -> int:
        if isinstance(item, dict):
            for k in ("id", "label", "mask_id"):
                if k in item:
                    try:
                        return int(item.get(k))  # type: ignore[arg-type]
                    except Exception:
                        break
        return int(idx1)

    if masks is not None:
        if not isinstance(masks, list):
            raise TypeError("masks must be a list or None")

        wanted: Optional[List[int]] = None
        if filter_segment is not None:
            if not isinstance(filter_segment, list):
                raise TypeError("filter_segment must be a list or None")
            wanted = []
            for v in filter_segment:
                try:
                    wanted.append(int(v))  # type: ignore[arg-type]
                except Exception:
                    continue

        roi = np.zeros((H, W), dtype=bool)
        for idx, item in enumerate(masks, start=1):
            mid = _id_from_item(item, idx)
            if wanted is not None and wanted and (mid not in wanted) and (idx not in wanted):
                continue
            m = _mask_from_item(item)
            if m is None:
                continue
            roi |= m
            selected_mask_count += 1

        if wanted is not None and wanted and selected_mask_count == 0:
            raise ValueError("filter_segment did not match any mask ids/indices")

        # If masks are supplied but none are valid, keep full-image ROI (best-effort).
        if selected_mask_count > 0:
            roi_mask = roi

    # Build a pixel selection mask by color or background-difference.
    bg = np.asarray(_estimate_background_rgb(img), dtype=np.int16)
    arr_i16 = arr.astype(np.int16)

    def _resolve_filter_rgb(v: Optional[RgbTuple]) -> Optional[np.ndarray]:
        if v is None:
            return None
        if isinstance(v, tuple) and len(v) == 3:
            try:
                r, g, b = [int(x) for x in v]
            except Exception as e:
                raise TypeError("filter_rgb must be a tuple[int,int,int] or None") from e
        elif isinstance(v, list) and len(v) == 3:
            try:
                r, g, b = [int(x) for x in v]
            except Exception as e:
                raise TypeError("filter_rgb must be a tuple[int,int,int] or None") from e
        else:
            raise TypeError("filter_rgb must be a tuple[int,int,int] or None")
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        return np.asarray([r, g, b], dtype=np.int16)

    rgb = _resolve_filter_rgb(filter_rgb)

    if rgb is None:
        # Default: select non-background pixels (best-effort "full chart" mask).
        diff = np.abs(arr_i16 - bg).sum(axis=2)
        color_mask = diff > 30
    else:
        # Color selection with tolerance for anti-aliasing.
        diff = np.abs(arr_i16 - rgb.reshape(1, 1, 3)).sum(axis=2)
        color_mask = diff <= 60

    target = np.logical_and(roi_mask, color_mask)

    # Visualization helper: overlay selected pixels.
    def _overlay(mask: np.ndarray) -> Image.Image:
        base = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
        if mask.shape != (H, W):
            return Image.fromarray(base)
        if int(mask.sum()) == 0:
            return Image.fromarray(base)
        overlay = base.copy()
        overlay[:, :, 0] = 255
        overlay[:, :, 1] = 0
        overlay[:, :, 2] = 0
        a = 0.35
        sel = mask.astype(bool)
        base[sel] = (np.round((1.0 - a) * base[sel] + a * overlay[sel])).astype(np.uint8)
        return Image.fromarray(base)

    if measure_s == "pixels":
        area = int(target.sum())
        return _overlay(target), int(area)

    # measure == "discrete-dots"
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError('compute_segment_area(measure="discrete-dots") requires opencv-python (cv2).') from e

    mask_u8 = (target.astype(np.uint8) * 255)
    if mask_u8.size == 0:
        return _overlay(target), 0

    # Light denoising: opening to remove single-pixel speckles without merging nearby dots.
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)

    n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n_labels <= 1:
        return _overlay(target), 0

    # Filter connected components to keep dot-like sizes (robust when most components are dots).
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int32)
    if areas.size == 0:
        return _overlay(target), 0
    med = float(np.median(areas))
    if med <= 0:
        return _overlay(target), 0

    min_a = max(5.0, 0.25 * med)
    max_a = max(min_a + 1.0, 3.0 * med)

    kept = []
    for lab in range(1, int(n_labels)):
        a0 = float(stats[lab, cv2.CC_STAT_AREA])
        if a0 < min_a or a0 > max_a:
            continue
        w0 = float(stats[lab, cv2.CC_STAT_WIDTH])
        h0 = float(stats[lab, cv2.CC_STAT_HEIGHT])
        if w0 <= 0 or h0 <= 0:
            continue
        ar = w0 / h0 if h0 != 0 else 999.0
        if ar < 0.35 or ar > 2.85:
            continue
        extent = a0 / float(max(1.0, w0 * h0))
        if extent < 0.15 or extent > 0.98:
            continue
        kept.append(lab)

    count = int(len(kept))

    # Visualization: overlay pixels + draw boxes for kept components.
    vis = np.asarray(_overlay(target).convert("RGB"), dtype=np.uint8).copy()
    for lab in kept:
        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        w0 = int(stats[lab, cv2.CC_STAT_WIDTH])
        h0 = int(stats[lab, cv2.CC_STAT_HEIGHT])
        x2 = int(min(W - 1, x + max(0, w0 - 1)))
        y2 = int(min(H - 1, y + max(0, h0 - 1)))
        cv2.rectangle(vis, (x, y), (x2, y2), (0, 255, 0), 1)

    return Image.fromarray(vis), count


def get_bar(
    image: Image.Image,
    rgb_of_interest: Optional[RgbTuple],
    ticker_label: Optional[str],
    segmentation_model: str = "SAM",
    bar_orientation: str = "vertical",
) -> BboxXyxy:
    """
    Detects and returns the bounding box of a bar in a chart image that matches a specified color and/or axis label.

    This implementation uses `segment_and_mark` (SAM1) to obtain cleaned masks, then filters for bar-like masks
    and selects the best match by:
      - proximity to the OCR-localized `ticker_label` (if provided), and/or
      - color similarity to `rgb_of_interest` (if provided).
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")

    if str(segmentation_model or "").upper() != "SAM":
        raise ValueError('segmentation_model must be "SAM" (SAM1 backend).')

    orient = str(bar_orientation or "").strip().lower()
    if orient not in ("vertical", "horizontal", "vertical-right"):
        raise ValueError('bar_orientation must be one of: "vertical", "horizontal", "vertical-right"')

    img = image.convert("RGB")
    W, H = img.size
    if W <= 1 or H <= 1:
        raise ValueError("Invalid image size")

    # 1) Segmentation masks (SAM1).
    try:
        _labeled, masks = segment_and_mark(img, segmentation_model="SAM")
    except Exception as e:
        raise RuntimeError(f"get_bar failed to segment the image with SAM1: {e}") from e

    if not masks:
        raise RuntimeError("get_bar found no segmentation masks")

    # 2) OCR localization for ticker_label (optional).
    label_bbox: Optional[BboxXyxy] = None
    label_anchor: Optional[Tuple[float, float]] = None

    def _ocr_lines(img_in: Image.Image) -> List[Dict[str, object]]:
        if _easyocr_available():
            # Upscale for better recall on small axis labels.
            w0, h0 = img_in.size
            scale = 2 if max(w0, h0) >= 900 else 3
            up = img_in.resize((max(2, w0 * scale), max(2, h0 * scale)), resample=Image.BICUBIC)
            lines = _easyocr_lines(up)
            # Downscale bboxes.
            for ln in lines:
                bb = ln.get("bbox_xyxy")
                if not isinstance(bb, tuple) or len(bb) != 4:
                    continue
                x1, y1, x2, y2 = bb
                ln["bbox_xyxy"] = (int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale))
            return lines
        if _tesseract_available():
            w0, h0 = img_in.size
            scale = 2
            up = img_in.resize((max(2, w0 * scale), max(2, h0 * scale)), resample=Image.BICUBIC)
            lines = _tesseract_lines(up, psm=6)
            for ln in lines:
                bb = ln.get("bbox_xyxy")
                if not isinstance(bb, tuple) or len(bb) != 4:
                    continue
                x1, y1, x2, y2 = bb
                ln["bbox_xyxy"] = (int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale))
            return lines
        raise RuntimeError(
            "get_bar requires an OCR backend when ticker_label is provided. "
            "Install EasyOCR (`pip install easyocr`) or install `tesseract` + `pytesseract`."
        )

    if isinstance(ticker_label, str) and ticker_label.strip():
        target = str(ticker_label).strip()
        if orient in ("vertical", "vertical-right"):
            roi = (0, int(round(0.62 * H)), W, H)
        else:
            roi = (0, 0, int(round(0.35 * W)), H)

        rx1, ry1, rx2, ry2 = _clip_bbox_xyxy(roi, W, H)
        roi_img = img.crop((rx1, ry1, rx2, ry2))
        lines = _ocr_lines(roi_img)
        bb0 = _find_best_line_bbox(lines, target, min_similarity=0.55)
        if bb0 is None:
            best = None
            best_score = -1.0
            for ln in lines:
                score = _similarity(target, str(ln.get("text", "") or ""))
                if score > best_score:
                    best_score = float(score)
                    best = str(ln.get("text", "") or "")
            raise RuntimeError(
                f'Failed to localize ticker_label="{target}" via OCR (best_score={best_score:.2f}, best_text="{best}").'
            )
        label_bbox = (bb0[0] + rx1, bb0[1] + ry1, bb0[2] + rx1, bb0[3] + ry1)
        cx = 0.5 * float(label_bbox[0] + label_bbox[2])
        cy = 0.5 * float(label_bbox[1] + label_bbox[3])
        label_anchor = (cx, cy)

    # 3) Candidate filtering: bar-like masks.
    arr = np.asarray(img, dtype=np.uint8)
    bg = np.asarray(_estimate_background_rgb(img), dtype=np.int16)
    def _rep_rgb_for_mask(seg: np.ndarray, bbox_xyxy: BboxXyxy) -> RgbTuple:
        x1, y1, x2, y2 = _clip_bbox_xyxy(bbox_xyxy, W, H)
        if x2 <= x1 or y2 <= y1:
            return (0, 0, 0)
        sub = arr[y1:y2, x1:x2, :]
        msub = seg[y1:y2, x1:x2].astype(bool)
        if sub.size == 0 or int(msub.sum()) == 0:
            return (0, 0, 0)
        pix = sub[msub]
        if pix.ndim != 2 or pix.shape[1] != 3:
            pix = pix.reshape(-1, 3)
        # Drop pixels close to background (helps when masks leak into whitespace).
        diff_bg = np.abs(pix.astype(np.int16) - bg.reshape(1, 3)).sum(axis=1)
        keep = diff_bg > 20
        if int(np.sum(keep)) >= 20:
            pix = pix[keep]
        if pix.shape[0] > 60000:
            # Deterministic downsample to keep tool output stable.
            step = int(max(1, pix.shape[0] // 60000))
            pix = pix[::step][:60000]
        q = (pix.astype(np.uint8) // 8) * 8
        vals, counts = np.unique(q, axis=0, return_counts=True)
        if vals.size == 0:
            med = np.median(pix, axis=0)
            return (int(med[0]), int(med[1]), int(med[2]))
        mode = vals[int(np.argmax(counts))]
        return (int(mode[0]), int(mode[1]), int(mode[2]))

    def _color_dist_l1(a: RgbTuple, b: RgbTuple) -> float:
        return float(abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])) + abs(int(a[2]) - int(b[2])))

    target_rgb: Optional[RgbTuple] = None
    if rgb_of_interest is not None:
        if isinstance(rgb_of_interest, tuple) and len(rgb_of_interest) == 3:
            target_rgb = tuple(int(max(0, min(255, int(x)))) for x in rgb_of_interest)  # type: ignore[misc]
        elif isinstance(rgb_of_interest, list) and len(rgb_of_interest) == 3:
            target_rgb = tuple(int(max(0, min(255, int(x)))) for x in rgb_of_interest)  # type: ignore[misc]
        else:
            raise TypeError("rgb_of_interest must be a tuple/list of 3 ints or None")

    candidates: List[Dict[str, object]] = []
    img_area = float(W * H)

    want_vertical = orient in ("vertical", "vertical-right")
    for m in masks:
        bb = m.get("bbox_xyxy")
        seg = m.get("segmentation")
        if not isinstance(bb, tuple) or len(bb) != 4 or not isinstance(seg, np.ndarray):
            continue
        x1, y1, x2, y2 = _clip_bbox_xyxy(bb, W, H)
        bw = int(x2 - x1)
        bh = int(y2 - y1)
        if bw <= 2 or bh <= 2:
            continue

        area = int(m.get("area", 0) or 0)
        if area <= 0:
            area = int(np.asarray(seg).astype(bool).sum())
        if area <= 0:
            continue
        if float(area) > 0.65 * img_area:
            continue

        box_area = float(bw * bh)
        fill = float(area) / float(max(1.0, box_area))
        if fill < 0.20:
            continue

        # Orientation filter (soft). Avoid selecting extremely thin axis/lines.
        major = float(bh) if want_vertical else float(bw)
        minor = float(bw) if want_vertical else float(bh)
        aspect = major / float(max(1.0, minor))
        if aspect < 0.15:
            continue

        # Drop axis-like lines (very long + very thin).
        minor_dim = float(min(bw, bh))
        major_dim = float(max(bw, bh))
        if minor_dim <= 3.0 and major_dim >= 0.60 * float(max(W, H)):
            continue

        # Extra guard for axis-like masks: very long and very thin.
        if aspect > 12.0 and fill < 0.60:
            continue

        cx = 0.5 * float(x1 + x2)
        cy = 0.5 * float(y1 + y2)

        cand: Dict[str, object] = {
            "bbox_xyxy": (int(x1), int(y1), int(x2), int(y2)),
            "segmentation": seg.astype(bool),
            "area": int(area),
            "fill": float(fill),
            "aspect": float(aspect),
            "center": (float(cx), float(cy)),
        }
        # Precompute representative color for later merging and/or color scoring.
        try:
            cand["rep_rgb"] = _rep_rgb_for_mask(cand["segmentation"], cand["bbox_xyxy"])  # type: ignore[arg-type]
        except Exception:
            cand["rep_rgb"] = (0, 0, 0)
        candidates.append(cand)

    if not candidates:
        raise RuntimeError("get_bar found no bar-like candidates from segmentation masks")

    # 4) Score + choose the best bar.
    dim = float(W if want_vertical else H)
    best = None
    best_score = -1e18

    for cand in candidates:
        bb = cand["bbox_xyxy"]
        fill = float(cand["fill"])
        aspect = float(cand["aspect"])
        cx, cy = cand["center"]

        # Shape score: prefer filled, elongated rectangles.
        aspect_score = min(3.0, max(0.0, aspect)) / 3.0
        shape_score = 0.65 * float(fill) + 0.35 * float(aspect_score)

        color_score = 0.0
        if target_rgb is not None:
            rep = cand.get("rep_rgb", (0, 0, 0))
            dist = _color_dist_l1(rep, target_rgb)
            # Normalize L1 distance to 0..1 (approx).
            color_score = 1.0 - min(1.0, dist / (3.0 * 255.0))

        label_score = 0.0
        if label_anchor is not None:
            ax, ay = label_anchor
            axis_dist = abs(cx - ax) if want_vertical else abs(cy - ay)
            # Within ~15% of the axis dimension is "close".
            label_score = 1.0 - min(1.0, axis_dist / float(max(1.0, 0.15 * dim)))
            # Penalize candidates that overlap the label region (common OCR/label artifacts).
            if label_bbox is not None:
                lx1, ly1, lx2, ly2 = label_bbox
                if want_vertical:
                    if int(bb[3]) > int(ly1):
                        label_score -= 0.25
                else:
                    if int(bb[0]) < int(lx2):
                        label_score -= 0.25

        if target_rgb is not None and label_anchor is not None:
            score = (0.10 * shape_score) + (0.55 * color_score) + (0.35 * label_score)
        elif target_rgb is not None:
            score = (0.20 * shape_score) + (0.80 * color_score)
        elif label_anchor is not None:
            score = (0.20 * shape_score) + (0.80 * label_score)
        else:
            score = float(shape_score)

        if score > best_score:
            best_score = float(score)
            best = cand

    if best is None:
        raise RuntimeError("get_bar failed to select a bar candidate")

    # 5) Merge fragmented masks that belong to the same bar (SAM often splits thin bars).
    # We start from the selected candidate and union nearby candidates that:
    #   - overlap strongly on the minor axis (same row/column), and
    #   - are adjacent/overlapping on the major axis (small gaps), and
    #   - have similar representative color.
    def _merge_bar_fragments(seed: Dict[str, object], all_cands: List[Dict[str, object]]) -> BboxXyxy:
        seed_bb = seed.get("bbox_xyxy")
        seed_seg = seed.get("segmentation")
        if not (isinstance(seed_bb, tuple) and len(seed_bb) == 4 and isinstance(seed_seg, np.ndarray)):
            return seed_bb if isinstance(seed_bb, tuple) and len(seed_bb) == 4 else (0, 0, 0, 0)

        merged_mask = np.asarray(seed_seg).astype(bool).copy()
        mx1, my1, mx2, my2 = [int(v) for v in seed_bb]
        seed_rgb = seed.get("rep_rgb", (0, 0, 0))
        if not (isinstance(seed_rgb, tuple) and len(seed_rgb) == 3):
            seed_rgb = (0, 0, 0)

        def _minor_overlap_ratio(a: BboxXyxy, b: BboxXyxy) -> float:
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            if want_vertical:
                overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
                denom = max(1, min(ax2 - ax1, bx2 - bx1))
            else:
                overlap = max(0, min(ay2, by2) - max(ay1, by1))
                denom = max(1, min(ay2 - ay1, by2 - by1))
            return float(overlap) / float(denom)

        def _major_gap_ok(a: BboxXyxy, b: BboxXyxy, max_gap: int) -> bool:
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            if want_vertical:
                # y-axis is major for vertical bars
                return not (by2 < (ay1 - max_gap) or by1 > (ay2 + max_gap))
            # x-axis is major for horizontal bars
            return not (bx2 < (ax1 - max_gap) or bx1 > (ax2 + max_gap))

        # Seed thickness guides allowed gaps.
        thickness = int(max(1, (mx2 - mx1) if want_vertical else (my2 - my1)))
        max_gap = int(max(2, round(2.5 * float(thickness))))
        color_tol = 80.0  # L1 distance threshold (robust to anti-aliasing + quantization)

        used_ids = set([id(seed)])
        changed = True
        while changed:
            changed = False
            merged_bb = (int(mx1), int(my1), int(mx2), int(my2))
            for cand in all_cands:
                if id(cand) in used_ids:
                    continue
                bb = cand.get("bbox_xyxy")
                seg = cand.get("segmentation")
                if not (isinstance(bb, tuple) and len(bb) == 4 and isinstance(seg, np.ndarray)):
                    continue
                bb_i = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
                if _minor_overlap_ratio(merged_bb, bb_i) < 0.60:
                    continue
                if not _major_gap_ok(merged_bb, bb_i, max_gap=max_gap):
                    continue
                rep = cand.get("rep_rgb", (0, 0, 0))
                if not (isinstance(rep, tuple) and len(rep) == 3):
                    rep = (0, 0, 0)
                if _color_dist_l1(rep, seed_rgb) > float(color_tol):
                    continue
                # Merge
                used_ids.add(id(cand))
                merged_mask |= np.asarray(seg).astype(bool)
                mx1 = min(mx1, int(bb_i[0]))
                my1 = min(my1, int(bb_i[1]))
                mx2 = max(mx2, int(bb_i[2]))
                my2 = max(my2, int(bb_i[3]))
                changed = True
        return _clip_bbox_xyxy((int(mx1), int(my1), int(mx2), int(my2)), W, H)

    merged_bb = _merge_bar_fragments(best, candidates)

    # 6) Final snap-to-color along the major axis: use the selected bar's representative color
    # to extend over small SAM gaps (gridlines/anti-aliasing can split bars).
    rep_rgb_best = best.get("rep_rgb", (0, 0, 0))
    if isinstance(rep_rgb_best, tuple) and len(rep_rgb_best) == 3:
        tol_l1 = 70  # tolerant to quantization; still avoids background/axes
        x1, y1, x2, y2 = [int(v) for v in merged_bb]
        if want_vertical:
            band_x1 = max(0, x1 - 1)
            band_x2 = min(W, x2 + 1)
            band = arr[:, band_x1:band_x2, :].astype(np.int16)
            rgb = np.asarray(rep_rgb_best, dtype=np.int16).reshape(1, 1, 3)
            diff = np.abs(band - rgb).sum(axis=2)
            close = diff <= int(tol_l1)
            ys = np.where(close.any(axis=1))[0]
            if ys.size >= 4:
                merged_bb = _clip_bbox_xyxy((x1, int(ys.min()), x2, int(ys.max() + 1)), W, H)
        else:
            band_y1 = max(0, y1 - 1)
            band_y2 = min(H, y2 + 1)
            band = arr[band_y1:band_y2, :, :].astype(np.int16)
            rgb = np.asarray(rep_rgb_best, dtype=np.int16).reshape(1, 1, 3)
            diff = np.abs(band - rgb).sum(axis=2)
            close = diff <= int(tol_l1)
            xs = np.where(close.any(axis=0))[0]
            if xs.size >= 6:
                merged_bb = _clip_bbox_xyxy((int(xs.min()), y1, int(xs.max() + 1), y2), W, H)

    return (int(merged_bb[0]), int(merged_bb[1]), int(merged_bb[2]), int(merged_bb[3]))


def compute_bar_height(
    image: Image.Image,
    bar_of_interest: Tuple[int, int, int, int],
    bar_orientation: str = "vertical",
    axis_threshold: float = 0.15,
    x_axis_tickers: Optional[List] = None,
    y_axis_tickers: Optional[List] = None,
    x_axis_title: Optional[str] = None,
    y_axis_title: Optional[str] = None,
) -> float:
    """
    Computes a bar’s value (height or length) by mapping its pixel bounding box to axis values using OCR-based
    axis localization.

    Notes:
      - This project primarily uses `bbox_xyxy` (x1,y1,x2,y2). However, the paper sometimes uses (x,y,w,h).
        This function accepts either format and picks the more plausible interpretation using the localized axis range.
      - `x_axis_title` / `y_axis_title` are accepted for API compatibility but are not used by the current OCR pipeline.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")

    orient = str(bar_orientation or "vertical").strip().lower()
    if orient not in ("vertical", "vertical-right", "horizontal"):
        raise ValueError('bar_orientation must be one of: "vertical", "vertical-right", "horizontal"')

    img = image.convert("RGB")
    W, H = img.size

    # 1) Axis mapping.
    axis = "x" if orient == "horizontal" else ("right_y" if orient == "vertical-right" else "y")
    tickers = x_axis_tickers if axis == "x" else y_axis_tickers
    axis_values, axis_pixel_positions = axis_localizer(
        img,
        axis=axis,
        axis_threshold=float(axis_threshold),
        axis_tickers=tickers,
    )
    if not axis_values or not axis_pixel_positions or len(axis_values) != len(axis_pixel_positions):
        raise RuntimeError("axis_localizer returned an invalid mapping")

    vmin = float(min(axis_values))
    vmax = float(max(axis_values))
    vrange = float(max(1e-9, vmax - vmin))

    # 2) Normalize bar bbox: try both xyxy and xywh interpretations; choose by plausibility.
    if not (isinstance(bar_of_interest, (list, tuple)) and len(bar_of_interest) == 4):
        raise TypeError("bar_of_interest must be a tuple[int,int,int,int] (xyxy or xywh)")

    try:
        bx0, by0, bx2_raw, by2_raw = [float(x) for x in bar_of_interest]
    except Exception as e:
        raise TypeError("bar_of_interest must contain 4 numeric values") from e

    def _clip_xyxy(bb: Tuple[float, float, float, float]) -> BboxXyxy:
        x1, y1, x2, y2 = bb
        return _clip_bbox_xyxy((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))), W, H)

    cand_xyxy = _clip_xyxy((bx0, by0, bx2_raw, by2_raw))
    cand_xywh = _clip_xyxy((bx0, by0, bx0 + bx2_raw, by0 + by2_raw))

    want_vertical = orient in ("vertical", "vertical-right")

    def _score_bbox(bb: BboxXyxy) -> Tuple[float, float]:
        x1, y1, x2, y2 = bb
        bw = float(max(1, x2 - x1))
        bh = float(max(1, y2 - y1))
        major = bh if want_vertical else bw
        minor = bw if want_vertical else bh
        aspect = major / float(max(1.0, minor))

        # Compute a value estimate (difference along the axis direction).
        if want_vertical:
            p0, p1 = float(y2), float(y1)
        else:
            p0, p1 = float(x1), float(x2)
        v0 = float(interpolate_pixel_to_value(p0, axis_values, axis_pixel_positions))
        v1 = float(interpolate_pixel_to_value(p1, axis_values, axis_pixel_positions))
        delta = float(v1 - v0)

        # Plausibility: prefer deltas within the axis range (allow slack).
        delta_abs = abs(delta)
        range_score = 1.0 - min(1.0, delta_abs / float(max(1e-9, 1.5 * vrange)))

        # Prefer elongated masks (bars are usually elongated along their value axis).
        aspect_score = min(4.0, max(0.0, aspect)) / 4.0

        # Penalize huge bboxes.
        area_frac = (bw * bh) / float(max(1.0, float(W * H)))
        area_pen = 0.0
        if area_frac > 0.55:
            area_pen = min(1.0, (area_frac - 0.55) / 0.35)

        score = (0.55 * range_score) + (0.45 * aspect_score) - (0.85 * area_pen)
        return float(score), float(delta)

    s_xyxy, d_xyxy = _score_bbox(cand_xyxy)
    s_xywh, d_xywh = _score_bbox(cand_xywh)

    chosen = cand_xyxy if s_xyxy >= s_xywh else cand_xywh
    delta = d_xyxy if chosen is cand_xyxy else d_xywh

    # Return magnitude as "height/length".
    return float(abs(delta))


def bar_value_consistency(
    image: Image.Image,
    bar_orientation: str = "vertical",
    *,
    debug_dir: Optional[str] = None,
) -> Tuple[Image.Image, Dict[str, object]]:
    """
    Check whether *annotated* bar values (numbers printed near bars) are consistent with bar sizes.

    This is a lightweight misrepresentation signal for bar charts when axis ticks are missing/unreliable.

    Returns:
      - preview: image with detected value-label bboxes + matched bar bboxes
      - result: dict with fields like:
          {
            "n_value_labels": int,
            "n_pairs": int,
            "pairs": [
              {"value": int, "value_text": str, "value_bbox_xyxy": [..], "bar_bbox_xyxy": [..], "bar_height_px": int}
            ],
            "kendall_tau": float | None,
            "discordant_ratio": float | None,
            "is_mismatch": bool,
          }
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")

    orient = str(bar_orientation or "vertical").strip().lower()
    if orient not in ("vertical", "vertical-right", "horizontal"):
        raise ValueError('bar_orientation must be one of: "vertical", "vertical-right", "horizontal"')

    img = image.convert("RGB")
    W, H = img.size
    if W <= 1 or H <= 1:
        raise ValueError("Invalid image size")

    arr = np.asarray(img, dtype=np.uint8)

    # ---- 1) OCR: find numeric value labels in a broad lower ROI ----
    # For annotated bars, value labels are typically near/above bars, not in the title region.
    if orient in ("vertical", "vertical-right"):
        roi = (0, int(round(0.25 * H)), W, int(round(0.95 * H)))
    else:
        roi = (int(round(0.05 * W)), int(round(0.05 * H)), W, int(round(0.95 * H)))
    rx1, ry1, rx2, ry2 = _clip_bbox_xyxy(roi, W, H)
    roi_img = img.crop((rx1, ry1, rx2, ry2))

    # A coarse background color from the OCR ROI (used to filter obvious background pixels).
    roi_arr = arr[ry1:ry2, rx1:rx2, :]
    if roi_arr.size == 0:
        roi_arr = arr
    sample = roi_arr[
        :: max(1, int(round(min(roi_arr.shape[0], roi_arr.shape[1]) / 120.0))),
        :: max(1, int(round(min(roi_arr.shape[0], roi_arr.shape[1]) / 120.0))),
        :,
    ]
    sample = sample.reshape(-1, 3).astype(np.uint8)
    q = (sample // 16) * 16
    vals, counts = np.unique(q, axis=0, return_counts=True)
    bg = vals[int(np.argmax(counts))].astype(np.int16) if vals.size else np.array([255, 255, 255], dtype=np.int16)

    def _l1_rgb_u8(rgb: np.ndarray, ref: np.ndarray) -> int:
        return int(np.abs(rgb.astype(np.int16) - ref.astype(np.int16)).sum())

    def _is_dark(rgb: np.ndarray) -> bool:
        try:
            return int(rgb[0]) + int(rgb[1]) + int(rgb[2]) < 60
        except Exception:
            return False

    def _ocr_words_roi(im: Image.Image) -> List[Dict[str, object]]:
        w0, h0 = im.size
        scale = 3 if max(w0, h0) <= 1300 else 2
        up = im.resize((max(2, w0 * scale), max(2, h0 * scale)), resample=Image.BICUBIC)
        out: List[Dict[str, object]] = []
        if _easyocr_available():
            reader = _get_easyocr_reader()
            det = reader.readtext(np.asarray(up.convert("RGB")), detail=1, paragraph=False)
            for item in det:
                if not isinstance(item, (list, tuple)) or len(item) < 3:
                    continue
                quad, text, conf = item[0], item[1], item[2]
                text_s = str(text or "").strip()
                if not text_s:
                    continue
                try:
                    conf_f = float(conf)
                except Exception:
                    conf_f = 0.0
                xs: List[float] = []
                ys: List[float] = []
                try:
                    for p in quad:
                        xs.append(float(p[0]))
                        ys.append(float(p[1]))
                except Exception:
                    continue
                if not xs or not ys:
                    continue
                x1 = int(max(0, min(xs) / float(scale)))
                y1 = int(max(0, min(ys) / float(scale)))
                x2 = int(max(xs) / float(scale))
                y2 = int(max(ys) / float(scale))
                if x2 <= x1 or y2 <= y1:
                    continue
                out.append({"text": text_s, "conf": conf_f, "bbox_xyxy": (x1, y1, x2, y2)})
            return out
        if _tesseract_available():
            det = _tesseract_words(up, psm=6)
            for w in det:
                bb = w.get("bbox_xyxy")
                if not isinstance(bb, tuple) or len(bb) != 4:
                    continue
                x1, y1, x2, y2 = [int(v / float(scale)) for v in bb]
                text_s = str(w.get("text", "") or "").strip()
                if not text_s:
                    continue
                try:
                    conf_f = float(w.get("conf", 0.0) or 0.0) / 100.0
                except Exception:
                    conf_f = 0.0
                out.append({"text": text_s, "conf": conf_f, "bbox_xyxy": (x1, y1, x2, y2)})
            return out
        raise RuntimeError(
            "bar_value_consistency requires an OCR backend. Install EasyOCR (`pip install easyocr`) "
            "or install `tesseract` + `pytesseract`."
        )

    words = _ocr_words_roi(roi_img)

    # Numeric label candidates: allow decimals and percents (and filter out axis-tick-like labels later).
    def _parse_numeric_label(text: str) -> Optional[Tuple[float, str]]:
        import math

        t = str(text or "").strip()
        if not t:
            return None
        # Reject common date-like tokens.
        if "/" in t or ":" in t:
            return None
        if re.search(r"[A-Za-z]", t):
            return None
        t = t.replace(" ", "")
        is_percent = False
        if t.endswith("%"):
            is_percent = True
            t = t[:-1]
        t = t.strip()
        if not t:
            return None
        if not re.fullmatch(
            r"[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?",
            t,
        ):
            return None
        t2 = t.replace(",", "")
        digits = re.sub(r"[^0-9]", "", t2)
        # Avoid pulling in single-digit ticks like "0" or "5" unless it's a decimal (e.g., "0.5").
        if len(digits) < 2 and "." not in t2:
            return None
        try:
            v = float(t2)
        except Exception:
            return None
        if not math.isfinite(v):
            return None
        if abs(float(v)) > 1e9:
            return None
        return float(v), ("percent" if is_percent else "number")

    def _nonbg_ratio(x1: int, y1: int, x2: int, y2: int, *, bg_exclude_l1: int = 45) -> float:
        x1c, y1c, x2c, y2c = _clip_bbox_xyxy((x1, y1, x2, y2), W, H)
        if x2c <= x1c or y2c <= y1c:
            return 0.0
        region = arr[y1c:y2c, x1c:x2c, :]
        if region.size == 0:
            return 0.0
        pix = region.astype(np.int16)
        diff_bg = np.abs(pix - bg.reshape(1, 1, 3)).sum(axis=2)
        dark = (pix[:, :, 0] + pix[:, :, 1] + pix[:, :, 2]) < 60
        m = (diff_bg >= int(bg_exclude_l1)) & (~dark)
        try:
            return float(m.mean())
        except Exception:
            return 0.0

    def _looks_like_value_label(label_bb: BboxXyxy) -> bool:
        lx1, ly1, lx2, ly2 = [int(v) for v in label_bb]
        lw = max(1, lx2 - lx1)
        lh = max(1, ly2 - ly1)
        pad = 2
        if orient in ("vertical", "vertical-right"):
            win = int(max(12, min(80, round(2.8 * float(lh) + 10))))
            # Value labels for vertical bars are typically placed on/above the bar, so the bar pixels should
            # appear immediately below the label. Using "above" as a fallback tends to admit x-axis tick labels.
            below = _nonbg_ratio(lx1 - 2, ly2 + pad, lx2 + 2, ly2 + pad + win)
            return bool(below >= 0.08)
        win = int(max(12, min(120, round(2.8 * float(lw) + 16))))
        left = _nonbg_ratio(lx1 - pad - win, ly1 - 2, lx1 - pad, ly2 + 2)
        right = _nonbg_ratio(lx2 + pad, ly1 - 2, lx2 + pad + win, ly2 + 2)
        return bool(max(left, right) >= 0.08)

    cand_labels: List[Dict[str, object]] = []
    for w in words:
        text_s = str(w.get("text", "") or "").strip()
        bb = w.get("bbox_xyxy")
        if not isinstance(bb, tuple) or len(bb) != 4:
            continue
        parsed = _parse_numeric_label(text_s)
        if parsed is None:
            continue
        v, v_kind = parsed
        try:
            conf_f = float(w.get("conf", 0.0) or 0.0)
        except Exception:
            conf_f = 0.0
        x1, y1, x2, y2 = [int(vv) for vv in bb]
        x1, y1, x2, y2 = _clip_bbox_xyxy((x1 + rx1, y1 + ry1, x2 + rx1, y2 + ry1), W, H)
        cand_labels.append(
            {
                "value": float(v),
                "value_kind": str(v_kind),
                "text": text_s,
                "conf": float(conf_f),
                "bbox_xyxy": (x1, y1, x2, y2),
            }
        )

    # Sort by confidence and de-duplicate near-identical boxes.
    cand_labels.sort(key=lambda d: float(d.get("conf", 0.0) or 0.0), reverse=True)

    def _iou(a: BboxXyxy, b: BboxXyxy) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = float(iw * ih)
        if inter <= 0:
            return 0.0
        area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
        area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
        denom = max(1e-9, area_a + area_b - inter)
        return float(inter / denom)

    dedup: List[Dict[str, object]] = []
    for c in cand_labels:
        bb = c.get("bbox_xyxy")
        if not isinstance(bb, tuple) or len(bb) != 4:
            continue
        if any(_iou(bb, d.get("bbox_xyxy")) > 0.65 for d in dedup if isinstance(d.get("bbox_xyxy"), tuple)):
            continue
        dedup.append(c)
        if len(dedup) >= 24:
            break

    # Filter out axis ticks / legends by requiring nearby bar-colored pixels.
    dedup = [c for c in dedup if isinstance(c.get("bbox_xyxy"), tuple) and _looks_like_value_label(c["bbox_xyxy"])]

    # ---- 2) Bar localization under each value label (OCR-only, pixel-based) ----
    # We intentionally do NOT rely on axis detection here; this is for annotated bar charts where
    # axes may be missing/unreliable.

    import cv2  # type: ignore

    def _pick_seed_color(band: np.ndarray, *, bg_exclude_l1: int = 60) -> Optional[np.ndarray]:
        if band.size == 0:
            return None
        # Sample to reduce cost.
        step_y = max(1, int(round(band.shape[0] / 36.0)))
        step_x = max(1, int(round(band.shape[1] / 72.0)))
        samp = band[::step_y, ::step_x, :].reshape(-1, 3).astype(np.uint8)
        if samp.size == 0:
            return None
        # Filter obvious background + very dark pixels.
        keep: List[np.ndarray] = []
        for pix in samp:
            if _is_dark(pix):
                continue
            if _l1_rgb_u8(pix, bg) <= int(bg_exclude_l1):
                continue
            keep.append(pix)
        if not keep:
            return None
        keep_arr = np.stack(keep, axis=0)
        qq = (keep_arr // 16) * 16
        v2, c2 = np.unique(qq, axis=0, return_counts=True)
        if v2.size == 0:
            return None
        # Require the seed color to have enough support; avoids picking thin gridlines.
        best_i = int(np.argmax(c2))
        if int(c2[best_i]) < max(6, int(round(0.01 * float(len(keep))))):
            return None
        return v2[best_i].astype(np.int16)

    def _mask_close_to_seed(region: np.ndarray, seed: np.ndarray, *, tol_l1: int = 70, bg_exclude_l1: int = 45) -> np.ndarray:
        if region.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)
        pix = region.astype(np.int16)
        diff_seed = np.abs(pix - seed.reshape(1, 1, 3)).sum(axis=2)
        diff_bg = np.abs(pix - bg.reshape(1, 1, 3)).sum(axis=2)
        m = (diff_seed <= int(tol_l1)) & (diff_bg >= int(bg_exclude_l1))
        # Exclude very dark pixels (black border / compression noise).
        if region.dtype == np.uint8:
            dark = (region[:, :, 0].astype(np.int16) + region[:, :, 1].astype(np.int16) + region[:, :, 2].astype(np.int16)) < 60
            m = m & (~dark)
        out = m.astype(np.uint8) * 255
        # Fill small holes; keep bars contiguous.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)
        return out

    def _best_component_bbox(mask_u8: np.ndarray, *, x0: int, y0: int, label_cx: float, min_w: int, min_h: int) -> Optional[BboxXyxy]:
        if mask_u8.size == 0:
            return None
        n, _, stats, cent = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if n <= 1:
            return None
        roi_w = float(mask_u8.shape[1])
        best = None
        best_score = -1e18
        for i in range(1, int(n)):
            x, y, w, h, area = [int(v) for v in stats[i]]
            if w < int(min_w) or h < int(min_h) or area < 30:
                continue
            fill = float(area) / float(max(1, w * h))
            if fill < 0.35:
                continue
            cx = float(x0) + float(cent[i][0])
            # Prefer components centered under the label.
            align_pen = abs(cx - float(label_cx)) / float(max(1.0, roi_w))
            score = (float(area) * (0.6 + fill)) + (45.0 * float(h)) - (1800.0 * align_pen)
            # Avoid selecting full-width *thin* strips (often gridlines), but keep actual bars.
            thin_h = max(3, int(round(0.08 * float(mask_u8.shape[0]))))
            if float(w) >= 0.92 * roi_w and int(h) <= thin_h and fill < 0.55:
                continue
            if score > best_score:
                best_score = score
                best = (int(x0 + x), int(y0 + y), int(x0 + x + w), int(y0 + y + h))
        if best is None:
            return None
        return _clip_bbox_xyxy(best, W, H)

    def _best_component_bbox_horizontal(
        mask_u8: np.ndarray, *, x0: int, y0: int, label_cy: float, min_w: int, min_h: int
    ) -> Optional[BboxXyxy]:
        if mask_u8.size == 0:
            return None
        n, _, stats, cent = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if n <= 1:
            return None
        roi_h = float(mask_u8.shape[0])
        best = None
        best_score = -1e18
        for i in range(1, int(n)):
            x, y, w, h, area = [int(v) for v in stats[i]]
            if w < int(min_w) or h < int(min_h) or area < 30:
                continue
            fill = float(area) / float(max(1, w * h))
            if fill < 0.35:
                continue
            cy = float(y0) + float(cent[i][1])
            align_pen = abs(cy - float(label_cy)) / float(max(1.0, roi_h))
            if align_pen > 0.35:
                continue
            score = (float(area) * (0.6 + fill)) + (45.0 * float(w)) - (1800.0 * align_pen)
            # Avoid selecting full-height *thin* strips (often axis/gridlines), but keep bars.
            thin_w = max(3, int(round(0.08 * float(mask_u8.shape[1]))))
            if float(h) >= 0.92 * roi_h and int(w) <= thin_w and fill < 0.55:
                continue
            if score > best_score:
                best_score = score
                best = (int(x0 + x), int(y0 + y), int(x0 + x + w), int(y0 + y + h))
        if best is None:
            return None
        return _clip_bbox_xyxy(best, W, H)

    debug_labels: List[Dict[str, object]] = []
    raw_pairs: List[Dict[str, object]] = []

    for c in dedup:
        bb = c.get("bbox_xyxy")
        if not isinstance(bb, tuple) or len(bb) != 4:
            continue
        lx1, ly1, lx2, ly2 = [int(v) for v in bb]
        lw = max(1, lx2 - lx1)
        lh = max(1, ly2 - ly1)
        cx = 0.5 * float(lx1 + lx2)

        bar_bb: Optional[BboxXyxy] = None
        seed: Optional[np.ndarray] = None

        if orient in ("vertical", "vertical-right"):
            half_w = int(max(28, min(110, round(1.9 * float(lw) + 18))))
            x0 = max(0, int(round(cx - float(half_w))))
            x1 = min(W, int(round(cx + float(half_w))))
            y0 = min(H - 1, max(0, int(ly2 + 1)))
            y1 = min(H, max(y0 + 2, int(round(0.96 * float(H)))))

            band_h = int(max(18, min(70, round(3.5 * float(lh)))))
            band = arr[y0 : min(H, y0 + band_h), x0:x1, :]
            seed = _pick_seed_color(band)
            if seed is not None:
                m = _mask_close_to_seed(arr[y0:y1, x0:x1, :], seed, tol_l1=75, bg_exclude_l1=45)
                bar_bb = _best_component_bbox(m, x0=x0, y0=y0, label_cx=cx, min_w=6, min_h=3)
        else:
            # Horizontal bars: label is typically to the left/right of the bar.
            half_h = int(max(14, min(60, round(1.2 * float(lh) + 8))))
            y0 = max(0, int(round(0.5 * float(ly1 + ly2) - float(half_h))))
            y1 = min(H, int(round(0.5 * float(ly1 + ly2) + float(half_h))))
            x1 = min(W - 1, max(0, int(lx1 - 1)))
            x0 = max(0, int(round(0.05 * float(W))))
            band = arr[y0:y1, x0 : min(W, x0 + int(round(0.25 * float(W)))), :]
            seed = _pick_seed_color(band)
            if seed is not None:
                m = _mask_close_to_seed(arr[y0:y1, x0:x1, :], seed, tol_l1=75, bg_exclude_l1=45)
                cy_label = 0.5 * float(ly1 + ly2)
                bar_bb = _best_component_bbox_horizontal(m, x0=x0, y0=y0, label_cy=cy_label, min_w=6, min_h=3)

        v = float(c.get("value", 0.0) or 0.0)
        text = str(c.get("text", "") or "")
        v_kind = str(c.get("value_kind", "number") or "number")
        height_px = 0
        bar_bb_list: Optional[List[int]] = None
        if bar_bb is not None:
            if orient in ("vertical", "vertical-right"):
                height_px = int(max(0, bar_bb[3] - bar_bb[1]))
            else:
                height_px = int(max(0, bar_bb[2] - bar_bb[0]))
            bar_bb_list = [int(bar_bb[0]), int(bar_bb[1]), int(bar_bb[2]), int(bar_bb[3])]

        raw_pairs.append(
            {
                "value": float(v),
                "value_kind": v_kind,
                "value_text": text,
                "value_bbox_xyxy": [int(lx1), int(ly1), int(lx2), int(ly2)],
                "bar_bbox_xyxy": bar_bb_list,
                "bar_height_px": int(height_px),
                "bar_found": bool(bar_bb is not None),
                "seed_rgb": seed.tolist() if seed is not None else None,
            }
        )
        debug_labels.append(
            {
                "value": float(v),
                "value_kind": v_kind,
                "value_text": text,
                "value_bbox_xyxy": [int(lx1), int(ly1), int(lx2), int(ly2)],
                "bar_bbox_xyxy": bar_bb_list,
                "bar_height_px": int(height_px),
                "bar_found": bool(bar_bb is not None),
                "seed_rgb": seed.tolist() if seed is not None else None,
            }
        )

    # Keep only successful matches for scoring/preview.
    pairs = [p for p in raw_pairs if bool(p.get("bar_found")) and isinstance(p.get("bar_bbox_xyxy"), list)]

    # ---- 3) Consistency score (order agreement), per-series ----
    def _score(values_in: List[float], lengths_in: List[float]) -> Tuple[Optional[float], Optional[float], bool]:
        concordant0 = 0
        discordant0 = 0
        for i in range(len(values_in)):
            for j in range(i + 1, len(values_in)):
                dv = float(values_in[i] - values_in[j])
                dh = float(lengths_in[i] - lengths_in[j])
                if dv == 0.0 or dh == 0.0:
                    continue
                if dv * dh > 0.0:
                    concordant0 += 1
                else:
                    discordant0 += 1
        denom0 = max(1, concordant0 + discordant0)
        tau0 = float(concordant0 - discordant0) / float(denom0) if denom0 > 0 else 0.0
        disc0 = float(discordant0) / float(denom0) if denom0 > 0 else 0.0

        mismatch0 = False
        n0 = int(len(values_in))
        if n0 >= 4:
            mismatch0 = bool(disc0 >= 0.25 or tau0 <= 0.4)
        elif n0 == 3:
            # With 3 points, a single inversion is common with OCR noise; require stronger disagreement.
            mismatch0 = bool(disc0 >= 0.5 or tau0 <= 0.0)
        elif n0 == 2:
            dv = abs(float(values_in[0]) - float(values_in[1]))
            dh = abs(float(lengths_in[0]) - float(lengths_in[1]))
            if values_in[0] != values_in[1] and lengths_in[0] != lengths_in[1]:
                flipped = (values_in[0] - values_in[1]) * (lengths_in[0] - lengths_in[1]) < 0.0
                v_rel = dv / float(max(1.0, max(values_in)))
                h_rel = dh / float(max(1.0, max(lengths_in)))
                if flipped and v_rel >= 0.04 and h_rel >= 0.25:
                    mismatch0 = True
        if n0 < 2:
            return None, None, False
        return float(tau0), float(disc0), bool(mismatch0)

    groups: Dict[Tuple[Optional[Tuple[int, int, int]], str], List[Dict[str, object]]] = {}
    for p in pairs:
        seed = p.get("seed_rgb")
        seed_t: Optional[Tuple[int, int, int]] = None
        if isinstance(seed, list) and len(seed) == 3:
            try:
                seed_t = (int(seed[0]), int(seed[1]), int(seed[2]))
            except Exception:
                seed_t = None
        kind = str(p.get("value_kind", "number") or "number")
        groups.setdefault((seed_t, kind), []).append(p)

    group_stats: List[Dict[str, object]] = []
    warning: Optional[str] = None
    is_mismatch = False
    best_tau: Optional[float] = None
    best_disc: Optional[float] = None
    best_n = 0

    if len(groups) >= 2:
        warning = "Multiple series detected; bar-value consistency is evaluated per series to avoid mixed-scale false positives."

    for (seed_t, kind), plist in groups.items():
        vals_in: List[float] = []
        lens_in: List[float] = []
        for p in plist:
            try:
                vals_in.append(float(p.get("value", 0.0) or 0.0))
            except Exception:
                vals_in.append(0.0)
            try:
                lens_in.append(float(p.get("bar_height_px", 0.0) or 0.0))
            except Exception:
                lens_in.append(0.0)
        tau0, disc0, mismatch0 = _score(vals_in, lens_in)

        # If mismatch is driven by a single OCR outlier, don't over-trigger misrepresentation.
        if mismatch0 and len(plist) >= 4:
            for drop in range(len(plist)):
                vals2 = [v for i, v in enumerate(vals_in) if i != drop]
                lens2 = [l for i, l in enumerate(lens_in) if i != drop]
                tau2, disc2, mismatch2 = _score(vals2, lens2)
                if not mismatch2:
                    mismatch0 = False
                    if warning:
                        warning = warning + " One point looked like an OCR outlier and was ignored for mismatch decision."
                    else:
                        warning = "One point looked like an OCR outlier and was ignored for mismatch decision."
                    break

        group_stats.append(
            {
                "seed_rgb": list(seed_t) if seed_t is not None else None,
                "value_kind": kind,
                "n_pairs": int(len(plist)),
                "kendall_tau": tau0,
                "discordant_ratio": disc0,
                "is_mismatch": bool(mismatch0),
            }
        )
        is_mismatch = bool(is_mismatch or mismatch0)

        if int(len(plist)) > best_n:
            best_n = int(len(plist))
            best_tau = tau0
            best_disc = disc0

    # ---- 4) Preview ----
    preview = img.copy()
    draw = ImageDraw.Draw(preview)
    for idx, p in enumerate(pairs, start=1):
        vb = p.get("value_bbox_xyxy")
        bb = p.get("bar_bbox_xyxy")
        if isinstance(vb, list) and len(vb) == 4:
            draw.rectangle([int(vb[0]), int(vb[1]), int(vb[2]), int(vb[3])], outline=(0, 255, 0), width=2)
        if isinstance(bb, list) and len(bb) == 4:
            draw.rectangle([int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])], outline=(255, 0, 0), width=2)
            label = "{}:{}".format(idx, str(p.get("value_text", "") or p.get("value", "")))
            draw.text((int(bb[0]), max(0, int(bb[1]) - 12)), label, fill=(255, 255, 0))

    if isinstance(debug_dir, str) and debug_dir.strip():
        os.makedirs(debug_dir, exist_ok=True)
        preview_path = os.path.join(debug_dir, "bar_value_consistency_preview.png")
        preview.save(preview_path)
        try:
            with open(os.path.join(debug_dir, "bar_value_consistency_debug.json"), "w") as f:
                json.dump({"labels": debug_labels}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    result: Dict[str, object] = {
        "n_value_labels": int(len(dedup)),
        "n_pairs": int(len(pairs)),
        "pairs": pairs,
        "kendall_tau": float(best_tau) if best_tau is not None else None,
        "discordant_ratio": float(best_disc) if best_disc is not None else None,
        "is_mismatch": bool(is_mismatch),
        "groups": group_stats,
        "warning": warning,
    }
    return preview, result


def get_boxplot(
    image: Image.Image,
    masks: List[Dict[str, object]],
    rgb_of_interest: Optional[RgbTuple] = None,
    ticker_label: Optional[str] = None,
    box_labels_of_interest: Optional[List[int]] = None,
    boxplot_orientation: str = "vertical",
    axis_threshold: float = 0.15,
) -> List[BboxXyxy]:
    """
    Detects and returns boxplot segments filtered by color, axis label, or segmentation indices.

    This tool is designed to be called after `segment_and_mark` and therefore expects `masks` in the
    same format (each item containing at least `id`, `bbox_xyxy`, and ideally `segmentation`).

    The returned list contains bounding boxes (xyxy) for the selected boxplot components, typically including:
      - IQR box
      - whiskers / caps
      - median line
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    if not isinstance(masks, list):
        raise TypeError("masks must be a list of mask dicts")

    orient = str(boxplot_orientation or "vertical").strip().lower()
    if orient not in ("vertical", "horizontal"):
        raise ValueError('boxplot_orientation must be one of: "vertical", "horizontal"')

    img = image.convert("RGB")
    W, H = img.size
    if W <= 1 or H <= 1:
        raise ValueError("Invalid image size")

    img_area = float(W * H)

    # 0) Explicit mask ids override all other filters.
    if box_labels_of_interest is not None:
        ids = []
        for x in box_labels_of_interest:
            try:
                ids.append(int(x))
            except Exception:
                continue
        id_set = set([i for i in ids if i > 0])
        if not id_set:
            raise ValueError("box_labels_of_interest must contain at least one positive integer id")

        out: List[BboxXyxy] = []
        for m in masks:
            try:
                mid = int(m.get("id", 0) or 0)
            except Exception:
                continue
            if mid not in id_set:
                continue
            bb = m.get("bbox_xyxy")
            if isinstance(bb, (list, tuple)) and len(bb) == 4:
                out.append(_clip_bbox_xyxy((int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])), W, H))
        if not out:
            raise RuntimeError("get_boxplot found no masks for the given box_labels_of_interest")
        return sorted(set(out), key=lambda b: (b[0], b[1], b[2], b[3]))

    # 1) Optional OCR localization of ticker_label to pick a category/group.
    label_bbox: Optional[BboxXyxy] = None
    label_anchor: Optional[Tuple[float, float]] = None

    thr = float(axis_threshold)
    thr = max(0.05, min(0.45, thr))

    def _ocr_units(img_in: Image.Image) -> List[Dict[str, object]]:
        """
        Word-level OCR units (preferred over line-level) so single-character tick labels
        like "A/B/C" are not merged into one string such as "A B".

        Returns items with at least: {text: str, bbox_xyxy: (x1,y1,x2,y2)}
        """
        if _easyocr_available():
            w0, h0 = img_in.size
            scale = 2 if max(w0, h0) >= 900 else 3
            up = img_in.resize((max(2, w0 * scale), max(2, h0 * scale)), resample=Image.BICUBIC)
            words = _easyocr_words(up)
            for w in words:
                bb = w.get("bbox_xyxy")
                if not isinstance(bb, tuple) or len(bb) != 4:
                    continue
                x1, y1, x2, y2 = bb
                w["bbox_xyxy"] = (int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale))
            return words
        if _tesseract_available():
            w0, h0 = img_in.size
            scale = 2 if max(w0, h0) >= 900 else 3
            up = img_in.resize((max(2, w0 * scale), max(2, h0 * scale)), resample=Image.BICUBIC)
            words = _tesseract_words(up, psm=6)
            for w in words:
                bb = w.get("bbox_xyxy")
                if not isinstance(bb, tuple) or len(bb) != 4:
                    continue
                x1, y1, x2, y2 = bb
                w["bbox_xyxy"] = (int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale))
            return words
        raise RuntimeError(
            "get_boxplot requires an OCR backend when ticker_label is provided. "
            "Install EasyOCR (`pip install easyocr`) or install `tesseract` + `pytesseract`."
        )

    def _norm_token(s: str) -> str:
        s = str(s or "").strip().lower()
        s = re.sub(r"[^0-9a-zA-Z]+", "", s)
        return s

    def _best_bbox_for_ticker(units: List[Dict[str, object]], target: str) -> Optional[BboxXyxy]:
        target_s = str(target or "").strip()
        if not target_s:
            return None
        t_norm = _norm_token(target_s)
        # For single-char tick labels, similarity-based matching is brittle; prefer token containment.
        min_sim = 0.55 if len(t_norm) > 2 else 0.35

        best_bb: Optional[BboxXyxy] = None
        best_score = -1.0

        for u in units:
            text_u = str(u.get("text", "") or "").strip()
            bb = u.get("bbox_xyxy")
            if not text_u or not (isinstance(bb, tuple) and len(bb) == 4):
                continue
            x1, y1, x2, y2 = [int(v) for v in bb]
            if x2 <= x1 or y2 <= y1:
                continue

            u_norm = _norm_token(text_u)
            if u_norm == t_norm and t_norm:
                # Exact normalized match.
                return (x1, y1, x2, y2)

            # Token containment: e.g., OCR returns "A B" but target is "B".
            tokens = [t for t in re.findall(r"[0-9A-Za-z]+", text_u) if t]
            tokens_norm = [_norm_token(t) for t in tokens]
            if t_norm and t_norm in tokens_norm:
                idx = int(tokens_norm.index(t_norm))
                total_len = float(sum(max(1, len(t)) for t in tokens_norm))
                if total_len <= 0:
                    total_len = float(len(tokens_norm))
                start = float(sum(max(1, len(t)) for t in tokens_norm[:idx]))
                end = start + float(max(1, len(tokens_norm[idx])))
                w = float(x2 - x1)
                xx1 = int(round(float(x1) + (start / total_len) * w))
                xx2 = int(round(float(x1) + (end / total_len) * w))
                # Slight padding so the box is visible.
                pad = int(round(0.10 * float(max(1, xx2 - xx1))))
                return (max(0, xx1 - pad), y1, min(W, xx2 + pad), y2)

            score = float(_similarity(target_s, text_u))
            if score > best_score:
                best_score = score
                best_bb = (x1, y1, x2, y2)

        if best_bb is not None and best_score >= float(min_sim):
            return best_bb
        return None

    if isinstance(ticker_label, str) and ticker_label.strip():
        target = str(ticker_label).strip()
        if orient == "vertical":
            roi = (0, int(round((1.0 - thr) * H)), W, H)
        else:
            roi = (0, 0, int(round(thr * W)), H)

        rx1, ry1, rx2, ry2 = _clip_bbox_xyxy(roi, W, H)
        roi_img = img.crop((rx1, ry1, rx2, ry2))
        units = _ocr_units(roi_img)
        bb0 = _best_bbox_for_ticker(units, target)
        if bb0 is None:
            best = None
            best_score = -1.0
            for u in units:
                score = _similarity(target, str(u.get("text", "") or ""))
                if score > best_score:
                    best_score = float(score)
                    best = str(u.get("text", "") or "")
            raise RuntimeError(
                f'Failed to localize ticker_label="{target}" via OCR (best_score={best_score:.2f}, best_text="{best}").'
            )
        label_bbox = (bb0[0] + rx1, bb0[1] + ry1, bb0[2] + rx1, bb0[3] + ry1)
        cx = 0.5 * float(label_bbox[0] + label_bbox[2])
        cy = 0.5 * float(label_bbox[1] + label_bbox[3])
        label_anchor = (cx, cy)

    # 2) Candidate filtering: keep boxplot-like masks (boxes + thin lines).
    arr = np.asarray(img, dtype=np.uint8)
    bg = np.asarray(_estimate_background_rgb(img), dtype=np.int16)

    def _rep_rgb_for_mask(seg: np.ndarray, bbox_xyxy: BboxXyxy) -> RgbTuple:
        x1, y1, x2, y2 = _clip_bbox_xyxy(bbox_xyxy, W, H)
        if x2 <= x1 or y2 <= y1:
            return (0, 0, 0)
        sub = arr[y1:y2, x1:x2, :]
        msub = seg[y1:y2, x1:x2].astype(bool)
        if sub.size == 0 or int(msub.sum()) == 0:
            return (0, 0, 0)
        pix = sub[msub]
        if pix.ndim != 2 or pix.shape[1] != 3:
            pix = pix.reshape(-1, 3)
        diff_bg = np.abs(pix.astype(np.int16) - bg.reshape(1, 3)).sum(axis=1)
        keep = diff_bg > 20
        if int(np.sum(keep)) >= 20:
            pix = pix[keep]
        if pix.shape[0] > 60000:
            step = int(max(1, pix.shape[0] // 60000))
            pix = pix[::step][:60000]
        q = (pix.astype(np.uint8) // 8) * 8
        vals, counts = np.unique(q, axis=0, return_counts=True)
        if vals.size == 0:
            med = np.median(pix, axis=0)
            return (int(med[0]), int(med[1]), int(med[2]))
        mode = vals[int(np.argmax(counts))]
        return (int(mode[0]), int(mode[1]), int(mode[2]))

    def _color_score(rep: RgbTuple, target: RgbTuple) -> float:
        dist = float(abs(int(rep[0]) - int(target[0])) + abs(int(rep[1]) - int(target[1])) + abs(int(rep[2]) - int(target[2])))
        return 1.0 - min(1.0, dist / (3.0 * 255.0))

    target_rgb: Optional[RgbTuple] = None
    if rgb_of_interest is not None:
        if isinstance(rgb_of_interest, tuple) and len(rgb_of_interest) == 3:
            target_rgb = tuple(int(max(0, min(255, int(x)))) for x in rgb_of_interest)  # type: ignore[misc]
        elif isinstance(rgb_of_interest, list) and len(rgb_of_interest) == 3:
            target_rgb = tuple(int(max(0, min(255, int(x)))) for x in rgb_of_interest)  # type: ignore[misc]
        else:
            raise TypeError("rgb_of_interest must be a tuple/list of 3 ints or None")

    cat_dim = float(W if orient == "vertical" else H)
    ref_dim = float(max(W, H))
    min_line_len = max(12.0, 0.03 * ref_dim)
    max_line_thick = max(4.0, 0.006 * ref_dim)
    min_box_minor = max(8.0, 0.015 * ref_dim)
    min_box_major = max(12.0, 0.02 * ref_dim)

    candidates: List[Dict[str, object]] = []
    for m in masks:
        bb0 = m.get("bbox_xyxy")
        seg = m.get("segmentation")
        if not (isinstance(bb0, (list, tuple)) and len(bb0) == 4):
            if isinstance(seg, np.ndarray):
                ys, xs = np.where(seg.astype(bool))
                if xs.size and ys.size:
                    bb0 = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
        if not (isinstance(bb0, (list, tuple)) and len(bb0) == 4):
            continue
        x1, y1, x2, y2 = _clip_bbox_xyxy((int(bb0[0]), int(bb0[1]), int(bb0[2]), int(bb0[3])), W, H)
        bw = int(x2 - x1)
        bh = int(y2 - y1)
        if bw <= 1 or bh <= 1:
            continue

        area = int(m.get("area", 0) or 0)
        if area <= 0 and isinstance(seg, np.ndarray):
            area = int(np.asarray(seg).astype(bool).sum())
        if area <= 0:
            continue
        if float(area) > 0.55 * img_area:
            continue

        # Filter out tiny OCR/legend noise.
        if area < 16:
            continue

        major = float(max(bw, bh))
        minor = float(min(bw, bh))
        if minor <= 0:
            continue
        aspect = major / minor
        fill = float(area) / float(max(1.0, float(bw * bh)))

        is_line = (minor <= float(max_line_thick)) and (major >= float(min_line_len)) and (aspect >= 3.0)
        is_boxish = (minor >= float(min_box_minor)) and (major >= float(min_box_major)) and (aspect <= 10.0)
        if not (is_line or is_boxish):
            continue

        cx = 0.5 * float(x1 + x2)
        cy = 0.5 * float(y1 + y2)
        axis_center = float(cx if orient == "vertical" else cy)

        cscore = None
        if target_rgb is not None and isinstance(seg, np.ndarray):
            rep = _rep_rgb_for_mask(np.asarray(seg).astype(bool), (x1, y1, x2, y2))
            cscore = float(_color_score(rep, target_rgb))

        candidates.append(
            {
                "id": int(m.get("id", 0) or 0),
                "bbox_xyxy": (int(x1), int(y1), int(x2), int(y2)),
                "axis_center": float(axis_center),
                "band": float(bw if orient == "vertical" else bh),
                "area": int(area),
                "fill": float(fill),
                "aspect": float(aspect),
                "is_boxish": bool(is_boxish),
                "is_line": bool(is_line),
                "color_score": float(cscore) if cscore is not None else None,
            }
        )

    if not candidates:
        raise RuntimeError("get_boxplot found no boxplot-like candidates from segmentation masks")

    # If ticker_label is provided, restrict to candidates near the label anchor along the categorical axis.
    if label_anchor is not None:
        anchor_axis = float(label_anchor[0] if orient == "vertical" else label_anchor[1])
        window = max(0.20 * cat_dim, 30.0)
        near = [c for c in candidates if abs(float(c["axis_center"]) - anchor_axis) <= window]
        if near:
            candidates = near

    # If rgb filter is provided, keep only the most color-consistent candidates (best-effort).
    if target_rgb is not None:
        scored = [c for c in candidates if c.get("color_score") is not None]
        if scored:
            scored.sort(key=lambda d: float(d.get("color_score") or 0.0), reverse=True)
            keep_n = min(60, max(12, int(round(0.35 * float(len(scored))))))
            cutoff = float(scored[min(len(scored) - 1, keep_n - 1)].get("color_score") or 0.0)
            candidates = [c for c in scored if float(c.get("color_score") or 0.0) >= max(0.35, cutoff - 0.08)]

    # 3) Cluster by categorical axis coordinate (x for vertical, y for horizontal).
    bands = np.asarray([float(c.get("band") or 0.0) for c in candidates], dtype=np.float64)
    band_p75 = float(np.percentile(bands, 75.0)) if bands.size else 12.0
    # Cluster gap is based on typical box width/height, with sensible clamps.
    cluster_gap = max(12.0, min(0.25 * cat_dim, max(0.9 * band_p75, 0.04 * cat_dim)))

    cand_sorted = sorted(candidates, key=lambda d: float(d.get("axis_center") or 0.0))
    clusters: List[List[Dict[str, object]]] = []
    cur: List[Dict[str, object]] = [cand_sorted[0]]
    for c in cand_sorted[1:]:
        if abs(float(c.get("axis_center") or 0.0) - float(cur[-1].get("axis_center") or 0.0)) <= cluster_gap:
            cur.append(c)
        else:
            clusters.append(cur)
            cur = [c]
    if cur:
        clusters.append(cur)

    # 4) Choose the best cluster.
    anchor_axis = None
    if label_anchor is not None:
        anchor_axis = float(label_anchor[0] if orient == "vertical" else label_anchor[1])

    def _cluster_center(cl: List[Dict[str, object]]) -> float:
        xs = np.asarray([float(x.get("axis_center") or 0.0) for x in cl], dtype=np.float64)
        return float(np.median(xs)) if xs.size else 0.0

    def _cluster_score(cl: List[Dict[str, object]]) -> float:
        score = 0.0
        # Prefer clusters with at least one box-ish component.
        if any(bool(x.get("is_boxish")) for x in cl):
            score += 0.75
        score += 0.10 * float(len(cl))

        if anchor_axis is not None:
            c0 = _cluster_center(cl)
            dist = abs(float(c0) - float(anchor_axis))
            score += 2.5 * (1.0 - min(1.0, dist / float(max(1.0, cluster_gap))))

        if target_rgb is not None:
            cs = [float(x.get("color_score") or 0.0) for x in cl if x.get("color_score") is not None]
            if cs:
                score += 1.8 * float(max(cs))
        # Prefer moderately sized masks over tiny speckles when ambiguous.
        areas = np.asarray([float(x.get("area") or 0.0) for x in cl], dtype=np.float64)
        if areas.size:
            score += 0.15 * float(np.log1p(np.median(areas) / max(1.0, img_area)))
        return float(score)

    best_cluster = max(clusters, key=_cluster_score)

    # 5) Return bboxes for the selected cluster (deterministic order).
    # Drop outlier dots / tiny text fragments that SAM often segments near the boxplot cluster.
    # These can corrupt min/max estimation (whiskers vs outliers) in `compute_boxplot_entity`.
    max_filled_area = 0.0
    for c in best_cluster:
        try:
            fill_f = float(c.get("fill") or 0.0)
            area_f = float(c.get("area") or 0.0)
        except Exception:
            continue
        if fill_f >= 0.35:
            max_filled_area = max(max_filled_area, area_f)
    if max_filled_area <= 0:
        for c in best_cluster:
            try:
                max_filled_area = max(max_filled_area, float(c.get("area") or 0.0))
            except Exception:
                continue

    min_keep_filled = max(0.0005 * float(img_area), 0.05 * float(max_filled_area))

    def _line_like(c: Dict[str, object]) -> bool:
        try:
            if bool(c.get("is_line")):
                return True
            fill_f = float(c.get("fill") or 0.0)
            aspect_f = float(c.get("aspect") or 0.0)
        except Exception:
            return False
        return bool((fill_f <= 0.18) and (aspect_f >= 2.5))

    filtered_cluster: List[Dict[str, object]] = []
    for c in best_cluster:
        try:
            area_f = float(c.get("area") or 0.0)
            fill_f = float(c.get("fill") or 0.0)
        except Exception:
            continue
        if _line_like(c):
            filtered_cluster.append(c)
            continue
        # Keep only sufficiently large filled components (IQR box, median line, etc.).
        if (fill_f >= 0.35) and (area_f >= float(min_keep_filled)):
            filtered_cluster.append(c)

    # Fallback: if filtering removed everything (edge-case), return the original cluster.
    if filtered_cluster:
        best_cluster = filtered_cluster

    out_boxes = [tuple(x["bbox_xyxy"]) for x in best_cluster if isinstance(x.get("bbox_xyxy"), tuple)]  # type: ignore[misc]
    # Drop duplicates.
    out_unique = sorted(set([_clip_bbox_xyxy(b, W, H) for b in out_boxes]), key=lambda b: (b[0], b[1], b[2], b[3]))
    if not out_unique:
        raise RuntimeError("get_boxplot failed to produce any output boxes")
    return out_unique


def compute_boxplot_entity(
    image: Image.Image,
    boxplot_of_interest: List[Tuple[int, int, int, int]],
    boxplot_orientation: str = "vertical",
    entity_of_interest: str = "median",
    axis_threshold: float = 0.15,
    x_axis_tickers: Optional[List] = None,
    y_axis_tickers: Optional[List] = None,
) -> float:
    """
    Computes a statistical entity (e.g., max, min, median, Q1, Q3, range, or IQR) of a boxplot by mapping
    pixel coordinates to value space using OCR-based axis localization.

    The input `boxplot_of_interest` is a list of bounding boxes for boxplot components (typically returned by
    `get_boxplot`). This repo primarily uses `bbox_xyxy`, but some references use (x, y, w, h); this function
    accepts either and auto-detects the format.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    if not isinstance(boxplot_of_interest, list) or not boxplot_of_interest:
        raise ValueError("boxplot_of_interest must be a non-empty list of bboxes")

    orient = str(boxplot_orientation or "vertical").strip().lower()
    if orient not in ("vertical", "horizontal"):
        raise ValueError('boxplot_orientation must be one of: "vertical", "horizontal"')

    ent = str(entity_of_interest or "median").strip().lower()
    if ent in ("q2", "q_2"):
        ent = "median"
    if ent not in ("median", "min", "max", "q1", "q3", "range", "iqr"):
        raise ValueError('entity_of_interest must be one of: "median","min","max","q1","q3","q2","range","iqr"')

    img = image.convert("RGB")
    W, H = img.size
    if W <= 1 or H <= 1:
        raise ValueError("Invalid image size")

    # 1) Axis mapping.
    axis = "y" if orient == "vertical" else "x"
    tickers = y_axis_tickers if axis == "y" else x_axis_tickers
    axis_values, axis_pixel_positions = axis_localizer(
        img,
        axis=axis,
        axis_threshold=float(axis_threshold),
        axis_tickers=tickers,
    )
    if not axis_values or not axis_pixel_positions or len(axis_values) != len(axis_pixel_positions):
        raise RuntimeError("axis_localizer returned an invalid mapping")

    # 2) Normalize input bboxes to xyxy (auto-detect xyxy vs xywh).
    raw: List[Tuple[float, float, float, float]] = []
    for b in boxplot_of_interest:
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            continue
        try:
            a, b0, c, d = [float(x) for x in b]
        except Exception:
            continue
        raw.append((a, b0, c, d))
    if not raw:
        raise ValueError("boxplot_of_interest contains no valid bboxes")

    xyxy_ok = sum(1 for (x1, y1, x2, y2) in raw if (x2 > x1) and (y2 > y1))
    fmt = "xyxy" if xyxy_ok >= int(math.ceil(0.75 * float(len(raw)))) else "xywh"

    bboxes: List[BboxXyxy] = []
    for x, y, c, d in raw:
        if fmt == "xyxy":
            if not (c > x and d > y):
                continue
            bb = (int(round(x)), int(round(y)), int(round(c)), int(round(d)))
        else:
            # xywh
            if c == 0 or d == 0:
                continue
            bb = (int(round(x)), int(round(y)), int(round(x + c)), int(round(y + d)))
        bb = _clip_bbox_xyxy(bb, W, H)
        if bb[2] <= bb[0] or bb[3] <= bb[1]:
            continue
        bboxes.append(bb)
    if not bboxes:
        raise RuntimeError("compute_boxplot_entity: no usable bboxes after normalization")

    ref_dim = float(max(W, H))
    # Heuristics to classify thin whisker/cap/median lines vs the IQR box.
    max_line_thick = max(4.0, 0.006 * ref_dim)
    min_line_len = max(12.0, 0.03 * ref_dim)
    min_box_minor = max(8.0, 0.015 * ref_dim)
    min_box_major = max(12.0, 0.02 * ref_dim)

    def _dims(bb: BboxXyxy) -> Tuple[float, float]:
        return float(max(1, bb[2] - bb[0])), float(max(1, bb[3] - bb[1]))

    def _is_line_like(bb: BboxXyxy) -> bool:
        bw, bh = _dims(bb)
        major = float(max(bw, bh))
        minor = float(min(bw, bh))
        if minor <= 0:
            return False
        aspect = major / minor
        return bool((minor <= max_line_thick) and (major >= min_line_len) and (aspect >= 3.0))

    def _is_boxish(bb: BboxXyxy) -> bool:
        bw, bh = _dims(bb)
        major = float(max(bw, bh))
        minor = float(min(bw, bh))
        if minor <= 0:
            return False
        aspect = major / minor
        return bool((minor >= min_box_minor) and (major >= min_box_major) and (aspect <= 12.0))

    # Pick an IQR box candidate.
    box_candidates = [bb for bb in bboxes if (not _is_line_like(bb)) and _is_boxish(bb)]
    if not box_candidates:
        # Fallback: pick the largest non-line bbox.
        box_candidates = [bb for bb in bboxes if not _is_line_like(bb)]
    if not box_candidates:
        box_candidates = list(bboxes)
    iqr_box = max(box_candidates, key=lambda bb: float((bb[2] - bb[0]) * (bb[3] - bb[1])))

    # Global whisker extrema (best-effort).
    if orient == "vertical":
        top_px = int(min(bb[1] for bb in bboxes))
        bot_px = int(max(bb[3] for bb in bboxes))
        q3_px = int(iqr_box[1])
        q1_px = int(iqr_box[3])
    else:
        left_px = int(min(bb[0] for bb in bboxes))
        right_px = int(max(bb[2] for bb in bboxes))
        q1_px = int(iqr_box[0])
        q3_px = int(iqr_box[2])

    # Median line (best-effort): find a thin line inside the IQR box.
    median_px: Optional[int] = None
    if orient == "vertical":
        bx1, by1, bx2, by2 = iqr_box
        bw = max(1, bx2 - bx1)

        best = None
        best_score = -1e18
        for bb in bboxes:
            if bb == iqr_box:
                continue
            if not _is_line_like(bb):
                continue
            x1, y1, x2, y2 = bb
            # Median in vertical boxplot is typically a horizontal line.
            if (x2 - x1) < (y2 - y1):
                continue
            cy = 0.5 * float(y1 + y2)
            if cy < float(by1) - 3.0 or cy > float(by2) + 3.0:
                continue
            ix1 = max(bx1, x1)
            ix2 = min(bx2, x2)
            overlap = float(max(0, ix2 - ix1)) / float(max(1, bw))
            if overlap < 0.40:
                continue
            thick = float(max(1, y2 - y1))
            score = (2.0 * overlap) - (0.25 * thick)
            if score > best_score:
                best_score = score
                best = bb
        if best is not None:
            median_px = int(round(0.5 * float(best[1] + best[3])))
        else:
            median_px = int(round(0.5 * float(q1_px + q3_px)))
    else:
        bx1, by1, bx2, by2 = iqr_box
        bh = max(1, by2 - by1)

        best = None
        best_score = -1e18
        for bb in bboxes:
            if bb == iqr_box:
                continue
            if not _is_line_like(bb):
                continue
            x1, y1, x2, y2 = bb
            # Median in horizontal boxplot is typically a vertical line.
            if (y2 - y1) < (x2 - x1):
                continue
            cx = 0.5 * float(x1 + x2)
            if cx < float(bx1) - 3.0 or cx > float(bx2) + 3.0:
                continue
            iy1 = max(by1, y1)
            iy2 = min(by2, y2)
            overlap = float(max(0, iy2 - iy1)) / float(max(1, bh))
            if overlap < 0.40:
                continue
            thick = float(max(1, x2 - x1))
            score = (2.0 * overlap) - (0.25 * thick)
            if score > best_score:
                best_score = score
                best = bb
        if best is not None:
            median_px = int(round(0.5 * float(best[0] + best[2])))
        else:
            median_px = int(round(0.5 * float(q1_px + q3_px)))

    def _pix_to_val(p: float) -> float:
        return float(interpolate_pixel_to_value(float(p), axis_values, axis_pixel_positions))

    if orient == "vertical":
        v_min_raw = _pix_to_val(float(bot_px))
        v_max_raw = _pix_to_val(float(top_px))
        v_q1_raw = _pix_to_val(float(q1_px))
        v_q3_raw = _pix_to_val(float(q3_px))
        v_med = _pix_to_val(float(median_px if median_px is not None else int(round(0.5 * float(q1_px + q3_px)))))
    else:
        v_min_raw = _pix_to_val(float(left_px))
        v_max_raw = _pix_to_val(float(right_px))
        v_q1_raw = _pix_to_val(float(q1_px))
        v_q3_raw = _pix_to_val(float(q3_px))
        v_med = _pix_to_val(float(median_px if median_px is not None else int(round(0.5 * float(q1_px + q3_px)))))

    # Normalize numeric ordering (handles inverted axes).
    v_min = float(min(v_min_raw, v_max_raw))
    v_max = float(max(v_min_raw, v_max_raw))
    v_q1 = float(min(v_q1_raw, v_q3_raw))
    v_q3 = float(max(v_q1_raw, v_q3_raw))

    if ent == "median":
        return float(v_med)
    if ent == "min":
        return float(v_min)
    if ent == "max":
        return float(v_max)
    if ent == "q1":
        return float(v_q1)
    if ent == "q3":
        return float(v_q3)
    if ent == "range":
        return float(abs(float(v_max) - float(v_min)))
    # ent == "iqr"
    return float(abs(float(v_q3) - float(v_q1)))


def get_edgepoints(
    image: Image.Image,
    masks: Optional[List[Dict[str, object]]] = None,
    rgb_of_interest: Optional[RgbTuple] = None,
    ticker_label: Optional[str] = None,
    mask_labels_of_interest: Optional[List[int]] = None,
    chart_orientation: str = "vertical",
    lineplot_get_dot: bool = False,
    axis_threshold: float = 0.15,
) -> List[Tuple[int, int]]:
    """
    Compute edge points of a chart segment filtered by color, label, and orientation.

    Behavior:
      - When `ticker_label` is provided, OCR is used to localize the label in the axis ROI and its center
        determines the scan line (vertical: x center; horizontal: y center).
      - Segment selection:
          * If `mask_labels_of_interest` is provided -> union those masks (requires `masks` with `segmentation`).
          * Else if `rgb_of_interest` is provided -> color-based pixel mask (optionally intersected with best mask).
      - Output:
          * `lineplot_get_dot=False`: return 2 edge endpoints (vertical: (x, top_y),(x,bottom_y); horizontal: (left_x,y),(right_x,y)).
          * `lineplot_get_dot=True`: return dot centroid(s). With `ticker_label`, returns the centroid near the scan line.
            Without `ticker_label`, returns centroids for all dot-like components.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")

    orient = str(chart_orientation or "vertical").strip().lower()
    if orient not in ("vertical", "horizontal"):
        raise ValueError('chart_orientation must be one of: "vertical", "horizontal"')

    img = image.convert("RGB")
    W, H = img.size
    if W <= 1 or H <= 1:
        raise ValueError("Invalid image size")

    # 1) Locate ticker label bbox (optional).
    label_bbox: Optional[BboxXyxy] = None
    label_anchor: Optional[Tuple[float, float]] = None

    thr = float(axis_threshold)
    thr = max(0.05, min(0.45, thr))

    def _ocr_lines(img_in: Image.Image) -> List[Dict[str, object]]:
        if _easyocr_available():
            w0, h0 = img_in.size
            scale = 2 if max(w0, h0) >= 900 else 3
            up = img_in.resize((max(2, w0 * scale), max(2, h0 * scale)), resample=Image.BICUBIC)
            lines = _easyocr_lines(up)
            for ln in lines:
                bb = ln.get("bbox_xyxy")
                if not isinstance(bb, tuple) or len(bb) != 4:
                    continue
                x1, y1, x2, y2 = bb
                ln["bbox_xyxy"] = (int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale))
            return lines
        if _tesseract_available():
            w0, h0 = img_in.size
            scale = 2 if max(w0, h0) >= 900 else 3
            up = img_in.resize((max(2, w0 * scale), max(2, h0 * scale)), resample=Image.BICUBIC)
            lines = _tesseract_lines(up, psm=6)
            for ln in lines:
                bb = ln.get("bbox_xyxy")
                if not isinstance(bb, tuple) or len(bb) != 4:
                    continue
                x1, y1, x2, y2 = bb
                ln["bbox_xyxy"] = (int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale))
            return lines
        raise RuntimeError(
            "get_edgepoints requires an OCR backend when ticker_label is provided. "
            "Install EasyOCR (`pip install easyocr`) or install `tesseract` + `pytesseract`."
        )

    if isinstance(ticker_label, str) and ticker_label.strip():
        target = str(ticker_label).strip()
        if orient == "vertical":
            roi = (0, int(round((1.0 - thr) * H)), W, H)
        else:
            roi = (0, 0, int(round(thr * W)), H)

        rx1, ry1, rx2, ry2 = _clip_bbox_xyxy(roi, W, H)
        roi_img = img.crop((rx1, ry1, rx2, ry2))
        lines = _ocr_lines(roi_img)
        bb0 = _find_best_line_bbox(lines, target, min_similarity=0.55)
        if bb0 is None:
            best = None
            best_score = -1.0
            for ln in lines:
                score = _similarity(target, str(ln.get("text", "") or ""))
                if score > best_score:
                    best_score = float(score)
                    best = str(ln.get("text", "") or "")
            raise RuntimeError(
                f'Failed to localize ticker_label="{target}" via OCR (best_score={best_score:.2f}, best_text="{best}").'
            )
        label_bbox = (bb0[0] + rx1, bb0[1] + ry1, bb0[2] + rx1, bb0[3] + ry1)
        cx = 0.5 * float(label_bbox[0] + label_bbox[2])
        cy = 0.5 * float(label_bbox[1] + label_bbox[3])
        label_anchor = (cx, cy)

    # 2) Build target mask (union masks or color-threshold).
    if mask_labels_of_interest is not None:
        if masks is None:
            raise ValueError("mask_labels_of_interest requires `masks`")
        if not isinstance(masks, list):
            raise TypeError("masks must be a list or None")
        wanted: List[int] = []
        for v in mask_labels_of_interest:
            try:
                wanted.append(int(v))
            except Exception:
                continue
        wanted_set = set([x for x in wanted if x > 0])
        if not wanted_set:
            raise ValueError("mask_labels_of_interest must contain at least one positive integer id")

        target_mask = np.zeros((H, W), dtype=bool)
        matched = 0
        for idx, m in enumerate(masks, start=1):
            if not isinstance(m, dict):
                continue
            mid = None
            for k in ("id", "label", "mask_id"):
                if k in m:
                    try:
                        mid = int(m.get(k))  # type: ignore[arg-type]
                    except Exception:
                        mid = None
                    break
            mid_i = int(mid) if isinstance(mid, int) else int(idx)
            if (mid_i not in wanted_set) and (idx not in wanted_set):
                continue
            seg = m.get("segmentation")
            if not isinstance(seg, np.ndarray):
                continue
            seg_b = np.asarray(seg).astype(bool)
            if seg_b.shape != (H, W):
                continue
            target_mask |= seg_b
            matched += 1
        if matched == 0 or int(target_mask.sum()) == 0:
            raise RuntimeError("get_edgepoints: mask_labels_of_interest did not match any usable masks")
    else:
        # No explicit mask ids -> use rgb-based selection (optionally aided by masks).
        if rgb_of_interest is None:
            raise ValueError("get_edgepoints requires either mask_labels_of_interest or rgb_of_interest")

        rgb = np.asarray([int(rgb_of_interest[0]), int(rgb_of_interest[1]), int(rgb_of_interest[2])], dtype=np.int16)
        rgb = np.clip(rgb, 0, 255)
        arr = np.asarray(img, dtype=np.uint8)
        arr_i16 = arr.astype(np.int16)

        # Start with a conservative tolerance; relax when nothing is found.
        tol = 60
        diff = np.abs(arr_i16 - rgb.reshape(1, 1, 3)).sum(axis=2)
        color_mask = diff <= tol
        if int(color_mask.sum()) == 0:
            tol = 90 if bool(lineplot_get_dot) else 80
            color_mask = diff <= tol

        target_mask = color_mask.astype(bool)

        # If masks are supplied, try to select a best mask by color similarity and intersect (best-effort).
        if isinstance(masks, list) and masks:
            bg = np.asarray(_estimate_background_rgb(img), dtype=np.int16)

            def _rep_rgb_for_mask(seg: np.ndarray, bbox_xyxy: BboxXyxy) -> RgbTuple:
                x1, y1, x2, y2 = _clip_bbox_xyxy(bbox_xyxy, W, H)
                if x2 <= x1 or y2 <= y1:
                    return (0, 0, 0)
                sub = arr[y1:y2, x1:x2, :]
                msub = seg[y1:y2, x1:x2].astype(bool)
                if sub.size == 0 or int(msub.sum()) == 0:
                    return (0, 0, 0)
                pix = sub[msub]
                if pix.ndim != 2 or pix.shape[1] != 3:
                    pix = pix.reshape(-1, 3)
                diff_bg = np.abs(pix.astype(np.int16) - bg.reshape(1, 3)).sum(axis=1)
                keep = diff_bg > 20
                if int(np.sum(keep)) >= 20:
                    pix = pix[keep]
                if pix.shape[0] > 60000:
                    step = int(max(1, pix.shape[0] // 60000))
                    pix = pix[::step][:60000]
                q = (pix.astype(np.uint8) // 8) * 8
                vals, counts = np.unique(q, axis=0, return_counts=True)
                if vals.size == 0:
                    med = np.median(pix, axis=0)
                    return (int(med[0]), int(med[1]), int(med[2]))
                mode = vals[int(np.argmax(counts))]
                return (int(mode[0]), int(mode[1]), int(mode[2]))

            def _l1(a: RgbTuple, b: Sequence[int]) -> int:
                return int(abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])) + abs(int(a[2]) - int(b[2])))

            best_m = None
            best_d = 10**9
            for m in masks:
                if not isinstance(m, dict):
                    continue
                seg = m.get("segmentation")
                bb = m.get("bbox_xyxy")
                if not isinstance(seg, np.ndarray):
                    continue
                if isinstance(bb, (list, tuple)) and len(bb) == 4:
                    bb_xyxy = _clip_bbox_xyxy((int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])), W, H)
                else:
                    ys, xs = np.where(np.asarray(seg).astype(bool))
                    if xs.size == 0 or ys.size == 0:
                        continue
                    bb_xyxy = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
                rep = _rep_rgb_for_mask(np.asarray(seg).astype(bool), bb_xyxy)
                d = _l1(rep, rgb.tolist())
                if d < best_d:
                    best_d = int(d)
                    best_m = np.asarray(seg).astype(bool)

            if best_m is not None and best_m.shape == (H, W):
                inter = np.logical_and(best_m, color_mask)
                if int(inter.sum()) >= int(0.05 * max(1, int(best_m.sum()))):
                    target_mask = inter

    if int(target_mask.sum()) == 0:
        raise RuntimeError("get_edgepoints: target mask is empty")

    # 3) Determine scan line coordinate (from label or mask center).
    if label_anchor is not None:
        scan_x = int(round(float(label_anchor[0])))
        scan_y = int(round(float(label_anchor[1])))
    else:
        ys, xs = np.where(target_mask)
        if xs.size == 0 or ys.size == 0:
            raise RuntimeError("get_edgepoints: cannot infer scan line without ticker_label (empty mask)")
        scan_x = int(round(float(np.median(xs))))
        scan_y = int(round(float(np.median(ys))))

    scan_x = max(0, min(W - 1, scan_x))
    scan_y = max(0, min(H - 1, scan_y))

    # 4) Dot mode: return centroid(s).
    if bool(lineplot_get_dot):
        try:
            import cv2  # type: ignore
        except Exception as e:
            raise RuntimeError("get_edgepoints(lineplot_get_dot=True) requires opencv-python (cv2).") from e

        # If no ticker_label: return all dot-like component centroids.
        if label_anchor is None:
            mask_u8 = (target_mask.astype(np.uint8) * 255)
            k = 3 if min(W, H) <= 1400 else 5
            kernel = np.ones((k, k), dtype=np.uint8)
            opened = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)

            n_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
            if n_labels <= 1:
                raise RuntimeError("get_edgepoints: no connected components found")
            areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int32)
            if areas.size == 0:
                raise RuntimeError("get_edgepoints: no connected components found")
            med = float(np.median(areas))
            if med <= 0:
                med = float(np.mean(areas)) if areas.size else 0.0
            min_a = max(5.0, 0.25 * med)
            max_a = max(min_a + 1.0, 3.0 * med)

            pts: List[Tuple[int, int]] = []
            for lab in range(1, int(n_labels)):
                a0 = float(stats[lab, cv2.CC_STAT_AREA])
                if a0 < min_a or a0 > max_a:
                    continue
                w0 = float(stats[lab, cv2.CC_STAT_WIDTH])
                h0 = float(stats[lab, cv2.CC_STAT_HEIGHT])
                if w0 <= 0 or h0 <= 0:
                    continue
                ar = w0 / h0 if h0 != 0 else 999.0
                if ar < 0.35 or ar > 2.85:
                    continue
                cx, cy = centroids[lab]
                pts.append((int(round(cx)), int(round(cy))))

            if not pts:
                raise RuntimeError("get_edgepoints: no dot-like components after filtering")

            pts_sorted = sorted(pts, key=lambda p: (p[0], p[1])) if orient == "vertical" else sorted(pts, key=lambda p: (p[1], p[0]))
            return pts_sorted

        # With ticker_label: find the dot near the scan line in a local ROI.
        # Scan around the axis coordinate to find an intersection pixel.
        if orient == "vertical":
            base = scan_x
            window = max(2, min(80, int(round(0.03 * float(W)))))
            min_count = 1
            chosen_x = None
            y_idxs = None
            for d in range(0, window + 1):
                for off in (d, -d) if d != 0 else (0,):
                    x = base + off
                    if x < 0 or x >= W:
                        continue
                    col = target_mask[:, x]
                    if int(col.sum()) < min_count:
                        continue
                    chosen_x = int(x)
                    y_idxs = np.where(col)[0]
                    break
                if chosen_x is not None:
                    break
            if chosen_x is None or y_idxs is None or y_idxs.size == 0:
                raise RuntimeError("get_edgepoints: no pixels found near the x-label scan line for dot localization")
            seed_y = int(y_idxs[int(y_idxs.size // 2)])
            seed_x = int(chosen_x)
        else:
            base = scan_y
            window = max(2, min(80, int(round(0.03 * float(H)))))
            min_count = 1
            chosen_y = None
            x_idxs = None
            for d in range(0, window + 1):
                for off in (d, -d) if d != 0 else (0,):
                    y = base + off
                    if y < 0 or y >= H:
                        continue
                    row = target_mask[y, :]
                    if int(row.sum()) < min_count:
                        continue
                    chosen_y = int(y)
                    x_idxs = np.where(row)[0]
                    break
                if chosen_y is not None:
                    break
            if chosen_y is None or x_idxs is None or x_idxs.size == 0:
                raise RuntimeError("get_edgepoints: no pixels found near the y-label scan line for dot localization")
            seed_x = int(x_idxs[int(x_idxs.size // 2)])
            seed_y = int(chosen_y)

        r = max(20, int(round(0.05 * float(min(W, H)))))
        x1 = max(0, seed_x - r)
        x2 = min(W, seed_x + r + 1)
        y1 = max(0, seed_y - r)
        y2 = min(H, seed_y + r + 1)

        patch = target_mask[y1:y2, x1:x2].astype(np.uint8) * 255
        if patch.size == 0:
            raise RuntimeError("get_edgepoints: empty dot ROI")

        k = 3 if min(W, H) <= 1400 else 5
        kernel = np.ones((k, k), dtype=np.uint8)
        opened = cv2.morphologyEx(patch, cv2.MORPH_OPEN, kernel, iterations=1)
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
        if n_labels <= 1:
            # Fallback to raw patch mask.
            n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(patch, connectivity=8)
        if n_labels <= 1:
            raise RuntimeError("get_edgepoints: failed to localize dot component in ROI")

        # Select component closest to the seed point (in ROI coords) and reasonably blob-like.
        sx = float(seed_x - x1)
        sy = float(seed_y - y1)
        best_lab = None
        best_score = 1e18
        for lab in range(1, int(n_labels)):
            a0 = float(stats[lab, cv2.CC_STAT_AREA])
            if a0 < 5.0:
                continue
            w0 = float(stats[lab, cv2.CC_STAT_WIDTH])
            h0 = float(stats[lab, cv2.CC_STAT_HEIGHT])
            if w0 <= 0 or h0 <= 0:
                continue
            ar = max(w0 / h0, h0 / w0)
            if ar > 3.0:
                continue
            cx, cy = centroids[lab]
            dist2 = (float(cx) - sx) ** 2 + (float(cy) - sy) ** 2
            # prefer closer; slight preference for larger components.
            score = dist2 - 0.15 * float(a0)
            if score < best_score:
                best_score = float(score)
                best_lab = int(lab)

        if best_lab is None:
            # fallback: component at seed pixel, else nearest centroid.
            lab0 = int(labels[int(round(sy)) if sy >= 0 else 0, int(round(sx)) if sx >= 0 else 0])
            best_lab = lab0 if lab0 != 0 else 1

        cx, cy = centroids[int(best_lab)]
        return [(int(round(float(cx) + float(x1))), int(round(float(cy) + float(y1))))]

    # 5) Edge endpoints mode.
    if orient == "vertical":
        base = scan_x
        window = max(2, min(80, int(round(0.03 * float(W)))))
        min_count = 3
        chosen_x = None
        y_idxs = None
        for d in range(0, window + 1):
            for off in (d, -d) if d != 0 else (0,):
                x = base + off
                if x < 0 or x >= W:
                    continue
                col = target_mask[:, x]
                if int(col.sum()) < min_count:
                    continue
                chosen_x = int(x)
                y_idxs = np.where(col)[0]
                break
            if chosen_x is not None:
                break
        if chosen_x is None or y_idxs is None or y_idxs.size == 0:
            raise RuntimeError("get_edgepoints: failed to find segment pixels near scan x")
        top_y = int(y_idxs.min())
        bot_y = int(y_idxs.max())
        return [(int(chosen_x), int(top_y)), (int(chosen_x), int(bot_y))]

    base = scan_y
    window = max(2, min(80, int(round(0.03 * float(H)))))
    min_count = 3
    chosen_y = None
    x_idxs = None
    for d in range(0, window + 1):
        for off in (d, -d) if d != 0 else (0,):
            y = base + off
            if y < 0 or y >= H:
                continue
            row = target_mask[y, :]
            if int(row.sum()) < min_count:
                continue
            chosen_y = int(y)
            x_idxs = np.where(row)[0]
            break
        if chosen_y is not None:
            break
    if chosen_y is None or x_idxs is None or x_idxs.size == 0:
        raise RuntimeError("get_edgepoints: failed to find segment pixels near scan y")
    left_x = int(x_idxs.min())
    right_x = int(x_idxs.max())
    return [(int(left_x), int(chosen_y)), (int(right_x), int(chosen_y))]


def get_radial(
    image: Image.Image,
    rgb_of_interest: Optional[RgbTuple] = None,
    ticker_label: Optional[str] = None,
    segmentation_model: str = "color",
) -> BboxXyxy:
    """
    Identify a radial-bar segment (radial bar / radial wedge) corresponding to a color and/or label and return
    its coordinates as a bounding box (xyxy).

    Supported modes:
      - segmentation_model="color": requires `rgb_of_interest` and finds the best connected component by color.
      - segmentation_model="SAM": uses `segment_and_mark` (SAM1) and selects the best mask by color/label.

    Notes:
      - This function returns `bbox_xyxy` (x1,y1,x2,y2) to match other tools in this repo.
      - If `ticker_label` is provided and an OCR backend is available, it is used as a tie-breaker to select the
        component/mask closest to that label.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")

    mode = str(segmentation_model or "color").strip().lower()
    if mode not in ("color", "sam"):
        raise ValueError('segmentation_model must be one of: "color", "SAM"')

    img = image.convert("RGB")
    W, H = img.size
    if W <= 1 or H <= 1:
        raise ValueError("Invalid image size")
    img_area = float(W * H)

    # Optional OCR localization for ticker_label (full-image search; radial labels can be anywhere).
    label_anchor: Optional[Tuple[float, float]] = None
    if isinstance(ticker_label, str) and ticker_label.strip():
        target = str(ticker_label).strip()

        def _ocr_lines(img_in: Image.Image) -> List[Dict[str, object]]:
            if _easyocr_available():
                w0, h0 = img_in.size
                scale = 2 if max(w0, h0) >= 900 else 3
                up = img_in.resize((max(2, w0 * scale), max(2, h0 * scale)), resample=Image.BICUBIC)
                lines = _easyocr_lines(up)
                for ln in lines:
                    bb = ln.get("bbox_xyxy")
                    if not isinstance(bb, tuple) or len(bb) != 4:
                        continue
                    x1, y1, x2, y2 = bb
                    ln["bbox_xyxy"] = (int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale))
                return lines
            if _tesseract_available():
                w0, h0 = img_in.size
                scale = 2 if max(w0, h0) >= 900 else 3
                up = img_in.resize((max(2, w0 * scale), max(2, h0 * scale)), resample=Image.BICUBIC)
                lines = _tesseract_lines(up, psm=6)
                for ln in lines:
                    bb = ln.get("bbox_xyxy")
                    if not isinstance(bb, tuple) or len(bb) != 4:
                        continue
                    x1, y1, x2, y2 = bb
                    ln["bbox_xyxy"] = (int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale))
                return lines
            return []

        lines = _ocr_lines(img)
        bb0 = _find_best_line_bbox(lines, target, min_similarity=0.55)
        if bb0 is not None:
            cx = 0.5 * float(bb0[0] + bb0[2])
            cy = 0.5 * float(bb0[1] + bb0[3])
            label_anchor = (cx, cy)

    if mode == "color":
        if rgb_of_interest is None:
            raise ValueError('get_radial(segmentation_model="color") requires rgb_of_interest')
        try:
            import cv2  # type: ignore
        except Exception as e:
            raise RuntimeError('get_radial(segmentation_model="color") requires opencv-python (cv2).') from e

        rgb = np.asarray([int(rgb_of_interest[0]), int(rgb_of_interest[1]), int(rgb_of_interest[2])], dtype=np.int16)
        rgb = np.clip(rgb, 0, 255)
        arr = np.asarray(img, dtype=np.uint8)
        arr_i16 = arr.astype(np.int16)
        bg = np.asarray(_estimate_background_rgb(img), dtype=np.int16)

        def _build_mask(tol: int) -> np.ndarray:
            diff_t = np.abs(arr_i16 - rgb.reshape(1, 1, 3)).sum(axis=2)
            diff_bg = np.abs(arr_i16 - bg.reshape(1, 1, 3)).sum(axis=2)
            # Ensure we are not selecting whitespace.
            return np.logical_and(diff_t <= int(tol), diff_bg > 25)

        tol = 70
        mask = _build_mask(tol)
        if int(mask.sum()) < 80:
            tol = 90
            mask = _build_mask(tol)
        if int(mask.sum()) < 80:
            tol = 120
            mask = _build_mask(tol)
        if int(mask.sum()) == 0:
            raise RuntimeError("get_radial(color) failed to find any pixels close to rgb_of_interest")

        mask_u8 = (mask.astype(np.uint8) * 255)
        # Clean up speckles without merging neighboring segments too aggressively.
        k_close = max(3, int(round(0.004 * float(max(W, H)))))
        k_close = min(k_close, 9)
        k_open = max(3, int(round(0.003 * float(max(W, H)))))
        k_open = min(k_open, 7)
        kern_c = np.ones((k_close, k_close), dtype=np.uint8)
        kern_o = np.ones((k_open, k_open), dtype=np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kern_c, iterations=1)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kern_o, iterations=1)

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if n_labels <= 1:
            raise RuntimeError("get_radial(color) found no connected components")

        diag = float(math.hypot(float(W), float(H)))
        best_lab = None
        best_score = -1e18
        for lab in range(1, int(n_labels)):
            area = float(stats[lab, cv2.CC_STAT_AREA])
            if area < max(80.0, 0.0002 * img_area):
                continue
            x = int(stats[lab, cv2.CC_STAT_LEFT])
            y = int(stats[lab, cv2.CC_STAT_TOP])
            bw = int(stats[lab, cv2.CC_STAT_WIDTH])
            bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
            if bw <= 0 or bh <= 0:
                continue
            # Reject near-global spill masks.
            if (float(bw * bh) / float(max(1.0, img_area))) > 0.80:
                continue

            cx, cy = centroids[lab]
            score = 0.65 * float(area / float(max(1.0, img_area)))
            if label_anchor is not None:
                lx, ly = label_anchor
                dist = float(math.hypot(float(cx) - float(lx), float(cy) - float(ly)))
                score += 0.35 * (1.0 - min(1.0, dist / float(max(1.0, diag))))
            if score > best_score:
                best_score = float(score)
                best_lab = int(lab)

        if best_lab is None:
            # fallback: take largest
            areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
            best_lab = 1 + int(np.argmax(areas)) if areas.size else 1

        x = int(stats[int(best_lab), cv2.CC_STAT_LEFT])
        y = int(stats[int(best_lab), cv2.CC_STAT_TOP])
        bw = int(stats[int(best_lab), cv2.CC_STAT_WIDTH])
        bh = int(stats[int(best_lab), cv2.CC_STAT_HEIGHT])
        bb = (x, y, x + bw, y + bh)
        return _clip_bbox_xyxy(bb, W, H)

    # mode == "SAM"
    if rgb_of_interest is None and (not (isinstance(ticker_label, str) and ticker_label.strip())):
        raise ValueError('get_radial(segmentation_model="SAM") requires rgb_of_interest and/or ticker_label')

    try:
        _labeled, masks = segment_and_mark(img, segmentation_model="SAM")
    except Exception as e:
        raise RuntimeError(f"get_radial failed to segment the image with SAM1: {e}") from e

    if not masks:
        raise RuntimeError("get_radial found no segmentation masks")

    target_rgb: Optional[RgbTuple] = None
    if rgb_of_interest is not None:
        if isinstance(rgb_of_interest, tuple) and len(rgb_of_interest) == 3:
            target_rgb = tuple(int(max(0, min(255, int(x)))) for x in rgb_of_interest)  # type: ignore[misc]
        elif isinstance(rgb_of_interest, list) and len(rgb_of_interest) == 3:
            target_rgb = tuple(int(max(0, min(255, int(x)))) for x in rgb_of_interest)  # type: ignore[misc]
        else:
            raise TypeError("rgb_of_interest must be a tuple/list of 3 ints or None")

    arr = np.asarray(img, dtype=np.uint8)
    bg = np.asarray(_estimate_background_rgb(img), dtype=np.int16)

    def _rep_rgb_for_mask(seg: np.ndarray, bbox_xyxy: BboxXyxy) -> RgbTuple:
        x1, y1, x2, y2 = _clip_bbox_xyxy(bbox_xyxy, W, H)
        if x2 <= x1 or y2 <= y1:
            return (0, 0, 0)
        sub = arr[y1:y2, x1:x2, :]
        msub = seg[y1:y2, x1:x2].astype(bool)
        if sub.size == 0 or int(msub.sum()) == 0:
            return (0, 0, 0)
        pix = sub[msub]
        if pix.ndim != 2 or pix.shape[1] != 3:
            pix = pix.reshape(-1, 3)
        diff_bg = np.abs(pix.astype(np.int16) - bg.reshape(1, 3)).sum(axis=1)
        keep = diff_bg > 20
        if int(np.sum(keep)) >= 20:
            pix = pix[keep]
        if pix.shape[0] > 60000:
            step = int(max(1, pix.shape[0] // 60000))
            pix = pix[::step][:60000]
        q = (pix.astype(np.uint8) // 8) * 8
        vals, counts = np.unique(q, axis=0, return_counts=True)
        if vals.size == 0:
            med = np.median(pix, axis=0)
            return (int(med[0]), int(med[1]), int(med[2]))
        mode = vals[int(np.argmax(counts))]
        return (int(mode[0]), int(mode[1]), int(mode[2]))

    def _l1(a: RgbTuple, b: RgbTuple) -> int:
        return int(abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])) + abs(int(a[2]) - int(b[2])))

    diag = float(math.hypot(float(W), float(H)))
    best = None
    best_score = -1e18

    for m in masks:
        bb = m.get("bbox_xyxy")
        seg = m.get("segmentation")
        if not isinstance(bb, tuple) or len(bb) != 4 or not isinstance(seg, np.ndarray):
            continue
        bb_xyxy = _clip_bbox_xyxy((int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])), W, H)
        bw = float(max(1, bb_xyxy[2] - bb_xyxy[0]))
        bh = float(max(1, bb_xyxy[3] - bb_xyxy[1]))
        area = float(m.get("area", 0) or 0)
        if area <= 0:
            area = float(np.asarray(seg).astype(bool).sum())
        if area <= 0:
            continue
        if (area / img_area) > 0.75:
            continue

        cx = 0.5 * float(bb_xyxy[0] + bb_xyxy[2])
        cy = 0.5 * float(bb_xyxy[1] + bb_xyxy[3])

        score = 0.0
        if target_rgb is not None:
            rep = _rep_rgb_for_mask(np.asarray(seg).astype(bool), bb_xyxy)
            dist = float(_l1(rep, target_rgb))
            score += 1.0 - min(1.0, dist / (3.0 * 255.0))

        if label_anchor is not None:
            lx, ly = label_anchor
            dist = float(math.hypot(float(cx) - float(lx), float(cy) - float(ly)))
            score += 0.6 * (1.0 - min(1.0, dist / float(max(1.0, diag))))

        # Penalize overly thin/line-like candidates.
        aspect = max(bw / float(max(1.0, bh)), bh / float(max(1.0, bw)))
        if aspect > 25.0 and (area / float(max(1.0, bw * bh))) < 0.60:
            score -= 0.35

        # Slight preference for moderately sized masks.
        score += 0.10 * float(math.log1p(area) / math.log1p(float(img_area)))

        if score > best_score:
            best_score = float(score)
            best = bb_xyxy

    if best is None:
        raise RuntimeError("get_radial(SAM) failed to select a radial segment mask")
    return _clip_bbox_xyxy(best, W, H)


def analyze_radial_geometry(
    image: Image.Image,
    contour_of_interest: np.ndarray,
) -> Tuple[Image.Image, int, int, float, float]:
    """
    Estimate radial geometry (center, outer radius) for a radial chart given a segment contour.

    Returns:
      - Visualization image (center + circles drawn)
      - center_x, center_y (ints)
      - r_outer (float): estimated outer reference circle radius
      - r_max (float): maximum radius from center to the contour
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    if not isinstance(contour_of_interest, np.ndarray):
        contour_of_interest = np.asarray(contour_of_interest)

    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("analyze_radial_geometry requires opencv-python (cv2).") from e

    img = image.convert("RGB")
    W, H = img.size
    if W <= 1 or H <= 1:
        raise ValueError("Invalid image size")

    cnt = np.asarray(contour_of_interest)
    if cnt.ndim == 3 and cnt.shape[1] == 1 and cnt.shape[2] == 2:
        pts = cnt[:, 0, :]
    elif cnt.ndim == 2 and cnt.shape[1] == 2:
        pts = cnt
    else:
        raise ValueError("contour_of_interest must have shape (N,1,2) or (N,2)")
    if pts.shape[0] < 3:
        raise ValueError("contour_of_interest must contain at least 3 points")
    pts_f = pts.astype(np.float64, copy=False)

    # 1) Estimate center by fitting a circle to (mostly) outer-arc points.
    mx = float(np.mean(pts_f[:, 0]))
    my = float(np.mean(pts_f[:, 1]))
    d0 = np.hypot(pts_f[:, 0] - mx, pts_f[:, 1] - my)
    fit_pts = pts_f
    if pts_f.shape[0] >= 30:
        try:
            thr = float(np.quantile(d0, 0.70))
            cand = pts_f[d0 >= thr]
            if cand.shape[0] >= 12:
                fit_pts = cand
        except Exception:
            fit_pts = pts_f

    def _fit_circle_algebraic(points: np.ndarray) -> Tuple[float, float, float]:
        x = points[:, 0]
        y = points[:, 1]
        A = np.column_stack([x, y, np.ones_like(x)])
        b = -(x * x + y * y)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        a_coef, b_coef, c_coef = float(sol[0]), float(sol[1]), float(sol[2])
        cx_f = -0.5 * a_coef
        cy_f = -0.5 * b_coef
        r2 = (cx_f * cx_f + cy_f * cy_f) - c_coef
        if not (r2 > 0.0 and math.isfinite(r2)):
            raise ValueError("invalid circle fit")
        r_f = float(math.sqrt(r2))
        return cx_f, cy_f, r_f

    cx, cy = mx, my
    try:
        cx, cy, _ = _fit_circle_algebraic(fit_pts)
    except Exception:
        # Keep mean as a last-resort fallback.
        cx, cy = mx, my

    # 2) Compute r_max from contour.
    d_contour = np.hypot(pts_f[:, 0] - cx, pts_f[:, 1] - cy)
    r_max = float(np.max(d_contour)) if d_contour.size else 0.0
    if not math.isfinite(r_max) or r_max <= 0.0:
        raise RuntimeError("analyze_radial_geometry failed to compute r_max")

    # 3) Estimate r_outer (outer reference circle radius).
    #
    # Prefer an outer-circle estimate based on the chart's outer circular border (robust to title/legend text,
    # which can otherwise get merged into a "non-background" component and inflate r_outer dramatically).
    arr = np.asarray(img, dtype=np.uint8)

    def _estimate_outer_circle_from_dark_ring(arr_rgb: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Try to localize the outer polar axis circle by finding a large, sparse, dark connected component
        (typically the black circular border) and fitting a circle to its pixels.
        """
        gray = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY)
        img_area = float(W * H)

        # Try a few thresholds: outer border is usually darker than most content.
        # NOTE: do not apply morphological open here; the outer circle can be 1–2px wide and will vanish.
        for thr in (70, 80, 90, 110, 130, 150):
            mask = (gray < int(thr)).astype(np.uint8)

            try:
                n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            except Exception:
                continue

            if int(n_labels) <= 1:
                continue

            best_lab = None
            best_score = -1e9
            for lab in range(1, int(n_labels)):
                x, y, ww, hh, area = [int(v) for v in stats[lab].tolist()]
                if area <= 0:
                    continue

                # Must be reasonably large and roughly centered in the image.
                if float(area) < 0.0015 * img_area:
                    continue
                if ww < int(0.35 * W) or hh < int(0.35 * H):
                    continue

                aspect = float(ww) / float(max(1, hh))
                if aspect < 0.50 or aspect > 2.00:
                    continue

                bbox_area = float(ww * hh)
                fill = float(area) / float(max(1.0, bbox_area))
                # Outer ring should be sparse in its bounding box.
                if fill > 0.18:
                    continue

                # Score: big bbox, sparse fill, aspect ~ 1.
                score = (2.0 * math.log1p(bbox_area)) - (8.0 * fill) - (2.0 * abs(math.log(aspect)))
                if score > best_score:
                    best_score = float(score)
                    best_lab = int(lab)

            if best_lab is None:
                continue

            ys_r, xs_r = np.where(labels == int(best_lab))
            if xs_r.size < 800:
                continue
            pts_ring = np.column_stack([xs_r.astype(np.float64), ys_r.astype(np.float64)])
            if pts_ring.shape[0] > 25000:
                rng = np.random.RandomState(0)
                pts_ring = pts_ring[rng.choice(pts_ring.shape[0], 25000, replace=False)]
            try:
                cx_f, cy_f, r_f = _fit_circle_algebraic(pts_ring)
            except Exception:
                continue
            if not (math.isfinite(cx_f) and math.isfinite(cy_f) and math.isfinite(r_f) and r_f > 0.0):
                continue
            return float(cx_f), float(cy_f), float(r_f)

        return None

    def _estimate_outer_circle_from_edges(
        arr_rgb: np.ndarray, cx0: float, cy0: float, r_min_hint: float
    ) -> Optional[Tuple[float, float, float]]:
        gray = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY)
        # Light blur helps Canny focus on the major circle boundary.
        gray = cv2.GaussianBlur(gray, (7, 7), 1.2)

        # Canny thresholds: adaptive-ish based on image intensity stats.
        med = float(np.median(gray))
        lo = int(max(10, 0.66 * med))
        hi = int(min(255, 1.33 * med))
        edges = cv2.Canny(gray, lo, hi, apertureSize=3, L2gradient=True)
        ys_e, xs_e = np.where(edges > 0)
        if xs_e.size < 2000:
            return None

        dx = xs_e.astype(np.float64) - float(cx0)
        dy = ys_e.astype(np.float64) - float(cy0)
        r = np.hypot(dx, dy)
        if not np.isfinite(r).all():
            return None

        # Ignore inner edges; we mainly want the outer reference circle.
        r_min = max(10.0, 0.80 * float(r_min_hint))
        keep = r >= r_min
        if int(np.count_nonzero(keep)) < 1200:
            return None
        r = r[keep]
        dx = dx[keep]
        dy = dy[keep]

        theta = np.arctan2(dy, dx)  # [-pi, pi]
        # Build a coarse radius bin → angular coverage map.
        n_ang = 72  # 5-degree bins
        bin_r = 2.0
        rb = (np.round(r / bin_r) * bin_r).astype(np.float32)
        ang_bin = np.floor((theta + math.pi) / (2.0 * math.pi) * float(n_ang)).astype(np.int32)
        ang_bin = np.clip(ang_bin, 0, n_ang - 1)

        # Dictionary: r_bin -> boolean coverage array.
        cover: Dict[float, np.ndarray] = {}
        counts: Dict[float, int] = {}
        for rv, ab in zip(rb.tolist(), ang_bin.tolist()):
            if rv not in cover:
                cover[rv] = np.zeros((n_ang,), dtype=bool)
                counts[rv] = 0
            cover[rv][ab] = True
            counts[rv] = int(counts[rv]) + 1

        # Pick the largest radius with strong angular coverage.
        best_r = None
        best_score = -1e9
        for rv, cov in cover.items():
            cov_n = int(np.count_nonzero(cov))
            cnt = int(counts.get(rv, 0))
            if cnt < 250:
                continue
            # Require decent angular coverage to avoid selecting localized text.
            if cov_n < int(0.45 * n_ang):
                continue
            score = (3.0 * cov_n) + (0.002 * cnt) + (0.05 * float(rv))
            if score > best_score:
                best_score = float(score)
                best_r = float(rv)

        if best_r is None:
            return None

        # Refine by fitting a circle to edge points near the selected radius.
        tol = max(2.0, 2.0 * bin_r)
        # Recompute r/theta on kept points, and pick those near best_r.
        # (Use original kept dx/dy arrays for speed.)
        r_kept = np.hypot(dx, dy)
        near = np.abs(r_kept - float(best_r)) <= float(tol)
        if int(np.count_nonzero(near)) < 500:
            return None
        pts_edge = np.column_stack([dx[near] + float(cx0), dy[near] + float(cy0)]).astype(np.float64)
        # Downsample for stability/perf.
        if pts_edge.shape[0] > 8000:
            idx = np.random.RandomState(0).choice(pts_edge.shape[0], 8000, replace=False)
            pts_edge = pts_edge[idx]
        try:
            cx_f, cy_f, r_f = _fit_circle_algebraic(pts_edge)
        except Exception:
            # Fallback: median radius w/ original center.
            cx_f, cy_f = float(cx0), float(cy0)
            r_f = float(np.median(r_kept[near]))
        if not (math.isfinite(cx_f) and math.isfinite(cy_f) and math.isfinite(r_f) and r_f > 0.0):
            return None
        return float(cx_f), float(cy_f), float(r_f)

    outer_fit = _estimate_outer_circle_from_dark_ring(arr)
    if outer_fit is None:
        outer_fit = _estimate_outer_circle_from_edges(arr, cx0=float(cx), cy0=float(cy), r_min_hint=float(r_max))
    if outer_fit is not None:
        cx, cy, r_outer = outer_fit
        # Recompute r_max using the refined center.
        d_contour = np.hypot(pts_f[:, 0] - cx, pts_f[:, 1] - cy)
        r_max = float(np.max(d_contour)) if d_contour.size else float(r_max)
    else:
        # Fallback: non-background connected-component heuristic.
        bg = np.asarray(_estimate_background_rgb(img), dtype=np.int16)
        diff_bg = np.abs(arr.astype(np.int16) - bg.reshape(1, 1, 3)).sum(axis=2)
        mask_u8 = (diff_bg > 30).astype(np.uint8) * 255

        k = max(3, int(round(0.004 * float(max(W, H)))))
        k = min(k, 11)
        kernel = np.ones((k, k), dtype=np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)

        comp_mask: np.ndarray
        try:
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
            if int(n_labels) <= 1:
                comp_mask = (mask_u8 > 0)
            else:
                xs_i = np.clip(np.round(pts_f[:, 0]).astype(np.int64), 0, W - 1).astype(np.int32)
                ys_i = np.clip(np.round(pts_f[:, 1]).astype(np.int64), 0, H - 1).astype(np.int32)
                labs = labels[ys_i, xs_i].astype(np.int32)
                labs = labs[labs > 0]
                if labs.size:
                    counts = np.bincount(labs)
                    sel = int(np.argmax(counts))
                else:
                    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
                    sel = 1 + int(np.argmax(areas)) if areas.size else 0
                comp_mask = (labels == sel) if sel > 0 else (mask_u8 > 0)
        except Exception:
            comp_mask = (mask_u8 > 0)

        ys, xs = np.where(comp_mask)
        if xs.size == 0 or ys.size == 0:
            r_outer = float(r_max)
        else:
            d_all = np.hypot(xs.astype(np.float32) - float(cx), ys.astype(np.float32) - float(cy))
            try:
                r_outer = float(np.percentile(d_all, 99.5))
            except Exception:
                r_outer = float(np.max(d_all))
            if not math.isfinite(r_outer) or r_outer <= 0.0:
                r_outer = float(r_max)

    if r_outer < r_max:
        r_outer = float(r_max)

    center_x = int(round(cx))
    center_y = int(round(cy))

    # 4) Visualization.
    vis = arr.copy()
    cnt_i32 = pts_f.reshape(-1, 1, 2).astype(np.int32)
    try:
        cv2.drawContours(vis, [cnt_i32], -1, (0, 200, 255), 2)
    except Exception:
        pass
    cv2.circle(vis, (center_x, center_y), max(2, int(round(0.008 * float(max(W, H))))), (255, 0, 0), -1)
    cv2.circle(vis, (center_x, center_y), int(round(r_max)), (255, 0, 0), 2)
    cv2.circle(vis, (center_x, center_y), int(round(r_outer)), (0, 200, 0), 3)
    vis_img = Image.fromarray(vis)

    return vis_img, int(center_x), int(center_y), float(r_outer), float(r_max)


def estimate_radial_value(
    image: Image.Image,
    center_x: int,
    center_y: int,
    r_outer: int,
    r_max: int,
    reference_circle_value: float = 100,
) -> float:
    """
    Estimate the value of a radial segment by scaling its radial length relative to an outer reference circle.

    Formula:
      value = reference_circle_value * (r_max / r_outer)
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")

    try:
        r_outer_f = float(r_outer)
        r_max_f = float(r_max)
        ref_f = float(reference_circle_value)
    except Exception as e:
        raise TypeError("center_x/center_y/r_outer/r_max/reference_circle_value must be numeric") from e

    if not math.isfinite(r_outer_f) or r_outer_f <= 0.0:
        raise ValueError("r_outer must be > 0")
    if not math.isfinite(r_max_f) or r_max_f < 0.0:
        raise ValueError("r_max must be >= 0")
    if not math.isfinite(ref_f):
        raise ValueError("reference_circle_value must be finite")

    # center_x/center_y are currently not used in this estimation, but validated to keep schema consistent.
    _ = int(center_x)
    _ = int(center_y)

    return float(ref_f * (r_max_f / r_outer_f))
