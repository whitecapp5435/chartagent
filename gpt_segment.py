"""
segment_and_mark.py

Segment an image with SAM (ViT-H), clean masks with several filters
(small / duplicate / composite / background / text / axis-like),
and draw contours + numbered labels.

Dependencies:
    pip install opencv-python-headless numpy easyocr torch torchvision torchaudio
    # For SAM (official repo):
    pip install git+https://github.com/facebookresearch/segment-anything.git

Usage example (after filling in SAM checkpoint path):

    python segment_and_mark.py input.png output.png

Or import and call segment_and_mark() from your own code.
"""

import sys
import os
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import torch
import easyocr

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from test2 import LegendDetectorEasyOCR

SAVE_MASKS = True

# Workaround for a PyTorch CUDA bug triggered inside SAM's RLE conversion
# (mask_to_rle_pytorch -> diff.nonzero on CUDA boolean tensors).
# We monkey-patch the function to run on CPU to avoid the crash.
try:
    import segment_anything.automatic_mask_generator as _sam_auto
    from segment_anything.utils import amg as _sam_amg

    _orig_mask_to_rle = _sam_amg.mask_to_rle_pytorch

    def _safe_mask_to_rle_pytorch(tensor: torch.Tensor):
        try:
            # Ensure operation occurs on CPU to avoid CUDA nonzero assert
            return _orig_mask_to_rle(tensor.detach().to("cpu"))
        except Exception as e:
            # Fall back: try original (helps if CPU move fails for some reason)
            return _orig_mask_to_rle(tensor)

    # Patch both the utility module and the automatic mask generator's reference
    _sam_amg.mask_to_rle_pytorch = _safe_mask_to_rle_pytorch  # type: ignore
    _sam_auto.mask_to_rle_pytorch = _safe_mask_to_rle_pytorch  # type: ignore
    print("[INFO] Patched SAM mask_to_rle_pytorch to run on CPU.")
except Exception as _patch_err:
    print(f"[WARN] Could not patch SAM RLE function: {_patch_err}")


# ---------------------------------------------------------
# 1. Utility functions
# ---------------------------------------------------------

def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Compute IoU between two boolean masks.
    mask_a, mask_b: (H, W) boolean arrays
    """
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return inter / union


def is_background_dominated(
    mask: np.ndarray,
    image_bgr: np.ndarray,
    brightness_thresh: int = 220,
    edge_ratio_thresh: float = 0.001,
) -> bool:
    """
    Heuristic: a mask is 'background-dominated' if
    - the mean brightness is very high (almost white), AND
    - there are very few edges inside the mask.
    """
    if mask.sum() == 0:
        return True

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    region_vals = gray[mask]
    mean_val = float(region_vals.mean())

    # If not bright enough, we don't treat it as background.
    if mean_val < brightness_thresh:
        return False

    # Edge density check
    edges = cv2.Canny(gray, 100, 200)
    edge_in_mask = edges[mask] > 0
    edge_ratio = edge_in_mask.sum() / float(mask.sum())

    return edge_ratio < edge_ratio_thresh


def build_text_mask(
    image_shape: Tuple[int, int, int],
    ocr_boxes: List[Tuple[int, int, int, int]],
) -> np.ndarray:
    """
    Build a boolean mask of 'text regions' from OCR bounding boxes.
    image_shape: (H, W, C)
    ocr_boxes: list of (x_min, y_min, x_max, y_max)
    """
    H, W = image_shape[:2]
    text_mask = np.zeros((H, W), dtype=bool)

    for (x_min, y_min, x_max, y_max) in ocr_boxes:
        # 약간 확장해서 텍스트 주변 영역까지 포함
        w_box = x_max - x_min
        h_box = y_max - y_min
        expand_x = int(0.15 * w_box)
        expand_y = int(0.3 * h_box)

        x_min = max(0, int(x_min) - expand_x)
        y_min = max(0, int(y_min) - expand_y)
        x_max = min(W, int(x_max) + expand_x)
        y_max = min(H, int(y_max) + expand_y)
        if x_max <= x_min or y_max <= y_min:
            continue
        text_mask[y_min:y_max, x_min:x_max] = True

    return text_mask


def is_axis_like(
    mask: np.ndarray,
    bbox: List[int],
    image_shape: Tuple[int, int, int],
    aspect_ratio_thresh: float = 10.0,
    margin_ratio: float = 0.05,
    max_area_ratio: float = 0.2,
) -> bool:
    """
    Heuristic: decide if a mask looks like an axis (long thin line near border).

    bbox: [x, y, w, h] from SAM
    """
    H, W = image_shape[:2]
    x, y, w, h = bbox

    if w <= 0 or h <= 0:
        return False

    area = int(mask.sum())
    if area == 0:
        return False

    # Too large overall? then unlikely to be a single axis line
    if area > max_area_ratio * H * W:
        return False

    aspect = max(w / h, h / w)

    # Must be very long and thin
    if aspect < aspect_ratio_thresh:
        return False

    # Near any of the borders
    near_left = x < margin_ratio * W
    near_right = (x + w) > (1.0 - margin_ratio) * W
    near_bottom = (y + h) > (1.0 - margin_ratio) * H
    near_top = y < margin_ratio * H

    return near_left or near_right or near_bottom or near_top


def is_gridline_like(
    mask: np.ndarray,
    bbox: List[int],
    image_bgr: np.ndarray,
    min_length_ratio: float = 0.5,
    max_thickness_px: int = 4,
) -> bool:
    """
    Heuristic: horizontal/vertical grid line inside the plotting area.
    Long thin, low-saturation, bright line, not restricted to image border.
    """
    H, W = image_bgr.shape[:2]
    x, y, w, h = bbox

    if w <= 0 or h <= 0:
        return False

    area = int(mask.sum())
    if area == 0:
        return False

    # Determine dominant orientation
    horiz = (w >= h)
    length = w if horiz else h
    thickness = h if horiz else w

    # Very long relative to image size, but extremely thin
    length_ratio = length / float(W if horiz else H)
    if length_ratio < min_length_ratio:
        return False
    if thickness > max_thickness_px:
        return False

    # Color check: grid lines are usually light / low-saturation
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1][mask]
    v = hsv[..., 2][mask]
    if s.size == 0 or v.size == 0:
        return False

    mean_s = float(s.mean())
    mean_v = float(v.mean())

    # Require fairly bright and weakly saturated (greyish)
    if mean_v < 180:
        return False
    if mean_s > 60:
        return False

    return True


# ---------------------------------------------------------
# 2. OCR (EasyOCR) wrapper
# ---------------------------------------------------------

def create_easyocr_reader(langs: List[str] = None, gpu: bool = False) -> easyocr.Reader:
    """
    Create an EasyOCR reader.
    langs: language codes (e.g., ['en'], ['en', 'ko'], ...)
    gpu: GPU 사용 여부 (기본 False - 메모리 안정성)
    """
    if langs is None:
        langs = ['en']
    # GPU 사용 시 메모리 충돌 위험 - 기본은 CPU 사용
    reader = easyocr.Reader(langs, gpu=gpu)
    return reader


def run_ocr_to_boxes(
    image_bgr: np.ndarray,
    reader: easyocr.Reader,
    min_confidence: float = 0.4,
) -> List[Tuple[int, int, int, int]]:
    """
    Run EasyOCR and return bounding boxes.
    Output: list of (x_min, y_min, x_max, y_max)
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = reader.readtext(image_rgb)  # each = [box, text, conf]

    boxes = []
    for item in results:
        if len(item) != 3:
            continue
        box, text, conf = item
        if conf < min_confidence:
            continue

        # box: 4 points [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        boxes.append((x_min, y_min, x_max, y_max))

    return boxes


# ---------------------------------------------------------
# 2-1. Chart cleaning (title / legend 제거)
# ---------------------------------------------------------

def clean_chart_image(
    image_bgr: np.ndarray,
    reader: easyocr.Reader,
    min_confidence: float = 0.4,
    title_region_height_ratio: float = 0.2,
    legend_region_left_ratio: float = 0.7,
    fill_color: Tuple[int, int, int] = (255, 255, 255),
) -> Tuple[np.ndarray, Dict[str, Optional[Tuple[int, int, int, int]]]]:
    """
    Detect and remove chart title and legend regions.

    Heuristics (matplotlib 스타일 차트 가정):
    - Title: 상단부 (height * title_region_height_ratio) 안에 있고,
      가로로 중앙(0.2W ~ 0.8W)에 위치한 텍스트 박스들의 합집합.
    - Legend: 오른쪽 영역 (x_center > legend_region_left_ratio * W)에
      위치한 텍스트 박스들의 합집합 + 그 왼쪽으로 일정 margin (legend marker 포함).
    """
    H, W = image_bgr.shape[:2]

    # 차트 본문이 시작되는 y 위치를 대략 추정
    # (위쪽에서부터 non-white 픽셀 비율이 충분히 큰 첫 번째 행)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    chart_top_y: Optional[int] = None
    for y in range(H):
        row = gray[y]
        # 거의 흰색(배경)이 아닌 픽셀 비율
        nonwhite_ratio = np.count_nonzero(row < 245) / float(W)
        if nonwhite_ratio > 0.02:
            chart_top_y = y
            break

    # 1) OCR 기반 타이틀 / 축 라벨 추정
    boxes = run_ocr_to_boxes(image_bgr, reader, min_confidence=min_confidence)
    title_candidates: List[Tuple[int, int, int, int]] = []
    x_axis_candidates: List[Tuple[int, int, int, int]] = []
    y_axis_candidates: List[Tuple[int, int, int, int]] = []
    legend_heights: List[int] = []

    for (x_min, y_min, x_max, y_max) in boxes:
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)

        # Title: 상단부 + 중앙 근처
        if (
            cy < title_region_height_ratio * H
            and 0.2 * W < cx < 0.8 * W
        ):
            title_candidates.append((x_min, y_min, x_max, y_max))

        # X-axis label / tick labels: 하단 영역
        if cy > 0.82 * H:
            x_axis_candidates.append((x_min, y_min, x_max, y_max))

        # Y-axis label / tick labels: 좌측 영역
        if cx < 0.2 * W and 0.1 * H < cy < 0.9 * H:
            y_axis_candidates.append((x_min, y_min, x_max, y_max))

    def _union_box(
        candidates: List[Tuple[int, int, int, int]]
    ) -> Optional[Tuple[int, int, int, int]]:
        if not candidates:
            return None
        xs_min = min(b[0] for b in candidates)
        ys_min = min(b[1] for b in candidates)
        xs_max = max(b[2] for b in candidates)
        ys_max = max(b[3] for b in candidates)
        return int(xs_min), int(ys_min), int(xs_max), int(ys_max)

    title_box = _union_box(title_candidates)
    x_axis_box = _union_box(x_axis_candidates)
    y_axis_box = _union_box(y_axis_candidates)

    # 2) LegendDetectorEasyOCR 기반 범례 탐지 (test2.py 재사용)
    legend_box = None
    legend_from_detector = False
    try:
        legend_detector = LegendDetectorEasyOCR(image=image_bgr, reader=reader, gpu=False)
        legend_bbox_xywh, _ = legend_detector.detect_legend(debug=False)
        if legend_bbox_xywh is not None:
            lx, ly, lw, lh = legend_bbox_xywh
            legend_box = (lx, ly, lx + lw, ly + lh)
            legend_from_detector = True
    except Exception as _legend_err:
        # fallback 은 아래 OCR-heuristic 으로 처리
        legend_box = None

    # 2-1) LegendDetector 가 실패했으면, 간단한 OCR 기반 오른쪽 영역 휴리스틱 사용
    if legend_box is None and boxes:
        legend_candidates: List[Tuple[int, int, int, int]] = []
        for (x_min, y_min, x_max, y_max) in boxes:
            cx = 0.5 * (x_min + x_max)
            cy = 0.5 * (y_min + y_max)
            if (
                cx > legend_region_left_ratio * W
                and 0.1 * H < cy < 0.9 * H
            ):
                legend_candidates.append((x_min, y_min, x_max, y_max))
                legend_heights.append(int(y_max - y_min))
        legend_box = _union_box(legend_candidates)

    cleaned = image_bgr.copy()

    # Title 영역 제거 (조금 padding 추가)
    if title_box is not None:
        tx1, ty1, tx2, ty2 = title_box
        pad_y = int(0.02 * H)
        pad_x = int(0.02 * W)
        tx1 = max(0, tx1 - pad_x)
        ty1 = max(0, ty1 - pad_y)
        tx2 = min(W, tx2 + pad_x)
        ty2 = min(H, ty2 + pad_y)

        # 차트 시작 y 위치를 넘어가지 않도록 하단을 제한
        if chart_top_y is not None:
            ty2 = min(ty2, max(0, chart_top_y - pad_y))

        if ty2 > ty1:
            cleaned[ty1:ty2, tx1:tx2] = fill_color

    # X-axis 라벨/눈금 영역 제거 (하단 좁은 띠)
    if x_axis_box is not None:
        xx1, xy1, xx2, xy2 = x_axis_box
        pad_y = int(0.02 * H)
        xx1 = max(0, xx1 - int(0.02 * W))
        xx2 = min(W, xx2 + int(0.02 * W))
        xy1 = max(0, xy1 - pad_y)
        xy2 = min(H, xy2 + pad_y)
        cleaned[xy1:xy2, xx1:xx2] = fill_color

    # Y-axis 라벨/눈금 영역 제거 (좌측 좁은 띠)
    if y_axis_box is not None:
        yx1, yy1, yx2, yy2 = y_axis_box
        pad_x = int(0.02 * W)
        yx1 = max(0, yx1 - pad_x)
        yx2 = min(W, yx2 + pad_x)
        yy1 = max(0, yy1 - int(0.02 * H))
        yy2 = min(H, yy2 + int(0.02 * H))
        cleaned[yy1:yy2, yx1:yx2] = fill_color

    # Legend 영역 제거
    if legend_box is not None:
        lx1, ly1, lx2, ly2 = legend_box
        pad_y = int(0.02 * H)

        if legend_from_detector:
            # test2.LegendDetectorEasyOCR가 이미 마커+텍스트를 포함하는
            # 꽤 타이트한 bbox 를 주므로, 너무 크게 확장하지 않고
            # 소량의 padding 만 적용해 차트 본문을 침범하지 않도록 한다.
            pad_x = int(0.02 * W)
            lx1 = max(0, lx1 - pad_x)
            lx2 = min(W, lx2 + pad_x)
        else:
            # 단순 OCR 휴리스틱으로 추정한 경우에는
            # legend marker 가 텍스트 왼쪽 3~4*h 근처에 있는 경우가 많으므로
            # 왼쪽으로 조금 더 확장해 marker 를 함께 지운다.
            if legend_heights:
                max_h = max(legend_heights)
            else:
                max_h = ly2 - ly1
            extra_left = int(3.0 * max_h)
            lx1 = max(0, lx1 - extra_left)

        ly1 = max(0, ly1 - pad_y)
        lx2 = min(W, lx2 + int(0.02 * W))
        ly2 = min(H, ly2 + pad_y)
        cleaned[ly1:ly2, lx1:lx2] = fill_color

    return cleaned, {
        "title_box": title_box,
        "legend_box": legend_box,
        "x_axis_box": x_axis_box,
        "y_axis_box": y_axis_box,
    }


# ---------------------------------------------------------
# 3. SAM mask filtering
# ---------------------------------------------------------

def filter_sam_masks(
    sam_masks: List[Dict],
    image_bgr: np.ndarray,
    min_area_ratio: float = 0.001,
    iou_dup_thresh: float = 0.9,
    composite_iou_thresh: float = 0.95,
    text_mask: Optional[np.ndarray] = None,
    text_overlap_thresh: float = 0.5,
    drop_axis_like: bool = True,
    ocr_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    drop_legend_like: bool = False,
) -> List[Dict]:
    """
    Extended cleaning pipeline:
    1) Remove too-small masks.
    2) Remove near-duplicate masks.
    3) Remove composite masks.
    4) Remove background-dominated masks.
    5) (optional) Remove text-dominated masks (axis title, chart title, labels).
    6) (optional) Remove axis-like elongated masks near borders.
    7) (optional) Remove legend-like markers next to OCR text (오른쪽 legend).
    8) Drop extremely large 'global' masks (e.g., donut ring made of many dots).
    """
    H, W = image_bgr.shape[:2]
    img_area = H * W

    # Step 1: size filter (너무 작은 것 제거)
    large_enough = [
        m for m in sam_masks
        if m.get("area", 0) >= min_area_ratio * img_area
    ]
    if not large_enough:
        return []

    # area 큰 것부터 정렬
    large_enough.sort(key=lambda m: m.get("area", 0), reverse=True)

    # 이전에는 median area 를 기반으로 '너무 큰 전역 링/배경' 마스크를
    # 직접 제거했지만, 파이/도넛 조각 자체를 잘못 제거하는 경우가 있어
    # 현재는 border-text 필터에서만 참조용으로 사용한다.
    areas_for_median = np.array(
        [m.get("area", 0) for m in large_enough], dtype=np.float32
    )
    median_area = float(np.median(areas_for_median)) if areas_for_median.size > 0 else 0.0

    kept: List[Dict] = []

    # Step 2: near-duplicate filter
    for m in large_enough:
        seg = m["segmentation"].astype(bool)
        is_duplicate = False
        for km in kept:
            iou = mask_iou(seg, km["segmentation"])
            if iou > iou_dup_thresh:
                is_duplicate = True
                break
        if is_duplicate:
            continue
        kept.append(m)

    # Step 3: composite / global-mask filter
    # 여러 개의 더 작은 마스크들을 거의 완전히 포함하는 큰 마스크
    # (예: 도넛 전체 링, 혹은 인접한 여러 wedge 를 한 번에 덮는 마스크)
    # 는 composite 으로 보고 제거한다.
    non_composite: List[Dict] = []
    for i, m in enumerate(kept):
        seg = m["segmentation"].astype(bool)
        area_m = seg.sum()
        if area_m == 0:
            continue

        contain_count = 0
        for j, other in enumerate(kept):
            if i == j:
                continue
            other_seg = other["segmentation"].astype(bool)
            area_o = other_seg.sum()
            if area_o == 0 or area_o >= area_m:
                continue

            inter = np.logical_and(seg, other_seg).sum()
            # 다른 마스크를 거의 완전히 포함하면 composite 후보로 센다
            if inter / float(area_o) > 0.98:
                contain_count += 1
                if contain_count >= 2:
                    break

        if contain_count >= 2:
            # 둘 이상의 작은 마스크를 포함하는 전역/합성 마스크 → 제거
            continue

        non_composite.append(m)

    final_masks: List[Dict] = []

    # Step 4~7: background / text / axis / legend / grid / border-text
    for m in non_composite:
        seg = m["segmentation"].astype(bool)
        area = seg.sum()
        if area == 0:
            continue

        # 한 마스크 안에 여러 개의 분리된 blob 이 있으면
        # 가장 큰 연결 요소만 남기고 작은 노이즈는 제거.
        seg_uint8 = seg.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            seg_uint8, connectivity=8
        )
        if num_labels > 2:
            # label 0 은 background 이므로 제외
            comp_areas = stats[1:, cv2.CC_STAT_AREA]
            largest_label = 1 + int(np.argmax(comp_areas))
            seg = (labels == largest_label)
            area = seg.sum()
            if area == 0:
                continue
            m["segmentation"] = seg.astype(np.uint8)

        # 4) background-dominated filter
        if is_background_dominated(seg, image_bgr):
            continue

        # 5) text-dominated filter (axis title, chart title 등 제거)
        if text_mask is not None:
            overlap_ratio = np.logical_and(seg, text_mask).sum() / float(area)
            if overlap_ratio > text_overlap_thresh:
                continue

        # 6) axis-like / grid-like filter (긴 축 제거 + 내부 그리드 제거)
        if "bbox" in m:
            x, y, w, h = m["bbox"]

            if drop_axis_like and is_axis_like(seg, m["bbox"], image_bgr.shape):
                continue
            if is_gridline_like(seg, m["bbox"], image_bgr):
                continue

            # 6-1) border small-text filter:
            # 매우 작은 마스크가 이미지 가장자리 근처에 있으면
            # 축 눈금/축 라벨 조각일 가능성이 높으므로 제거.
            area_ratio = area / float(img_area)
            near_left = x < 0.06 * W
            near_right = (x + w) > 0.94 * W
            near_bottom = (y + h) > 0.94 * H
            near_top = y < 0.06 * H
            if area < 0.2 * median_area and (near_left or near_right or near_bottom or near_top):
                continue

        # 7) legend-like filter: 오른쪽에 있고 OCR 텍스트 옆에 붙은 작은 마커 제거
        if drop_legend_like and ocr_boxes is not None and "bbox" in m:
            x, y, w, h = m["bbox"]
            cx = x + 0.5 * w
            cy = y + 0.5 * h

            # 그림 오른쪽 20% 영역만 legend 후보로 본다
            if cx > 0.8 * W:
                is_legend = False
                for (tx1, ty1, tx2, ty2) in ocr_boxes:
                    tx_min, tx_max = min(tx1, tx2), max(tx1, tx2)
                    ty_min, ty_max = min(ty1, ty2), max(ty1, ty2)
                    text_cy = 0.5 * (ty_min + ty_max)

                    # 수직 위치가 비슷하고, 텍스트 왼쪽 근처에 있으면 legend marker 로 간주
                    if abs(cy - text_cy) < 1.5 * h and (tx_min - 4*h) <= cx <= tx_min + 1.5*h:
                        is_legend = True
                        break
                if is_legend:
                    continue

        final_masks.append(m)

    # Step 8: 너무 작은 잔여 마스크 제거
    if not final_masks:
        return final_masks

    areas_final = np.array(
        [m["segmentation"].astype(bool).sum() for m in final_masks],
        dtype=np.float32,
    )
    median_area_final = float(np.median(areas_final))
    # median 의 5% 미만인 작은 마스크는 노이즈로 간주
    min_keep_area = max(5.0, 0.05 * median_area_final)

    filtered_final: List[Dict] = []
    for m, a in zip(final_masks, areas_final):
        if a >= min_keep_area:
            filtered_final.append(m)

    return filtered_final

# ---------------------------------------------------------
# 4. Main segmentation + drawing function
# ---------------------------------------------------------

def segment_and_mark(
    image_bgr: np.ndarray,
    mask_generator: SamAutomaticMaskGenerator,
    ocr_reader: Optional[easyocr.Reader] = None,
    draw_labels: bool = True,
    contour_thickness: int = 2,
    font_scale: float = 0.5,
    font_thickness: int = 1,
    text_overlap_thresh: float = 0.5,
) -> Tuple[np.ndarray, List[np.ndarray], List[Dict]]:
    """
    Segment an input image using SAM (mask_generator),
    clean the masks (small, duplicate, composite, background, text, axis),
    and draw contours + (optional) numbered labels.

    Returns:
        vis_image_bgr: original image with drawn contours and labels.
        cleaned_masks: list of boolean (H, W) masks.
        cleaned_mask_dicts: original SAM mask dicts (filtered).
    """

    # 1) SAM segmentation (expects RGB)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    raw_masks: List[Dict] = mask_generator.generate(image_rgb)

    if len(raw_masks) == 0:
        return image_bgr.copy(), [], []

    # 2) OCR -> text_mask
    if ocr_reader is not None:
        ocr_boxes = run_ocr_to_boxes(image_bgr, ocr_reader)
        text_mask = build_text_mask(image_bgr.shape, ocr_boxes)
    else:
        ocr_boxes = None
        text_mask = None

    # 3) Filter masks
    cleaned_mask_dicts = filter_sam_masks(
        raw_masks,
        image_bgr,
        min_area_ratio=0.00005,
        text_mask=text_mask,
        # 텍스트와 조금만 겹쳐도 텍스트로 간주해서 제거
        text_overlap_thresh=0.3,
        drop_axis_like=True,
        ocr_boxes=ocr_boxes,
        drop_legend_like=True,
    )

    # 만약 필터가 너무 aggressive 해서 하나도 남지 않았다면,
    # 자동으로 더 느슨한 설정으로 한 번 더 필터링해서
    # 최소한 몇 개의 마스크는 남도록 한다.
    if len(cleaned_mask_dicts) == 0 and len(raw_masks) > 0:
        cleaned_mask_dicts = filter_sam_masks(
            raw_masks,
            image_bgr,
            min_area_ratio=0.00002,
            text_mask=None,              # 텍스트/legend 기반 삭제 끔
            text_overlap_thresh=0.9,
            drop_axis_like=False,
            ocr_boxes=None,
            drop_legend_like=False,
        )

    cleaned_masks = [m["segmentation"].astype(bool) for m in cleaned_mask_dicts]

    # 4) Draw contours + labels
    vis = image_bgr.copy()

    for idx, mask in enumerate(cleaned_masks, start=1):
        mask_uint8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        # Color: deterministic pseudo-random by index
        color = (
            (37 * idx) % 255,
            (97 * idx) % 255,
            (173 * idx) % 255,
        )
        cv2.drawContours(vis, contours, -1, color, contour_thickness)

        if draw_labels:
            largest_cnt = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
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

    return vis, cleaned_masks, cleaned_mask_dicts


# ---------------------------------------------------------
# 5. SAM model loader
# ---------------------------------------------------------

def load_sam_vith(
    checkpoint_path: str,
    device: Optional[str] = None,
) -> SamAutomaticMaskGenerator:
    """
    Load SAM ViT-H model and create a SamAutomaticMaskGenerator.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=64,
        crop_n_layers=1,
        box_nms_thresh=0.9,
        pred_iou_thresh=0.80,
        stability_score_thresh=0.85,
        min_mask_region_area=0,  # small junk removed early
    )
    return mask_generator


# ---------------------------------------------------------
# 6. Simple CLI example
# ---------------------------------------------------------

def main():

    print("[INFO] Starting segment_and_mark example...")

    IMAGE = '3'
    sam_ckpt = './sam_vit_h_4b8939.pth'
    for i in range(1,4):
        IMAGE = i
        in_path = f'./legend_test/{IMAGE}.png'
        out_path = f'./seg_output/{IMAGE}_output.png'
        clean_path = f'./seg_output/{IMAGE}_clean.png'

        image_bgr = cv2.imread(in_path)
        if image_bgr is None:
            print(f"Failed to read image: {in_path}")
            sys.exit(1)

        print("[INFO] Creating EasyOCR reader...")
        # 필요 언어에 맞게 수정 (예: ['en', 'ko'])
        ocr_reader = create_easyocr_reader(['en'])

        print("[INFO] Cleaning chart (title / legend 제거)...")
        cleaned_bgr, _ = clean_chart_image(image_bgr, ocr_reader)
        cv2.imwrite(clean_path, cleaned_bgr)
        print(f"[INFO] Saved cleaned chart to: {clean_path}")

        print("[INFO] Loading SAM (ViT-H)...")
        mask_generator = load_sam_vith(sam_ckpt)

        print("[INFO] Running segment_and_mark...")
        vis, cleaned_masks, _ = segment_and_mark(
            cleaned_bgr,
            mask_generator,
            ocr_reader=ocr_reader,
            draw_labels=True,
        )

        print(f"[INFO] # of cleaned masks: {len(cleaned_masks)}")

        if SAVE_MASKS and len(cleaned_masks) > 0:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            # 개별 마스크 저장 + 전체 union 마스크 저장
            union_mask = np.zeros_like(cleaned_masks[0], dtype=np.uint8)
            for idx, mask in enumerate(cleaned_masks, start=1):
                mask_uint8 = (mask.astype(np.uint8) * 255)
                union_mask = np.logical_or(union_mask > 0, mask_uint8 > 0).astype(
                    np.uint8
                ) * 255
                os.makedirs(f"./seg_output/{IMAGE}", exist_ok=True)
                mask_path = f"./seg_output/{IMAGE}/{IMAGE}_mask_{idx:03d}.png"
                cv2.imwrite(mask_path, mask_uint8)
            union_path = f"./seg_output/{IMAGE}/{IMAGE}_mask_all.png"
            cv2.imwrite(union_path, union_mask)
            print(f"[INFO] Saved masks to: ./seg_output/{IMAGE}/{IMAGE}_mask_###.png and _mask_all.png")

        cv2.imwrite(out_path, vis)
        print(f"[INFO] Saved result to: {out_path}")


if __name__ == "__main__":
    main()
