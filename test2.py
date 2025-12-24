import cv2
import numpy as np
import easyocr
import os
import re

from legend_detector.chartreader_port import (
    detect_axes,
    merge_text_boxes,
    group_aligned,
    is_numeric_text,
    filter_left_color_boxes,
)

output_dir = "./out"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class LegendDetectorEasyOCR:
    """EasyOCR 기반 범례 탐지기 - 더 정확한 텍스트 인식

    Note:
        - 기존에는 image_path만 받아 내부에서 항상 EasyOCR Reader를 새로 생성했으나,
          gpt_segment 등 다른 모듈에서 이미 생성한 Reader와 이미지 배열을 재사용할 수 있도록
          image / reader 인젝션도 지원한다.
    """
    
    def __init__(
        self,
        image_path: str = None,
        image: np.ndarray = None,
        reader: easyocr.Reader = None,
        gpu: bool = True,
    ):
        # 이미지 소스: 우선순위 image (ndarray) > image_path
        if image is not None:
            self.image = image.copy()
        elif image_path is not None:
            self.image = cv2.imread(image_path)
            if self.image is None:
                raise ValueError(f"Failed to read image: {image_path}")
        else:
            raise ValueError("Either image_path or image must be provided.")

        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.image.shape[:2]
        
        # EasyOCR 초기화 (한 번만)
        if reader is not None:
            self.reader = reader
            print("  EasyOCR 재사용 (외부에서 주입된 Reader)")
        else:
            print("  EasyOCR 초기화 중...")
            self.reader = easyocr.Reader(['en'], gpu=gpu)
            print("  EasyOCR 준비 완료")
    
    def find_color_markers(self):
        """작은 색상 마커들 찾기"""
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        _, saturation, value = cv2.split(hsv)

        sat_mask = cv2.inRange(saturation, 16, 255)
        val_mask = cv2.inRange(value, 31, 255)
        color_mask = cv2.bitwise_and(sat_mask, val_mask)
        
        kernel = np.ones((3, 3), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        markers = []
        
        # 과검출 억제: 소형·정사각·실체(솔리디티) 조건 + 하단 범례 특례
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h

            # 너무 작은 점 제거 (6x6)
            if area < 36:
                continue

            # 큰 막대/영역 제거 (이미지의 1% 초과)
            if area > self.width * self.height * 0.01:
                continue

            # 정사각/근사 사각 비율 (기본)
            ar = w / float(h) if h > 0 else 999.0
            
            # 솔리디티(채움 정도)로 선/윤곽 억제
            hull = cv2.convexHull(cnt)
            cnt_area = cv2.contourArea(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = cnt_area / (hull_area + 1e-6) if hull_area > 0 else 0

            cy = y + h / 2.0

            # 1) 기본: 정사각/근사 사각 + 실체
            square_like = (0.55 <= ar <= 2.6 and solidity >= 0.75)

            # 2) 하단 범례 특례: 얇은 라인 샘플(가로로 길쭉, 매우 얇음)
            line_like_bottom = (
                cy > self.height * 0.55 and   # 하단 영역에서만 허용
                h <= 18 and                   # 얇은 선/샘플
                w <= 200 and                  # 너무 긴 축/격자선 제외
                6.0 <= ar <= 80.0 and         # 범례용 짧은 라인 비율 허용
                solidity >= 0.85
            )

            # 3) 하단 범례 특례: 직사각 패치(뉴욕 바 샘플)
            rect_patch_bottom = (
                cy > self.height * 0.55 and
                h <= 48 and w <= 200 and
                1.2 <= ar <= 5.0 and          # 작은 가로 직사각
                solidity >= 0.85
            )

            if not (square_like or line_like_bottom or rect_patch_bottom):
                continue

            markers.append({
                'bbox': (x, y, w, h),
                'center': (x + w//2, y + h//2),
                'area': area
            })
        
        print(f"  색상 마커 발견: {len(markers)}개")
        return markers
    
    def find_aligned_marker_groups(self, markers, min_group_size=2):
        """정렬된 마커 그룹 찾기"""
        if len(markers) < min_group_size:
            return []
        
        groups = []
        horizontal_groups = self._find_horizontal_groups(markers, y_tolerance=60)
        vertical_groups = self._find_vertical_groups(markers, x_tolerance=90)
        
        groups.extend(horizontal_groups)
        groups.extend(vertical_groups)
        
        valid_groups = [g for g in groups if len(g) >= min_group_size]
        print(f"  정렬된 마커 그룹: {len(valid_groups)}개")
        
        return valid_groups
    
    def _find_horizontal_groups(self, markers, y_tolerance=60):
        """y 좌표가 비슷하고 x 간격이 너무 벌어지지 않는 마커들 그룹화"""
        if not markers:
            return []

        # 1) y 레벨별 1차 그룹화
        sorted_by_y = sorted(markers, key=lambda m: m['center'][1])
        y_groups = []
        current = [sorted_by_y[0]]
        for i in range(1, len(sorted_by_y)):
            if abs(sorted_by_y[i]['center'][1] - sorted_by_y[i-1]['center'][1]) < y_tolerance:
                current.append(sorted_by_y[i])
            else:
                if len(current) >= 2:
                    y_groups.append(current)
                current = [sorted_by_y[i]]
        if len(current) >= 2:
            y_groups.append(current)

        # 2) 각 y-그룹을 x 간격 기준으로 세분화
        groups = []
        for g in y_groups:
            g_sorted = sorted(g, key=lambda m: m['center'][0])
            widths = [m['bbox'][2] for m in g_sorted]
            med_w = np.median(widths) if widths else 12
            # x 갭 임계: 해상도/마커 크기 기반
            gap_thr = min(int(0.3 * self.width), max(80, int(12 * med_w)))

            sub = [g_sorted[0]]
            for i in range(1, len(g_sorted)):
                if (g_sorted[i]['center'][0] - g_sorted[i-1]['center'][0]) <= gap_thr:
                    sub.append(g_sorted[i])
                else:
                    if len(sub) >= 2:
                        groups.append(sub)
                    sub = [g_sorted[i]]
            if len(sub) >= 2:
                groups.append(sub)

        # 3) 너무 넓게 퍼진 그룹 드랍 (축/격자/플롯 영역 억제)
        filtered = []
        max_span = int(0.45 * self.width)
        for g in groups:
            xs = [m['center'][0] for m in g]
            span = max(xs) - min(xs)
            if span <= max_span:
                filtered.append(g)
        return filtered
    
    def _find_vertical_groups(self, markers, x_tolerance=90):
        """x 좌표가 비슷한 마커들 그룹화"""
        if not markers:
            return []
        
        sorted_markers = sorted(markers, key=lambda m: m['center'][0])
        groups = []
        current_group = [sorted_markers[0]]
        
        for i in range(1, len(sorted_markers)):
            if abs(sorted_markers[i]['center'][0] - sorted_markers[i-1]['center'][0]) < x_tolerance:
                current_group.append(sorted_markers[i])
            else:
                if len(current_group) >= 2:
                    groups.append(current_group)
                current_group = [sorted_markers[i]]
        
        if len(current_group) >= 2:
            groups.append(current_group)
        
        return groups
    
    def get_marker_region(self, marker_group):
        """마커 그룹의 영역 계산"""
        xs = [m['center'][0] for m in marker_group]
        ys = [m['center'][1] for m in marker_group]
        
        x_min = min([m['bbox'][0] for m in marker_group])
        y_min = min([m['bbox'][1] for m in marker_group])
        x_max = max([m['bbox'][0] + m['bbox'][2] for m in marker_group])
        y_max = max([m['bbox'][1] + m['bbox'][3] for m in marker_group])
        
        # 배치 방향 판단
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        is_horizontal = x_range > y_range * 1.5
        
        # 텍스트 영역까지 확장
        if is_horizontal:
            # 가로 배치: x 확장은 충분히, 과도한 확장 제한
            expand_x = min(max(340, x_range), int(0.35 * self.width))
            expand_y = 26  # 40 -> 26
        else:
            expand_x = 400
            expand_y = max(60, y_range // len(marker_group))
        
        x_min = max(0, x_min - 30)
        y_min = max(0, y_min - expand_y)
        x_max = min(self.width, x_max + expand_x)
        y_max = min(self.height, y_max + expand_y)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min), is_horizontal
    
    def find_texts_in_region(self, region_bbox, marker_group=None, is_horizontal=True):
        """EasyOCR로 영역 내 텍스트 찾기"""
        try:
            rx, ry, rw, rh = region_bbox
            
            # 영역 크롭
            region_img = self.image[ry:ry+rh, rx:rx+rw]
            
            # EasyOCR 실행
            results = self.reader.readtext(region_img)
            
            # 결과 처리
            texts = []
            for (bbox, text, conf) in results:
                if conf < 0.3:  # 신뢰도 30% 이상만
                    continue
                
                # bbox는 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] 형식
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                x_min = int(min(x_coords))
                y_min = int(min(y_coords))
                x_max = int(max(x_coords))
                y_max = int(max(y_coords))
                
                # 전역 좌표로 변환
                gx, gy = rx + x_min, ry + y_min
                gw, gh = x_max - x_min, y_max - y_min
                
                texts.append({
                    'bbox': (gx, gy, gw, gh),
                    'text': text.strip(),
                    'conf': conf
                })
            
            print(f"    EasyOCR 감지: {len(texts)}개 텍스트")
            
            # 필터링
            filtered_texts = []
            
            if marker_group:
                marker_centers = [m['center'] for m in marker_group]
                mx1 = min([m['bbox'][0] for m in marker_group])
                my1 = min([m['bbox'][1] for m in marker_group])
                mx2 = max([m['bbox'][0] + m['bbox'][2] for m in marker_group])
                my2 = max([m['bbox'][1] + m['bbox'][3] for m in marker_group])
                mhs = [m['bbox'][3] for m in marker_group]
                med_h = int(np.median(mhs)) if mhs else 12
                
                # Y축 밴드
                # band_y 대역을 더 타이트하게 (3.0*med_h -> 2.0*med_h, min 20)
                y_pad = max(20, int(2.0 * med_h))
                band_y1 = max(0, my1 - y_pad)
                band_y2 = min(self.height, my2 + y_pad)
            
            # 텍스트 평균 높이 (참고용)
            median_text_h = np.median([t['bbox'][3] for t in texts]) if texts else 0
            
            for item in texts:
                text = item['text']
                gx, gy, gw, gh = item['bbox']
                cx, cy = gx + gw // 2, gy + gh // 2
                
                # === 다층 필터링: 정상 범례 보존 우선 ===
                
                # Tier 1: 명확한 노이즈
                # 1. 매우 큰 텍스트 (차트 제목)
                if gh > 100:
                    print(f"      제외 (매우 큰): '{text}' (h={gh})")
                    continue

                # 타이틀 가드 제거: 상단의 합법적 범례(예: Country …) 보존
                # 큰 제목은 아래 '매우 큰 텍스트' 규칙(gh>100)으로 억제됩니다.
                
                # 2. 알파벳/숫자 계산 (나중에 사용)
                alnum = sum(ch.isalpha() for ch in text)
                digits = sum(ch.isdigit() for ch in text)
                
                # 숫자만 있는 텍스트도 범례일 수 있으므로 제외하지 않음!
                # 예: "2020", "2025", "100", "200" 등
                
                # Tier 2: 짧은 노이즈
                # 3. 매우 짧고 낮은 신뢰도
                if len(text) <= 3 and item['conf'] < 0.7:
                    print(f"      제외 (짧은 노이즈): '{text}' (len={len(text)}, conf={item['conf']:.2f})")
                    continue
                
                # Tier 3: 낮은 신뢰도 중간 길이
                # 4. 중간 길이 + 낮은 신뢰도
                if 4 <= len(text) <= 7 and item['conf'] < 0.4:
                    print(f"      제외 (낮은 신뢰도): '{text}' (len={len(text)}, conf={item['conf']:.2f})")
                    continue
                
                # Tier 4: 비정상 패턴
                # 알파벳 비율로 강제 제외하던 규칙 제거 (위치/거리 기준으로 대체)
                alpha_ratio = alnum / len(text) if len(text) > 0 else 0
                
                # 6. 중간 길이 + 낮은 알파벳 비율 + 낮은 신뢰도
                if 4 <= len(text) <= 6 and alpha_ratio < 0.8 and item['conf'] < 0.8:
                    print(f"      제외 (비정상 패턴): '{text}' (len={len(text)}, alpha={alpha_ratio:.2f}, conf={item['conf']:.2f})")
                    continue
                
                # === 핵심: 마커와의 거리만 체크! ===
                if marker_group:
                    # 가장 가까운 마커와의 거리
                    min_distance = float('inf')
                    closest_marker = None
                    
                    for m in marker_group:
                        mx, my = m['center']
                        # 유클리드 거리
                        distance = np.sqrt((cx - mx)**2 + (cy - my)**2)
                        if distance < min_distance:
                            min_distance = distance
                            closest_marker = (mx, my)
                    
                    # 거리 임계값: 마커 평균 높이/텍스트 높이에 비례 (해상도 보정)
                    m_hs = [m['bbox'][3] for m in marker_group]
                    med_mh = np.median(m_hs) if m_hs else 12
                    # 알파벳 포함
                    MAX_DISTANCE = max(3.0 * med_mh, 1.5 * gh, 90.0)
                    # 숫자만
                    if alnum == 0:
                        MAX_DISTANCE = max(2.0 * med_mh, 1.2 * gh, 60.0)
                    
                    if min_distance > MAX_DISTANCE:
                        reason = "숫자만" if alnum == 0 else "거리 멀음"
                        print(f"      제외 ({reason}, 거리 {min_distance:.0f}px > {MAX_DISTANCE}): '{text}'")
                        continue
                    
                    # 추가 체크: Y축 범위 (같은 수평선상)
                    if not (band_y1 <= cy <= band_y2):
                        print(f"      제외 (y축 벗어남): '{text}' (cy={cy})")
                        continue
                
                print(f"      ✓ 포함: '{text}' (conf={item['conf']:.2f}, 거리={min_distance:.0f}px)")
                filtered_texts.append((gx, gy, gw, gh, text, item['conf']))
            
            # === 후처리: 마커 수와 텍스트 수 검증 ===
            if marker_group and len(filtered_texts) > len(marker_group):
                print(f"    ⚠️  텍스트({len(filtered_texts)}) > 마커({len(marker_group)}): 신뢰도 순 정리")
                # 신뢰도 순으로 정렬 (높은 순)
                filtered_texts.sort(key=lambda t: t[5], reverse=True)
                # 마커 수 + 1개까지만 허용 (여유)
                max_texts = len(marker_group) + 1
                filtered_texts = filtered_texts[:max_texts]
                print(f"    → {len(filtered_texts)}개로 제한")
            
            # 반환 형식 맞추기 (conf 제거)
            result_texts = [(t[0], t[1], t[2], t[3], t[4]) for t in filtered_texts]
            
            print(f"    최종 텍스트: {len(result_texts)}개")
            return result_texts
            
        except Exception as e:
            print(f"  텍스트 탐지 오류: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def find_markers_in_bbox(self, bbox, all_markers):
        """bbox 내의 모든 마커 찾기"""
        x, y, w, h = bbox
        markers_in_bbox = []
        
        for marker in all_markers:
            mx, my = marker['center']
            if x <= mx <= x + w and y <= my <= y + h:
                markers_in_bbox.append(marker)
        
        return markers_in_bbox
    
    def create_final_bbox(self, marker_group, texts, is_horizontal=True, all_markers=None):
        """최종 바운딩 박스 생성"""
        all_elements = []
        
        for m in marker_group:
            all_elements.append(m['bbox'])
        
        for x, y, w, h, _ in texts:
            all_elements.append((x, y, w, h))
        
        if not all_elements:
            return None, marker_group
        
        xs = [x for x, y, w, h in all_elements]
        ys = [y for x, y, w, h in all_elements]
        x_ends = [x + w for x, y, w, h in all_elements]
        y_ends = [y + h for x, y, w, h in all_elements]
        
        pad_x, pad_y = 32, 18
        x_min = max(0, min(xs) - pad_x)
        y_min = max(0, min(ys) - pad_y)
        x_max = min(self.width, max(x_ends) + pad_x)
        y_max = min(self.height, max(y_ends) + pad_y)
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        # 세로 레이아웃: 우측 제한
        if not is_horizontal:
            mx1 = min([m['bbox'][0] for m in marker_group])
            mx2 = max([m['bbox'][0] + m['bbox'][2] for m in marker_group])
            mhs = [m['bbox'][3] for m in marker_group]
            med_h = int(np.median(mhs)) if mhs else 14
            text_widths = [w for _, _, w, _, _ in texts] if texts else []
            obs = (sorted(text_widths)[-2] if len(text_widths) >= 2 else (text_widths[0] if text_widths else 0))
            limit = max(160, min(480, int(12.0 * med_h)))
            band = max(limit, int(1.2 * obs))
            right_limit = min(self.width, mx2 + band)
            bx, by, bw, bh = bbox
            bx2 = min(bx + bw, right_limit)
            left_guard = max(0, mx1 - int(1.2 * np.median([m['bbox'][2] for m in marker_group])) - 6)
            bx1 = max(bx, left_guard)
            bbox = (bx1, by, bx2 - bx1, bh)
        
        # 크기 검증
        area_ratio = (bbox[2] * bbox[3]) / (self.width * self.height)
        if area_ratio > 0.3:
            print(f"    ⚠️  영역 과대 (비율: {area_ratio:.2%})")
            return None, marker_group
        
        # bbox 내의 모든 마커 포함
        included_markers = marker_group
        if all_markers is not None:
            included_markers = self.find_markers_in_bbox(bbox, all_markers)
            if len(included_markers) > len(marker_group):
                print(f"      추가 마커 발견: {len(included_markers) - len(marker_group)}개")
        
        return bbox, included_markers

    # ===== ChartReader식 축/텍스트 기반 탐지 (포팅) =====
    def _ocr_full_image(self, min_conf=0.3):
        results = self.reader.readtext(self.image)
        texts = []
        for (bbox, text, conf) in results:
            if conf < min_conf:
                continue
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
            texts.append((text.strip(), (x1, y1, x2 - x1, y2 - y1), conf))
        return texts

    def _find_color_rects_loose(self):
        """컬러 패치/라인 후보: 범례 연결용 느슨한 컬러 사각형 목록."""
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        sat_mask = cv2.inRange(s, 16, 255)
        val_mask = cv2.inRange(v, 36, 255)
        cmask = cv2.bitwise_and(sat_mask, val_mask)
        cmask = cv2.morphologyEx(cmask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(cmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        img_area = self.width * self.height
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            a = w * h
            if a < 20:
                continue
            if a > img_area * 0.02:
                continue
            rects.append((x, y, w, h))
        return rects

    def detect_legend_via_axes(self, allow_below=True, debug=False):
        """ChartReader식: 축 검출 + 텍스트 병합/정렬 그룹화 기반 범례 탐지.
        반환: (bbox 또는 None, legend_rects, color_rects)
        """
        # 1) 축 검출
        try:
            xaxis, yaxis = detect_axes(self.image)
        except Exception as e:
            print(f"  축 검출 실패: {e}")
            return None, [], []

        (x1, y1, x2, y2) = xaxis
        (yx1, yy1, yx2, yy2) = yaxis
        x_row = y1
        y_col = yx1

        # 2) 전역 OCR
        ocr = self._ocr_full_image(min_conf=0.4)
        # Clean 'I'
        image_text = [(t, r) for (t, r, c) in ocr if t.strip() != 'I']

        # 3) 범례 후보 텍스트 필터
        legend_candidates = []
        for text, (tx, ty, tw, th) in image_text:
            # Right of y-axis
            right_of_y = (tx >= y_col + 4)
            above_x = (ty + th) <= x_row - 2
            below_x = ty >= x_row + 2
            # 숫자만은 제외 (ChartReader 규칙)
            if is_numeric_text(text):
                continue
            if right_of_y and (above_x or (allow_below and below_x)):
                legend_candidates.append((text, (tx, ty, tw, th)))

        if not legend_candidates:
            return None, [], []

        # 4) 텍스트 병합
        merged = merge_text_boxes(legend_candidates, x_thr=12, y_thr=2)

        # 5) 정렬 그룹화 → 최장 그룹 선택
        rects = [r for _, r in merged]
        groups = group_aligned(rects, y_thr=6, x_thr=6)
        if not groups:
            return None, [], []
        max_group = max(groups, key=len)

        # 6) 좌측 인접 컬러 박스 연결
        color_rects = self._find_color_rects_loose()
        linked_colors = []
        for r in max_group:
            tx, ty, tw, th = r
            # 허용 거리: 텍스트 높이에 비례해 동적 설정
            dx_lim = min(280, max(90, int(6.0 * th)))
            linked = filter_left_color_boxes(color_rects, r, min_overlap=0.3, max_dx=dx_lim, y_thr=12)
            # 가장 가까운 좌측 상자 하나 선택
            best = None
            bestd = 1e9
            for (x, y, w, h) in linked:
                d = abs((x + w) - tx)
                if d < bestd:
                    bestd = d
                    best = (x, y, w, h)
            if best is not None:
                linked_colors.append(best)
            else:
                # ROI 근방에서 추가 탐색 시도
                alt = self._find_near_color_for_text(r)
                if alt is not None:
                    linked_colors.append(alt)

        # 7) 최종 bbox
        elems = list(max_group) + linked_colors
        xs = [x for x, y, w, h in elems]
        ys = [y for x, y, w, h in elems]
        x2s = [x + w for x, y, w, h in elems]
        y2s = [y + h for x, y, w, h in elems]
        pad_x, pad_y = 24, 14
        bx1 = max(0, min(xs) - pad_x)
        by1 = max(0, min(ys) - pad_y)
        bx2 = min(self.width, max(x2s) + pad_x)
        by2 = min(self.height, max(y2s) + pad_y)
        bbox = (bx1, by1, bx2 - bx1, by2 - by1)

        # 면적 sanity check
        area_ratio = (bbox[2] * bbox[3]) / float(self.width * self.height)
        if area_ratio > 0.3 or area_ratio < 0.001:
            return None, [], []

        return bbox, max_group, linked_colors

    def _find_near_color_for_text(self, text_rect):
        """텍스트 좌측 근방에서 컬러 패치/라인을 로컬 탐색.
        반환: (x,y,w,h) 또는 None
        """
        tx, ty, tw, th = text_rect
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        sat_mask = cv2.inRange(s, 16, 255)
        val_mask = cv2.inRange(v, 36, 255)
        cmask = cv2.bitwise_and(sat_mask, val_mask)
        cmask = cv2.morphologyEx(cmask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        ypad = max(6, int(0.6 * th))
        xpad = min(240, max(60, int(6.0 * th)))
        x1 = max(0, tx - xpad)
        y1 = max(0, ty - ypad)
        x2 = tx
        y2 = min(self.height, ty + th + ypad)
        if x2 <= x1 or y2 <= y1:
            return None
        roi = cmask[y1:y2, x1:x2]
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        bestd = 1e9
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            gx, gy = x + x1, y + y1
            # 필터: 너무 작은/큰 것 제외
            a = w * h
            if a < 12:
                continue
            if a > self.width * self.height * 0.02:
                continue
            # 선형 또는 소형 직사각만 허용
            ar = w / float(h) if h > 0 else 999.0
            if not (ar >= 4.0 or (1.1 <= ar <= 6.0)):
                continue
            # 텍스트와 수직 중첩 필요
            top_o = max(gy, ty)
            bot_o = min(gy + h, ty + th)
            if bot_o <= top_o:
                continue
            d = tx - (gx + w)
            if d < 0:
                continue
            if d < bestd:
                bestd = d
                best = (gx, gy, w, h)
        return best
    
    def score_candidate(self, bbox, marker_count, text_count, alpha_text_count=None):
        """후보 점수 계산"""
        if bbox is None:
            return -1000
        
        x, y, w, h = bbox
        score = 0
        
        # 크기 점수
        area_ratio = (w * h) / (self.width * self.height)
        if 0.005 < area_ratio < 0.15:
            score += 20
        elif 0.15 < area_ratio < 0.25:
            score += 8
        else:
            return -1000
        
        # 위치 점수 보정: 하단 보너스, 상단 패널티
        if y > self.height * 0.75:
            score += 16
        elif y > self.height * 0.6:
            score += 10

        if x > self.width * 0.65:
            score += 10
        elif x > self.width * 0.45:
            score += 5

        # 상단(타이틀 가능) 패널티
        if y < self.height * 0.2:
            score -= 12
        
        # 가로세로 비율
        aspect = w / h if h > 0 else 0
        if 2 < aspect < 15:
            score += 8
        elif 0.25 < aspect < 0.7:
            score += 8
        elif 0.7 < aspect < 2:
            score += 4
        
        # 마커/텍스트 개수
        if marker_count > 0:
            coverage = min(text_count, marker_count) / float(marker_count)
            score += int(24 * coverage)
        
        if alpha_text_count is not None:
            # 알파벳이 없다고 바로 탈락시키지 않음 (숫자 범례 허용)
            if text_count > 0:
                score += int(8 * (alpha_text_count / float(text_count)))
        
        # 마커 개수 보너스/페널티 (큰 그룹 과다 보정)
        if 2 <= marker_count <= 8:
            score += marker_count * 5
        elif marker_count > 8:
            penalty = (marker_count - 8) * 10
            score -= penalty
            print(f"      ⚠️ 마커 과다 페널티: -{penalty}")
        
        if marker_count >= 4 and text_count <= 1:
            score -= 6

        # 색상 점유율 기반 보정: 범례는 컬러 픽셀이 적음
        try:
            sub = self.image[y:y+h, x:x+w]
            if sub.size > 0:
                hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
                _, s, v = cv2.split(hsv)
                sat_mask = cv2.inRange(s, 40, 255)
                val_mask = cv2.inRange(v, 50, 255)
                color_mask = cv2.bitwise_and(sat_mask, val_mask)
                color_ratio = float(cv2.countNonZero(color_mask)) / max(1, w * h)
                # 컬러가 많으면 차트 본문일 가능성 → 패널티
                if color_ratio > 0.14:
                    score -= int(140 * (color_ratio - 0.14))
                elif color_ratio < 0.06:
                    score += 12
        except Exception:
            pass

        return score
    
    def refine_bbox_tight(self, bbox, orient_horizontal=True):
        """bbox 타이트하게 보정"""
        if bbox is None:
            return bbox
        x, y, w, h = bbox
        sub = self.gray[y:y+h, x:x+w]
        _, thr = cv2.threshold(sub, 240, 255, cv2.THRESH_BINARY_INV)
        coords = cv2.findNonZero(thr)
        if coords is not None and len(coords) > 10:
            cx, cy, cw, ch = cv2.boundingRect(coords)
            pad = 8
            nx = max(0, x + cx - pad)
            ny = max(0, y + cy - pad)
            nw = min(self.width - nx, cw + 2*pad)
            nh = min(self.height - ny, ch + 2*pad)
            return (nx, ny, nw, nh)
        return bbox
    
    def detect_legend(self, debug=False):
        """범례 탐지 메인 로직"""
        print("\n" + "="*60)
        print("범례 탐지 시작 (EasyOCR)")
        print("="*60)
        
        # 1. 색상 마커
        print("\n[단계 1] 색상 마커 탐지...")
        markers = self.find_color_markers()
        
        if len(markers) < 2:
            print("  ⚠️  충분한 마커 없음 → 축+텍스트 기반으로 시도")
            ax_bbox, ax_text_rects, ax_color_rects = self.detect_legend_via_axes(allow_below=True, debug=debug)
            if ax_bbox:
                print("  ✓ 축 기반으로 범례 후보 확보")
                return ax_bbox, [ax_bbox]
            else:
                return None, []
        
        # 2. 마커 그룹
        print("\n[단계 2] 마커 그룹 탐지...")
        marker_groups = self.find_aligned_marker_groups(markers)
        
        if not marker_groups:
            print("  ⚠️  그룹 없음 → 축+텍스트 기반으로 시도")
            ax_bbox, ax_text_rects, ax_color_rects = self.detect_legend_via_axes(allow_below=True, debug=debug)
            if ax_bbox:
                print("  ✓ 축 기반으로 범례 후보 확보")
                return ax_bbox, [ax_bbox]
            else:
                return None, []
        
        # 3. 후보 생성
        print("\n[단계 3] 범례 후보 생성...")
        candidates = []
        
        for i, group in enumerate(marker_groups):
            print(f"  그룹 {i+1}: {len(group)}개 마커")
            
            region, is_horiz = self.get_marker_region(group)
            print(f"    방향: {'가로' if is_horiz else '세로'}")
            
            texts = self.find_texts_in_region(region, marker_group=group, is_horizontal=is_horiz)
            
            bbox, included_markers = self.create_final_bbox(group, texts, is_horizontal=is_horiz, all_markers=markers)
            
            if bbox:
                alpha_texts = sum(1 for _, _, _, _, t in texts if any(ch.isalpha() for ch in t))
                score = self.score_candidate(bbox, len(included_markers), len(texts), alpha_text_count=alpha_texts)
                
                bx, by, bw, bh = bbox
                inside = sum(1 for mk in markers if bx <= mk['center'][0] <= bx+bw and by <= mk['center'][1] <= by+bh)
                # 내부 마커 수에 대한 보너스는 비율로 제한
                if len(group) > 0:
                    ratio = min(1.0, inside / float(len(group)))
                    score += int(12 * ratio)
                
                if score > 0:
                    candidates.append((score, bbox, is_horiz))
                    print(f"    ✓ 점수: {score}, 마커:{len(group)}, 텍스트:{len(texts)}, 포함:{inside}")
        
        if not candidates:
            print("  ⚠️  유효 후보 없음 (마커 기반)")
        
        # 축/텍스트 기반 후보도 생성하여 비교
        print("\n[보조 경로] 축+텍스트 기반 후보 생성...")
        ax_bbox, ax_text_rects, ax_color_rects = self.detect_legend_via_axes(allow_below=True, debug=debug)
        if ax_bbox:
            ax_text_count = len(ax_text_rects)
            ax_marker_count = len(ax_color_rects)
            alpha_texts = ax_text_count  # 숫자 제외만 남김
            ax_score = self.score_candidate(ax_bbox, ax_marker_count, ax_text_count, alpha_text_count=alpha_texts)
            if ax_score > 0:
                candidates.append((ax_score, ax_bbox, True))
                print(f"    ✓(축기반) 점수: {ax_score}, 마커:{ax_marker_count}, 텍스트:{ax_text_count}")
        else:
            print("    축기반 후보 없음")
        
        if not candidates:
            print("  ⚠️  유효 후보 없음")
            return None, []
        
        # 4. 최고 점수 선택
        print("\n[단계 4] 최적 후보 선택...")
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        if debug:
            for i, (score, bbox, _) in enumerate(candidates[:3]):
                area_pct = (bbox[2] * bbox[3]) / (self.width * self.height) * 100
                print(f"    {i+1}위. 점수:{score}, 면적:{area_pct:.1f}%")
        
        best_score, best_bbox, best_horiz = candidates[0]
        best_bbox = self.refine_bbox_tight(best_bbox, orient_horizontal=best_horiz)
        all_bboxes = [bbox for _, bbox, _ in candidates]
        
        area_pct = (best_bbox[2] * best_bbox[3]) / (self.width * self.height) * 100
        print(f"\n✅ 최종 선택: 점수 {best_score}, 면적 {area_pct:.1f}%")
        print("="*60)
        
        return best_bbox, all_bboxes
    
    def crop_legend(self, bbox):
        """범례 영역 크롭"""
        if bbox is None:
            return None
        
        x, y, w, h = bbox
        x = max(0, x)
        y = max(0, y)
        w = min(w, self.width - x)
        h = min(h, self.height - y)
        
        return self.image[y:y+h, x:x+w]
    
    def visualize_detection(self, final_bbox, all_candidates=None, show_markers=False):
        """탐지 결과 시각화"""
        result = self.image.copy()
        
        if show_markers:
            markers = self.find_color_markers()
            for m in markers:
                x, y, w, h = m['bbox']
                cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 255), 2)
                cv2.circle(result, m['center'], 3, (255, 0, 255), -1)
        
        if all_candidates and len(all_candidates) > 1:
            for i, (x, y, w, h) in enumerate(all_candidates[1:], 1):
                cv2.rectangle(result, (x, y), (x+w, y+h), (255, 120, 0), 2)
                cv2.putText(result, f"#{i+1}", (x+5, y+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)
        
        if final_bbox:
            x, y, w, h = final_bbox
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 5)
            cv2.putText(result, "LEGEND", (x+10, y+35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        return result


# 사용 예제
if __name__ == "__main__":
    #test_images = [5,8]
    test_images = range(1,11)
    
    for i in test_images:
        print(f"\n{'='*70}")
        print(f"이미지 {i} 처리")
        print(f"{'='*70}")
        
        try:
            detector = LegendDetectorEasyOCR(f"./legend_test/{i}.png")
            legend_bbox, all_candidates = detector.detect_legend(debug=True)
            
            if legend_bbox:
                print(f"\n📍 위치: ({legend_bbox[0]}, {legend_bbox[1]})")
                print(f"📏 크기: {legend_bbox[2]} x {legend_bbox[3]}")
                
                legend_crop = detector.crop_legend(legend_bbox)
                cv2.imwrite(f"{output_dir}/legend_cropped{i}_easyocr.png", legend_crop)
                
                visualization = detector.visualize_detection(legend_bbox, all_candidates, show_markers=True)
                cv2.imwrite(f"{output_dir}/legend_detection_result{i}_easyocr.png", visualization)
                
                print("✅ 저장 완료")
            else:
                print("❌ 범례 미발견")
                
        except Exception as e:
            print(f"❌ 오류: {e}")
            import traceback
            traceback.print_exc()
