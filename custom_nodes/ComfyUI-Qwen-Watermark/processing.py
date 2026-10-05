import json
import math

import cv2
import numpy as np


OVERLAY_TYPES = {
    "text_logo",
    "translucent_text_logo",
    "rectangle",
    "mosaic",
    "irregular_sticker",
    "unknown",
}


def normalize_overlay_type(value):
    text = str(value or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
    if text in OVERLAY_TYPES:
        return text
    if any(token in text for token in ("transparent", "translucent", "stock", "credit", "透明", "半透明")):
        return "translucent_text_logo"
    if any(token in text for token in ("mosaic", "pixel", "马赛克")):
        return "mosaic"
    if any(token in text for token in ("rectangle", "block", "patch", "矩形", "色块", "方块")):
        return "rectangle"
    if any(token in text for token in ("sticker", "pasted_object", "贴纸", "贴片", "遮挡物")):
        return "irregular_sticker"
    if any(token in text for token in ("logo", "watermark", "wordmark", "text", "文字", "水印", "平台标记")):
        return "text_logo"
    return "unknown"


def route_for_overlay(item):
    overlay_type = normalize_overlay_type(item.get("overlay_type", item.get("type")))
    if overlay_type in {"text_logo", "translucent_text_logo", "rectangle", "mosaic"}:
        return "box"
    if overlay_type == "irregular_sticker":
        return "sam"
    requested = str(item.get("mask_method", "")).strip().lower()
    return requested if requested in {"box", "sam"} else "sam"


def _adaptive_padding_ratio(overlay_type, requested):
    caps = {
        "text_logo": 0.08,
        "translucent_text_logo": 0.35,
        "rectangle": 0.06,
        "mosaic": 0.06,
        "irregular_sticker": min(requested, 0.25),
        "unknown": min(requested, 0.20),
    }
    return min(requested, caps[overlay_type])


def _box_intersection_ratio(first, second):
    x0, y0 = max(first[0], second[0]), max(first[1], second[1])
    x1, y1 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    first_area = max(1, (first[2] - first[0]) * (first[3] - first[1]))
    second_area = max(1, (second[2] - second[0]) * (second[3] - second[1]))
    return intersection / min(first_area, second_area)


def _box_area(box):
    return max(1, (box[2] - box[0]) * (box[3] - box[1]))


def _boxes_are_related(first, second, width, height, overlap_threshold):
    if _box_intersection_ratio(first, second) >= overlap_threshold:
        return True
    horizontal_gap = max(0, max(first[0], second[0]) - min(first[2], second[2]))
    vertical_gap = max(0, max(first[1], second[1]) - min(first[3], second[3]))
    horizontal_overlap = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    vertical_overlap = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    min_width = max(1, min(first[2] - first[0], second[2] - second[0]))
    min_height = max(1, min(first[3] - first[1], second[3] - second[1]))
    same_row = vertical_overlap / min_height >= 0.3 and horizontal_gap <= max(8, width * 0.03)
    same_column = horizontal_overlap / min_width >= 0.3 and vertical_gap <= max(8, height * 0.03)
    return same_row or same_column


def merge_detection_results(results, width, height, overlap_threshold=0.20):
    """Merge repeated detections so partial Logo boxes become one complete region."""
    merged = []
    for pass_index, result in enumerate(results):
        for source in result.get("boxes", []):
            values = source.get("source_bbox", source.get("bbox"))
            if not isinstance(values, (list, tuple)) or len(values) != 4:
                continue
            box = [max(0, int(values[0])), max(0, int(values[1])),
                   min(width, int(values[2])), min(height, int(values[3]))]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            confidence = source.get("confidence", 0.5)
            confidence = float(confidence) if isinstance(confidence, (int, float)) else 0.5
            candidate = {
                "label": str(source.get("label", "watermark")),
                "bbox": box,
                "overlay_type": normalize_overlay_type(source.get("overlay_type", source.get("type"))),
                "confidence": max(0.0, min(1.0, confidence)),
                "source_passes": [pass_index],
            }
            candidate["mask_method"] = route_for_overlay({**source, **candidate})
            duplicate = next((item for item in merged
                              if _boxes_are_related(item["bbox"], candidate["bbox"],
                                                    width, height, overlap_threshold)), None)
            if duplicate is None:
                merged.append(candidate)
                continue
            later_recovery_is_more_complete = (
                pass_index > max(duplicate["source_passes"])
                and _box_area(box) >= _box_area(duplicate["bbox"]) * 1.5
                and candidate["confidence"] >= duplicate["confidence"] - 0.05
            )
            if later_recovery_is_more_complete:
                duplicate["bbox"] = box
            else:
                duplicate["bbox"] = [min(duplicate["bbox"][0], box[0]), min(duplicate["bbox"][1], box[1]),
                                     max(duplicate["bbox"][2], box[2]), max(duplicate["bbox"][3], box[3])]
            duplicate["confidence"] = max(duplicate["confidence"], candidate["confidence"])
            duplicate["source_passes"] = sorted(set(duplicate["source_passes"] + [pass_index]))
            if duplicate["overlay_type"] == "unknown" or candidate["overlay_type"] == "translucent_text_logo":
                duplicate["overlay_type"] = candidate["overlay_type"]
            if "box" in (duplicate["mask_method"], candidate["mask_method"]):
                duplicate["mask_method"] = "box"
            if len(candidate["label"]) > len(duplicate["label"]):
                duplicate["label"] = candidate["label"]
    return {"coordinate_space": "pixels", "image_size": [width, height], "boxes": merged}


def build_edit_prompt(result, hint=""):
    types = {normalize_overlay_type(item.get("overlay_type")) for item in result.get("boxes", [])}
    instructions = [
        "编辑<image1>。只移除检测区域内的后期叠加物，并依据紧邻区域恢复被遮挡的内容。",
        "严格保持原图的相机、透视、物体位置、人物身份、轮廓、材质、光照、真实文字和整体颜色。不得改变构图、移动物体、裁剪或缩放画面。",
    ]
    if types & {"text_logo", "translucent_text_logo"}:
        instructions.append("完整移除文字水印、平台标记、Logo、网址、所有笔画、描边、阴影和低透明度残留；恢复连续的建筑线条、道路、皮肤、服装或背景纹理。")
    if types & {"rectangle", "mosaic"}:
        instructions.append("矩形花纹、纯色色块或马赛克是后期遮挡；移除整个硬边区域和接缝，不得用模糊色块代替背景。")
    if "irregular_sticker" in types or "unknown" in types:
        instructions.append("不规则对象是后期贴纸；完整移除其主体、四肢、耳朵、白边、半透明边缘和投影，不要重画或保留其形状。")
    if hint.strip():
        instructions.append("补充要求：" + hint.strip())
    instructions.append("输出与输入尺寸相同的完整局部图像，不添加无关内容。")
    return "".join(instructions)


def calibrate_large_translucent_boxes(result, width, height):
    """Correct the common Qwen coordinate bias for large central translucent credits."""
    calibrated = json.loads(json.dumps(result))
    for item in calibrated.get("boxes", []):
        if normalize_overlay_type(item.get("overlay_type")) != "translucent_text_logo":
            continue
        values = item.get("source_bbox", item.get("bbox"))
        if not isinstance(values, (list, tuple)) or len(values) != 4:
            continue
        x0, y0, x1, y1 = [int(round(float(value))) for value in values]
        box_width = max(1, x1 - x0)
        center_x = (x0 + x1) / (2 * max(1, width))
        center_y = (y0 + y1) / (2 * max(1, height))
        if box_width < width * 0.18 or not 0.25 <= center_x <= 0.75 or not 0.30 <= center_y <= 0.60:
            continue
        shift = min(max(24, round(height * 0.08)), max(0, height - y1))
        if shift <= 0:
            continue
        for key in ("bbox", "source_bbox"):
            current = item.get(key)
            if isinstance(current, (list, tuple)) and len(current) == 4:
                item[key] = [current[0], min(height, current[1] + shift),
                             current[2], min(height, current[3] + shift)]
        item["coordinate_calibration"] = {"axis": "y", "shift_px": shift,
                                           "reason": "large_central_translucent_logo"}
    return calibrated


def _finite_number(value, field):
    if isinstance(value, bool):
        raise ValueError(f"{field} 必须是有限数字。")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 必须是有限数字。") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} 必须是有限数字。")
    return number


def _integer_value(value, field, minimum=0):
    number = _finite_number(value, field)
    if not number.is_integer() or number < minimum:
        raise ValueError(f"{field} 必须是大于等于 {minimum} 的整数。")
    return int(number)


def parse_boxes(text, width, height, expand=6, padding_ratio=0, adaptive_padding=False):
    if not isinstance(text, str):
        raise ValueError("定位结果必须是 JSON 文本。")
    if not isinstance(width, (int, np.integer)) or not isinstance(height, (int, np.integer)) or width <= 0 or height <= 0:
        raise ValueError("图像宽度和高度必须为正整数。")
    expand = _integer_value(expand, "expand")
    padding_ratio = _finite_number(padding_ratio, "padding_ratio")
    if padding_ratio < 0:
        raise ValueError("padding_ratio 不能为负数。")
    decoder = json.JSONDecoder()
    documents = []
    list_documents = []
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("boxes"), list):
            documents.append(value)
        elif isinstance(value, list) and all(
            isinstance(item, dict) and isinstance(item.get("bbox"), (list, tuple))
            for item in value
        ):
            list_documents.append({"coordinate_space": "xyxy_1000", "boxes": value})
    if documents:
        document = documents[-1]
    elif list_documents:
        document = list_documents[-1]
    else:
        raise ValueError('定位结果不是有效 JSON，需要 {"coordinate_space":"xyxy_1000","boxes":[{"bbox":[x0,y0,x1,y1]}]}。')
    space = document.get("coordinate_space", "xyxy_1000")
    scales = {"xyxy_1000": (width / 1000, height / 1000), "normalized": (width, height), "pixels": (1, 1)}
    if space not in scales:
        raise ValueError("coordinate_space 必须为 xyxy_1000、normalized 或 pixels。")
    sx, sy = scales[space]
    mask = np.zeros((height, width), dtype=np.float32)
    boxes = []
    for item in document["boxes"]:
        if not isinstance(item, dict):
            raise ValueError("boxes 中的每一项必须是对象。")
        values = item.get("bbox")
        if not isinstance(values, (list, tuple)) or len(values) != 4:
            raise ValueError("每个 bbox 必须包含四个有限数字。")
        x0, y0, x1, y1 = [_finite_number(v, "bbox") for v in values]
        x0, x1 = sorted((x0 * sx, x1 * sx))
        y0, y1 = sorted((y0 * sy, y1 * sy))
        if x1 <= 0 or y1 <= 0 or x0 >= width or y0 >= height or x1 <= x0 or y1 <= y0:
            continue
        source_values = item.get("source_bbox", values)
        if not isinstance(source_values, (list, tuple)) or len(source_values) != 4:
            raise ValueError("source_bbox 必须包含四个有限数字。")
        sx0, sy0, sx1, sy1 = [_finite_number(v, "source_bbox") for v in source_values]
        sx0, sx1 = sorted((sx0 * sx, sx1 * sx))
        sy0, sy1 = sorted((sy0 * sy, sy1 * sy))
        source_box = [max(0, math.floor(sx0)), max(0, math.floor(sy0)),
                      min(width, math.ceil(sx1)), min(height, math.ceil(sy1))]
        overlay_type = normalize_overlay_type(item.get("overlay_type", item.get("type")))
        item_padding = _adaptive_padding_ratio(overlay_type, padding_ratio) if adaptive_padding else padding_ratio
        ex = math.ceil(expand + (x1 - x0) * item_padding)
        ey = math.ceil(expand + (y1 - y0) * item_padding)
        box = [max(0, math.floor(x0) - ex), max(0, math.floor(y0) - ey),
               min(width, math.ceil(x1) + ex), min(height, math.ceil(y1) + ey)]
        mask[box[1]:box[3], box[0]:box[2]] = 1
        confidence = item.get("confidence", 0.5)
        confidence = float(confidence) if isinstance(confidence, (int, float)) else 0.5
        boxes.append({"label": str(item.get("label", "watermark")), "bbox": box,
                      "source_bbox": source_box, "overlay_type": overlay_type,
                      "mask_method": route_for_overlay(item),
                      "confidence": round(max(0.0, min(1.0, confidence)), 4)})
    result = {"coordinate_space": "pixels", "image_size": [width, height], "boxes": boxes}
    return mask, result


def refine_manual_mask(mask, dilation=0, adaptive_dilation=False, dilation_ratio=0.08):
    """Expand a painted mask so translucent borders and shadows are editable."""
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("手绘遮罩必须是二维数组。")
    dilation = _integer_value(dilation, "dilation")
    dilation_ratio = _finite_number(dilation_ratio, "dilation_ratio")
    if dilation_ratio < 0:
        raise ValueError("dilation_ratio 不能为负数。")
    selected = mask > 0
    if not np.any(selected):
        return selected.astype(np.float32), {
            "effective_dilation": dilation,
            "base_pixels": 0,
            "refined_pixels": 0,
        }

    ys, xs = np.nonzero(selected)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    effective_dilation = dilation
    if adaptive_dilation:
        short_side = min(x1 - x0, y1 - y0)
        effective_dilation = min(64, max(dilation, math.ceil(short_side * dilation_ratio)))
    if effective_dilation:
        kernel_size = effective_dilation * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        selected = cv2.dilate(selected.astype(np.uint8), kernel, iterations=1).astype(bool)

    report = {
        "source_bbox": [x0, y0, x1, y1],
        "effective_dilation": effective_dilation,
        "base_pixels": int(np.count_nonzero(mask > 0)),
        "refined_pixels": int(np.count_nonzero(selected)),
    }
    return selected.astype(np.float32), report


def refine_sam_candidate(candidate, safety_box, prompt_box, dilation=0, adaptive_dilation=False,
                         dilation_ratio=0.08, recovery_ratio=0.12):
    """Keep a SAM contour that slightly exceeds detection while bounding false selections."""
    if not isinstance(candidate, np.ndarray) or candidate.ndim != 2:
        raise ValueError("SAM 候选遮罩必须是二维数组。")
    height, width = candidate.shape
    dilation = _integer_value(dilation, "dilation")
    dilation_ratio = _finite_number(dilation_ratio, "dilation_ratio")
    recovery_ratio = _finite_number(recovery_ratio, "recovery_ratio")
    if dilation_ratio < 0 or recovery_ratio < 0:
        raise ValueError("dilation_ratio 和 recovery_ratio 不能为负数。")

    def normalize_box(values, field):
        if not isinstance(values, (list, tuple)) or len(values) != 4:
            raise ValueError(f"{field} 必须包含四个有限数字。")
        x0, y0, x1, y1 = [_finite_number(value, field) for value in values]
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        box = [max(0, math.floor(x0)), max(0, math.floor(y0)),
               min(width, math.ceil(x1)), min(height, math.ceil(y1))]
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"{field} 为空或超出图像范围。")
        return box

    safety = normalize_box(safety_box, "safety_box")
    prompt = normalize_box(prompt_box, "prompt_box")
    prompt_width, prompt_height = prompt[2] - prompt[0], prompt[3] - prompt[1]
    prompt_extent = max(prompt_width, prompt_height)
    effective_dilation = dilation
    if adaptive_dilation:
        effective_dilation = min(64, max(dilation, math.ceil(min(prompt_width, prompt_height) * dilation_ratio)))
    recovery_margin = max(8, math.ceil(prompt_extent * recovery_ratio))
    guard_margin = recovery_margin + effective_dilation
    guard = [max(0, safety[0] - guard_margin), max(0, safety[1] - guard_margin),
             min(width, safety[2] + guard_margin), min(height, safety[3] + guard_margin)]

    raw = candidate.astype(bool)
    selected = np.zeros_like(raw)
    x0, y0, x1, y1 = guard
    selected[y0:y1, x0:x1] = raw[y0:y1, x0:x1]
    inside_safety = np.zeros_like(raw)
    sx0, sy0, sx1, sy1 = safety
    inside_safety[sy0:sy1, sx0:sx1] = True
    recovered_pixels = int(np.count_nonzero(selected & ~inside_safety))
    clipped_pixels = int(np.count_nonzero(raw)) - int(np.count_nonzero(selected))

    if effective_dilation:
        kernel_size = effective_dilation * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        selected = cv2.dilate(selected.astype(np.uint8), kernel, iterations=1).astype(bool)
        bounded = np.zeros_like(selected)
        bounded[y0:y1, x0:x1] = selected[y0:y1, x0:x1]
        selected = bounded

    report = {"safety_box": safety, "guard_box": guard, "recovery_margin": recovery_margin,
              "effective_dilation": effective_dilation,
              "recovered_outside_safety_pixels": recovered_pixels,
              "guard_clipped_pixels": max(0, clipped_pixels)}
    return selected.astype(np.float32), report


def prepare_crop(image, mask, context_pixels=96, max_side=768):
    if not isinstance(image, np.ndarray) or image.ndim < 2:
        raise ValueError("原图必须是至少二维数组。")
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("遮罩必须是二维数组。")
    if mask.shape != image.shape[:2]:
        raise ValueError("遮罩尺寸与原图不同。请使用原图尺寸的遮罩。")
    context_pixels = _integer_value(context_pixels, "context_pixels")
    max_side = _integer_value(max_side, "max_side", minimum=1)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("未识别到水印。先查看定位预览，或改用 manual_boxes / manual_mask。")
    height, width = image.shape[:2]
    x0, y0 = max(0, int(xs.min()) - context_pixels), max(0, int(ys.min()) - context_pixels)
    x1, y1 = min(width, int(xs.max()) + 1 + context_pixels), min(height, int(ys.max()) + 1 + context_pixels)
    crop = image[y0:y1, x0:x1].copy()
    scale = min(1.0, max_side / max(crop.shape[:2]))
    rw, rh = max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))
    crop = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    pw, ph = math.ceil(rw / 32) * 32, math.ceil(rh / 32) * 32
    crop = cv2.copyMakeBorder(crop, 0, ph - rh, 0, pw - rw, cv2.BORDER_REPLICATE)
    local_mask = cv2.resize(mask[y0:y1, x0:x1], (rw, rh), interpolation=cv2.INTER_NEAREST)
    local_mask = np.pad(local_mask, ((0, ph - rh), (0, pw - rw)))
    geometry = {"roi": [x0, y0, x1, y1], "content_size": [rw, rh], "model_size": [pw, ph], "image_size": [width, height]}
    return crop, local_mask, geometry


def restore_crop(edited, geometry):
    if not isinstance(edited, np.ndarray) or edited.ndim < 2:
        raise ValueError("编辑输出必须是至少二维数组。")
    required = ("model_size", "content_size", "roi")
    if not isinstance(geometry, dict) or any(key not in geometry for key in required):
        raise ValueError("编辑几何信息不完整。请直接连接 QWMPrepare 的 geometry 输出。")
    pw, ph = geometry["model_size"]
    if edited.shape[:2] != (ph, pw):
        raise ValueError(f"编辑输出尺寸 {edited.shape[1]}x{edited.shape[0]} 与参考 {pw}x{ph} 不同。请连接 TextEncodeQwenImage21 的 latent 输出。")
    rw, rh = geometry["content_size"]
    x0, y0, x1, y1 = geometry["roi"]
    return cv2.resize(edited[:rh, :rw], (x1 - x0, y1 - y0), interpolation=cv2.INTER_LANCZOS4)


def align_crop(original, edited, mask, enabled=True):
    report = {"enabled": enabled, "applied": False, "shift_px": [0.0, 0.0], "warning": ""}
    if not enabled:
        return edited, report
    valid = (cv2.dilate((mask > 0).astype(np.uint8), np.ones((17, 17), np.uint8)) == 0).astype(np.uint8)
    if valid.sum() < 256:
        report["warning"] = "背景上下文不足，无法检查对齐。"
        return edited, report
    ref = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float32)
    src = cv2.cvtColor(edited, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if np.std(ref[valid > 0]) < 0.005:
        report["warning"] = "背景过于平坦，无法估计偏移。"
        return edited, report
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        score, warp = cv2.findTransformECC(ref, src, warp, cv2.MOTION_TRANSLATION,
                                          (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 60, 1e-5), valid, 5)
    except cv2.error:
        report["warning"] = "对齐估计未收敛，请检查对比图。"
        return edited, report
    shift = warp[:, 2]
    report.update({"correlation": round(float(score), 5), "shift_px": [round(float(v), 3) for v in shift]})
    if max(abs(shift)) > 12 or score < 0.5:
        report["warning"] = "编辑区域可能存在结构变化，请检查对比图或重试。未自动校正。"
        return edited, report
    aligned = cv2.warpAffine(edited, warp, (edited.shape[1], edited.shape[0]),
                             flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    before = float(np.mean((ref[valid > 0] - src[valid > 0]) ** 2))
    after_gray = cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY)
    after = float(np.mean((ref[valid > 0] - after_gray[valid > 0]) ** 2))
    if after < before:
        report["applied"] = True
        return aligned, report
    return edited, report


def composite(original, edited, mask, geometry, feather=6, align=True):
    if not isinstance(original, np.ndarray) or original.ndim < 3 or original.shape[2] < 3:
        raise ValueError("原图必须是 HxWxC 数组。")
    if not isinstance(mask, np.ndarray) or mask.shape != original.shape[:2]:
        raise ValueError("遮罩尺寸与原图不同。请连接原图尺寸的 MASK。")
    if not isinstance(edited, np.ndarray) or edited.ndim < 3 or edited.shape[2] < 3:
        raise ValueError("编辑结果必须是 HxWxC 数组。")
    original = original.astype(np.float32)
    edited = restore_crop(edited.astype(np.float32), geometry)
    x0, y0, x1, y1 = geometry["roi"]
    height, width = original.shape[:2]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("编辑几何信息中的 ROI 超出原图范围。")
    feather = _integer_value(feather, "feather")
    local_mask = np.clip(mask[y0:y1, x0:x1], 0, 1).astype(np.float32)
    edited, alignment = align_crop(original[y0:y1, x0:x1], edited, local_mask, align)
    if feather > 0:
        # Feather inward, so pixels outside the selection remain exactly original.
        padded_mask = np.pad((local_mask > 0).astype(np.uint8), 1)
        distance = cv2.distanceTransform(padded_mask, cv2.DIST_L2, 5)[1:-1, 1:-1]
        local_mask *= np.clip(distance / feather, 0, 1)
    output = original.copy()
    region = output[y0:y1, x0:x1]
    changed = local_mask > 0
    alpha = local_mask[changed, None]
    region[changed] = region[changed] * (1 - alpha) + edited[changed] * alpha
    output = np.clip(output, 0, 1)
    outside = mask <= 0
    difference = np.max(np.abs(output[outside] - original[outside])) if np.any(outside) else 0.0
    report = {"geometry": geometry, "alignment": alignment, "selection_pixels": int(np.count_nonzero(mask)),
              "preserved_fraction": round(float(np.mean(outside)), 6), "outside_mask_max_difference": float(difference)}
    return output, report
