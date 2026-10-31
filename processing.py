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
        # Detector boxes are estimates. Leave enough room for antialiasing,
        # shadows and the smaller URL that often follows a logo.
        "text_logo": 0.12,
        # Faint credits are often under-boxed on the right/bottom edge. Keep a
        # generous safety margin because compositing still protects every pixel
        # outside the final mask.
        "translucent_text_logo": 0.50,
        # Hard-edged patches are frequently reported one or two pattern cells
        # short. A small border is safer than feeding the model a clipped edge.
        "rectangle": 0.12,
        "mosaic": 0.12,
        "irregular_sticker": min(requested, 0.25),
        "unknown": min(requested, 0.25),
    }
    return min(requested, caps[overlay_type])


def coerce_rectangular_overlay_types(result):
    """Correct common VLM misclassification of wide flat patches as stickers."""
    corrected = json.loads(json.dumps(result))
    for item in corrected.get("boxes", []):
        if normalize_overlay_type(item.get("overlay_type")) != "irregular_sticker":
            continue
        values = item.get("source_bbox", item.get("bbox"))
        if not isinstance(values, (list, tuple)) or len(values) != 4:
            continue
        width = abs(float(values[2]) - float(values[0]))
        height = abs(float(values[3]) - float(values[1]))
        if width <= 0 or height <= 0:
            continue
        label = str(item.get("label", "")).lower()
        flat_hint = any(token in label for token in ("rectangle", "patch", "block", "mosaic", "pattern", "色块", "花纹", "方块", "马赛克"))
        # Shape alone cannot distinguish a wide cutout from a rectangular patch.
        if flat_hint:
            item["overlay_type"] = "rectangle"
            item["mask_method"] = "box"
            item["type_correction"] = "wide_flat_patch"
    return corrected


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
                "background": str(source.get("background", "")),
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
            if candidate["background"]:
                duplicate["background"] = candidate["background"]
    return {"coordinate_space": "pixels", "image_size": [width, height], "boxes": merged}


def build_edit_prompt(result, hint="", edit_prompt=""):
    types = {normalize_overlay_type(item.get("overlay_type")) for item in result.get("boxes", [])}
    instructions = [
        "编辑<image1>。只修复局部填充痕迹和后期叠加物，恢复为周围可见材质与纹理的连续延伸。",
        "严格保持原图的相机、透视、物体位置、人物身份、轮廓、材质、光照、真实文字和整体颜色。不得改变构图、移动物体、裁剪或缩放画面。",
    ]
    if result.get("source") in {"manual_mask", "manual_boxes"}:
        instructions.append("手动选区内全部属于需要移除的后期叠加物；清理半透明边缘和残影，延续选区外的原场景纹理。")
    if types & {"text_logo", "translucent_text_logo"}:
        instructions.append("完整移除文字水印、平台标记、Logo、网址、所有笔画、描边、阴影和低透明度残留；恢复连续的建筑线条、道路、皮肤、服装或背景纹理。")
    if types & {"rectangle", "mosaic"}:
        instructions.append("恢复矩形填充区域下原本的场景纹理，清除马赛克和硬边接缝。")
    if "irregular_sticker" in types:
        instructions.append("不规则对象是后期贴纸；完整移除其主体、四肢、耳朵、白边、半透明边缘和投影，不要重画或保留其形状。")
    backgrounds = list(dict.fromkeys(item.get("background", "").strip() for item in result.get("boxes", []) if item.get("background", "").strip()))
    if backgrounds:
        instructions.append("周围可见的原场景材质：" + "；".join(backgrounds) + "。据此延续纹理和颜色。")
    if hint.strip():
        instructions.append("补充要求：" + hint.strip())
    if edit_prompt.strip():
        instructions.append("用户编辑指令：" + edit_prompt.strip())
    instructions.append("输出与输入尺寸相同的完整局部图像，不添加无关内容。")
    return "\n".join(instructions)


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
                      "background": str(item.get("background", "")),
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


def select_sam_recovery(seed, candidates, scores, seed_score):
    """Recover a complete cutout without accepting an unrelated larger object."""
    seed = seed.astype(bool)
    seed_pixels = int(seed.sum())
    selected = seed
    best_score = -1.0
    if seed_pixels:
        for candidate, score in zip(candidates, scores):
            count, labels = cv2.connectedComponents(candidate.astype(np.uint8))
            anchored = np.unique(labels[seed])
            anchored = anchored[anchored != 0]
            candidate = np.isin(labels, anchored) if count > 1 else np.zeros_like(seed)
            coverage = np.count_nonzero(candidate & seed) / seed_pixels
            growth = np.count_nonzero(candidate) / seed_pixels
            if coverage >= 0.9 and 1.02 <= growth <= 2.5 and score >= max(0.8, seed_score - 0.15) and score > best_score:
                selected = candidate | seed
                best_score = float(score)
    recovered = int(np.count_nonzero(selected & ~seed))
    return selected, {"outline_recovery_applied": bool(recovered), "outline_recovered_pixels": recovered,
                      "outline_recovery_score": best_score if recovered else None}


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
              "dilation_ratio": dilation_ratio, "effective_dilation": effective_dilation,
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
    if not (0 < rw <= pw and 0 < rh <= ph and x0 < x1 and y0 < y1):
        raise ValueError("编辑几何信息的有效内容或 ROI 尺寸无效。")
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
    if max(abs(shift)) > 12 or score < 0.85:
        report["warning"] = "编辑区域可能存在结构变化，请检查对比图或重试。未自动校正。"
        return edited, report
    if max(abs(shift)) < 0.25:
        report["reason"] = "subpixel_noise_ignored"
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


def match_context_color(original, edited, selected, enabled=True):
    """Fit bounded RGB gains using paired, unedited context and held-out pixels."""
    valid = cv2.dilate(selected.astype(np.uint8), np.ones((7, 7), np.uint8)) == 0
    report = {"enabled": enabled, "applied": False, "samples": int(valid.sum())}
    if not enabled or valid.sum() < 256:
        report["reason"] = "disabled" if not enabled else "insufficient_context"
        return edited, report
    ref = cv2.GaussianBlur(original, (0, 0), 1.2)[valid][::max(1, int(valid.sum()) // 32000)]
    src = cv2.GaussianBlur(edited, (0, 0), 1.2)[valid][::max(1, int(valid.sum()) // 32000)]
    train = np.arange(len(ref)) % 3 != 0
    before = np.abs(ref[~train] - src[~train])
    report["context_mae_before"] = float(before.mean())
    if report["context_mae_before"] > 0.1:
        report.update(reason="context_changed_too_much", context_mae_after=report["context_mae_before"])
        return edited, report
    if float(np.median(before)) <= 1 / 255:
        report["reason"] = "already_matched"
        report["context_mae_after"] = report["context_mae_before"]
        return edited, report
    gain, bias = np.ones(3, np.float32), np.zeros(3, np.float32)
    for channel in range(3):
        x, y = src[train, channel], ref[train, channel]
        weights = np.ones_like(x)
        for _ in range(4):
            xm, ym = np.average(x, weights=weights), np.average(y, weights=weights)
            variance = np.average((x - xm) ** 2, weights=weights)
            g = np.clip(np.average((x - xm) * (y - ym), weights=weights) / variance, 0.8, 1.25) if variance > 0.0001 else 1.0
            b = np.clip(np.median(y - g * x), -0.08, 0.08)
            residual = y - (g * x + b)
            scale = max(1 / 255, float(np.median(np.abs(residual))) * 1.4826)
            weights = np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(residual), 1e-6))
        gain[channel], bias[channel] = g, b
    after = np.abs(ref[~train] - np.clip(src[~train] * gain + bias, 0, 1))
    report.update(gain=gain.tolist(), bias=bias.tolist())
    if after.mean() < before.mean() * 0.85 and np.percentile(after, 90) <= np.percentile(before, 90):
        report.update(applied=True, reason="validated_on_context", context_mae_after=float(after.mean()))
        return np.clip(edited * gain + bias, 0, 1), report
    report.update(reason="no_reliable_improvement", context_mae_after=report["context_mae_before"])
    return edited, report


def boundary_color_correction(original, edited, selected, width):
    """Extend only exterior RGB discrepancies into a narrow interior band."""
    if width <= 0 or not np.any(~selected) or not np.any(selected):
        return edited, {"applied": False, "width": width}
    # Never sample original pixels inside the selection: those contain the overlay.
    valid = (~selected).astype(np.float32)
    sigma = max(1.0, width / 2)
    delta = np.clip(original - edited, -0.08, 0.08) * valid[..., None]
    weights = cv2.GaussianBlur(valid, (0, 0), sigma)
    field = cv2.GaussianBlur(delta, (0, 0), sigma) / np.maximum(weights[..., None], 1e-6)
    distance = cv2.distanceTransform(np.pad(selected.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    strength = np.clip(1 - distance / max(1, width), 0, 1)
    strength = strength * strength * (3 - 2 * strength)
    strength *= selected & (weights > 0.01)
    adjustment = field * strength[..., None]
    maximum = float(np.max(np.abs(adjustment)))
    if maximum <= 1 / 255:
        return edited, {"applied": False, "width": width, "max_adjustment": maximum}
    return np.clip(edited + adjustment, 0, 1), {"applied": True, "width": width, "max_adjustment": maximum}


def boundary_jump(image, mask):
    selected = mask > 0
    samples = []
    for axis in (0, 1):
        edge = np.diff(selected.astype(np.int8), axis=axis) != 0
        values = np.mean(np.abs(np.diff(image, axis=axis)), axis=2)[edge]
        if len(values):
            samples.append(values)
    return float(np.percentile(np.concatenate(samples), 90)) if samples else 0.0


def constrain_residuals(result, allowed_mask, expand=24, min_confidence=0.8):
    """Residual detection may repair only the original selection and its guard."""
    height, width = allowed_mask.shape
    allowed = cv2.dilate((allowed_mask > 0).astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * expand + 1, 2 * expand + 1))) > 0
    accepted, rejected = [], []
    uncertain = False
    for item in result.get("boxes", []):
        x0, y0, x1, y1 = item.get("source_bbox", item["bbox"])
        overlap = float(np.mean(allowed[y0:y1, x0:x1])) if x1 > x0 and y1 > y0 else 0
        if overlap >= 0.5 and item.get("confidence", 0) >= min_confidence:
            accepted.append(item)
        else:
            rejected.append({**item, "reason": "outside_repair_scope" if overlap < 0.5 else "low_confidence"})
            uncertain |= overlap >= 0.5
    mask, parsed = parse_boxes(json.dumps({"coordinate_space": "pixels", "boxes": accepted}),
                               width, height, expand=expand, padding_ratio=0)
    mask *= allowed
    parsed.update(rejected_boxes=rejected, uncertain=uncertain, scope_pixels=int(allowed.sum()))
    return mask, parsed


def choose_final(first, second, first_mask, second_mask, second_report):
    """Reject a second pass that increases boundary discontinuity or leaves its scope."""
    report = {"selected_pass": 1, "second_pass_accepted": False, "reasons": []}
    if second is None:
        return first, first_mask, report
    before, after = boundary_jump(first, second_mask), boundary_jump(second, second_mask)
    report.update(boundary_jump_before=before, boundary_jump_after=after)
    if after > max(4 / 255, before * 1.3 + 2 / 255):
        report["reasons"].append("second_pass_boundary_regression")
    if second_report.get("alignment", {}).get("warning"):
        report["reasons"].append("second_pass_alignment_uncertain")
    if np.any(second_mask <= 0) and np.max(np.abs(second[second_mask <= 0] - first[second_mask <= 0])) > 0:
        report["reasons"].append("second_pass_changed_protected_pixels")
    if report["reasons"]:
        return first, first_mask, report
    report.update(selected_pass=2, second_pass_accepted=True)
    return second, np.maximum(first_mask, second_mask), report


def composite(original, edited, mask, geometry, feather=6, align=True, color_match=True):
    if not isinstance(original, np.ndarray) or original.ndim < 3 or original.shape[2] < 3:
        raise ValueError("原图必须是 HxWxC 数组。")
    if not isinstance(mask, np.ndarray) or mask.shape != original.shape[:2]:
        raise ValueError("遮罩尺寸与原图不同。请连接原图尺寸的 MASK。")
    if not isinstance(edited, np.ndarray) or edited.ndim < 3 or edited.shape[2] < 3:
        raise ValueError("编辑结果必须是 HxWxC 数组。")
    if not np.isfinite(original).all() or not np.isfinite(edited).all() or not np.isfinite(mask).all():
        raise ValueError("图像或遮罩包含 NaN/Inf，无法安全贴回。")
    original = original.astype(np.float32)
    edited = restore_crop(edited.astype(np.float32), geometry)
    x0, y0, x1, y1 = geometry["roi"]
    height, width = original.shape[:2]
    if geometry.get("image_size", [width, height]) != [width, height]:
        raise ValueError("geometry 的原图尺寸不匹配，请连接同一次裁剪结果。")
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("编辑几何信息中的 ROI 超出原图范围。")
    covered = np.zeros(mask.shape, bool)
    covered[y0:y1, x0:x1] = True
    if np.any((mask > 0) & ~covered):
        raise ValueError("遮罩超出裁剪 ROI。请连接同一次 QWMPrepare 的 geometry 和 MASK。")
    feather = _integer_value(feather, "feather")
    local_mask = np.clip(mask[y0:y1, x0:x1], 0, 1).astype(np.float32)
    edited, alignment = align_crop(original[y0:y1, x0:x1], edited, local_mask, align)
    region = original[y0:y1, x0:x1]
    changed = local_mask > 0
    edited, color = match_context_color(region, edited, changed, color_match)
    scale = max((x1 - x0) / geometry["content_size"][0], (y1 - y0) / geometry["content_size"][1])
    edited, boundary = boundary_color_correction(region, edited, changed, round(feather * scale))
    output = original.copy()
    region = output[y0:y1, x0:x1]
    region[changed] = np.clip(edited[changed], 0, 1)
    outside = mask <= 0
    difference = np.max(np.abs(output[outside] - original[outside])) if np.any(outside) else 0.0
    report = {"geometry": geometry, "alignment": alignment, "selection_pixels": int(np.count_nonzero(mask)),
              "preserved_fraction": round(float(np.mean(outside)), 6), "outside_mask_max_difference": float(difference)}
    seam_before = boundary_jump(original[y0:y1, x0:x1], local_mask)
    seam_after = boundary_jump(output[y0:y1, x0:x1], local_mask)
    report.update(color=color, boundary=boundary, edge_blend="exterior_color_field",
                  boundary_jump_p90=seam_after, boundary_jump_original_p90=seam_before,
                  boundary_jump_excess_p90=max(0.0, seam_after - seam_before),
                  working_precision="float32_rgb", source_pixels_mixed_inside_mask=0)
    return output, report
