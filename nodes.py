import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from comfy_execution.graph import ExecutionBlocker
from PIL import Image, ImageDraw

try:
    import comfy.model_prefetch as model_prefetch
except ImportError:  # Older ComfyUI versions do not have the memory compiler.
    model_prefetch = None

from .processing import (
    build_edit_prompt,
    choose_final,
    composite,
    constrain_residuals,
    coerce_rectangular_overlay_types,
    merge_detection_results,
    parse_boxes,
    prepare_crop,
    refine_manual_mask,
    refine_sam_candidate,
    route_for_overlay,
    select_sam_recovery,
)

try:
    import folder_paths
except ImportError:  # Allows the processing tests to import this module outside ComfyUI.
    folder_paths = None

try:
    from segment_anything import SamPredictor, sam_model_registry
except ImportError:
    SamPredictor = None
    sam_model_registry = {}


DETECTION_PROMPT = """Locate visible overlays added after this photo was taken: text watermarks, translucent credits, logos/URLs, rectangular solid or patterned patches, mosaics and cutout stickers. Inspect the whole image including edges and corners. Preserve real signs, garment prints, branding and objects that belong to the photographed scene.
Only report overlays supported by visible pixels; do not invent a target or guess its location. Trace the complete outline including letters, URL, border and shadow. For cutout stickers include ALL protruding parts: arms, legs, feet, paws, ears and tails, even near the image edge; do not stop at the torso or clothing. Coordinates must follow the actual visible position, never a fixed central-position rule. Use separate boxes for distant overlays. A rectangular flat patch is rectangle/mosaic; an irregular animal/person/object cutout is irregular_sticker regardless of aspect ratio.
Return ONLY JSON: {"coordinate_space":"xyxy_1000","boxes":[{"label":"what is visibly overlaid","bbox":[left,top,right,bottom],"overlay_type":"text_logo","confidence":0.9,"background":"visible surrounding material"}]}. Coordinates are 0..1000 relative to THIS image. Allowed types: text_logo, translucent_text_logo, rectangle, mosaic, irregular_sticker, unknown. background describes the visible surrounding surface in a few words, such as road, wall, fabric or skin; never guess hidden details. Use confidence below 0.65 when evidence is uncertain. Return {"coordinate_space":"xyxy_1000","boxes":[]} when no overlay is visible. No markdown or explanation."""


RESIDUAL_PROMPT = """Locate visible post-production overlays or fragments in this photo crop: text, logos, cutout stickers, pixel/bead-art patches, illustration fragments and white cutout borders. Inspect every edge and corner. A previous edit may have removed only part of an overlay; do not assume it succeeded. Isolated cartoon arms, legs, socks, shoes or tails still count as sticker remnants even when the head/body is gone. Distinguish pasted flat or pixel/bead-art texture from photographed real clothing and objects. Preserve real skin, garments, signs, furniture, natural shadows and architecture. Only report evidence visible in the crop; do not invent examples or hidden details. Color seams without an identifiable overlay require color correction, not another generative edit.
Return ONLY JSON: {"coordinate_space":"xyxy_1000","boxes":[{"label":"specific visible remnant","bbox":[left,top,right,bottom],"overlay_type":"text_logo","confidence":0.9}]}. Coordinates are 0..1000 relative to THIS crop, not the original whole image. Allowed types: text_logo, translucent_text_logo, rectangle, mosaic, irregular_sticker, unknown. If no identifiable overlay remains return {"coordinate_space":"xyxy_1000","boxes":[]}. No markdown, speculative boxes or example lists."""


RESIDUAL_REFERENCE_PROMPT = """Image 1 is the original photo crop containing a pasted overlay. Image 2 is a proposed repair of the same crop. Compare them: locate any visible pieces of the pasted overlay that STILL REMAIN in image 2, including isolated legs, feet, white borders or patterned sticker fragments whose head or body was removed. Preserve photographed real clothing, skin and furniture. Report only remnants actually visible in image 2, with coordinates relative to image 2. Return ONLY JSON: {"coordinate_space":"xyxy_1000","boxes":[{"label":"visible remaining overlay fragment","bbox":[left,top,right,bottom],"overlay_type":"irregular_sticker","confidence":0.9}]}. Allowed types: text_logo, translucent_text_logo, rectangle, mosaic, irregular_sticker, unknown. Classify visible text/logo remnants as text_logo, not as a cutout sticker. If no overlay remains return {"coordinate_space":"xyxy_1000","boxes":[]}."""


_SAM_CACHE = {}


def _sam_model_choices():
    if folder_paths is not None:
        try:
            choices = list(folder_paths.get_filename_list("sams"))
            if choices:
                return choices
        except Exception:
            pass
    return ["sam_vit_b_01ec64.pth"]


def _sam_checkpoint(model_name):
    if folder_paths is not None:
        try:
            path = folder_paths.get_full_path("sams", model_name)
            if path:
                return Path(path)
        except Exception:
            pass
        try:
            path = Path(folder_paths.models_dir) / "sams" / model_name
            if path.is_file():
                return path
        except Exception:
            pass
    return Path(model_name)


def _load_sam(model_name, device):
    if SamPredictor is None:
        raise RuntimeError("未找到 segment-anything。请安装 ComfyUI 的 SAM 支持，或把细化方式改为 box。")
    checkpoint = _sam_checkpoint(model_name)
    if not checkpoint.is_file():
        raise RuntimeError(f"找不到 SAM 模型 {model_name}。请将模型放到 ComfyUI/models/sams，或把细化方式改为 box。")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    key = (str(checkpoint.resolve()), device)
    predictor = _SAM_CACHE.get(key)
    if predictor is not None:
        return predictor, device
    stem = checkpoint.stem.lower()
    variant = next((name for name in ("vit_h", "vit_l", "vit_b") if name in stem), None)
    if variant is None or variant not in sam_model_registry:
        raise RuntimeError(f"暂不支持 SAM 模型 {checkpoint.name}；请使用 sam_vit_b_01ec64.pth。")
    model = sam_model_registry[variant](checkpoint=str(checkpoint))
    model.eval().to(device)
    predictor = SamPredictor(model)
    _SAM_CACHE[key] = predictor
    return predictor, device


def to_array(image):
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ValueError("IMAGE 必须是 [batch,height,width,channels] 张量。")
    if image.shape[0] != 1:
        raise ValueError("每次处理一张图；批量请通过运行脚本逐张提交。")
    if image.shape[-1] < 3:
        raise ValueError("IMAGE 至少需要 RGB 三个通道。")
    return image[0, :, :, :3].detach().cpu().numpy().astype(np.float32)


def to_tensor(image):
    return torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None]


def _detect_with_clip(clip, tensor, prompt, max_tokens):
    tokens = clip.tokenize(prompt, images=tensor if isinstance(tensor, list) else [tensor])
    try:
        output = clip.generate(tokens, do_sample=False, max_length=max_tokens, temperature=0.0, seed=0)
    finally:
        # Each region has new KV buffers; captured graphs must not cross generations.
        if model_prefetch is not None:
            model_prefetch.cleanup_prefetch_queues()
    return str(clip.decode(output, skip_special_tokens=True))


class QWMDetect:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mode": (["auto_qwen", "manual_boxes", "manual_mask"],),
            "boxes_json": ("STRING", {"default": '{"coordinate_space":"pixels","boxes":[]}', "multiline": True,
                                         "tooltip": "手动框 JSON。坐标可用像素、xyxy_1000 或 normalized；auto_qwen 模式会用模型输出覆盖它。"}),
            "detection_side": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 32,
                                         "tooltip": "送入 Qwen3-VL 的定位图最长边。越大越容易看清小 Logo，显存和耗时也随之增加。"}),
            "expand": ("INT", {"default": 8, "min": 0, "max": 64,
                                 "tooltip": "矩形框向外扩的像素数，用于覆盖描边、阴影和接缝。它只影响 box 范围，不负责羽化。"}),
            "max_tokens": ("INT", {"default": 512, "min": 64, "max": 2048,
                                     "tooltip": "定位模型输出 JSON 的最大 token 数。多水印或复杂描述时增加；普通图片保持 512。"}),
            "hint": ("STRING", {"default": "", "multiline": True,
                                  "tooltip": "给定位模型的补充线索，例如“左下角灰色文字”和“右下角圆形 Logo”；不改变编辑指令。"}),
            "edit_prompt": ("STRING", {"default": "", "multiline": True,
                                         "tooltip": "给 Qwen Image 的补充编辑要求。写保留结构、材质和位置约束；不要只写“去水印”。"}),
            "padding_ratio": ("FLOAT", {"default": 0.35, "min": 0, "max": 1, "step": 0.05,
                                          "tooltip": "自动检测框按比例扩边；程序会按水印类型设上限。手动框模式不使用此比例。"}),
        }, "optional": {"clip": ("CLIP",), "manual_mask": ("MASK",),
                        "min_confidence": ("FLOAT", {"default": 0.65, "min": 0, "max": 1,
                                                         "tooltip": "保留自动候选的最低置信度。调高可减少误检，调低可减少低对比水印漏检；低于 0.65 的结果应人工确认。"})}}

    RETURN_TYPES = ("MASK", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("watermark_mask", "boxes_json", "preview", "edit_prompt")
    FUNCTION = "detect"
    CATEGORY = "Qwen Watermark"

    def detect(self, image, mode, boxes_json, detection_side, expand, max_tokens, hint,
               padding_ratio=0.35, clip=None, manual_mask=None, edit_prompt="", min_confidence=0.65):
        array = to_array(image)
        height, width = array.shape[:2]
        raw = ""
        if mode == "manual_mask":
            if manual_mask is None:
                raise ValueError("manual_mask 模式需要连接遮罩。")
            if not isinstance(manual_mask, torch.Tensor) or manual_mask.ndim != 3 or manual_mask.shape[0] < 1:
                raise ValueError("手动遮罩必须是 [batch,height,width] 张量。")
            mask = manual_mask[0].detach().cpu().numpy().astype(np.float32)
            if mask.shape != (height, width):
                raise ValueError("手动遮罩尺寸必须与原图相同。")
            mask = np.clip(mask, 0, 1)
            result = {"coordinate_space": "pixels", "image_size": [width, height], "boxes": [], "source": mode}
        else:
            if mode == "auto_qwen":
                if clip is None:
                    raise ValueError("auto_qwen 模式需要连接 Qwen3-VL CLIP。")
                picture = Image.fromarray(np.clip(array * 255, 0, 255).astype(np.uint8))
                picture.thumbnail((detection_side, detection_side), Image.Resampling.LANCZOS)
                tensor = to_tensor(np.array(picture).astype(np.float32) / 255)
                suffix = "\nAdditional watermark description: " + hint if hint.strip() else ""
                raw_general = _detect_with_clip(clip, tensor, DETECTION_PROMPT + suffix, max_tokens)
                try:
                    _, parsed = parse_boxes(raw_general, width, height, expand=0, padding_ratio=0)
                except ValueError as exc:
                    result = merge_detection_results([], width, height)
                    errors = [str(exc)]
                else:
                    result = coerce_rectangular_overlay_types(merge_detection_results([parsed], width, height))
                    errors = []
                rejected = [item for item in result["boxes"] if item["confidence"] < min_confidence]
                result["boxes"] = [item for item in result["boxes"] if item["confidence"] >= min_confidence]
                boxes_json = json.dumps(result, ensure_ascii=False)
                raw = json.dumps({"strategy": "single_inference_high_recall_geometry_recovery",
                                  "combined": raw_general,
                                  "parse_warnings": errors}, ensure_ascii=False)
            mask, result = parse_boxes(boxes_json, width, height, expand,
                                       padding_ratio if mode == "auto_qwen" else 0,
                                       adaptive_padding=mode == "auto_qwen")
            result["source"] = mode
            if raw:
                result["detection_passes"] = json.loads(raw)
                result["rejected_boxes"] = rejected
                result["status"] = "indeterminate" if errors else "needs_review" if rejected else "detected" if result["boxes"] else "no_overlay_detected"
        picture = Image.fromarray(np.clip(array * 255, 0, 255).astype(np.uint8))
        if mode == "manual_mask":
            preview = array.copy()
            selected = mask > 0
            preview[selected] = preview[selected] * 0.65 + np.array([1, 0.25, 0.15]) * 0.35
        else:
            draw = ImageDraw.Draw(picture)
            for index, item in enumerate(result["boxes"]):
                x0, y0, x1, y1 = item["bbox"]
                draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(255, 65, 35), width=3)
                draw.text((x0, max(0, y0 - 14)), str(index + 1), fill=(255, 65, 35))
            preview = np.array(picture).astype(np.float32) / 255
        report = json.dumps(result, ensure_ascii=False, indent=2)
        edit_prompt = build_edit_prompt(result, hint, edit_prompt)
        return {"ui": {"text": [report]},
                "result": (torch.from_numpy(mask)[None], report, to_tensor(preview), edit_prompt)}


class QWMResidualDetect:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "clip": ("CLIP",),
            "detection_side": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 32,
                                         "tooltip": "残留分区送入 Qwen3-VL 的最长边。小残片漏检时提高到 1280；8GB 显存保持 1024。"}),
            "expand": ("INT", {"default": 24, "min": 0, "max": 96,
                                 "tooltip": "残留候选向外扩的安全边，随后与第一次修复范围取交集。它不会扩大到原遮罩之外。"}),
            "max_tokens": ("INT", {"default": 384, "min": 64, "max": 2048,
                                     "tooltip": "残留复检 JSON 的最大 token 数。只检查残片，通常 384 足够。"}),
            "edit_prompt": ("STRING", {"default": "", "multiline": True,
                                         "tooltip": "二采追加的局部修复要求。只描述残留和需要保持的结构；留空使用自动提示词。"}),
        }, "optional": {"allowed_mask": ("MASK",),
                        "enabled": ("BOOLEAN", {"default": True,
                                                 "tooltip": "是否运行第一次修复后的残留复检。关闭后不会触发二采，最终报告会保留 needs_review。"}),
                        "first_report": ("STRING", {"default": ""}),
                        "min_confidence": ("FLOAT", {"default": 0.8, "min": 0, "max": 1,
                                                         "tooltip": "触发二采的残留置信度阈值。默认 0.8，只处理高置信度残片，避免把自然纹理再次生成。"}),
                        "original_image": ("IMAGE",)}}

    RETURN_TYPES = ("MASK", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("residual_mask", "boxes_json", "preview", "edit_prompt")
    FUNCTION = "detect"
    CATEGORY = "Qwen Watermark"

    def detect(self, image, clip, detection_side, expand, max_tokens, edit_prompt="", allowed_mask=None,
               enabled=True, min_confidence=0.8, first_report="", original_image=None):
        array = to_array(image)
        height, width = array.shape[:2]
        original = to_array(original_image) if original_image is not None else None
        if original is not None and original.shape != array.shape:
            raise ValueError("复检原图参照必须与修复图尺寸一致。")
        scope = np.ones((height, width), np.float32) if allowed_mask is None else allowed_mask[0].detach().cpu().numpy()
        if scope.shape != (height, width):
            raise ValueError("复检范围遮罩必须与图片尺寸一致。")
        first_data = json.loads(first_report) if first_report else {}
        invalid = (first_data.get("alignment", {}).get("correlation", 1) < 0.85 or bool(first_data.get("alignment", {}).get("warning"))) and (
                   first_data.get("color", {}).get("context_mae_before", 0) > 0.12)
        if not enabled or not np.any(scope > 0) or invalid:
            result = {"coordinate_space": "pixels", "boxes": [], "source": "residual_qwen",
                      "status": "skipped_invalid_first_pass" if invalid else "skipped" if not enabled else "no_overlay_detected"}
            report = json.dumps(result, ensure_ascii=False)
            return {"ui": {"text": [report]}, "result": (torch.zeros((1, height, width)), report, image, edit_prompt)}
        inspection = cv2.dilate((scope > 0).astype(np.uint8),
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (129, 129)))
        count, _, stats, _ = cv2.connectedComponentsWithStats(inspection)
        inspection_rois, raw_detections, parsed_results, warnings = [], [], [], []
        for index in range(1, count):
            x0, y0, rw, rh, _ = [int(value) for value in stats[index]]
            x1, y1 = x0 + rw, y0 + rh
            inspection_rois.append([x0, y0, x1, y1])
            picture = Image.fromarray(np.clip(array[y0:y1, x0:x1] * 255, 0, 255).astype(np.uint8))
            picture.thumbnail((detection_side, detection_side), Image.Resampling.LANCZOS)
            tensor = to_tensor(np.array(picture).astype(np.float32) / 255)
            prompt = RESIDUAL_PROMPT
            if original is not None:
                reference = Image.fromarray(np.clip(original[y0:y1, x0:x1] * 255, 0, 255).astype(np.uint8))
                reference.thumbnail((detection_side, detection_side), Image.Resampling.LANCZOS)
                tensor = [to_tensor(np.array(reference).astype(np.float32) / 255), tensor]
                prompt = RESIDUAL_REFERENCE_PROMPT
            raw = _detect_with_clip(clip, tensor, prompt, max_tokens)
            raw_detections.append(raw)
            try:
                _, parsed = parse_boxes(raw, rw, rh, expand=0, padding_ratio=0)
            except ValueError as exc:
                warnings.append(str(exc))
                continue
            for item in parsed["boxes"]:
                for key in ("bbox", "source_bbox"):
                    b = item[key]
                    item[key] = [b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0]
            parsed_results.append(parsed)
        merged = coerce_rectangular_overlay_types(merge_detection_results(parsed_results, width, height))
        mask, result = constrain_residuals(merged, scope, expand=expand, min_confidence=min_confidence)
        if result["boxes"]:
            status = "residual_detected"
        elif warnings or result["uncertain"] or result["rejected_boxes"]:
            status = "indeterminate"
        else:
            status = "clean"
        result.update({"source": "residual_qwen", "raw_detection": raw_detections, "status": status,
                       "inspection_rois": inspection_rois, "reference_comparison": original is not None})
        if len(inspection_rois) == 1:
            result["inspection_roi"] = inspection_rois[0]
        if warnings:
            result["parse_warning"] = "; ".join(warnings)
        preview_image = Image.fromarray(np.clip(array * 255, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(preview_image)
        for index, item in enumerate(result["boxes"]):
            x0, y0, x1, y1 = item["bbox"]
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(255, 178, 0), width=3)
            draw.text((x0, max(0, y0 - 14)), f"R{index + 1}", fill=(255, 178, 0))
        preview = np.array(preview_image).astype(np.float32) / 255
        report = json.dumps(result, ensure_ascii=False, indent=2)
        prompt = build_edit_prompt(result, "只清理第一次修复后仍可见的残留，不改变已经修复干净的区域。", edit_prompt)
        return {"ui": {"text": [report]},
                "result": (torch.from_numpy(mask)[None], report, to_tensor(preview), prompt)}


class QWMRefineMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "boxes_json": ("STRING", {"multiline": True}),
            "method": (["auto", "sam", "box"],),
            "sam_model": ("STRING", {"default": _sam_model_choices()[0],
                                       "tooltip": "SAM 权重文件名；放在 ComfyUI/models/sams/。"}),
            "device": (["auto", "cpu", "cuda"],),
            "dilation": ("INT", {"default": 12, "min": 0, "max": 64,
                                  "tooltip": "SAM 或手绘遮罩向外扩的基础像素数，用于覆盖白边、透明边和投影。它不等于 feather。"}),
            "adaptive_dilation": ("BOOLEAN", {"default": True,
                                                 "tooltip": "按大型贴纸短边自动增加外扩，最多 64px。人物、头发或建筑边缘敏感时可关闭。"}),
        }}

    RETURN_TYPES = ("MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("refined_mask", "preview", "status")
    FUNCTION = "refine"
    CATEGORY = "Qwen Watermark"

    def refine(self, image, mask, boxes_json, method, sam_model, device, dilation, adaptive_dilation=True):
        array = to_array(image)
        if not isinstance(mask, torch.Tensor) or mask.ndim != 3 or mask.shape[0] < 1:
            raise ValueError("MASK 必须是 [batch,height,width] 张量。")
        base = mask[0].detach().cpu().numpy().astype(np.float32)
        height, width = base.shape
        if base.shape != array.shape[:2]:
            raise ValueError("细化遮罩尺寸与原图不同。")
        if not np.any(base > 0):
            status = {"status": "box", "method": method, "refined_pixels": int(np.count_nonzero(base))}
            report = json.dumps(status, ensure_ascii=False, indent=2)
            return {"ui": {"text": [report]}, "result": (torch.from_numpy(base)[None], to_tensor(array), report)}
        try:
            _, parsed = parse_boxes(boxes_json, width, height, expand=0, padding_ratio=0)
            # The detector's padded box is the safety boundary. Prompt SAM with
            # the tighter source box so it selects the pasted object instead of
            # a larger underlying person or surface.
            boxes = parsed["boxes"]
        except (TypeError, ValueError) as exc:
            boxes = []
            parse_warning = str(exc)
        else:
            parse_warning = ""
        if not boxes:
            refined, refinement = refine_manual_mask(
                base, dilation=dilation, adaptive_dilation=adaptive_dilation
            )
            status = {"status": "manual_mask", "method": "manual_mask",
                      "dilation": dilation, "adaptive_dilation": adaptive_dilation,
                      **refinement}
            if parse_warning:
                status["warning"] = parse_warning
            preview = array.copy()
            selected = refined > 0
            preview[selected] = preview[selected] * 0.55 + np.array([1.0, 0.18, 0.08], dtype=np.float32) * 0.45
            report = json.dumps(status, ensure_ascii=False, indent=2)
            return {"ui": {"text": [report]}, "result": (torch.from_numpy(refined)[None], to_tensor(preview), report)}
        if method == "box":
            status = {"status": "box", "method": method, "refined_pixels": int(np.count_nonzero(base))}
            report = json.dumps(status, ensure_ascii=False, indent=2)
            return {"ui": {"text": [report]}, "result": (torch.from_numpy(base)[None], to_tensor(array), report)}
        planned_route_counts = {
            "box": sum(1 for item in boxes if method == "auto" and route_for_overlay(item) == "box"),
            "sam": sum(1 for item in boxes if method != "auto" or route_for_overlay(item) == "sam"),
        }
        try:
            refined = np.zeros_like(base, dtype=np.float32)
            refined_count = 0
            refinements = []
            route_counts = {"box": 0, "sam": 0}
            sam_boxes = [item for item in boxes if method == "auto" and route_for_overlay(item) == "sam"]
            if method == "sam":
                sam_boxes = boxes
            predictor = None
            selected_device = None
            if sam_boxes:
                predictor, selected_device = _load_sam(sam_model, device)
                rgb = np.clip(array * 255, 0, 255).astype(np.uint8)
                predictor.set_image(rgb)
            for item in boxes:
                selected_method = route_for_overlay(item) if method == "auto" else "sam"
                if selected_method == "box":
                    x0, y0, x1, y1 = item["bbox"]
                    selected = np.zeros_like(base, dtype=np.float32)
                    selected[y0:y1, x0:x1] = 1
                    refined = np.maximum(refined, selected)
                    route_counts["box"] += 1
                    refinements.append({"mask_method": "box", "overlay_type": item.get("overlay_type", "unknown"),
                                        "safety_box": item["bbox"], "effective_dilation": 0})
                    refined_count += 1
                    continue
                values = item.get("source_bbox", item["bbox"])
                x0, y0, x1, y1 = [float(v) for v in values]
                box = np.array([max(0, x0), max(0, y0), min(width, x1), min(height, y1)], dtype=np.float32)
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                masks, scores, _ = predictor.predict(box=box, multimask_output=True)
                best = int(np.argmax(scores))
                selected = masks[best]
                recovery = {}
                if item.get("overlay_type") == "irregular_sticker" and item["bbox"] != list(values) and np.any(selected):
                    # Anchor the wider prompt inside the cutout already found,
                    # so missed limbs can join it without selecting the person.
                    distance = cv2.distanceTransform(selected.astype(np.uint8), cv2.DIST_L2, 5)
                    py, px = np.unravel_index(np.argmax(distance), distance.shape)
                    candidates, recovery_scores, _ = predictor.predict(
                        box=np.array(item["bbox"], dtype=np.float32),
                        point_coords=np.array([[px, py]], dtype=np.float32),
                        point_labels=np.array([1]), multimask_output=True)
                    selected, recovery = select_sam_recovery(selected, candidates, recovery_scores, scores[best])
                selected, selection_report = refine_sam_candidate(
                    selected, item["bbox"], values, dilation=dilation,
                    adaptive_dilation=adaptive_dilation
                )
                refined = np.maximum(refined, selected)
                route_counts["sam"] += 1
                refinements.append({"mask_method": "sam", "overlay_type": item.get("overlay_type", "unknown"),
                                    **selection_report, **recovery})
                refined_count += 1
            if not np.any(refined):
                raise RuntimeError("SAM 未生成有效轮廓。")
            status = {"status": "routed" if method == "auto" else "sam", "method": method,
                      "model": sam_model if sam_boxes else None, "device": selected_device,
                      "boxes_refined": refined_count, "base_pixels": int(np.count_nonzero(base)),
                      "refined_pixels": int(np.count_nonzero(refined)), "dilation": dilation,
                      "adaptive_dilation": adaptive_dilation,
                      "route_counts": route_counts,
                      "prompt_boxes": [item.get("source_bbox", item["bbox"]) for item in boxes],
                      "refinements": refinements}
        except Exception as exc:
            status = {"status": "box_fallback", "method": method, "warning": str(exc),
                      "base_pixels": int(np.count_nonzero(base)),
                      "route_counts": planned_route_counts}
            refined = base
        if json.loads(boxes_json).get("source") == "residual_qwen":
            refined = np.minimum(refined, base)
        preview = array.copy()
        selected = refined > 0
        preview[selected] = preview[selected] * 0.55 + np.array([1.0, 0.18, 0.08], dtype=np.float32) * 0.45
        report = json.dumps(status, ensure_ascii=False, indent=2)
        return {"ui": {"text": [report]}, "result": (torch.from_numpy(refined)[None], to_tensor(preview), report)}


class QWMPrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "mask": ("MASK",),
                             "context_pixels": ("INT", {"default": 96, "min": 0, "max": 2048,
                                                          "tooltip": "裁剪区四周保留的原图上下文像素。普通文字用 96；贴近人体或建筑结构用 256~384。"}),
                             "max_side": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 32,
                                                   "tooltip": "局部裁剪送入编辑模型的最长边。8GB 显存建议 768；细节复杂且显存足够时用 1024。"}),
                             "pre_inpaint": (["auto", "none", "telea", "ns"],),
                             "inpaint_radius": ("INT", {"default": 3, "min": 1, "max": 15,
                                                          "tooltip": "Telea / NS 预清理的半径，仅作用于遮罩内。色块或贴纸用 3~5；纹理复杂时可关闭预清理。"})},
                "optional": {"edit_prompt": ("STRING", {"default": "", "multiline": True,
                                                              "tooltip": "裁剪节点接收的编辑要求；通常由“02 自动定位 / 手动改框”连线，不需手动填写。"}),
                             "boxes_json": ("STRING", {"default": "", "multiline": True,
                                                          "tooltip": "检测报告 JSON，用于按 overlay_type 选择预清理策略；通常由定位或残留节点连线。"})}}

    RETURN_TYPES = ("IMAGE", "MASK", "QWM_GEOMETRY", "STRING", "STRING")
    RETURN_NAMES = ("reference_crop", "crop_mask", "geometry", "status", "edit_prompt")
    FUNCTION = "prepare"
    CATEGORY = "Qwen Watermark"

    def prepare(self, image, mask, context_pixels, max_side, pre_inpaint="none", inpaint_radius=3, edit_prompt="", boxes_json=""):
        array = to_array(image)
        if not isinstance(mask, torch.Tensor) or mask.ndim != 3 or mask.shape[0] < 1:
            raise ValueError("MASK 必须是 [batch,height,width] 张量。")
        mask_array = mask[0].detach().cpu().numpy()
        if mask_array.shape != array.shape[:2]:
            raise ValueError("遮罩尺寸与原图不同。请使用原图尺寸的遮罩。")
        if not np.any(mask_array > 0):
            status = json.dumps({
                "status": "no_overlay_detected",
                "message": "自动定位未发现水印或遮挡，后续编辑已跳过。请查看定位坐标；漏检时填写 hint、改用 manual_boxes 或 manual_mask。",
            }, ensure_ascii=False, indent=2)
            blocker = ExecutionBlocker(None)
            return {"ui": {"text": [status]}, "result": (blocker, blocker, blocker, status, blocker)}
        crop, local_mask, geometry = prepare_crop(array, mask_array, context_pixels, max_side)
        requested_prefill = pre_inpaint
        if pre_inpaint == "auto":
            boxes = json.loads(boxes_json).get("boxes", []) if boxes_json else []
            types = {item.get("overlay_type", "unknown") for item in boxes}
            pre_inpaint = "none" if types and types <= {"text_logo", "translucent_text_logo"} else "telea"
        prefill_status = "none"
        if pre_inpaint != "none":
            method = cv2.INPAINT_NS if pre_inpaint == "ns" else cv2.INPAINT_TELEA
            source = np.clip(np.round(crop * 255), 0, 255).astype(np.uint8)
            mask_u8 = (local_mask > 0).astype(np.uint8) * 255
            bgr = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
            cleaned = cv2.inpaint(bgr, mask_u8, inpaint_radius, method)
            filled = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            crop[local_mask > 0] = filled[local_mask > 0]
            prefill_status = pre_inpaint
        geometry["pre_inpaint"] = prefill_status
        geometry["requested_pre_inpaint"] = requested_prefill
        _, _, stats, _ = cv2.connectedComponentsWithStats((local_mask > 0).astype(np.uint8), connectivity=8)
        boxes = []
        ch, cw = local_mask.shape
        for x, y, w, h, area in stats[1:]:
            if area:
                boxes.append([round(x * 1000 / cw), round(y * 1000 / ch),
                              round((x + w) * 1000 / cw), round((y + h) * 1000 / ch)])
        prompt = edit_prompt or build_edit_prompt({"source": "manual_mask", "boxes": []})
        if prefill_status != "none":
            generated_prefix = "编辑<image1>。只修复局部填充痕迹"
            supplements = []
            if prompt.startswith(generated_prefix):
                for name in ("周围可见的原场景材质", "补充要求", "用户编辑指令"):
                    match = re.search(name + r"：(.*?)(?=\n(?:补充要求|用户编辑指令|输出与)|$)", prompt, re.DOTALL)
                    if match:
                        supplements.append(name + "：" + match.group(1))
            else:
                supplements.append(prompt)
            prompt = ("编辑<image1>。将粗糙、模糊或拉丝状填充区域恢复为原本的场景，延续紧邻区域可见的材质、颜色、纹理和线条。其他部分保持原样。"
                      + "\n" + "\n".join(supplements))
        if len(boxes) > 32:
            boxes = [[min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]]
        geometry["local_regions_1000"] = boxes
        status = json.dumps({"status": "ready", "geometry": geometry}, ensure_ascii=False, indent=2)
        return {"ui": {"text": [status]}, "result": (to_tensor(crop), torch.from_numpy(local_mask)[None], geometry, status, prompt)}


class QWMComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"original": ("IMAGE",), "edited_crop": ("IMAGE",), "mask": ("MASK",),
                             "geometry": ("QWM_GEOMETRY",),
                             "feather": ("INT", {"default": 6, "min": 0, "max": 64,
                                                  "tooltip": "贴回边缘的颜色过渡宽度（原图像素）。只缓和边界色差，不能补齐漏选范围；残影应调 expand/dilation。"}),
                             "align": ("BOOLEAN", {"default": True,
                                                    "tooltip": "根据遮罩外上下文估计小幅位移并对齐。报告出现 warning 时应人工检查。"})},
                "optional": {"color_match": ("BOOLEAN", {"default": True,
                                                             "tooltip": "用遮罩外配对像素做有限 RGB 校正，减少局部色漂；不把原遮挡像素混回生成结果。"})}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("restored_image", "comparison", "report")
    FUNCTION = "paste"
    CATEGORY = "Qwen Watermark"

    def paste(self, original, edited_crop, mask, geometry, feather, align, color_match=True):
        array = to_array(original)
        restored, report = composite(array, to_array(edited_crop), mask[0].detach().cpu().numpy(), geometry, feather, align, color_match)
        report_json = json.dumps(report, ensure_ascii=False, indent=2)
        comparison = np.concatenate((array, restored), axis=1)
        return {"ui": {"text": [report_json]}, "result": (to_tensor(restored), to_tensor(comparison), report_json)}


class QWMFinal:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"original": ("IMAGE",), "initial_mask": ("MASK",), "detection_report": ("STRING",)},
                "optional": {"first_pass": ("IMAGE", {"lazy": True}), "first_report": ("STRING", {"lazy": True}),
                             "residual_report": ("STRING", {"lazy": True}), "second_pass": ("IMAGE", {"lazy": True}),
                             "second_report": ("STRING", {"lazy": True}), "residual_mask": ("MASK", {"lazy": True})}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("final_image", "comparison", "quality_report")
    FUNCTION = "select"
    CATEGORY = "Qwen Watermark"

    def check_lazy_status(self, original, initial_mask, detection_report, first_pass=None, first_report=None,
                          residual_report=None, second_pass=None, second_report=None, residual_mask=None):
        if not torch.any(initial_mask > 0):
            return []
        pending = [name for name, value in (("first_pass", first_pass), ("first_report", first_report),
                                            ("residual_report", residual_report)) if value is None]
        if pending:
            return pending
        if json.loads(residual_report).get("status") == "residual_detected":
            return [name for name, value in (("second_pass", second_pass), ("second_report", second_report),
                                             ("residual_mask", residual_mask)) if value is None]
        return []

    def select(self, original, initial_mask, detection_report, first_pass=None, first_report=None,
               residual_report=None, second_pass=None, second_report=None, residual_mask=None):
        array = to_array(original)
        detection = json.loads(detection_report)
        selected = initial_mask[0].detach().cpu().numpy()
        if not np.any(selected > 0):
            final = array.copy()
            status = detection.get("status", "no_overlay_detected")
            report = {"selected_pass": 0, "status": status, "needs_review": status in ("indeterminate", "needs_review")}
            final_mask = selected
        else:
            first = to_array(first_pass)
            first_data, residual = json.loads(first_report), json.loads(residual_report)
            second = to_array(second_pass) if second_pass is not None else None
            second_data = json.loads(second_report) if second_report is not None else {}
            second_mask = residual_mask[0].detach().cpu().numpy() if residual_mask is not None else np.zeros_like(selected)
            final, final_mask, report = choose_final(first, second, selected, second_mask, second_data)
            report["residual_status"] = residual.get("status")
            if first_data.get("alignment", {}).get("warning"):
                report["reasons"].append("first_pass_alignment_uncertain")
            if first_data.get("boundary_jump_excess_p90", 0) > 0.04:
                report["reasons"].append("boundary_discontinuity")
            if first_data.get("color", {}).get("context_mae_after", 0) > 0.04:
                report["reasons"].append("context_color_uncertain")
            if detection.get("status") in ("indeterminate", "needs_review"):
                report["reasons"].append("detection_uncertain")
            if residual.get("status") != "clean":
                report["reasons"].append("residual_unverified")
            report["needs_review"] = bool(report["reasons"])
            report["status"] = "needs_review" if report["needs_review"] else "completed_clean_first_pass"
            report["first_pass_quality"] = first_data
            if second_data:
                report["second_pass_quality"] = second_data
            if ((first_data.get("alignment", {}).get("correlation", 1) < 0.85 or bool(first_data.get("alignment", {}).get("warning"))) and
                    first_data.get("color", {}).get("context_mae_before", 0) > 0.12):
                final, final_mask = array.copy(), np.zeros_like(selected)
                report.update(selected_pass=0, second_pass_accepted=False, status="rejected_first_pass",
                              needs_review=True, reasons=["first_pass_context_regression"])
        outside = final_mask <= 0
        difference = float(np.max(np.abs(final[outside] - array[outside]))) if np.any(outside) else 0.0
        report.update(outside_mask_max_difference=difference, preserved_fraction=float(np.mean(outside)), version="1.3.0")
        if difference > 0:
            raise ValueError("最终结果改变了修复范围外像素，请检查工作流连接。")
        report_json = json.dumps(report, ensure_ascii=False, indent=2)
        return {"ui": {"text": [report_json]},
                "result": (to_tensor(final), to_tensor(np.concatenate((array, final), axis=1)), report_json)}


NODE_CLASS_MAPPINGS = {"QWMDetect": QWMDetect, "QWMResidualDetect": QWMResidualDetect, "QWMFinal": QWMFinal,
                       "QWMRefineMask": QWMRefineMask, "QWMPrepare": QWMPrepare,
                       "QWMComposite": QWMComposite}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QWMDetect": "02 自动定位 / 手动改框",
    "QWMRefineMask": "03 不规则轮廓细化（SAM）",
    "QWMPrepare": "04 裁剪并记录坐标",
    "QWMComposite": "06 对齐 / 保色贴回",
    "QWMResidualDetect": "07 第一次修复后检查残留",
    "QWMFinal": "12 质量检查 / 选择最终结果",
}
