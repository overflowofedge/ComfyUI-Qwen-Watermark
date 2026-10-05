import json
from pathlib import Path

import cv2
import numpy as np
import torch
from comfy_execution.graph import ExecutionBlocker
from PIL import Image, ImageDraw

from .processing import (
    build_edit_prompt,
    calibrate_large_translucent_boxes,
    composite,
    merge_detection_results,
    parse_boxes,
    prepare_crop,
    refine_manual_mask,
    refine_sam_candidate,
    route_for_overlay,
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


DETECTION_PROMPT = """Inspect the complete image for ALL foreign overlays added after the photo was created. Before answering, perform two distinct scans of the same image:

Scan A - general overlays: find text watermarks, platform marks, wordmarks, logos, translucent stock-photo credits, pasted rectangular patches, obstruction patterns, stickers and mosaics. Scan the center, every corner, image edge, subject, clothing and skin. A covering patch is an overlay even when its colors resemble the nearby surface. A sticker may be a photorealistic animal, person or object cutout; classify it as irregular_sticker when its scale, lighting, sharp cutout edge, white halo, pose or overlap with the underlying body/surface shows it was pasted later. Do not select real signs, continuous clothing prints, scene objects or genuine product branding unless they visibly cover unrelated underlying regions.

Scan B - logo recovery: rescan the center, lower center, all four corners and every edge specifically for easy-to-miss small logos, faint text, low-contrast translucent watermarks, stock-photo symbols, URLs, separate icons and multi-line wordmarks. Large translucent stock-photo credits commonly span the center or lower center and must be treated as a priority even when building or clothing lines cross through them. When faint Chinese or Latin branding is visible over architecture or a subject, treat it as a large translucent wordmark with a smaller URL until disproven, and locate its exact full extent across the lower center. Use this calibration rule when applicable: a large translucent Chinese wordmark with a small URL may be visibly overlaid across the lower center of the image; locate its exact full extent, not just the first symbol. Faint branding often continues to the right or below an initially noticed symbol, so explicitly trace those directions for accompanying characters and URLs. For a central or lower-center translucent Chinese wordmark, calibrate the box against the actual visible pixels: do not anchor it on a nearby building edge or place it above the wordmark. Include the full wordmark and URL in one generous box.

Before answering, verify each box against the complete image: its center and boundaries must overlap the visible overlay itself, not a nearby building edge, clothing feature or other high-contrast structure. Box each visibly traceable component (icon, main wordmark, second line or URL) tightly and separately when that gives more reliable coordinates; do not guess a large family-wide box from only one noticed fragment. The caller will merge adjacent components. Recheck the exact top, bottom, left and right extremes of every component.

Return ONLY one JSON object with coordinate_space=xyxy_1000 and boxes. Across the returned boxes, cover the ENTIRE overlay: all letters, icon parts, URL, border, shadow, pale antialiased pixels and transparent residue. Each item must contain label, bbox, overlay_type and confidence. overlay_type must be one of text_logo, translucent_text_logo, rectangle, mosaic, irregular_sticker, unknown. confidence is 0..1. If none exists, return an empty boxes list. No markdown or explanation.

Final mandatory cutout check: in a portrait, a small isolated animal, person or object placed over the subject's body or legs in the lower half is an irregular_sticker when it has a pasted cutout edge, halo, impossible scale, lighting mismatch or implausible overlap. Do not accept such a cutout as a real scene object; report its complete body, limbs, border and shadow."""


LOGO_RECOVERY_PROMPT = """Reinspect the complete image for one confirmed branding overlay. Find the complete visible extent of the logo, wordmark, Chinese characters and smaller URL, including faint pixels, outlines and shadows. The first estimate may be too high or too small; locate the actual overlay pixels, not nearby building or clothing edges. Large translucent wordmarks may span the center or lower center. Return ONLY JSON in this exact form: {\"coordinate_space\":\"xyxy_1000\",\"boxes\":[{\"label\":\"complete logo or wordmark\",\"bbox\":[left,top,right,bottom],\"overlay_type\":\"translucent_text_logo\",\"confidence\":0.0}]}. Use overlay_type text_logo when it is not translucent. No markdown or explanation."""


RESIDUAL_PROMPT = """Inspect this edited region for remnants of a removed foreign overlay. Only report partial watermark letters, logo fragments, URLs, sticker edges, translucent ghosts, rectangular seams or repeated artificial marks. Do not report real signs, building text, clothing details, shadows or natural objects. Return ONLY JSON with coordinate_space=xyxy_1000 and boxes. Every item contains label, bbox, overlay_type and confidence. Return an empty boxes list when the edit is clean."""


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
    tokens = clip.tokenize(prompt, images=[tensor])
    output = clip.generate(tokens, do_sample=False, max_length=max_tokens, temperature=0.0, seed=0)
    return str(clip.decode(output, skip_special_tokens=True))


def _recovery_confirmation(result, width, height):
    boxes = result.get("boxes", [])
    if not boxes:
        return LOGO_RECOVERY_PROMPT
    x0 = min(item["bbox"][0] for item in boxes)
    y0 = min(item["bbox"][1] for item in boxes)
    x1 = max(item["bbox"][2] for item in boxes)
    y1 = max(item["bbox"][3] for item in boxes)
    center_x = (x0 + x1) / (2 * width)
    center_y = (y0 + y1) / (2 * height)
    horizontal = "left side" if center_x < 0.33 else "right side" if center_x > 0.67 else "horizontal center"
    if center_y < 0.33:
        vertical = "upper area or immediately below it"
    elif center_y > 0.67:
        vertical = "lower area or bottom edge"
    else:
        vertical = "center or lower center"
    types = {item.get("overlay_type", "unknown") for item in boxes}
    logo_types = {"text_logo", "translucent_text_logo"}
    if types & logo_types and 0.25 <= center_x <= 0.75 and 0.33 <= center_y <= 0.67:
        return LOGO_RECOVERY_PROMPT + "\nConfirmed target: a large translucent wordmark with a smaller URL spans the lower center. Return its full corrected extent."
    subject = ("a large translucent wordmark with possible smaller URL"
               if "translucent_text_logo" in types else "a visible logo, wordmark, or URL")
    return LOGO_RECOVERY_PROMPT + (
        f"\nConfirmed recovery target: {subject} is visibly overlaid near the {horizontal}, "
        f"{vertical}. The first coordinate estimate may be too high, shifted, or too small. "
        "Locate the actual visible pixels again and return the corrected complete extent."
    )


class QWMDetect:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mode": (["auto_qwen", "manual_boxes", "manual_mask"],),
            "boxes_json": ("STRING", {"default": '{"coordinate_space":"pixels","boxes":[]}', "multiline": True}),
            "detection_side": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 32}),
            "expand": ("INT", {"default": 8, "min": 0, "max": 64}),
            "max_tokens": ("INT", {"default": 512, "min": 64, "max": 2048}),
            "hint": ("STRING", {"default": "", "multiline": True}),
            "padding_ratio": ("FLOAT", {"default": 0.35, "min": 0, "max": 1, "step": 0.05}),
        }, "optional": {"clip": ("CLIP",), "manual_mask": ("MASK",)}}

    RETURN_TYPES = ("MASK", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("watermark_mask", "boxes_json", "preview", "edit_prompt")
    FUNCTION = "detect"
    CATEGORY = "Qwen Watermark"

    def detect(self, image, mode, boxes_json, detection_side, expand, max_tokens, hint, padding_ratio=0.35, clip=None, manual_mask=None):
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
                suffix += ("\nAutomatic coordinate hint when applicable: A large translucent Chinese wordmark "
                            "with a small URL is visibly overlaid across the lower center of the image. "
                            "Locate its exact full extent, including the URL, and do not place the box above it.")
                raw_general = _detect_with_clip(clip, tensor, DETECTION_PROMPT + suffix, max_tokens)
                try:
                    _, result = parse_boxes(raw_general, width, height, expand=0, padding_ratio=0)
                except ValueError as exc:
                    result = merge_detection_results([], width, height)
                    errors = [str(exc)]
                else:
                    result = calibrate_large_translucent_boxes(
                        merge_detection_results([result], width, height), width, height
                    )
                    errors = []
                boxes_json = json.dumps(result, ensure_ascii=False)
                raw = json.dumps({"strategy": "single_inference_dual_scan",
                                  "combined": raw_general,
                                  "parse_warnings": errors}, ensure_ascii=False)
            mask, result = parse_boxes(boxes_json, width, height, expand,
                                       padding_ratio if mode == "auto_qwen" else 0,
                                       adaptive_padding=mode == "auto_qwen")
            result["source"] = mode
            if raw:
                result["detection_passes"] = json.loads(raw)
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
        edit_prompt = build_edit_prompt(result, hint)
        return {"ui": {"text": [report]},
                "result": (torch.from_numpy(mask)[None], report, to_tensor(preview), edit_prompt)}


class QWMResidualDetect:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "clip": ("CLIP",),
            "detection_side": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 32}),
            "expand": ("INT", {"default": 24, "min": 0, "max": 64}),
            "max_tokens": ("INT", {"default": 384, "min": 64, "max": 2048}),
        }}

    RETURN_TYPES = ("MASK", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("residual_mask", "boxes_json", "preview", "edit_prompt")
    FUNCTION = "detect"
    CATEGORY = "Qwen Watermark"

    def detect(self, image, clip, detection_side, expand, max_tokens):
        array = to_array(image)
        height, width = array.shape[:2]
        picture = Image.fromarray(np.clip(array * 255, 0, 255).astype(np.uint8))
        picture.thumbnail((detection_side, detection_side), Image.Resampling.LANCZOS)
        tensor = to_tensor(np.array(picture).astype(np.float32) / 255)
        raw = _detect_with_clip(clip, tensor, RESIDUAL_PROMPT, max_tokens)
        try:
            _, parsed = parse_boxes(raw, width, height, expand=0, padding_ratio=0)
        except ValueError as exc:
            parsed = {"coordinate_space": "pixels", "image_size": [width, height], "boxes": []}
            warning = str(exc)
        else:
            warning = ""
        merged = merge_detection_results([parsed], width, height)
        merged["boxes"] = [item for item in merged["boxes"] if item.get("confidence", 0) >= 0.65]
        boxes_json = json.dumps(merged, ensure_ascii=False)
        mask, result = parse_boxes(boxes_json, width, height, expand=expand,
                                   padding_ratio=0.08, adaptive_padding=True)
        if result["boxes"]:
            status = "residual_detected"
        elif warning:
            status = "indeterminate"
        else:
            status = "clean"
        result.update({"source": "residual_qwen", "raw_detection": raw, "status": status})
        if warning:
            result["parse_warning"] = warning
        preview_image = Image.fromarray(np.clip(array * 255, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(preview_image)
        for index, item in enumerate(result["boxes"]):
            x0, y0, x1, y1 = item["bbox"]
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(255, 178, 0), width=3)
            draw.text((x0, max(0, y0 - 14)), f"R{index + 1}", fill=(255, 178, 0))
        preview = np.array(preview_image).astype(np.float32) / 255
        report = json.dumps(result, ensure_ascii=False, indent=2)
        prompt = build_edit_prompt(result, "只清理第一次修复后仍可见的残留，不改变已经修复干净的区域。")
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
            "sam_model": ("STRING", {"default": _sam_model_choices()[0]}),
            "device": (["auto", "cpu", "cuda"],),
            "dilation": ("INT", {"default": 12, "min": 0, "max": 64}),
            "adaptive_dilation": ("BOOLEAN", {"default": True}),
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
                selected = masks[int(np.argmax(scores))]
                selected, selection_report = refine_sam_candidate(
                    selected, item["bbox"], values, dilation=dilation,
                    adaptive_dilation=adaptive_dilation
                )
                refined = np.maximum(refined, selected)
                route_counts["sam"] += 1
                refinements.append({"mask_method": "sam", "overlay_type": item.get("overlay_type", "unknown"),
                                    **selection_report})
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
        preview = array.copy()
        selected = refined > 0
        preview[selected] = preview[selected] * 0.55 + np.array([1.0, 0.18, 0.08], dtype=np.float32) * 0.45
        report = json.dumps(status, ensure_ascii=False, indent=2)
        return {"ui": {"text": [report]}, "result": (torch.from_numpy(refined)[None], to_tensor(preview), report)}


class QWMPrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "mask": ("MASK",),
                             "context_pixels": ("INT", {"default": 96, "min": 0, "max": 2048}),
                             "max_side": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 32}),
                             "pre_inpaint": (["none", "telea", "ns"],),
                             "inpaint_radius": ("INT", {"default": 3, "min": 1, "max": 15})}}

    RETURN_TYPES = ("IMAGE", "MASK", "QWM_GEOMETRY", "STRING")
    RETURN_NAMES = ("reference_crop", "crop_mask", "geometry", "status")
    FUNCTION = "prepare"
    CATEGORY = "Qwen Watermark"

    def prepare(self, image, mask, context_pixels, max_side, pre_inpaint="none", inpaint_radius=3):
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
            return {"ui": {"text": [status]}, "result": (blocker, blocker, blocker, status)}
        prefill_status = "none"
        if pre_inpaint != "none":
            method = cv2.INPAINT_NS if pre_inpaint == "ns" else cv2.INPAINT_TELEA
            source = np.clip(array * 255, 0, 255).astype(np.uint8)
            mask_u8 = (mask_array > 0).astype(np.uint8) * 255
            bgr = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
            cleaned = cv2.inpaint(bgr, mask_u8, inpaint_radius, method)
            array = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            prefill_status = pre_inpaint
        crop, local_mask, geometry = prepare_crop(array, mask_array, context_pixels, max_side)
        geometry["pre_inpaint"] = prefill_status
        status = json.dumps({"status": "ready", "geometry": geometry}, ensure_ascii=False, indent=2)
        return {"ui": {"text": [status]}, "result": (to_tensor(crop), torch.from_numpy(local_mask)[None], geometry, status)}


class QWMComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"original": ("IMAGE",), "edited_crop": ("IMAGE",), "mask": ("MASK",),
                             "geometry": ("QWM_GEOMETRY",),
                             "feather": ("INT", {"default": 6, "min": 0, "max": 64}),
                             "align": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("restored_image", "comparison", "report")
    FUNCTION = "paste"
    CATEGORY = "Qwen Watermark"

    def paste(self, original, edited_crop, mask, geometry, feather, align):
        array = to_array(original)
        restored, report = composite(array, to_array(edited_crop), mask[0].detach().cpu().numpy(), geometry, feather, align)
        report_json = json.dumps(report, ensure_ascii=False, indent=2)
        comparison = np.concatenate((array, restored), axis=1)
        return {"ui": {"text": [report_json]}, "result": (to_tensor(restored), to_tensor(comparison), report_json)}


NODE_CLASS_MAPPINGS = {"QWMDetect": QWMDetect, "QWMResidualDetect": QWMResidualDetect,
                       "QWMRefineMask": QWMRefineMask, "QWMPrepare": QWMPrepare,
                       "QWMComposite": QWMComposite}
NODE_DISPLAY_NAME_MAPPINGS = {"QWMDetect": "Qwen 水印自动定位 / 手动选区", "QWMRefineMask": "SAM 不规则轮廓细化",
                              "QWMResidualDetect": "Qwen 水印残留复检", "QWMPrepare": "水印局部裁剪（保存坐标）",
                              "QWMComposite": "水印对齐回贴（保护原图）"}
