import json
from pathlib import Path

import cv2
import numpy as np
import torch
from comfy_execution.graph import ExecutionBlocker
from PIL import Image, ImageDraw

from .processing import composite, parse_boxes, prepare_crop, refine_manual_mask, refine_sam_candidate

try:
    import folder_paths
except ImportError:  # Allows the processing tests to import this module outside ComfyUI.
    folder_paths = None

try:
    from segment_anything import SamPredictor, sam_model_registry
except ImportError:
    SamPredictor = None
    sam_model_registry = {}


DETECTION_PROMPT = """Locate ALL foreign overlays that cover the original photo. This includes text watermarks, platform marks, logos, translucent stock-photo credits, pasted rectangular image patches, obstruction patterns, stickers and mosaics. Scan the center, subjects, clothing, skin and every edge. Look for hard rectangular seams, duplicated texture, a patch that does not follow the body or surface perspective, and a pattern placed across multiple underlying regions. A covering patch is an overlay even when its colors resemble nearby clothing. Do not select a normal continuous clothing print, real sign, product branding or scene object unless it has clear pasted boundaries or covers unrelated underlying content. Each box must cover the ENTIRE overlay including all strokes, border, shadow and pasted area. If there are no foreign overlays, return an empty boxes list. Return ONLY one JSON object. coordinate_space must be xyxy_1000. boxes is a list of objects containing label and bbox. Each bbox is [left,top,right,bottom], normalized to 0..1000 relative to the complete input image. No markdown or explanation."""


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


class QWMDetect:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mode": (["auto_qwen", "manual_boxes", "manual_mask"],),
            "boxes_json": ("STRING", {"default": '{"coordinate_space":"pixels","boxes":[]}', "multiline": True}),
            "detection_side": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 32}),
            "expand": ("INT", {"default": 8, "min": 0, "max": 64}),
            "max_tokens": ("INT", {"default": 384, "min": 64, "max": 2048}),
            "hint": ("STRING", {"default": "", "multiline": True}),
            "padding_ratio": ("FLOAT", {"default": 0.25, "min": 0, "max": 1, "step": 0.05}),
        }, "optional": {"clip": ("CLIP",), "manual_mask": ("MASK",)}}

    RETURN_TYPES = ("MASK", "STRING", "IMAGE")
    RETURN_NAMES = ("watermark_mask", "boxes_json", "preview")
    FUNCTION = "detect"
    CATEGORY = "Qwen Watermark"

    def detect(self, image, mode, boxes_json, detection_side, expand, max_tokens, hint, padding_ratio=0.25, clip=None, manual_mask=None):
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
                prompt = DETECTION_PROMPT + ("\nAdditional watermark description: " + hint if hint.strip() else "")
                tokens = clip.tokenize(prompt, images=[tensor])
                output = clip.generate(tokens, do_sample=False, max_length=max_tokens, temperature=0.0, seed=0)
                raw = str(clip.decode(output, skip_special_tokens=True))
                boxes_json = raw
            mask, result = parse_boxes(boxes_json, width, height, expand, padding_ratio if mode == "auto_qwen" else 0)
            result["source"] = mode
            if raw:
                result["raw_detection"] = raw
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
        return {"ui": {"text": [report]}, "result": (torch.from_numpy(mask)[None], report, to_tensor(preview))}


class QWMRefineMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "boxes_json": ("STRING", {"multiline": True}),
            "method": (["sam", "box"],),
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
        try:
            predictor, selected_device = _load_sam(sam_model, device)
            rgb = np.clip(array * 255, 0, 255).astype(np.uint8)
            predictor.set_image(rgb)
            refined = np.zeros_like(base, dtype=np.float32)
            refined_count = 0
            refinements = []
            for item in boxes:
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
                refinements.append(selection_report)
                refined_count += 1
            if not np.any(refined):
                raise RuntimeError("SAM 未生成有效轮廓。")
            status = {"status": "sam", "method": "sam", "model": sam_model, "device": selected_device,
                      "boxes_refined": refined_count, "base_pixels": int(np.count_nonzero(base)),
                      "refined_pixels": int(np.count_nonzero(refined)), "dilation": dilation,
                      "adaptive_dilation": adaptive_dilation,
                      "prompt_boxes": [item.get("source_bbox", item["bbox"]) for item in boxes],
                      "refinements": refinements}
        except Exception as exc:
            status = {"status": "box_fallback", "method": "sam", "warning": str(exc),
                      "base_pixels": int(np.count_nonzero(base))}
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


NODE_CLASS_MAPPINGS = {"QWMDetect": QWMDetect, "QWMRefineMask": QWMRefineMask,
                       "QWMPrepare": QWMPrepare, "QWMComposite": QWMComposite}
NODE_DISPLAY_NAME_MAPPINGS = {"QWMDetect": "Qwen 水印自动定位 / 手动选区", "QWMRefineMask": "SAM 不规则轮廓细化",
                              "QWMPrepare": "水印局部裁剪（保存坐标）", "QWMComposite": "水印对齐回贴（保护原图）"}
