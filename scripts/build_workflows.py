import json
from pathlib import Path
import uuid

import requests


ROOT = Path(__file__).resolve().parents[1]
EDIT_PROMPT = """编辑<image1>。只移除图像中后期叠加的文字水印、平台标记、Logo、矩形粘贴图案、贴纸、马赛克或其他遮挡贴片，依据紧邻区域和主体结构恢复被遮挡的内容。输入中待修复位置可能已经被预处理成粗糙、模糊或拉丝状色块；这是贴片被预先擦除后的痕迹，必须重建为连续、清晰、自然的背景，不能保留该痕迹。待移除的前景叠加物可能是不规则轮廓，甚至像动物、人物或其他真实物体；它一定是需要移除的后期贴片，不是原始场景对象。完整移除该叠加物的所有可见像素，包括内部细节、外轮廓、四肢、耳朵、投影、描边和半透明残留，不要重画或保留其形状。严格保持原图的相机、透视、物体位置、人物身份、轮廓、材质、光照和整体颜色。人物皮肤、服装边缘、道路标线、建筑边缘和纹理应自然连续。保留场景中的真实文字、招牌、服装本身的连续图案和物体商标。不得改变构图，不得移动任何原有物体，不得裁剪或缩放画面，不添加无关内容，不用模糊色块遮盖。输出与输入尺寸相同的完整局部图像。"""


def api_graph():
    def node(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}

    return {
        "1": node("LoadImage", image="qwm_demo_watermarked.png"),
        "2": node("CLIPLoader", clip_name="qwen3vl_8b_int8_convrot.safetensors", type="qwen_image", device="default"),
        "3": node("QWMDetect", image=["1", 0], clip=["2", 0], mode="auto_qwen",
                  boxes_json='{"coordinate_space":"pixels","boxes":[]}', detection_side=768,
                  expand=16, max_tokens=384, hint="", padding_ratio=0.25),
        "4": node("QWMRefineMask", image=["1", 0], mask=["3", 0], boxes_json=["3", 1], method="sam",
                  sam_model="sam_vit_b_01ec64.pth", device="auto", dilation=12, adaptive_dilation=True),
        "5": node("QWMPrepare", image=["1", 0], mask=["4", 0], context_pixels=96, max_side=768,
                  pre_inpaint="telea", inpaint_radius=5),
        "6": node("UNETLoader", unet_name="qwen_image_2.1_bf16.safetensors", weight_dtype="default"),
        "7": node("ModelAttentionBackend", model=["6", 0], attention="comfy kitchen attention"),
        "8": node("QwenImage21Cache", model=["7", 0], device="off", dtype="default"),
        "9": node("VAELoader", vae_name="qwen_image_2.1_vae_bf16.safetensors"),
        "10": node("TextEncodeQwenImage21", **{"clip": ["2", 0], "vae": ["9", 0],
                  "images.image_1": ["5", 0], "prompt": EDIT_PROMPT, "negative_prompt": "残留水印，残留贴纸，残留动物或人物，模糊涂抹，接缝，构图变化，物体移位", "resolution": 0}),
        # Qwen Image Edit already returns a size-matched edit latent. Feeding it
        # through generic inpaint conditioning encourages reconstruction of the
        # pasted object; use the native latent and enforce locality at composite.
        "12": node("KSampler", model=["8", 0], positive=["10", 0], negative=["10", 1], latent_image=["10", 2],
                   seed=802146, steps=20, cfg=1.0, sampler_name="euler_ancestral", scheduler="sgm_uniform", denoise=1.0),
        "13": node("VAEDecode", samples=["12", 0], vae=["9", 0]),
        "14": node("QWMComposite", original=["1", 0], edited_crop=["13", 0], mask=["4", 0], geometry=["5", 2], feather=6, align=True),
        "15": node("SaveImage", images=["14", 0], filename_prefix="QwenWatermark/restored"),
        "16": node("SaveImage", images=["14", 1], filename_prefix="QwenWatermark/comparison"),
        "17": node("SaveImage", images=["3", 2], filename_prefix="QwenWatermark/detection"),
        "18": node("PreviewImage", images=["5", 0]),
        "19": node("MaskToImage", mask=["4", 0]),
        "20": node("PreviewImage", images=["19", 0]),
        "21": node("SaveImage", images=["13", 0], filename_prefix="QwenWatermark/edited_crop"),
        "22": node("PreviewAny", source=["14", 2]),
        "23": node("PreviewAny", source=["3", 1]),
        "24": node("PreviewAny", source=["5", 3]),
        "25": node("PreviewAny", source=["4", 2]),
    }


POSITIONS = {"1": (40, 100), "2": (40, 400), "3": (430, 100), "4": (870, 100), "5": (1260, 100),
             "6": (40, 900), "7": (400, 900), "8": (720, 900), "9": (40, 1100), "10": (1740, 100),
             "12": (2200, 100), "13": (2200, 620), "14": (2580, 100), "15": (3000, 100),
             "16": (3000, 700), "17": (430, 650), "18": (1260, 480), "19": (1260, 850), "20": (1620, 850),
             "21": (2580, 680), "22": (3400, 100), "23": (430, 1180), "24": (1260, 1180), "25": (870, 1180)}
TITLES = {"1": "01 上传原图", "2": "共享本机 Qwen3-VL", "3": "02 自动定位 / 手动改框",
          "4": "03 不规则轮廓细化（SAM）", "5": "04 裁剪并记录坐标", "10": "05 Qwen 2.1 编辑（尺寸匹配）",
          "12": "局部编辑采样", "14": "06 校验对齐并贴回", "15": "保存修复结果",
          "16": "左右对比：原图 / 结果", "17": "定位预览", "18": "送入模型的局部图", "20": "细化遮罩预览",
          "21": "编辑模型的局部输出", "22": "对齐与回贴报告（检查 warning）", "23": "定位坐标（可复制并修改）",
          "24": "裁剪状态（空检测会静默停止）", "25": "SAM 细化状态（失败会回退矩形框）"}


def ui_graph(graph, infos):
    nodes, links = [], []
    link_id = 0
    for key, spec in graph.items():
        info = infos[spec["class_type"]]
        schema = {**info["input"].get("required", {}), **info["input"].get("optional", {})}
        inputs, widgets, named = [], [], {}
        for name, value in spec["inputs"].items():
            is_link = isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and value[0] in graph
            entry = schema.get(name)
            if entry is None and name.startswith("images."):
                entry = ["IMAGE", {}]
            kind = entry[0]
            options = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
            widget = isinstance(kind, list) or kind in ["INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"]
            if is_link:
                source, slot = value
                link_id += 1
                input_type = infos[graph[source]["class_type"]]["output"][slot]
                links.append([link_id, int(source), slot, int(key), len(inputs), input_type])
                inputs.append({"name": name, "type": input_type, "link": link_id})
                # ComfyUI keeps a placeholder widget for a linked STRING input.
                # Preserve that slot so later widget values do not shift on import.
                if entry is not None and entry[0] == "STRING":
                    widgets.append(None)
            elif widget:
                inputs.append({"name": name, "type": "COMBO" if isinstance(kind, list) else kind,
                               "widget": {"name": name}, "link": None})
                widgets.append(value)
                named[name] = value
                if options.get("control_after_generate"):
                    widgets.append("fixed")
                    named["control_after_generate"] = "fixed"
            else:
                inputs.append({"name": name, "type": kind, "link": None})
        outputs = [{"name": name, "type": kind, "links": []} for name, kind in
                   zip(info.get("output_name", info["output"]), info["output"])]
        height = 210 if spec["class_type"] in ["QWMDetect", "QWMRefineMask"] else 180
        if spec["class_type"] in ["SaveImage", "PreviewImage", "PreviewAny"]:
            height = 440
        if spec["class_type"] in ["QWMDetect", "QWMRefineMask", "TextEncodeQwenImage21"]:
            height = 510
        node = {"id": int(key), "type": spec["class_type"], "title": TITLES.get(key, spec["class_type"]),
                "pos": POSITIONS[key], "size": [400 if spec["class_type"] in ["QWMDetect", "TextEncodeQwenImage21"] else 310, height],
                "flags": {}, "order": len(nodes), "mode": 0, "inputs": inputs, "outputs": outputs,
                "properties": {"Node name for S&R": spec["class_type"]}, "widgets_values": widgets,
                "widgets_values_named": named}
        if spec["class_type"] == "LoadImage":
            node["widgets_values"].append("image")
        nodes.append(node)
    by_id = {node["id"]: node for node in nodes}
    for link in links:
        by_id[link[1]]["outputs"][link[2]]["links"].append(link[0])
    return {"id": str(uuid.uuid4()), "revision": 0, "last_node_id": max(by_id), "last_link_id": link_id,
            "nodes": nodes, "links": links, "groups": [], "config": {},
            "extra": {"ds": {"scale": 0.55, "offset": [20, 30]}}, "version": 0.4}


def main():
    session = requests.Session()
    session.trust_env = False
    response = session.get("http://127.0.0.1:8188/object_info", timeout=30)
    response.raise_for_status()
    infos = response.json()
    folder = ROOT / "workflows"
    folder.mkdir(exist_ok=True)
    graph = api_graph()
    detection = {k: {**graph[k], "inputs": dict(graph[k]["inputs"])}
                 for k in ["1", "2", "3", "17", "19", "20", "23"]}
    # The detection-only graph omits SAM refinement, so its mask preview must
    # consume QWMDetect directly instead of retaining the full graph's node 4.
    detection["19"]["inputs"]["mask"] = ["3", 0]
    for name, subset in [("qwen21_watermark", graph), ("qwen_watermark_detection", detection)]:
        (folder / f"{name}.api.json").write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / f"{name}.json").write_text(json.dumps(ui_graph(subset, infos), ensure_ascii=False, indent=2), encoding="utf-8")
    print("Created importable ComfyUI workflows and API graphs.")


if __name__ == "__main__":
    main()
