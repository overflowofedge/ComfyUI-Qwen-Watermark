import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("processing", ROOT / "custom_nodes/ComfyUI-Qwen-Watermark/processing.py")
processing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(processing)


def main():
    parser = argparse.ArgumentParser(description="用保存的裁剪图和坐标重新贴回，无需重新采样")
    parser.add_argument("run_dir", type=Path, help="包含 report.json、edited_crop.png、comparison*.png 的任务目录")
    parser.add_argument("--mask", type=Path, help="原图尺寸的实际 MASK；SAM/手绘任务必须提供")
    parser.add_argument("--feather", type=int, default=6)
    parser.add_argument("--no-color-match", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run_dir
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    geometry = report["composite"]["geometry"]
    width, height = geometry["image_size"]
    comparison = next((p for p in (run / "comparison_first_pass.png", run / "comparison.png") if p.is_file()), None)
    if comparison is None:
        parser.error("缺少包含实际 ComfyUI 输入像素的 comparison 图。")
    original = np.array(Image.open(comparison).convert("RGB"), dtype=np.float32)[:height, :width] / 255
    edited = np.array(Image.open(run / "edited_crop.png").convert("RGB"), dtype=np.float32) / 255
    if args.mask:
        mask = np.array(Image.open(args.mask).convert("L"), dtype=np.float32) / 255
    else:
        detection = report["detection"]
        refine = report.get("refine", {})
        if detection.get("source") == "manual_mask" or refine.get("route_counts", {}).get("sam", 0) or refine.get("method") == "sam":
            parser.error("SAM/手绘任务必须用 --mask 指定实际遮罩，不能使用检测框代替。")
        mask, _ = processing.parse_boxes(json.dumps(detection), width, height, expand=0)
    restored, quality = processing.composite(original, edited, mask, geometry, feather=args.feather,
                                              align=True, color_match=not args.no_color_match)
    output = args.output or run / "recomposited_v13"
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(restored * 255).astype(np.uint8)).save(output / "restored.png")
    Image.fromarray(np.rint(np.concatenate((original, restored), axis=1) * 255).astype(np.uint8)).save(output / "comparison.png")
    (output / "quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)
    print(json.dumps(quality, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
