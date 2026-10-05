import json
import argparse
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(r"E:\ComfyUI-aki-v3\ComfyUI-aki-v3\ComfyUI\input\dancing.jpg")


def _load_font(size, requested=None):
    candidates = [requested] if requested else []
    candidates.extend([
        r"C:\Windows\Fonts\arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ])
    for candidate in candidates:
        if candidate:
            try:
                return ImageFont.truetype(str(candidate), size)
            except (OSError, ValueError):
                continue
    return ImageFont.load_default()


def main(argv=None):
    parser = argparse.ArgumentParser(description="从一张本机图片生成可复现的水印测试样图")
    parser.add_argument("source", nargs="?", type=Path, default=DEFAULT_SOURCE,
                        help="原始图片，默认读取 ComfyUI input/dancing.jpg")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "samples",
                        help="输出目录，默认是项目 samples")
    parser.add_argument("--font", type=Path, help="可选的粗体 TTF 字体")
    args = parser.parse_args(argv)
    source = args.source.expanduser()
    if not source.is_file():
        parser.error(f"原始图片不存在：{source}。请传入 source 参数。")
    folder = args.output_dir.expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    base = Image.open(source).convert("RGB")
    base.thumbnail((768, 512), Image.Resampling.LANCZOS)
    base.save(folder / "demo_clean.png")
    layer = Image.new("RGBA", base.size)
    draw = ImageDraw.Draw(layer)
    font = _load_font(28, args.font)
    draw.multiline_text((60, 350), "DEMO\nWATERMARK", font=font, fill=(255, 255, 255, 185), stroke_width=1,
                        stroke_fill=(35, 35, 35, 160), spacing=2)
    draw.ellipse((292, 350, 358, 416), fill=(255, 255, 255, 155), outline=(20, 20, 20, 200), width=3)
    draw.text((309, 362), "W", font=_load_font(34, args.font), fill=(15, 15, 15, 230))
    watermarked = Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")
    watermarked.save(folder / "demo_watermarked.png")
    layer.getchannel("A").point(lambda x: 255 if x else 0).save(folder / "demo_exact_mask.png")
    boxes = {"coordinate_space": "pixels", "boxes": [{"label": "DEMO WATERMARK", "bbox": [58, 350, 250, 418]},
                                                        {"label": "W logo", "bbox": [292, 350, 359, 417]}]}
    (folder / "demo_boxes.json").write_text(json.dumps(boxes, indent=2), encoding="utf-8")
    print(folder / "demo_watermarked.png")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
