import argparse
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="仅安装本项目节点，不修改 ComfyUI 核心或其他插件")
    parser.add_argument("--comfy-root", type=Path, default=Path(r"E:\ComfyUI-aki-v3\ComfyUI-aki-v3\ComfyUI"))
    args = parser.parse_args()
    if not (args.comfy_root / "main.py").is_file():
        parser.error("指定目录不是 ComfyUI 根目录。")
    name = "ComfyUI-Qwen-Watermark"
    target = args.comfy_root / "custom_nodes" / name
    target.mkdir(exist_ok=True)
    for filename in ("__init__.py", "nodes.py", "processing.py", "requirements.txt", "README.md"):
        source = ROOT / "custom_nodes" / name / filename
        shutil.copy2(source, target / filename)
    print("Installed:", target)
    print("Reload ComfyUI to register the five nodes.")


if __name__ == "__main__":
    main()
