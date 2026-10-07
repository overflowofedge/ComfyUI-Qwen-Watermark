import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


PLUGIN = "ComfyUI-Qwen-Watermark"
MODELS = (
    ("qwen_image_2.1_bf16.safetensors", ("unet", "diffusion_models")),
    ("qwen3vl_8b_int8_convrot.safetensors", ("clip", "text_encoders")),
    ("qwen_image_2.1_vae_bf16.safetensors", ("vae",)),
    ("sam_vit_b_01ec64.pth", ("sams",)),
)


def find_python(comfy_root):
    candidates = [
        comfy_root.parent / "python" / "python.exe",
        comfy_root.parent / "python_embeded" / "python.exe",
    ]
    return next((path for path in candidates if path.is_file()), Path(sys.executable))


def model_exists(comfy_root, filename, folders):
    return any((comfy_root / "models" / folder / filename).is_file() for folder in folders)


def main():
    parser = argparse.ArgumentParser(description="安装 Qwen Watermark 自定义节点并检查模型")
    parser.add_argument("comfy_root", type=Path, help="包含 main.py 的 ComfyUI 根目录")
    parser.add_argument("--skip-deps", action="store_true", help="不执行 pip 依赖安装")
    parser.add_argument("--check-only", action="store_true", help="只检查环境，不复制或安装")
    args = parser.parse_args()
    root = args.comfy_root.expanduser().resolve()
    if not (root / "main.py").is_file():
        parser.error(f"不是有效的 ComfyUI 根目录：{root}")

    package_root = Path(__file__).resolve().parent
    source = package_root / "custom_nodes" / PLUGIN
    target = root / "custom_nodes" / PLUGIN
    if not args.check_only:
        manifest_path = package_root / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for name, expected in manifest["files"].items():
                path = (package_root / name).resolve()
                if not path.is_relative_to(package_root.resolve()) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f"分享包文件校验失败：{name}")
        if target.is_dir():
            backup = root / ".qwm-backups" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            shutil.copytree(target, backup, ignore=shutil.ignore_patterns("__pycache__", ".git"))
            print(f"旧节点已备份：{backup}")
        shutil.copytree(source, target, dirs_exist_ok=True)
        if not args.skip_deps:
            python = find_python(root)
            subprocess.run(
                [str(python), "-m", "pip", "install", "-r", str(target / "requirements.txt")],
                check=True,
            )
        print(f"节点已安装：{target}")

    missing = [filename for filename, folders in MODELS if not model_exists(root, filename, folders)]
    if missing:
        print("缺少模型：")
        for filename in missing:
            print(f"  - {filename}")
    else:
        print("四个模型文件均已找到。")

    if args.check_only:
        print("检查完成，未修改文件。")
    else:
        print("请重启 ComfyUI，并在浏览器按 Ctrl+F5。")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
