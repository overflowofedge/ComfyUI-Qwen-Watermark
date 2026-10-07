import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(description="顺序处理目录图片；单张失败继续，逐张保存报告")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "batch")
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--skip-residual", action="store_true")
    parser.add_argument("--resume", action="store_true", help="跳过已有最终结果和报告的图片")
    args = parser.parse_args()
    if not args.folder.is_dir():
        parser.error("输入路径必须是图片目录。")
    output = args.output.resolve()
    if args.folder.resolve() == output:
        parser.error("输出目录必须与输入目录不同。")
    images = sorted(path for path in args.folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        parser.error("目录内没有可处理的图片。")
    output.mkdir(parents=True, exist_ok=True)
    workflow_hash = hashlib.sha256((ROOT / "workflows" / "qwen21_watermark.api.json").read_bytes()).hexdigest()
    items = []
    for index, image in enumerate(images):
        target = output / image.name
        report_path = target / "report.json"
        if args.resume and report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            restored = report.get("files", {}).get("restored")
            if (restored and Path(restored).is_file() and
                    report.get("input_sha256") == hashlib.sha256(image.read_bytes()).hexdigest() and
                    report.get("workflow_sha256") == workflow_hash and
                    report.get("options", {}).get("skip_residual") == args.skip_residual):
                items.append({"image": str(image), "status": report["status"], "resumed": True})
                continue
        target.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(ROOT / "scripts" / "run_workflow.py"), str(image),
                   "--server", args.server, "--output", str(target)]
        if args.skip_residual:
            command.append("--skip-residual")
        print(f"[{index + 1}/{len(images)}] {image.name}", flush=True)
        start = time.monotonic()
        with (target / "run.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        status = "failed"
        if completed.returncode == 0 and report_path.is_file():
            status = json.loads(report_path.read_text(encoding="utf-8"))["status"]
        items.append({"image": str(image), "status": status, "exit_code": completed.returncode,
                      "elapsed_seconds": round(time.monotonic() - start, 2), "output": str(target)})
        (output / "batch_report.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "batch_report.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(items, ensure_ascii=False, indent=2))
    return int(any(item["status"] == "failed" for item in items))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
