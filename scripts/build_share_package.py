from pathlib import Path
import hashlib
import json
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.3.0"
DIST = ROOT / "dist"
PACKAGE = DIST / f"Qwen-Watermark-Workflow-{VERSION}"


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main():
    if PACKAGE.resolve().parent != DIST.resolve():
        raise RuntimeError("分享包路径必须位于 dist 目录内。")
    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)
    PACKAGE.mkdir(parents=True)

    copy_file(ROOT / "workflows" / "qwen21_watermark.json",
              PACKAGE / "workflows" / "qwen21_watermark.json")
    for name in ("qwen21_watermark.api.json", "qwen_watermark_detection.json", "qwen_watermark_detection.api.json",
                 "qwen21_watermark_structure.json", "qwen21_watermark_structure.api.json"):
        copy_file(ROOT / "workflows" / name, PACKAGE / "workflows" / name)
    for name in ("run_workflow.py", "run_batch.py", "replay_composite.py"):
        copy_file(ROOT / "scripts" / name, PACKAGE / "scripts" / name)
    plugin_source = ROOT
    plugin_target = PACKAGE / "custom_nodes" / "ComfyUI-Qwen-Watermark"
    for name in ("__init__.py", "nodes.py", "processing.py", "requirements.txt", "README.md"):
        copy_file(plugin_source / name, plugin_target / name)
    copy_file(ROOT / "packaging" / "README.md", PACKAGE / "README.md")
    copy_file(ROOT / "packaging" / "install.py", PACKAGE / "install.py")
    copy_file(ROOT / "packaging" / "LICENSE", PACKAGE / "LICENSE")
    copy_file(ROOT / "docs" / "参数与提示词.md", PACKAGE / "docs" / "参数与提示词.md")
    copy_file(ROOT / "docs" / "产品化与验收指南.md", PACKAGE / "docs" / "产品化与验收指南.md")
    for name in ("demo-before.png", "demo-after.png", "demo-mask.png"):
        copy_file(ROOT / "docs" / "images" / name, PACKAGE / "docs" / "images" / name)

    manifest = {"version": VERSION, "files": {path.relative_to(PACKAGE).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(PACKAGE.rglob("*")) if path.is_file()}}
    (PACKAGE / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    archive = DIST / f"Qwen-Watermark-Workflow-{VERSION}.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(PACKAGE.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(DIST))
    print(PACKAGE)
    print(archive)
    print("SHA256:", hashlib.sha256(archive.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
