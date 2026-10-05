from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.1.0"
DIST = ROOT / "dist"
PACKAGE = DIST / f"Qwen-Watermark-Workflow-{VERSION}"


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main():
    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)
    PACKAGE.mkdir(parents=True)

    copy_file(ROOT / "workflows" / "qwen21_watermark.json",
              PACKAGE / "workflows" / "qwen21_watermark.json")
    plugin_source = ROOT / "custom_nodes" / "ComfyUI-Qwen-Watermark"
    plugin_target = PACKAGE / "custom_nodes" / "ComfyUI-Qwen-Watermark"
    for name in ("__init__.py", "nodes.py", "processing.py", "requirements.txt", "README.md"):
        copy_file(plugin_source / name, plugin_target / name)
    copy_file(ROOT / "packaging" / "README.md", PACKAGE / "README.md")
    copy_file(ROOT / "packaging" / "install.py", PACKAGE / "install.py")
    copy_file(ROOT / "packaging" / "LICENSE", PACKAGE / "LICENSE")
    copy_file(ROOT / "docs" / "参数与提示词.md", PACKAGE / "docs" / "参数与提示词.md")
    for name in ("demo-before.png", "demo-after.png", "demo-mask.png"):
        copy_file(ROOT / "docs" / "images" / name, PACKAGE / "docs" / "images" / name)

    archive = DIST / f"Qwen-Watermark-Workflow-{VERSION}.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(PACKAGE.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(DIST))
    print(PACKAGE)
    print(archive)


if __name__ == "__main__":
    main()
