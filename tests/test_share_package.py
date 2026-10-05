from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_share_package_sources_include_docs_and_public_demo_assets():
    required = [
        ROOT / "packaging" / "README.md",
        ROOT / "packaging" / "install.py",
        ROOT / "packaging" / "LICENSE",
        ROOT / "docs" / "参数与提示词.md",
        ROOT / "docs" / "images" / "demo-before.png",
        ROOT / "docs" / "images" / "demo-after.png",
        ROOT / "docs" / "images" / "demo-mask.png",
        ROOT / "workflows" / "qwen21_watermark.json",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    assert not missing, f"share package sources missing: {missing}"


def test_share_package_does_not_reference_private_output_images():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    packaging_readme = (ROOT / "packaging" / "README.md").read_text(encoding="utf-8")
    for text in (readme, packaging_readme):
        assert "outputs/" not in text
        assert "diagnostics/" not in text
