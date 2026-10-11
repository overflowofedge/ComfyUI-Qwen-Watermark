from pathlib import Path
import ast


ROOT = Path(__file__).resolve().parents[1]


def test_repository_root_is_a_comfyui_custom_node():
    required = [ROOT / "__init__.py", ROOT / "nodes.py", ROOT / "processing.py", ROOT / "requirements.txt"]
    assert all(path.is_file() for path in required)
    assert not (ROOT / "custom_nodes" / "ComfyUI-Qwen-Watermark" / "__init__.py").exists()
    tree = ast.parse((ROOT / "__init__.py").read_text(encoding="utf-8"))
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                for alias in node.names}
    assert {"NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"} <= imported


def test_readme_documents_git_url_installation():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Install via Git URL" in readme
    assert "https://github.com/overflowofedge/ComfyUI-Qwen-Watermark.git" in readme


def test_share_package_sources_include_docs_and_public_demo_assets():
    required = [
        ROOT / "packaging" / "README.md",
        ROOT / "packaging" / "install.py",
        ROOT / "packaging" / "LICENSE",
        ROOT / "LICENSE",
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
