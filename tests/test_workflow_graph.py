import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    return json.loads((ROOT / "workflows" / name).read_text(encoding="utf-8"))


def _references(graph):
    for node_id, node in graph.items():
        for input_name, value in node.get("inputs", {}).items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                yield node_id, input_name, value


def test_api_workflow_references_are_present():
    for name in ("qwen21_watermark.api.json", "qwen_watermark_detection.api.json"):
        graph = _load(name)
        missing = [(node_id, input_name, value[0])
                   for node_id, input_name, value in _references(graph)
                   if value[0] not in graph]
        assert not missing, f"{name} has dangling links: {missing}"


def test_detection_workflow_preview_uses_detector_mask():
    graph = _load("qwen_watermark_detection.api.json")
    assert graph["19"]["inputs"]["mask"] == ["3", 0]
    assert graph["3"]["class_type"] == "QWMDetect"


def test_full_workflow_uses_qwen_native_edit_latent_and_irregular_defaults():
    graph = _load("qwen21_watermark.api.json")
    assert "11" not in graph
    assert graph["12"]["inputs"]["positive"] == ["10", 0]
    assert graph["12"]["inputs"]["negative"] == ["10", 1]
    assert graph["12"]["inputs"]["latent_image"] == ["10", 2]
    assert graph["4"]["inputs"]["method"] == "sam"
    assert graph["4"]["inputs"]["dilation"] == 12
    assert graph["4"]["inputs"]["adaptive_dilation"] is True
    assert graph["5"]["inputs"]["pre_inpaint"] == "telea"
    assert graph["5"]["inputs"]["inpaint_radius"] == 5


def test_ui_workflow_links_are_bidirectional():
    for name in ("qwen21_watermark.json", "qwen_watermark_detection.json"):
        workflow = _load(name)
        nodes = {str(node["id"]): node for node in workflow["nodes"]}
        links = {link[0]: link for link in workflow["links"]}
        assert workflow["last_link_id"] == max(links, default=0)
        for link_id, source_id, source_slot, target_id, target_slot, _ in workflow["links"]:
            assert str(source_id) in nodes, (name, link_id, "missing source")
            assert str(target_id) in nodes, (name, link_id, "missing target")
            source = nodes[str(source_id)]["outputs"][source_slot]
            target = nodes[str(target_id)]["inputs"][target_slot]
            assert link_id in (source.get("links") or []), (name, link_id, "missing source backlink")
            assert target.get("link") == link_id, (name, link_id, "missing target backlink")
