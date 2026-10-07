import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ("qwen21_watermark", "qwen_watermark_detection", "qwen21_watermark_structure")


def _load(name):
    return json.loads((ROOT / "workflows" / name).read_text(encoding="utf-8"))


def _references(graph):
    for node_id, node in graph.items():
        for input_name, value in node.get("inputs", {}).items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                yield node_id, input_name, value


def test_api_workflow_references_are_present():
    for name in (workflow + ".api.json" for workflow in WORKFLOWS):
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
    assert graph["3"]["inputs"]["detection_side"] == 1024
    assert graph["3"]["inputs"]["max_tokens"] == 512
    assert graph["4"]["inputs"]["method"] == "auto"
    assert graph["4"]["inputs"]["dilation"] == 12
    assert graph["4"]["inputs"]["adaptive_dilation"] is True
    assert graph["3"]["inputs"]["manual_mask"] == ["1", 1]
    assert graph["3"]["inputs"]["edit_prompt"] == ""
    assert graph["5"]["inputs"]["pre_inpaint"] == "auto"
    assert graph["5"]["inputs"]["inpaint_radius"] == 5
    assert graph["10"]["inputs"]["prompt"] == ["5", 4]
    assert graph["5"]["inputs"]["edit_prompt"] == ["3", 3]


def test_full_workflow_rechecks_residuals_and_runs_second_pass_when_needed():
    graph = _load("qwen21_watermark.api.json")
    assert graph["26"]["class_type"] == "QWMResidualDetect"
    assert graph["26"]["inputs"]["image"] == ["14", 0]
    assert graph["27"]["inputs"]["method"] == "auto"
    assert graph["28"]["inputs"]["image"] == ["14", 0]
    assert graph["29"]["inputs"]["prompt"] == ["28", 4]
    assert graph["28"]["inputs"]["edit_prompt"] == ["26", 3]
    assert graph["32"]["inputs"]["original"] == ["14", 0]
    assert graph["33"]["inputs"]["images"] == ["37", 0]
    assert graph["37"]["class_type"] == "QWMFinal"
    assert graph["26"]["inputs"]["allowed_mask"] == ["4", 0]
    assert graph["37"]["inputs"]["residual_mask"] == ["27", 0]
    assert graph["26"]["inputs"]["edit_prompt"] == ""


def test_ui_workflow_links_are_bidirectional():
    for name in (workflow + ".json" for workflow in WORKFLOWS):
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


def test_ui_workflow_widgets_never_contain_null_text_placeholders():
    for name in (workflow + ".json" for workflow in WORKFLOWS):
        for node in _load(name)["nodes"]:
            assert all(value is not None for value in node.get("widgets_values", [])), (
                name, node["id"], "null widget crashes presetText when queueing"
            )


def test_linked_strings_remain_serializable_without_shifting_later_widgets():
    spec = importlib.util.spec_from_file_location("build_workflows", ROOT / "scripts" / "build_workflows.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    graph = {
        "1": {"class_type": "TextSource", "inputs": {"text": "source text"}},
        "26": {"class_type": "ReportConsumer", "inputs": {
            "first_report": ["1", 0], "min_confidence": 0.8,
        }},
    }
    infos = {
        "TextSource": {"input": {"required": {"text": ["STRING", {}]}}, "output": ["STRING"]},
        "ReportConsumer": {"input": {"required": {
            "first_report": ["STRING", {}], "min_confidence": ["FLOAT", {}],
        }}, "output": ["STRING"]},
    }
    workflow = builder.ui_graph(graph, infos)
    consumer = workflow["nodes"][1]
    text, confidence = consumer["widgets_values"]
    assert isinstance(text, str)  # The preset-text extension calls text.replace().
    assert text.replace("unused", "value") == ""
    assert confidence == 0.8
    assert consumer["inputs"][0]["link"] == workflow["links"][0][0]


def test_structure_sampling_freezes_encoded_context_instead_of_an_empty_latent():
    graph = _load("qwen21_watermark_structure.api.json")
    for sampler, crop, encoder, mask in (("12", "5", "39", "40"), ("30", "28", "43", "44")):
        # A mask attached to Qwen's empty latent would freeze black context.
        assert graph[sampler]["inputs"]["latent_image"] == [mask, 0]
        assert graph[mask]["class_type"] == "SetLatentNoiseMask"
        assert graph[mask]["inputs"]["mask"] == [crop, 1]
        assert graph[mask]["inputs"]["samples"] == [encoder, 0]
        assert graph[encoder]["class_type"] == "VAEEncode"
        assert graph[encoder]["inputs"]["pixels"] == [crop, 0]
        assert graph[crop]["inputs"]["pre_inpaint"] == "none"
        assert graph[crop]["inputs"]["context_pixels"] == 384
    assert graph["26"]["inputs"]["enabled"] is False


def test_visible_prompt_reports_are_the_exact_strings_connected_to_edit_encoders():
    for name in ("qwen21_watermark", "qwen21_watermark_structure"):
        graph = _load(name + ".api.json")
        for encoder, preview in (("10", "41"), ("29", "42")):
            assert graph[preview]["class_type"] == "PreviewAny"
            assert graph[preview]["inputs"]["source"] == graph[encoder]["inputs"]["prompt"]
