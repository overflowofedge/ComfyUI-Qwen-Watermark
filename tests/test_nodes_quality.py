import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch


@pytest.fixture
def nodes(monkeypatch):
    # Only the ComfyUI execution boundary is stubbed; image operations are real.
    graph = types.ModuleType("comfy_execution.graph")
    graph.ExecutionBlocker = type("ExecutionBlocker", (), {"__init__": lambda self, value: None})
    monkeypatch.setitem(sys.modules, "comfy_execution.graph", graph)
    root = Path(__file__).resolve().parents[1] / "custom_nodes/ComfyUI-Qwen-Watermark"
    spec = importlib.util.spec_from_file_location("qwm_test", root / "__init__.py", submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "qwm_test", module)
    spec.loader.exec_module(module)
    return sys.modules["qwm_test.nodes"]


def test_prepare_prefill_preserves_float_context_and_only_processes_crop(nodes):
    image = torch.rand((1, 300, 400, 3))
    mask = torch.zeros((1, 300, 400))
    mask[:, 100:140, 160:200] = 1
    direct = nodes.QWMPrepare().prepare(image, mask, 25, 256, "none")["result"]
    filled = nodes.QWMPrepare().prepare(image, mask, 25, 256, "telea", 3)["result"]
    np.testing.assert_array_equal(direct[0].numpy()[0][direct[1].numpy()[0] == 0],
                                  filled[0].numpy()[0][filled[1].numpy()[0] == 0])
    assert filled[0].shape[1] < image.shape[1]


def test_prefill_prompt_requests_completion_and_keeps_multiline_user_instructions(nodes):
    image = torch.rand((1, 128, 128, 3))
    mask = torch.zeros((1, 128, 128))
    mask[:, 50:70, 50:70] = 1
    prompt = nodes.build_edit_prompt({"boxes": [{"overlay_type": "text_logo", "background": "road surface"}]},
                                     edit_prompt="保留道路标线。\n保持地面颜色。")
    result = nodes.QWMPrepare().prepare(image, mask, 20, 256, "telea", 3, prompt)["result"]
    assert "完整移除文字水印" not in result[4]
    assert "恢复为原本的场景" in result[4]
    assert "road surface" in result[4]
    assert "保留道路标线。\n保持地面颜色。" in result[4]
    direct = nodes.QWMPrepare().prepare(image, mask, 20, 256, "none", 3, prompt)["result"]
    assert "完整移除文字水印" in direct[4]


def test_auto_prefill_preserves_translucent_context_and_clears_opaque_patches(nodes):
    image = torch.rand((1, 128, 128, 3))
    mask = torch.zeros((1, 128, 128))
    mask[:, 50:70, 50:70] = 1
    text = nodes.QWMPrepare().prepare(image, mask, 20, 256, "auto", boxes_json='{"boxes":[{"overlay_type":"translucent_text_logo"}]}')["result"]
    patch = nodes.QWMPrepare().prepare(image, mask, 20, 256, "auto", boxes_json='{"boxes":[{"overlay_type":"rectangle"}]}')["result"]
    assert text[2]["pre_inpaint"] == "none"
    assert patch[2]["pre_inpaint"] == "telea"


def test_empty_final_does_not_request_model_branches_and_returns_original(nodes):
    image = torch.rand((1, 32, 48, 3))
    mask = torch.zeros((1, 32, 48))
    node = nodes.QWMFinal()
    detection = json.dumps({"status": "indeterminate"})
    assert node.check_lazy_status(image, mask, detection) == []
    restored, _, quality = node.select(image, mask, detection)["result"]
    assert torch.equal(restored, image)
    assert json.loads(quality)["needs_review"]


def test_final_clean_first_pass_does_not_request_second_pass(nodes):
    image = torch.rand((1, 32, 48, 3))
    mask = torch.ones((1, 32, 48))
    node = nodes.QWMFinal()
    assert node.check_lazy_status(image, mask, "{}", image, "{}", '{"status":"clean"}') == []
    assert set(node.check_lazy_status(image, mask, "{}", image, "{}", '{"status":"residual_detected"}')) == {"second_pass", "second_report", "residual_mask"}


def test_residual_coordinates_are_mapped_from_crop_and_clipped_to_guard(nodes, monkeypatch):
    image = torch.rand((1, 800, 1000, 3))
    mask = torch.zeros((1, 800, 1000))
    mask[:, 350:450, 400:500] = 1
    raw = json.dumps({"coordinate_space": "xyxy_1000", "boxes": [
        {"bbox": [400, 400, 600, 600], "confidence": 0.95, "overlay_type": "text_logo"}]})
    monkeypatch.setattr(nodes, "_detect_with_clip", lambda *args: raw)
    residual, report, _, _ = nodes.QWMResidualDetect().detect(image, None, 1024, 24, 384, allowed_mask=mask)["result"]
    report = json.loads(report)
    assert report["inspection_roi"] == [336, 286, 564, 514]
    assert report["boxes"][0]["source_bbox"] == [427, 377, 473, 423]
    assert report["status"] == "residual_detected"
    assert torch.count_nonzero(residual[:, :320]) == 0


def test_detector_parse_failure_is_never_reported_as_clean(nodes, monkeypatch):
    image = torch.rand((1, 64, 64, 3))
    monkeypatch.setattr(nodes, "_detect_with_clip", lambda *args: "truncated invalid answer")
    mask, report, _, _ = nodes.QWMDetect().detect(image, "auto_qwen", "{}", 256, 8, 512, "", clip=object())["result"]
    assert not torch.any(mask)
    assert json.loads(report)["status"] == "indeterminate"


def test_auto_detector_never_shifts_central_logo_by_a_fixed_offset(nodes, monkeypatch):
    image = torch.rand((1, 747, 500, 3))
    raw = '{"coordinate_space":"pixels","boxes":[{"bbox":[162,281,288,334],"overlay_type":"translucent_text_logo","confidence":0.98}]}'
    monkeypatch.setattr(nodes, "_detect_with_clip", lambda *args: raw)
    _, report, _, _ = nodes.QWMDetect().detect(image, "auto_qwen", "{}", 1024, 0, 512, "", padding_ratio=0, clip=object())["result"]
    assert json.loads(report)["boxes"][0]["source_bbox"] == [162, 281, 288, 334]


def test_final_rejects_catastrophic_context_change(nodes):
    image = torch.full((1, 64, 64, 3), 0.4)
    mask = torch.zeros((1, 64, 64))
    mask[:, 20:40, 20:40] = 1
    bad = image.clone()
    bad[:, 20:40, 20:40] = 0.9
    quality = json.dumps({"alignment": {"correlation": 0.3}, "color": {"context_mae_before": 0.25}})
    final, _, report = nodes.QWMFinal().select(image, mask, "{}", bad, quality, '{"status":"clean"}')["result"]
    assert torch.equal(final, image)
    assert json.loads(report)["status"] == "rejected_first_pass"


def test_nonconverged_alignment_with_bad_context_also_rejects_first_pass(nodes):
    image = torch.full((1, 32, 32, 3), 0.4)
    mask = torch.zeros((1, 32, 32))
    mask[:, 10:20, 10:20] = 1
    quality = json.dumps({"alignment": {"warning": "not converged"}, "color": {"context_mae_before": 0.25}})
    final, _, report = nodes.QWMFinal().select(image, mask, "{}", image * 0.5, quality, '{"status":"clean"}')["result"]
    assert torch.equal(final, image)
    assert json.loads(report)["status"] == "rejected_first_pass"
