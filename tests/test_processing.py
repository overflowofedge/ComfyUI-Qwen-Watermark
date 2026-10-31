import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest


path = Path(__file__).resolve().parents[1] / "processing.py"
spec = importlib.util.spec_from_file_location("processing", path)
processing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(processing)


def test_detector_coordinates_map_back_to_original():
    text = json.dumps({"coordinate_space": "xyxy_1000", "boxes": [{"label": "text", "bbox": [100, 200, 300, 400]}]})
    mask, boxes = processing.parse_boxes(text, 2000, 1000, expand=0)
    assert boxes["boxes"][0]["bbox"] == [200, 200, 600, 400]
    assert boxes["boxes"][0]["source_bbox"] == [200, 200, 600, 400]
    assert mask.sum() == 400 * 200


def test_detector_preserves_tight_source_box_when_mask_is_padded():
    text = json.dumps({"coordinate_space": "pixels", "boxes": [{"bbox": [100, 200, 300, 400]}]})
    _, boxes = processing.parse_boxes(text, 1000, 1000, expand=10, padding_ratio=0.25)
    assert boxes["boxes"][0]["bbox"] == [40, 140, 360, 460]
    assert boxes["boxes"][0]["source_bbox"] == [100, 200, 300, 400]

    # A second parse models the detector report entering QWMRefineMask.
    _, reparsed = processing.parse_boxes(json.dumps(boxes), 1000, 1000, expand=0)
    assert reparsed["boxes"][0]["bbox"] == [40, 140, 360, 460]
    assert reparsed["boxes"][0]["source_bbox"] == [100, 200, 300, 400]


def test_wide_sticker_like_patch_is_routed_to_rectangular_box_inpaint():
    result = {"boxes": [{"label": "printed pattern patch", "bbox": [100, 200, 500, 340],
                          "overlay_type": "irregular_sticker", "confidence": 0.9}]}
    corrected = processing.coerce_rectangular_overlay_types(result)
    item = corrected["boxes"][0]
    assert item["overlay_type"] == "rectangle"
    assert item["mask_method"] == "box"
    assert item["type_correction"] == "wide_flat_patch"


def test_sam_outline_recovery_adds_missing_limbs_and_discards_unrelated_islands():
    seed = np.zeros((160, 120), bool)
    seed[30:90, 40:80] = True
    complete = seed.copy()
    complete[90:135, 48:72] = True
    complete[5:15, 5:15] = True
    selected, report = processing.select_sam_recovery(seed, [complete], [0.96], 0.98)
    assert selected[130, 60]
    assert not selected[10, 10]
    assert np.all(selected[seed])
    assert report["outline_recovered_pixels"] == 45 * 24


def test_sam_outline_recovery_rejects_background_and_weak_or_unrelated_candidates():
    seed = np.zeros((160, 120), bool)
    seed[30:90, 40:80] = True
    unrelated = np.zeros_like(seed)
    unrelated[95:145, 10:50] = True
    weak = seed.copy()
    weak[90:135, 48:72] = True
    selected, report = processing.select_sam_recovery(
        seed, [np.ones_like(seed), unrelated, weak], [0.99, 0.99, 0.7], 0.98)
    np.testing.assert_array_equal(selected, seed)
    assert not report["outline_recovery_applied"]


def test_sam_refinement_recovers_contour_outside_detection_box():
    candidate = np.zeros((320, 320), np.uint8)
    candidate[110:190, 130:222] = 1
    refined, report = processing.refine_sam_candidate(
        candidate, safety_box=[100, 90, 210, 210], prompt_box=[110, 100, 200, 200], dilation=0
    )
    assert refined[150, 220] == 1
    assert report["recovered_outside_safety_pixels"] == 80 * 12
    assert report["guard_clipped_pixels"] == 0


def test_sam_refinement_limits_implausibly_large_selection():
    candidate = np.ones((320, 320), np.uint8)
    refined, report = processing.refine_sam_candidate(
        candidate, safety_box=[100, 90, 210, 210], prompt_box=[110, 100, 200, 200], dilation=0
    )
    x0, y0, x1, y1 = report["guard_box"]
    assert refined.sum() == (x1 - x0) * (y1 - y0)
    assert not refined[:y0].any()
    assert not refined[y1:].any()
    assert report["guard_clipped_pixels"] > 0


def test_sam_refinement_adapts_halo_to_large_irregular_overlay():
    candidate = np.zeros((900, 900), np.uint8)
    candidate[150:650, 200:600] = 1
    _, report = processing.refine_sam_candidate(
        candidate, safety_box=[170, 120, 630, 680], prompt_box=[200, 150, 600, 650],
        dilation=12, adaptive_dilation=True
    )
    assert report["effective_dilation"] == 32

    _, small_report = processing.refine_sam_candidate(
        candidate, safety_box=[190, 140, 310, 210], prompt_box=[200, 150, 300, 200],
        dilation=12, adaptive_dilation=True
    )
    assert small_report["effective_dilation"] == 12


def test_sam_refinement_can_use_wider_sticker_halo_ratio():
    candidate = np.zeros((900, 900), np.uint8)
    candidate[150:650, 200:600] = 1
    _, report = processing.refine_sam_candidate(
        candidate, safety_box=[170, 120, 630, 680], prompt_box=[200, 150, 600, 650],
        dilation=12, adaptive_dilation=True, dilation_ratio=0.12
    )
    assert report["effective_dilation"] == 48
    assert report["dilation_ratio"] == 0.12


def test_manual_mask_applies_adaptive_halo_for_overlay_edges_and_shadow():
    mask = np.zeros((900, 900), np.float32)
    mask[150:650, 200:600] = 1
    refined, report = processing.refine_manual_mask(
        mask, dilation=12, adaptive_dilation=True
    )
    assert report["source_bbox"] == [200, 150, 600, 650]
    assert report["effective_dilation"] == 32
    assert report["refined_pixels"] > report["base_pixels"]
    assert refined[150, 180] == 1

    _, small_report = processing.refine_manual_mask(
        mask[150:200, 200:300], dilation=12, adaptive_dilation=True
    )
    assert small_report["effective_dilation"] == 12


def test_crop_padding_is_removed_before_restore():
    image = np.zeros((297, 413, 3), np.float32)
    image[:, :, 0] = np.linspace(0, 1, 413)[None]
    image[:, :, 1] = np.linspace(0, 1, 297)[:, None]
    mask = np.zeros(image.shape[:2], np.float32)
    mask[130:164, 176:209] = 1
    crop, _, geometry = processing.prepare_crop(image, mask, context_pixels=53, max_side=512)
    restored = processing.restore_crop(crop, geometry)
    x0, y0, x1, y1 = geometry["roi"]
    assert crop.shape[0] % 32 == crop.shape[1] % 32 == 0
    np.testing.assert_allclose(restored, image[y0:y1, x0:x1], atol=1e-6)


def test_composite_never_changes_unselected_pixels():
    rng = np.random.default_rng(10)
    image = rng.random((180, 250, 3), dtype=np.float32)
    mask = np.zeros(image.shape[:2], np.float32)
    mask[35:75, 45:115] = 1
    mask[90:135, 145:190] = 1
    crop, _, geometry = processing.prepare_crop(image, mask, context_pixels=20, max_side=512)
    edited = np.ones_like(crop) * 0.75
    result, report = processing.composite(image, edited, mask, geometry, feather=6, align=False)
    np.testing.assert_array_equal(result[mask == 0], image[mask == 0])
    assert report["outside_mask_max_difference"] == 0
    assert np.mean(np.abs(result[mask > 0] - image[mask > 0])) > 0.1


def test_composite_feather_does_not_reintroduce_overlay_at_mask_edge():
    image = np.zeros((96, 128, 3), np.float32)
    image[:] = [0.12, 0.32, 0.52]
    # Simulate a bright watermark that reaches the right edge of the selected
    # region, which is where inward feathering used to leak the source back in.
    image[32:64, 48:80] = 1.0
    mask = np.zeros(image.shape[:2], np.float32)
    mask[30:66, 46:82] = 1
    crop, _, geometry = processing.prepare_crop(image, mask, context_pixels=12, max_side=256)
    edited = np.zeros_like(crop)
    edited[:] = [0.12, 0.32, 0.52]
    result, report = processing.composite(image, edited, mask, geometry, feather=6, align=False)
    # The selected edge should be clean even though the source overlay touches
    # the interior; unselected pixels must remain byte-for-byte unchanged.
    assert float(np.max(result[mask > 0])) < 0.95
    np.testing.assert_array_equal(result[mask == 0], image[mask == 0])
    assert report["edge_blend"] == "exterior_color_field"
    assert report["source_pixels_mixed_inside_mask"] == 0


def test_alignment_corrects_small_shift_using_background():
    rng = np.random.default_rng(50)
    image = cv2.GaussianBlur(rng.random((240, 280, 3), dtype=np.float32), (9, 9), 0)
    image *= 0.7
    shifted = cv2.warpAffine(image, np.array([[1, 0, 3], [0, 1, -2]], np.float32), (280, 240), borderMode=cv2.BORDER_REPLICATE)
    mask = np.zeros((240, 280), np.float32)
    mask[90:140, 100:160] = 1
    shifted[90:140, 100:160] = 0.8
    result, report = processing.align_crop(image, shifted, mask)
    assert report["applied"]
    np.testing.assert_allclose(report["shift_px"], [3, -2], atol=0.25)
    valid = cv2.dilate(mask.astype(np.uint8), np.ones((21, 21), np.uint8)) == 0
    assert np.mean((result[valid] - image[valid]) ** 2) < np.mean((shifted[valid] - image[valid]) ** 2) * 0.2


def test_no_watermark_does_not_create_full_image_selection():
    mask, boxes = processing.parse_boxes('{"boxes":[]}', 100, 100)
    assert not boxes["boxes"]
    with pytest.raises(ValueError, match="未识别到水印"):
        processing.prepare_crop(np.zeros((100, 100, 3), np.float32), mask)


def test_wrong_model_output_size_is_not_silently_stretched():
    geometry = {"model_size": [320, 320], "content_size": [300, 300], "roi": [0, 0, 300, 300]}
    with pytest.raises(ValueError, match="latent"):
        processing.restore_crop(np.zeros((256, 256, 3), np.float32), geometry)


def test_parse_boxes_rejects_malformed_items_with_actionable_error():
    with pytest.raises(ValueError, match="每个 bbox"):
        processing.parse_boxes('{"boxes":[{"label":"bad"}]}', 100, 100)
    with pytest.raises(ValueError, match="有限数字"):
        processing.parse_boxes('{"boxes":[{"bbox":[0, 0, "not-a-number", 10]}]}', 100, 100)


def test_parse_boxes_accepts_integer_like_expand_without_float_slice_indices():
    mask, boxes = processing.parse_boxes(
        '{"coordinate_space":"pixels","boxes":[{"bbox":[10, 10, 20, 20]}]}',
        100,
        100,
        expand=2.0,
    )
    assert boxes["boxes"][0]["bbox"] == [8, 8, 22, 22]
    assert mask[8:22, 8:22].all()


def test_parse_boxes_accepts_markdown_wrapped_top_level_array():
    text = """```json
[{"label":"partial watermark","bbox":[100,200,300,400],"overlay_type":"watermark","confidence":0.98}]
```"""
    _, boxes = processing.parse_boxes(text, 500, 1000, expand=0)
    assert boxes["boxes"][0]["bbox"] == [50, 200, 150, 400]
    assert boxes["boxes"][0]["overlay_type"] == "text_logo"


def test_composite_rejects_mask_with_wrong_image_dimensions():
    original = np.zeros((20, 30, 3), np.float32)
    edited = np.zeros((32, 32, 3), np.float32)
    geometry = {"model_size": [32, 32], "content_size": [20, 20], "roi": [0, 0, 20, 20]}
    with pytest.raises(ValueError, match="遮罩尺寸"):
        processing.composite(original, edited, np.zeros((10, 10), np.float32), geometry)


def test_overlay_types_route_to_box_or_sam_automatically():
    assert processing.route_for_overlay({"overlay_type": "text_logo"}) == "box"
    assert processing.route_for_overlay({"overlay_type": "translucent_text_logo"}) == "box"
    assert processing.route_for_overlay({"overlay_type": "mosaic"}) == "box"
    assert processing.route_for_overlay({"overlay_type": "rectangle"}) == "box"
    assert processing.route_for_overlay({"overlay_type": "irregular_sticker"}) == "sam"


def test_two_pass_logo_detections_merge_partial_and_complete_boxes():
    general = {"boxes": [{"label": "logo icon", "bbox": [100, 200, 180, 270],
                            "overlay_type": "text_logo", "confidence": 0.73}]}
    recovery = {"boxes": [{"label": "complete logo and URL", "bbox": [90, 190, 310, 285],
                             "overlay_type": "translucent_text_logo", "confidence": 0.94}]}
    merged = processing.merge_detection_results([general, recovery], 500, 700)
    assert len(merged["boxes"]) == 1
    assert merged["boxes"][0]["bbox"] == [90, 190, 310, 285]
    assert merged["boxes"][0]["overlay_type"] == "translucent_text_logo"
    assert merged["boxes"][0]["source_passes"] == [0, 1]
    assert merged["boxes"][0]["mask_method"] == "box"


def test_larger_recovery_box_replaces_shifted_partial_logo_box():
    general = {"boxes": [{"label": "faint fragment", "bbox": [160, 276, 260, 322],
                            "overlay_type": "translucent_text_logo", "confidence": 0.98}]}
    recovery = {"boxes": [{"label": "complete wordmark and URL", "bbox": [160, 313, 335, 374],
                             "overlay_type": "translucent_text_logo", "confidence": 0.98}]}
    merged = processing.merge_detection_results([general, recovery], 500, 747)
    assert len(merged["boxes"]) == 1
    assert merged["boxes"][0]["bbox"] == [160, 313, 335, 374]
    assert merged["boxes"][0]["source_passes"] == [0, 1]


def test_logo_icon_and_nearby_wordmark_merge_into_one_complete_region():
    result = {"boxes": [
        {"label": "icon", "bbox": [50, 50, 90, 100], "overlay_type": "text_logo", "confidence": 0.8},
        {"label": "wordmark", "bbox": [97, 52, 180, 98], "overlay_type": "text_logo", "confidence": 0.9},
    ]}
    merged = processing.merge_detection_results([result], 500, 500)
    assert len(merged["boxes"]) == 1
    assert merged["boxes"][0]["bbox"] == [50, 50, 180, 100]


def test_auto_padding_is_type_aware():
    text = json.dumps({"coordinate_space": "pixels", "boxes": [
        {"bbox": [100, 100, 300, 200], "overlay_type": "text_logo", "confidence": 0.9},
        {"bbox": [500, 100, 700, 300], "overlay_type": "irregular_sticker", "confidence": 0.9},
    ]})
    _, parsed = processing.parse_boxes(text, 1000, 1000, expand=0, padding_ratio=0.25,
                                       adaptive_padding=True)
    assert parsed["boxes"][0]["bbox"] == [76, 88, 324, 212]
    assert parsed["boxes"][1]["bbox"] == [450, 50, 750, 350]


def test_translucent_logo_gets_extra_room_for_faint_text_and_url():
    text = json.dumps({"coordinate_space": "pixels", "boxes": [
        {"bbox": [100, 400, 300, 500], "overlay_type": "translucent_text_logo", "confidence": 0.9},
    ]})
    _, parsed = processing.parse_boxes(text, 1000, 1000, expand=0, padding_ratio=0.50,
                                       adaptive_padding=True)
    assert parsed["boxes"][0]["bbox"] == [0, 350, 400, 550]


def test_dynamic_prompt_includes_detected_overlay_routes():
    prompt = processing.build_edit_prompt({"boxes": [
        {"overlay_type": "translucent_text_logo"},
        {"overlay_type": "mosaic"},
        {"overlay_type": "irregular_sticker"},
    ]})
    assert "低透明度残留" in prompt
    assert "马赛克" in prompt
    assert "不规则对象" in prompt


def test_edit_prompt_accepts_user_instruction_without_dropping_safety_rules():
    prompt = processing.build_edit_prompt(
        {"boxes": [{"overlay_type": "text_logo"}]},
        edit_prompt="保留人物脸部和衣服纹理，只修复右下角。",
    )
    assert "完整移除文字水印" in prompt
    assert "用户编辑指令：保留人物脸部和衣服纹理，只修复右下角。" in prompt
    assert "保持身体部位数量、姿态、连接关系" in prompt
    assert "不要猜测或绘制具体敏感细节" in prompt


def test_manual_selection_prompt_explicitly_removes_full_overlay_halo():
    prompt = processing.build_edit_prompt({"source": "manual_mask", "boxes": []})
    assert "手动选区内全部属于需要移除的后期叠加物" in prompt
    assert "半透明边缘" in prompt
