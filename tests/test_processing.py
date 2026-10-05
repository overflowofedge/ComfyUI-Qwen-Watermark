import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest


path = Path(__file__).resolve().parents[1] / "custom_nodes/ComfyUI-Qwen-Watermark/processing.py"
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


def test_composite_rejects_mask_with_wrong_image_dimensions():
    original = np.zeros((20, 30, 3), np.float32)
    edited = np.zeros((32, 32, 3), np.float32)
    geometry = {"model_size": [32, 32], "content_size": [20, 20], "roi": [0, 0, 20, 20]}
    with pytest.raises(ValueError, match="遮罩尺寸"):
        processing.composite(original, edited, np.zeros((10, 10), np.float32), geometry)
