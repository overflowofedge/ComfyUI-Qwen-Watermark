import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest


path = Path(__file__).resolve().parents[1] / "processing.py"
spec = importlib.util.spec_from_file_location("processing", path)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def scene():
    y, x = np.mgrid[:160, :192].astype(np.float32)
    clean = np.stack((0.2 + x / 600, 0.25 + y / 500, 0.3 + (x + y) / 1000), axis=-1)
    # Fine geometry near the mask edge must survive an identity edit.
    clean[::4, :, :] += 0.08
    mask = np.zeros((160, 192), np.float32)
    mask[55:100, 60:115] = 1
    return clean, mask


def test_identity_edit_preserves_color_and_edge_texture():
    clean, mask = scene()
    crop, _, geometry = p.prepare_crop(clean, mask, context_pixels=40, max_side=512)
    result, report = p.composite(clean, crop, mask, geometry, feather=12, align=False)
    np.testing.assert_allclose(result, clean, atol=1e-6)
    assert not report["color"]["applied"]
    assert not report["boundary"]["applied"]


@pytest.mark.parametrize("alpha", [1.0, 0.15])
def test_perfect_repair_never_mixes_corrupted_source_even_with_soft_mask(alpha):
    clean, mask = scene()
    damaged = clean.copy()
    damaged[mask > 0] = [0.95, 0.08, 0.8]
    mask *= alpha
    crop, _, geometry = p.prepare_crop(clean, mask, context_pixels=40, max_side=512)
    result, report = p.composite(damaged, crop, mask, geometry, feather=12, align=False)
    np.testing.assert_allclose(result, clean, atol=1e-6)
    np.testing.assert_array_equal(result[mask == 0], damaged[mask == 0])
    assert report["source_pixels_mixed_inside_mask"] == 0


def test_rgb_exposure_drift_is_corrected_using_only_context():
    clean, mask = scene()
    damaged = clean.copy()
    damaged[mask > 0] = 1
    crop, _, geometry = p.prepare_crop(clean, mask, context_pixels=40, max_side=512)
    drifted = crop * np.array([1.06, 0.95, 1.03]) + np.array([0.012, 0.016, -0.01])
    result, report = p.composite(damaged, drifted.astype(np.float32), mask, geometry, feather=6, align=False)
    assert report["color"]["applied"]
    assert np.mean(np.abs(result[mask > 0] - clean[mask > 0])) < 0.003
    assert report["color"]["context_mae_after"] < report["color"]["context_mae_before"] * 0.15


def test_full_selection_with_no_context_keeps_generated_color():
    original = np.ones((64, 64, 3), np.float32)
    mask = np.ones((64, 64), np.float32)
    crop, _, geometry = p.prepare_crop(original, mask, context_pixels=0)
    edited = crop * np.array([0.2, 0.35, 0.5], np.float32)
    result, report = p.composite(original, edited, mask, geometry, feather=20, align=False)
    np.testing.assert_array_equal(result, edited)
    assert report["color"]["reason"] == "insufficient_context"


def test_disconnected_selection_preserves_gap_and_outer_image():
    clean, mask = scene()
    mask[110:125, 130:145] = 1
    damaged = clean.copy()
    damaged[mask > 0] = 0.95
    crop, _, geometry = p.prepare_crop(clean, mask, context_pixels=20)
    result, _ = p.composite(damaged, crop, mask, geometry, align=False)
    np.testing.assert_allclose(result, clean, atol=1e-6)
    np.testing.assert_array_equal(result[mask == 0], damaged[mask == 0])


def test_roi_mismatch_and_nonfinite_images_are_rejected():
    clean, mask = scene()
    crop, _, geometry = p.prepare_crop(clean, mask, context_pixels=5)
    mask[0, 0] = 1
    with pytest.raises(ValueError, match="ROI"):
        p.composite(clean, crop, mask, geometry)
    mask[0, 0] = 0
    crop[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        p.composite(clean, crop, mask, geometry)


def test_residual_scope_excludes_unrelated_regions_and_low_confidence():
    _, mask = scene()
    result = {"boxes": [
        {"bbox": [65, 60, 85, 80], "source_bbox": [65, 60, 85, 80], "confidence": 0.95},
        {"bbox": [150, 10, 170, 30], "source_bbox": [150, 10, 170, 30], "confidence": 0.99},
        {"bbox": [90, 70, 110, 90], "source_bbox": [90, 70, 110, 90], "confidence": 0.6},
    ]}
    residual, report = p.constrain_residuals(result, mask, expand=8)
    guard = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    assert not np.any(residual[guard == 0])
    assert len(report["boxes"]) == 1
    assert len(report["rejected_boxes"]) == 2
    assert report["uncertain"]


def test_second_pass_regression_is_rejected_and_improvement_is_kept():
    first, mask = scene()
    bad = first.copy()
    bad[mask > 0] = [1, 0, 1]
    result, _, report = p.choose_final(first, bad, mask, mask, {})
    np.testing.assert_array_equal(result, first)
    assert "second_pass_boundary_regression" in report["reasons"]
    result, _, report = p.choose_final(bad, first, mask, mask, {})
    np.testing.assert_array_equal(result, first)
    assert report["second_pass_accepted"]


def test_wide_animal_cutout_stays_on_sam_route():
    result = {"boxes": [{"bbox": [0, 0, 200, 50], "label": "cat sticker", "overlay_type": "irregular_sticker"}]}
    assert p.coerce_rectangular_overlay_types(result)["boxes"][0]["overlay_type"] == "irregular_sticker"
