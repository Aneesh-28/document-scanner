"""
Test suite for the document scanner pipeline.

Run with:  pytest tests/ -v
"""

import os
import sys
import numpy as np
import pytest
import cv2

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC_DIR)

from preprocessing import load_image, resize_image, to_grayscale, blur, detect_edges, preprocess
from detect_and_warp import (
    find_document_corners,
    order_points,
    compute_output_dimensions,
    detect_and_warp,
    DocumentNotFoundError,
)
from enhance import enhance

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "samples")
GOOD_SAMPLE = os.path.join(SAMPLES_DIR, "photo1.jpg")
BAD_SAMPLE = os.path.join(SAMPLES_DIR, "not_an_image.jpg")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def test_preprocessing_returns_correct_shape_and_type():
    """Preprocessing returns correct output shape/type."""
    resized, edged, scale_ratio = preprocess(GOOD_SAMPLE)

    # resized_original: color image, HxWx3, uint8
    assert isinstance(resized, np.ndarray)
    assert resized.ndim == 3 and resized.shape[2] == 3
    assert resized.dtype == np.uint8
    assert max(resized.shape[:2]) <= 900  # resized to target_dim

    # edged: single-channel binary edge map, same H/W as resized
    assert isinstance(edged, np.ndarray)
    assert edged.ndim == 2
    assert edged.shape[:2] == resized.shape[:2]
    assert edged.dtype == np.uint8

    # scale_ratio: positive float, >= 1.0 since original is larger
    assert isinstance(scale_ratio, float)
    assert scale_ratio >= 1.0


def test_resize_image_preserves_aspect_ratio():
    img = np.zeros((1300, 1000, 3), dtype=np.uint8)
    resized, scale_ratio = resize_image(img, target_dim=900)
    h, w = resized.shape[:2]
    assert max(h, w) == 900
    orig_aspect = 1300 / 1000
    new_aspect = h / w
    assert abs(orig_aspect - new_aspect) < 0.01


def test_grayscale_and_blur_shapes():
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    gray = to_grayscale(img)
    assert gray.ndim == 2
    assert gray.shape == (200, 300)

    blurred = blur(gray)
    assert blurred.shape == gray.shape


def test_load_image_missing_file_raises():
    with pytest.raises(ValueError):
        load_image(os.path.join(SAMPLES_DIR, "does_not_exist.jpg"))


def test_load_image_invalid_file_raises():
    """A non-image file raises a handled error, not a crash."""
    with pytest.raises(ValueError):
        load_image(BAD_SAMPLE)


# ---------------------------------------------------------------------------
# Contour detection & warp
# ---------------------------------------------------------------------------

def test_order_points_orders_correctly():
    # Deliberately shuffled: bottom-right, top-left, bottom-left, top-right
    pts = np.array([[100, 100], [0, 0], [0, 100], [100, 0]], dtype="float32")
    ordered = order_points(pts)
    tl, tr, br, bl = ordered
    assert list(tl) == [0, 0]
    assert list(tr) == [100, 0]
    assert list(br) == [100, 100]
    assert list(bl) == [0, 100]


def test_compute_output_dimensions_is_positive():
    rect = np.array([[0, 0], [200, 0], [200, 300], [0, 300]], dtype="float32")
    w, h = compute_output_dimensions(rect)
    assert w == 200
    assert h == 300


def test_known_good_sample_produces_warped_output_of_expected_dimensions():
    """A known-good sample image produces a warped output of expected dimensions."""
    resized, edged, scale_ratio = preprocess(GOOD_SAMPLE)
    full_res = load_image(GOOD_SAMPLE)

    warped, used_fallback = detect_and_warp(resized, edged, scale_ratio, full_res_original=full_res)

    assert isinstance(warped, np.ndarray)
    h, w = warped.shape[:2]
    # The synthetic page is portrait (1000x1300), so the recovered warp
    # should also be taller than it is wide, and reasonably large since we
    # warped at full resolution.
    assert h > w
    assert h > 500 and w > 300


def test_find_document_corners_returns_four_points():
    resized, edged, _ = preprocess(GOOD_SAMPLE)
    corners, used_fallback = find_document_corners(edged)
    assert corners.shape == (4, 2)


def test_no_contours_raises_document_not_found_error():
    """An edge map with nothing in it should raise a handled error, not crash."""
    blank_edges = np.zeros((400, 400), dtype=np.uint8)
    with pytest.raises(DocumentNotFoundError):
        find_document_corners(blank_edges)


# ---------------------------------------------------------------------------
# Enhancement
# ---------------------------------------------------------------------------

def test_enhance_produces_binary_looking_image():
    resized, edged, scale_ratio = preprocess(GOOD_SAMPLE)
    full_res = load_image(GOOD_SAMPLE)
    warped, _ = detect_and_warp(resized, edged, scale_ratio, full_res_original=full_res)

    scanned = enhance(warped)
    assert scanned.ndim == 2
    unique_vals = np.unique(scanned)
    # adaptiveThreshold output should only contain 0 and 255
    assert set(unique_vals.tolist()).issubset({0, 255})


# ---------------------------------------------------------------------------
# End-to-end / CLI-level behavior
# ---------------------------------------------------------------------------

def test_end_to_end_pipeline_runs_without_crashing_on_all_samples(tmp_path):
    """Smoke test: every real sample image in data/samples should make it
    through the full pipeline without raising, regardless of whether the
    4-point contour path or the fallback path is used."""
    for fname in os.listdir(SAMPLES_DIR):
        path = os.path.join(SAMPLES_DIR, fname)
        if fname == "not_an_image.jpg":
            continue
        resized, edged, scale_ratio = preprocess(path)
        full_res = load_image(path)
        warped, _ = detect_and_warp(resized, edged, scale_ratio, full_res_original=full_res)
        scanned = enhance(warped)
        assert scanned.size > 0
