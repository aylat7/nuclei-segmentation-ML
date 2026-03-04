"""Unit tests for instance extraction, matching, overlays, and metrics."""

import numpy as np
import pytest

from src.visualize import (
    compute_instance_metrics,
    count_nuclei,
    create_overlay,
    extract_instances,
    match_instances,
    watershed_separate,
    watershed_to_binary_instances,
)


# ---------------------------------------------------------------------------
# count_nuclei
# ---------------------------------------------------------------------------

def test_count_nuclei():
    """Mask with 3 separate blobs should return count of 3."""
    mask = _make_mask_with_blobs(50, [(0, 0, 5, 5), (20, 20, 5, 5), (40, 40, 5, 5)])
    assert count_nuclei(mask) == 3


def test_count_nuclei_empty():
    """Empty mask should return 0."""
    mask = np.zeros((20, 20), dtype=bool)
    assert count_nuclei(mask) == 0


def test_count_nuclei_single():
    """Single connected region should return 1."""
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    assert count_nuclei(mask) == 1


# ---------------------------------------------------------------------------
# extract_instances
# ---------------------------------------------------------------------------

def _make_mask_with_blobs(size: int, blobs: list) -> np.ndarray:
    """Create a binary mask with rectangular blobs at given positions."""
    mask = np.zeros((size, size), dtype=bool)
    for (r, c, h, w) in blobs:
        mask[r:r + h, c:c + w] = True
    return mask


def test_extract_instances_count():
    """Mask with 3 separate blobs should yield 3 instances."""
    # Three non-overlapping 5x5 blobs in a 50x50 image
    mask = _make_mask_with_blobs(50, [(0, 0, 5, 5), (20, 20, 5, 5), (40, 40, 5, 5)])
    instances = extract_instances(mask)
    assert len(instances) == 3


def test_extract_instances_single():
    """One connected region should yield exactly 1 instance."""
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    instances = extract_instances(mask)
    assert len(instances) == 1


def test_extract_instances_empty():
    """Empty mask should yield an empty list."""
    mask = np.zeros((20, 20), dtype=bool)
    instances = extract_instances(mask)
    assert len(instances) == 0


def test_extract_instances_full():
    """Fully foreground mask should yield 1 instance."""
    mask = np.ones((10, 10), dtype=bool)
    instances = extract_instances(mask)
    assert len(instances) == 1


# ---------------------------------------------------------------------------
# match_instances
# ---------------------------------------------------------------------------

def test_match_instances_perfect():
    """Identical pred and GT masks should yield all TP, no FP or FN."""
    mask = _make_mask_with_blobs(50, [(0, 0, 10, 10), (30, 30, 10, 10)])
    instances = extract_instances(mask)
    tp, fp, fn = match_instances(instances, instances, iou_threshold=0.5)
    assert len(tp) == 2
    assert len(fp) == 0
    assert len(fn) == 0


def test_match_instances_no_overlap():
    """Completely non-overlapping pred and GT should yield all FP and FN."""
    pred_mask = _make_mask_with_blobs(50, [(0, 0, 10, 10)])
    gt_mask = _make_mask_with_blobs(50, [(35, 35, 10, 10)])
    pred_inst = extract_instances(pred_mask)
    gt_inst = extract_instances(gt_mask)
    tp, fp, fn = match_instances(pred_inst, gt_inst, iou_threshold=0.5)
    assert len(tp) == 0
    assert len(fp) == 1
    assert len(fn) == 1


def test_match_instances_partial():
    """2 TP, 1 FP, 1 FN configuration."""
    # Two matching pairs + one extra pred + one extra GT
    pred_mask = _make_mask_with_blobs(100, [
        (0, 0, 10, 10),   # matches GT[0]
        (30, 30, 10, 10), # matches GT[1]
        (60, 60, 10, 10), # FP - no GT counterpart
    ])
    gt_mask = _make_mask_with_blobs(100, [
        (0, 0, 10, 10),   # matches pred[0]
        (30, 30, 10, 10), # matches pred[1]
        (85, 85, 10, 10), # FN - no pred counterpart
    ])
    pred_inst = extract_instances(pred_mask)
    gt_inst = extract_instances(gt_mask)
    tp, fp, fn = match_instances(pred_inst, gt_inst, iou_threshold=0.5)
    assert len(tp) == 2
    assert len(fp) == 1
    assert len(fn) == 1


def test_match_instances_empty_pred():
    """Empty prediction list should yield all GT as FN."""
    gt_mask = _make_mask_with_blobs(30, [(5, 5, 8, 8)])
    gt_inst = extract_instances(gt_mask)
    tp, fp, fn = match_instances([], gt_inst, iou_threshold=0.5)
    assert len(tp) == 0
    assert len(fp) == 0
    assert len(fn) == 1


# ---------------------------------------------------------------------------
# create_overlay
# ---------------------------------------------------------------------------

def test_overlay_shape():
    """Output should have the same H x W x 3 shape as the input image."""
    image = np.random.rand(64, 64, 3).astype(np.float32)
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:20, 10:20] = True
    instances = [mask]
    result = create_overlay(image, instances, [], [])
    assert result.shape == (64, 64, 3)


def test_overlay_modifies_image():
    """Overlay with a TP region should differ from the plain input."""
    image = np.ones((64, 64, 3), dtype=np.float32) * 0.5
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:20, 10:20] = True
    result = create_overlay(image, [mask], [], [])
    plain = (image * 255).astype(np.uint8)
    assert not np.array_equal(result, plain)


def test_overlay_output_dtype():
    """Output should be uint8."""
    image = np.random.rand(32, 32, 3).astype(np.float32)
    result = create_overlay(image, [], [], [])
    assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# compute_instance_metrics
# ---------------------------------------------------------------------------

def test_instance_metrics_perfect():
    """Perfect overlap should yield precision=1, recall=1, F1=1."""
    mask = _make_mask_with_blobs(50, [(5, 5, 10, 10), (30, 30, 10, 10)])
    pred = mask.astype(np.float32)
    gt = mask.astype(np.float32)
    metrics = compute_instance_metrics(pred, gt)
    assert abs(metrics["precision"] - 1.0) < 1e-6
    assert abs(metrics["recall"] - 1.0) < 1e-6
    assert abs(metrics["f1"] - 1.0) < 1e-6


def test_instance_metrics_partial():
    """2 TP, 1 FP, 1 FN -> precision=2/3, recall=2/3."""
    pred_mask = _make_mask_with_blobs(100, [
        (0, 0, 10, 10),
        (30, 30, 10, 10),
        (60, 60, 10, 10),  # FP
    ])
    gt_mask = _make_mask_with_blobs(100, [
        (0, 0, 10, 10),
        (30, 30, 10, 10),
        (85, 85, 10, 10),  # FN
    ])
    metrics = compute_instance_metrics(
        pred_mask.astype(np.float32), gt_mask.astype(np.float32)
    )
    assert abs(metrics["precision"] - 2 / 3) < 0.01
    assert abs(metrics["recall"] - 2 / 3) < 0.01


def test_instance_metrics_no_overlap():
    """Zero overlap should yield precision=0, recall=0, F1=0."""
    pred_mask = _make_mask_with_blobs(50, [(0, 0, 10, 10)])
    gt_mask = _make_mask_with_blobs(50, [(35, 35, 10, 10)])
    metrics = compute_instance_metrics(
        pred_mask.astype(np.float32), gt_mask.astype(np.float32)
    )
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


# ---------------------------------------------------------------------------
# watershed_separate and watershed_to_binary_instances
# ---------------------------------------------------------------------------

def _make_circle_mask(size: int, cx: int, cy: int, r: int) -> np.ndarray:
    """Create a binary mask with a filled circle (cx=col center, cy=row center)."""
    mask = np.zeros((size, size), dtype=bool)
    y, x = np.ogrid[:size, :size]
    mask[(x - cx) ** 2 + (y - cy) ** 2 <= r ** 2] = True
    return mask


def test_watershed_separate_splits_touching():
    """Two overlapping circles that form one connected component should be split into 2."""
    size = 60
    # Radius 12, centers 22px apart -> they overlap, forming a single connected component
    circle1 = _make_circle_mask(size, cx=15, cy=30, r=12)
    circle2 = _make_circle_mask(size, cx=37, cy=30, r=12)
    merged = (circle1 | circle2).astype(np.float32)

    # Confirm extract_instances sees only 1 blob
    assert len(extract_instances(merged)) == 1

    # Watershed should recover 2 separate regions
    labeled = watershed_separate(merged, min_distance=5)
    assert len(np.unique(labeled[labeled > 0])) == 2


def test_watershed_separate_single_blob():
    """A single isolated circle should remain as exactly 1 region after watershed."""
    mask = _make_circle_mask(60, cx=30, cy=30, r=12).astype(np.float32)
    labeled = watershed_separate(mask, min_distance=5)
    assert len(np.unique(labeled[labeled > 0])) == 1


def test_watershed_to_binary_instances_count():
    """watershed_to_binary_instances should produce one boolean mask per labeled region."""
    size = 60
    circle1 = _make_circle_mask(size, cx=15, cy=30, r=12)
    circle2 = _make_circle_mask(size, cx=37, cy=30, r=12)
    merged = (circle1 | circle2).astype(np.float32)

    labeled = watershed_separate(merged, min_distance=5)
    instances = watershed_to_binary_instances(labeled)

    assert len(instances) == 2
    assert instances[0].dtype == bool
