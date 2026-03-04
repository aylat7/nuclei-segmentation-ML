"""Unit tests for Dice coefficient and IoU score functions."""

import pytest
import torch

from src.evaluate import dice_coefficient, iou_score


def test_dice_perfect_overlap():
    """Identical binary tensors should yield Dice close to 1."""
    t = torch.ones(1, 1, 8, 8)
    score = dice_coefficient(t, t)
    assert abs(score.item() - 1.0) < 1e-4


def test_dice_no_overlap():
    """Fully disjoint predictions should yield Dice close to 0."""
    pred = torch.zeros(1, 1, 8, 8)
    pred[:, :, :4, :] = 1.0
    target = torch.zeros(1, 1, 8, 8)
    target[:, :, 4:, :] = 1.0
    score = dice_coefficient(pred, target)
    assert score.item() < 0.01


def test_dice_known_value():
    """Hand-calculated Dice for partial overlap.

    Pred: 6 pixels positive, Target: 4 pixels positive, Intersection: 3.
    Dice = 2*3 / (6+4) = 0.6
    """
    pred = torch.zeros(1, 1, 10, 1)
    pred[:, :, :6, :] = 1.0

    target = torch.zeros(1, 1, 10, 1)
    target[:, :, 3:7, :] = 1.0

    score = dice_coefficient(pred, target)
    # Intersection = 3, sum_pred = 6, sum_target = 4
    expected = (2 * 3 + 1e-6) / (6 + 4 + 1e-6)
    assert abs(score.item() - expected) < 1e-3


def test_iou_perfect_overlap():
    """Identical binary tensors should yield IoU close to 1."""
    t = torch.ones(1, 1, 8, 8)
    score = iou_score(t, t)
    assert abs(score.item() - 1.0) < 1e-4


def test_iou_no_overlap():
    """Fully disjoint predictions should yield IoU close to 0."""
    pred = torch.zeros(1, 1, 8, 8)
    pred[:, :, :4, :] = 1.0
    target = torch.zeros(1, 1, 8, 8)
    target[:, :, 4:, :] = 1.0
    score = iou_score(pred, target)
    assert score.item() < 0.01


def test_iou_known_value():
    """Hand-calculated IoU for partial overlap.

    Pred: 6 pixels positive, Target: 4 pixels positive, Intersection: 3.
    Union = 6 + 4 - 3 = 7
    IoU = 3 / 7 ~ 0.4286
    """
    pred = torch.zeros(1, 1, 10, 1)
    pred[:, :, :6, :] = 1.0

    target = torch.zeros(1, 1, 10, 1)
    target[:, :, 3:7, :] = 1.0

    score = iou_score(pred, target)
    expected = (3 + 1e-6) / (7 + 1e-6)
    assert abs(score.item() - expected) < 1e-3


def test_dice_all_zeros():
    """All-zero tensors should still return a finite score (smoothing prevents NaN)."""
    pred = torch.zeros(1, 1, 8, 8)
    target = torch.zeros(1, 1, 8, 8)
    score = dice_coefficient(pred, target)
    assert torch.isfinite(score)


def test_iou_all_zeros():
    """All-zero tensors should return a finite score."""
    pred = torch.zeros(1, 1, 8, 8)
    target = torch.zeros(1, 1, 8, 8)
    score = iou_score(pred, target)
    assert torch.isfinite(score)
