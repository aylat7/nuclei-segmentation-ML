"""Unit tests for NucleiDataset loading, mask combining, and augmentation."""

import numpy as np
import pytest

from src.dataset import NucleiDataset


@pytest.fixture
def dataset(synthetic_data_dir):
    return NucleiDataset(str(synthetic_data_dir), image_size=64, augment=False)


def test_dataset_length(dataset, synthetic_data_dir):
    """Dataset length should equal the number of sample directories."""
    num_dirs = len(list(synthetic_data_dir.iterdir()))
    assert len(dataset) == num_dirs


def test_dataset_item_shapes(dataset):
    """Each item should return (3, H, W) image and (1, H, W) mask."""
    image, mask = dataset[0]
    assert image.shape == (3, 64, 64), f"Image shape: {image.shape}"
    assert mask.shape == (1, 64, 64), f"Mask shape: {mask.shape}"


def test_mask_is_binary(dataset):
    """Combined mask should contain only 0.0 or 1.0 values."""
    for idx in range(len(dataset)):
        _, mask = dataset[idx]
        unique_vals = mask.unique()
        for val in unique_vals:
            assert val.item() in (0.0, 1.0), (
                f"Non-binary value {val.item()} found in mask"
            )


def test_image_normalized(dataset):
    """Image pixel values should be in [0, 1]."""
    image, _ = dataset[0]
    assert image.min().item() >= 0.0
    assert image.max().item() <= 1.0


def test_mask_combining(synthetic_data_dir):
    """Combined mask should include pixels from all individual nucleus masks."""
    dataset = NucleiDataset(str(synthetic_data_dir), image_size=64)

    for idx in range(len(dataset)):
        _, combined_mask = dataset[idx]
        individual_masks = dataset.get_individual_masks(idx)

        if not individual_masks:
            continue

        # Every individual mask should have at least some overlap with combined
        combined_np = combined_mask.squeeze().numpy()
        for i, ind_mask in enumerate(individual_masks):
            overlap = (ind_mask.astype(bool) & (combined_np > 0.5)).sum()
            assert overlap > 0, (
                f"Sample {idx}, mask {i}: no overlap with combined mask"
            )


def test_get_individual_masks(dataset):
    """get_individual_masks should return a list of boolean arrays."""
    masks = dataset.get_individual_masks(0)
    assert isinstance(masks, list)
    assert len(masks) >= 1
    for mask in masks:
        assert mask.dtype == bool
        assert mask.ndim == 2


def test_augmented_dataset_same_shapes(synthetic_data_dir):
    """Augmented dataset should still return tensors with correct shapes."""
    aug_dataset = NucleiDataset(str(synthetic_data_dir), image_size=64, augment=True)
    image, mask = aug_dataset[0]
    assert image.shape == (3, 64, 64)
    assert mask.shape == (1, 64, 64)
