"""Shared pytest fixtures for all test modules."""

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="session")
def synthetic_data_dir(tmp_path_factory) -> Path:
    """Create a minimal synthetic dataset with 3 samples.

    Each sample has one RGB image and 2-3 individual nucleus mask PNGs.
    Images are 64x64 to keep tests fast.
    """
    base = tmp_path_factory.mktemp("stage1_train")

    for sample_idx in range(3):
        sample_dir = base / f"sample_{sample_idx:03d}"
        images_dir = sample_dir / "images"
        masks_dir = sample_dir / "masks"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)

        # Random RGB image
        img_arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(img_arr).save(images_dir / f"sample_{sample_idx}.png")

        # 2-3 non-overlapping nucleus masks
        num_masks = 2 + (sample_idx % 2)
        for mask_idx in range(num_masks):
            mask_arr = np.zeros((64, 64), dtype=np.uint8)
            # Place a small filled rectangle as a nucleus
            row_start = 5 + mask_idx * 20
            col_start = 5 + mask_idx * 15
            mask_arr[row_start:row_start + 10, col_start:col_start + 10] = 255
            Image.fromarray(mask_arr).save(masks_dir / f"mask_{mask_idx}.png")

    return base
