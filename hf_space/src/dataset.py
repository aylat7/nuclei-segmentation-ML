"""Dataset class for the 2018 Data Science Bowl nuclei segmentation dataset."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


class NucleiDataset(Dataset):
    """Dataset for cell nuclei segmentation using Data Science Bowl 2018 structure.

    Each sample is a folder containing:
    - images/: one PNG image of the microscopy slide
    - masks/: multiple PNG files, one binary mask per nucleus

    Individual masks are combined into a single binary mask by taking the
    element-wise maximum across all mask files.

    Args:
        data_dir: Root directory containing per-sample subdirectories.
        image_size: Target (H, W) size after resizing.
        augment: Whether to apply random flips for data augmentation.

    Example:
        >>> dataset = NucleiDataset("data/stage1_train", image_size=128)
        >>> image, mask = dataset[0]
        >>> image.shape, mask.shape
        (torch.Size([3, 128, 128]), torch.Size([1, 128, 128]))
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int = 128,
        augment: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augment = augment

        # Each subfolder is one sample
        self.sample_dirs = sorted(
            [d for d in self.data_dir.iterdir() if d.is_dir()]
        )
        if not self.sample_dirs:
            raise ValueError(f"No sample directories found in {data_dir}")

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and return (image, combined_mask) for a sample.

        Args:
            idx: Index of the sample.

        Returns:
            Tuple of:
            - image tensor of shape (3, H, W) in [0, 1]
            - binary mask tensor of shape (1, H, W) with values 0.0 or 1.0
        """
        sample_dir = self.sample_dirs[idx]

        image = self._load_image(sample_dir)
        mask = self._load_combined_mask(sample_dir)

        image_tensor = TF.to_tensor(image)  # (3, H, W), [0, 1]
        mask_tensor = TF.to_tensor(mask)    # (1, H, W), [0, 1]

        if self.augment:
            image_tensor, mask_tensor = self._apply_augmentations(
                image_tensor, mask_tensor
            )

        # Binarize mask after any resizing/augmentation that might blur it
        mask_tensor = (mask_tensor >= 0.5).float()

        return image_tensor, mask_tensor

    def get_individual_masks(self, idx: int) -> List[np.ndarray]:
        """Return list of individual nucleus mask arrays for instance evaluation.

        Each element is a boolean 2D numpy array of shape (H, W) after resize.

        Args:
            idx: Index of the sample.

        Returns:
            List of numpy boolean arrays, one per nucleus.
        """
        sample_dir = self.sample_dirs[idx]
        masks_dir = sample_dir / "masks"
        individual_masks = []

        for mask_path in sorted(masks_dir.glob("*.png")):
            mask = Image.open(mask_path).convert("L")
            mask = mask.resize(
                (self.image_size, self.image_size), Image.NEAREST
            )
            mask_arr = np.array(mask) > 127
            if mask_arr.any():  # Skip empty masks
                individual_masks.append(mask_arr)

        return individual_masks

    def _load_image(self, sample_dir: Path) -> Image.Image:
        """Load and resize the microscopy image to RGB."""
        images_dir = sample_dir / "images"
        image_path = next(images_dir.glob("*.png"))
        image = Image.open(image_path).convert("RGB")
        return image.resize((self.image_size, self.image_size), Image.BILINEAR)

    def _load_combined_mask(self, sample_dir: Path) -> Image.Image:
        """Combine all individual nucleus masks into one binary mask."""
        masks_dir = sample_dir / "masks"
        combined = None

        for mask_path in sorted(masks_dir.glob("*.png")):
            mask = Image.open(mask_path).convert("L")
            mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
            mask_arr = np.array(mask)

            if combined is None:
                combined = mask_arr
            else:
                # Union of all nuclei masks
                combined = np.maximum(combined, mask_arr)

        if combined is None:
            combined = np.zeros(
                (self.image_size, self.image_size), dtype=np.uint8
            )

        return Image.fromarray(combined.astype(np.uint8))

    def _apply_augmentations(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply identical random flips to image and mask.

        Stacks them as a 4-channel tensor so the same random transform applies
        to both, then splits them apart.

        Args:
            image: (3, H, W) image tensor.
            mask: (1, H, W) mask tensor.

        Returns:
            Augmented (image, mask) tensors.
        """
        # Stack to 4 channels so one random transform hits both
        combined = torch.cat([image, mask], dim=0)  # (4, H, W)

        if torch.rand(1).item() > 0.5:
            combined = TF.hflip(combined)
        if torch.rand(1).item() > 0.5:
            combined = TF.vflip(combined)

        return combined[:3], combined[3:]
