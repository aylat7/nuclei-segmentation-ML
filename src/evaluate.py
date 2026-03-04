"""Evaluation metrics and prediction visualization for segmentation models."""

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.model import UNet
from src.attention_unet import AttentionUNet


def dice_coefficient(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Compute the Dice coefficient between predictions and targets.

    Args:
        preds: Predicted probabilities (after sigmoid), shape (B, 1, H, W).
        targets: Ground truth binary masks, shape (B, 1, H, W).
        threshold: Binarization threshold for predictions.
        smooth: Smoothing constant to avoid division by zero.

    Returns:
        Scalar Dice coefficient tensor.

    Example:
        >>> p = torch.ones(1, 1, 4, 4)
        >>> t = torch.ones(1, 1, 4, 4)
        >>> dice_coefficient(p, t).item()  # doctest: +ELLIPSIS
        1.0...
    """
    preds_binary = (preds >= threshold).float()
    intersection = (preds_binary * targets).sum()
    return (2.0 * intersection + smooth) / (preds_binary.sum() + targets.sum() + smooth)


def iou_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Compute the Intersection over Union (IoU / Jaccard index).

    Args:
        preds: Predicted probabilities (after sigmoid), shape (B, 1, H, W).
        targets: Ground truth binary masks, shape (B, 1, H, W).
        threshold: Binarization threshold for predictions.
        smooth: Smoothing constant to avoid division by zero.

    Returns:
        Scalar IoU tensor.

    Example:
        >>> p = torch.ones(1, 1, 4, 4)
        >>> t = torch.ones(1, 1, 4, 4)
        >>> iou_score(p, t).item()  # doctest: +ELLIPSIS
        1.0...
    """
    preds_binary = (preds >= threshold).float()
    intersection = (preds_binary * targets).sum()
    union = preds_binary.sum() + targets.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def load_model(
    checkpoint_path: str,
    in_channels: int = 3,
    out_channels: int = 1,
    features: List[int] = [64, 128, 256, 512],
    device: Optional[torch.device] = None,
    model_type: str = "unet",
) -> nn.Module:
    """Load a segmentation model from a checkpoint and return it in eval mode.

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        features: Encoder feature sizes.
        device: Target device; defaults to CPU if not specified.
        model_type: Architecture to instantiate - "unet" or "attention_unet".

    Returns:
        Model in eval mode with loaded weights.
    """
    if device is None:
        device = torch.device("cpu")

    if model_type == "attention_unet":
        model = AttentionUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
        )
    else:
        model = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
        )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def plot_predictions(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    save_path: str,
    num_samples: int = 6,
    threshold: float = 0.5,
) -> None:
    """Save a grid of [Input | Ground Truth | Prediction] figures.

    Args:
        model: Trained segmentation model in eval mode.
        dataset: Dataset to sample from (returns image, mask tensors).
        device: Compute device.
        save_path: File path to save the PNG figure.
        num_samples: Number of sample rows to display.
        threshold: Binarization threshold for predictions.
    """
    model.eval()
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input Image", "Ground Truth", "Prediction"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold")

    with torch.no_grad():
        for row, idx in enumerate(indices):
            image, mask = dataset[idx]
            logit = model(image.unsqueeze(0).to(device))
            pred = (torch.sigmoid(logit) >= threshold).float().squeeze().cpu().numpy()

            img_np = image.permute(1, 2, 0).numpy()
            mask_np = mask.squeeze().numpy()

            axes[row, 0].imshow(img_np)
            axes[row, 0].axis("off")

            axes[row, 1].imshow(mask_np, cmap="gray")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(pred, cmap="gray")
            axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_evaluation_panels(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    save_path: str,
    num_samples: int = 4,
    threshold: float = 0.5,
    use_watershed: bool = True,
) -> None:
    """Generate color-coded 4-panel evaluation figures for best/worst samples.

    Selects samples by Dice score, generates panels via plot_evaluation_panels,
    and saves a combined figure.

    Args:
        model: Trained segmentation model in eval mode.
        dataset: Dataset to evaluate on.
        device: Compute device.
        save_path: File path to save the PNG figure.
        num_samples: How many samples to show.
        threshold: Binarization threshold for predictions.
    """
    # Import here to avoid circular imports
    from src.visualize import compute_instance_metrics, plot_evaluation_panels

    model.eval()
    scored_samples = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            image, mask = dataset[idx]
            logit = model(image.unsqueeze(0).to(device))
            pred = (torch.sigmoid(logit) >= threshold).float().squeeze()
            d = dice_coefficient(
                pred.unsqueeze(0).unsqueeze(0),
                mask.unsqueeze(0).to(device),
            ).item()
            scored_samples.append((d, idx))

    scored_samples.sort(key=lambda x: x[0])

    # Take a mix of worst and best samples
    half = num_samples // 2
    selected = scored_samples[:half] + scored_samples[-half:]

    save_dir = Path(save_path).parent
    panel_paths = []

    for rank, (dice_val, idx) in enumerate(selected):
        image, mask = dataset[idx]
        with torch.no_grad():
            logit = model(image.unsqueeze(0).to(device))
            pred = (torch.sigmoid(logit) >= threshold).float().squeeze().cpu().numpy()

        img_np = image.permute(1, 2, 0).numpy()
        mask_np = mask.squeeze().numpy()

        panel_path = save_dir / f"_tmp_panel_{rank}.png"
        plot_evaluation_panels(img_np, pred, mask_np, str(panel_path), use_watershed=use_watershed)
        panel_paths.append((dice_val, panel_path))

    # Assemble into one figure
    from PIL import Image as PILImage

    images_pil = [PILImage.open(str(p)) for _, p in panel_paths]
    widths, heights = zip(*(img.size for img in images_pil))
    total_height = sum(heights)
    max_width = max(widths)

    combined = PILImage.new("RGB", (max_width, total_height), color=(255, 255, 255))
    y_offset = 0
    for img_pil in images_pil:
        combined.paste(img_pil, (0, y_offset))
        y_offset += img_pil.size[1]

    combined.save(save_path)

    # Clean up temp files
    for _, p in panel_paths:
        p.unlink(missing_ok=True)


def print_instance_metrics(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    num_samples: int = 20,
    threshold: float = 0.5,
    use_watershed: bool = True,
) -> None:
    """Compute and print instance-level precision, recall, F1 on a subset.

    Args:
        model: Trained model in eval mode.
        dataset: Dataset with get_individual_masks() support.
        device: Compute device.
        num_samples: Number of samples to evaluate.
        threshold: Binarization threshold.
    """
    from src.visualize import compute_instance_metrics

    model.eval()
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    count = 0

    with torch.no_grad():
        for idx in indices:
            image, mask = dataset[idx]
            logit = model(image.unsqueeze(0).to(device))
            pred = (torch.sigmoid(logit) >= threshold).float().squeeze().cpu().numpy()
            mask_np = mask.squeeze().numpy()

            metrics = compute_instance_metrics(pred, mask_np, use_watershed=use_watershed)
            total_precision += metrics["precision"]
            total_recall += metrics["recall"]
            total_f1 += metrics["f1"]
            count += 1

    print(
        f"Instance Metrics (n={count}):  "
        f"Precision: {total_precision / count:.4f}  "
        f"Recall: {total_recall / count:.4f}  "
        f"F1: {total_f1 / count:.4f}"
    )
