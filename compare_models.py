"""Side-by-side comparison of standard U-Net and Attention U-Net."""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import random_split

from src.attention_unet import AttentionUNet
from src.dataset import NucleiDataset
from src.evaluate import dice_coefficient, iou_score
from src.model import UNet
from src.utils import load_config, get_device, ensure_output_dir


def load_checkpoint(model: torch.nn.Module, path: str, device: torch.device) -> torch.nn.Module:
    """Load weights from checkpoint into model."""
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def compare_predictions(
    unet: torch.nn.Module,
    attn_unet: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    save_path: str,
    num_samples: int = 5,
    threshold: float = 0.5,
) -> None:
    """Generate a side-by-side comparison figure of both model predictions.

    Args:
        unet: Trained standard U-Net.
        attn_unet: Trained Attention U-Net.
        dataset: Validation dataset.
        device: Compute device.
        save_path: Output PNG path.
        num_samples: Number of rows in the grid.
        threshold: Binarization threshold.
    """
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

    fig, axes = plt.subplots(num_samples, 4, figsize=(14, 3.5 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input Image", "Ground Truth", "U-Net Prediction", "Attention U-Net"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    with torch.no_grad():
        for row, idx in enumerate(indices):
            image, mask = dataset[idx]
            img_batch = image.unsqueeze(0).to(device)

            logit_unet = unet(img_batch)
            pred_unet = (torch.sigmoid(logit_unet) >= threshold).float().squeeze().cpu().numpy()

            logit_attn = attn_unet(img_batch)
            pred_attn = (torch.sigmoid(logit_attn) >= threshold).float().squeeze().cpu().numpy()

            img_np = image.permute(1, 2, 0).numpy()
            mask_np = mask.squeeze().numpy()

            axes[row, 0].imshow(img_np)
            axes[row, 1].imshow(mask_np, cmap="gray")
            axes[row, 2].imshow(pred_unet, cmap="gray")
            axes[row, 3].imshow(pred_attn, cmap="gray")

            for col in range(4):
                axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison figure: {save_path}")


def print_comparison_metrics(
    unet: torch.nn.Module,
    attn_unet: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    threshold: float = 0.5,
) -> None:
    """Compute and print a metric comparison table for both models.

    Args:
        unet: Standard U-Net in eval mode.
        attn_unet: Attention U-Net in eval mode.
        dataset: Validation dataset.
        device: Compute device.
        threshold: Binarization threshold.
    """
    models = {"U-Net": unet, "Attention U-Net": attn_unet}
    results = {}

    for name, model in models.items():
        total_dice = 0.0
        total_iou = 0.0

        with torch.no_grad():
            for idx in range(len(dataset)):
                image, mask = dataset[idx]
                logit = model(image.unsqueeze(0).to(device))
                pred = torch.sigmoid(logit)
                total_dice += dice_coefficient(pred, mask.unsqueeze(0).to(device)).item()
                total_iou += iou_score(pred, mask.unsqueeze(0).to(device)).item()

        results[name] = {
            "dice": total_dice / len(dataset),
            "iou": total_iou / len(dataset),
        }

    print("\n" + "=" * 50)
    print(f"{'Model':<20}  {'Dice':>8}  {'IoU':>8}")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"{name:<20}  {metrics['dice']:>8.4f}  {metrics['iou']:>8.4f}")
    print("=" * 50)


def main() -> None:
    cfg = load_config("configs/config.yaml")
    device = get_device(cfg["device"])
    output_dir = ensure_output_dir(cfg["output_dir"])

    unet_ckpt = output_dir / "best_unet.pth"
    attn_ckpt = output_dir / "best_attention_unet.pth"

    if not unet_ckpt.exists() or not attn_ckpt.exists():
        print("Both model checkpoints must exist. Run train_model.py --model both first.")
        return

    # Validation split (same seed as training)
    full_dataset = NucleiDataset(
        data_dir=cfg["data_dir"], image_size=cfg["image_size"], augment=False
    )
    total = len(full_dataset)
    train_size = int(cfg["train_split"] * total)
    val_size = total - train_size
    generator = torch.Generator().manual_seed(42)
    _, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Load models
    unet = load_checkpoint(
        UNet(cfg["in_channels"], cfg["out_channels"], cfg["features"]),
        str(unet_ckpt),
        device,
    )
    attn_unet = load_checkpoint(
        AttentionUNet(cfg["in_channels"], cfg["out_channels"], cfg["features"]),
        str(attn_ckpt),
        device,
    )

    # Comparison figure
    compare_predictions(
        unet, attn_unet, val_subset, device,
        save_path=str(output_dir / "comparison.png"),
        num_samples=5,
    )

    # Metrics table
    print_comparison_metrics(unet, attn_unet, val_subset, device)


if __name__ == "__main__":
    main()
