"""Main entry point: train U-Net or Attention U-Net on the nuclei dataset."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import random_split, DataLoader

from src.dataset import NucleiDataset
from src.evaluate import plot_predictions, generate_evaluation_panels, print_instance_metrics
from src.model import UNet
from src.attention_unet import AttentionUNet
from src.train import train_model
from src.utils import load_config, get_device, plot_training_curves, ensure_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train segmentation model on nuclei dataset")
    parser.add_argument(
        "--model",
        choices=["unet", "attention_unet", "both"],
        default="both",
        help="Which model to train (default: both)",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config YAML file",
    )
    return parser.parse_args()


def build_model(model_name: str, cfg: dict) -> torch.nn.Module:
    """Instantiate a model by name using config settings."""
    if model_name == "attention_unet":
        return AttentionUNet(
            in_channels=cfg["in_channels"],
            out_channels=cfg["out_channels"],
            features=cfg["features"],
        )
    return UNet(
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        features=cfg["features"],
    )


def run_training(model_name: str, cfg: dict, device: torch.device) -> None:
    """Train a single model and generate output files.

    Args:
        model_name: "unet" or "attention_unet".
        cfg: Configuration dictionary.
        device: Compute device.
    """
    output_dir = ensure_output_dir(cfg["output_dir"])

    print(f"\n{'='*60}")
    print(f"Training: {model_name.upper()}")
    print(f"{'='*60}")

    # Data loading with reproducible split
    full_dataset = NucleiDataset(
        data_dir=cfg["data_dir"],
        image_size=cfg["image_size"],
        augment=True,
    )
    val_dataset = NucleiDataset(
        data_dir=cfg["data_dir"],
        image_size=cfg["image_size"],
        augment=False,
    )

    total = len(full_dataset)
    train_size = int(cfg["train_split"] * total)
    val_size = total - train_size

    generator = torch.Generator().manual_seed(42)
    train_subset, _ = random_split(full_dataset, [train_size, val_size], generator=generator)
    _, val_subset = random_split(val_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = build_model(model_name, cfg)
    checkpoint_path = str(output_dir / f"best_{model_name}.pth")

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=cfg["num_epochs"],
        learning_rate=cfg["learning_rate"],
        checkpoint_path=checkpoint_path,
    )

    # Save training curve
    curve_path = str(output_dir / f"training_curve_{model_name}.png")
    plot_training_curves(
        history["train_losses"],
        history["val_losses"],
        history["val_dices"],
        history["val_ious"],
        save_path=curve_path,
    )
    print(f"\nSaved training curve: {curve_path}")

    # Load best model for visualization
    from src.evaluate import load_model
    best_model = load_model(
        checkpoint_path,
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        features=cfg["features"],
        device=device,
        model_type=model_name,
    )

    # Generate predictions grid
    pred_path = str(output_dir / f"predictions_{model_name}.png")
    plot_predictions(best_model, val_subset, device, pred_path, num_samples=6)
    print(f"Saved predictions: {pred_path}")

    # Generate color-coded evaluation panels
    panels_path = str(output_dir / f"evaluation_panels_{model_name}.png")
    generate_evaluation_panels(best_model, val_subset, device, panels_path, num_samples=4)
    print(f"Saved evaluation panels: {panels_path}")

    # Print instance metrics
    print_instance_metrics(best_model, val_subset, device, num_samples=20)

    # Symlink best_model.pth to the last trained model for the demo
    default_ckpt = output_dir / "best_model.pth"
    if default_ckpt.exists() or default_ckpt.is_symlink():
        default_ckpt.unlink()
    default_ckpt.symlink_to(Path(checkpoint_path).name)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg["device"])
    print(f"Using device: {device}")

    models_to_train = (
        ["unet", "attention_unet"] if args.model == "both" else [args.model]
    )

    for model_name in models_to_train:
        run_training(model_name, cfg, device)

    print("\nTraining complete. Check the outputs/ directory for results.")


if __name__ == "__main__":
    main()
