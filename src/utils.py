"""Utility functions: config loading, device setup, and plotting helpers."""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file into a dictionary.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary of configuration values.

    Example:
        >>> cfg = load_config("configs/config.yaml")
        >>> cfg["image_size"]
        128
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_device(device_str: str = "auto") -> torch.device:
    """Resolve the compute device string to a torch.device.

    Args:
        device_str: One of "auto", "cuda", "mps", or "cpu". "auto" picks
            CUDA first, then MPS (Apple Silicon), then CPU.

    Returns:
        Resolved torch.device.

    Example:
        >>> device = get_device("auto")
        >>> isinstance(device, torch.device)
        True
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_dices: List[float],
    val_ious: List[float],
    save_path: str,
) -> None:
    """Plot and save training loss and validation metric curves.

    Args:
        train_losses: Per-epoch training losses.
        val_losses: Per-epoch validation losses.
        val_dices: Per-epoch validation Dice scores.
        val_ious: Per-epoch validation IoU scores.
        save_path: File path to save the PNG plot.
    """
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curves
    axes[0].plot(epochs, train_losses, label="Train Loss", color="steelblue")
    axes[0].plot(epochs, val_losses, label="Val Loss", color="tomato")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Metric curves
    axes[1].plot(epochs, val_dices, label="Val Dice", color="forestgreen")
    axes[1].plot(epochs, val_ious, label="Val IoU", color="darkorange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Metrics")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def ensure_output_dir(output_dir: str) -> Path:
    """Create the output directory if it does not exist.

    Args:
        output_dir: Path string for the output directory.

    Returns:
        Path object for the output directory.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path
