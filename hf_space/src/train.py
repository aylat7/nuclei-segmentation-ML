"""Training loop with validation, checkpointing, and metric tracking."""

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluate import dice_coefficient, iou_score


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch and return the average loss.

    Args:
        model: The segmentation model (outputs raw logits).
        loader: DataLoader for the training split.
        optimizer: Optimizer to update model parameters.
        criterion: Loss function (BCEWithLogitsLoss).
        device: Compute device.

    Returns:
        Average training loss over all batches.
    """
    model.train()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Run validation and return average loss, Dice, and IoU.

    Args:
        model: The segmentation model.
        loader: DataLoader for the validation split.
        criterion: Loss function.
        device: Compute device.

    Returns:
        Dictionary with keys "loss", "dice", "iou".
    """
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += loss.item()

            # Sigmoid + threshold to get binary predictions for metrics
            preds = torch.sigmoid(logits)
            total_dice += dice_coefficient(preds, masks).item()
            total_iou += iou_score(preds, masks).item()

    n = len(loader)
    return {
        "loss": total_loss / n,
        "dice": total_dice / n,
        "iou": total_iou / n,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    learning_rate: float,
    checkpoint_path: str,
) -> Dict[str, List[float]]:
    """Full training loop with validation and best-model checkpointing.

    Saves the model weights whenever validation Dice improves.

    Args:
        model: The segmentation model.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        device: Compute device.
        num_epochs: Total number of training epochs.
        learning_rate: Initial learning rate for Adam optimizer.
        checkpoint_path: File path to save the best model weights.

    Returns:
        Dictionary with lists "train_losses", "val_losses", "val_dices", "val_ious".
    """
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_dice = 0.0
    history: Dict[str, List[float]] = {
        "train_losses": [],
        "val_losses": [],
        "val_dices": [],
        "val_ious": [],
    }

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, criterion, device)

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_metrics["loss"])
        history["val_dices"].append(val_metrics["dice"])
        history["val_ious"].append(val_metrics["iou"])

        print(
            f"Epoch [{epoch:>3}/{num_epochs}]  "
            f"Train Loss: {train_loss:.4f}  "
            f"Val Loss: {val_metrics['loss']:.4f}  "
            f"Val Dice: {val_metrics['dice']:.4f}  "
            f"Val IoU: {val_metrics['iou']:.4f}"
        )

        # Save checkpoint whenever validation Dice improves
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Saved best model (Dice: {best_dice:.4f})")

    return history
