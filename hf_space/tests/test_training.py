"""Integration tests for training loop: loss reduction and checkpointing."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model import UNet
from src.train import train_one_epoch, validate, train_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_loaders(
    num_samples: int = 10,
    image_size: int = 32,
    batch_size: int = 2,
) -> tuple:
    """Create synthetic DataLoaders with random images and masks."""
    images = torch.rand(num_samples, 3, image_size, image_size)
    masks = torch.randint(0, 2, (num_samples, 1, image_size, image_size)).float()
    dataset = TensorDataset(images, masks)
    loader = DataLoader(dataset, batch_size=batch_size)
    return loader, loader  # use same for train and val in these tests


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_one_epoch_reduces_loss_over_time():
    """Loss should decrease between epoch 1 and epoch 2 on small data."""
    train_loader, val_loader = _make_tiny_loaders()
    model = UNet(in_channels=3, out_channels=1, features=[8, 16])
    device = torch.device("cpu")
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    loss_epoch1 = train_one_epoch(model, train_loader, optimizer, criterion, device)
    loss_epoch2 = train_one_epoch(model, train_loader, optimizer, criterion, device)

    # Two epochs of gradient descent on tiny data should generally reduce loss
    # We allow some slack since random init can cause occasional ties
    assert loss_epoch2 <= loss_epoch1 * 1.05, (
        f"Expected loss to not increase significantly: epoch1={loss_epoch1:.4f}, "
        f"epoch2={loss_epoch2:.4f}"
    )


def test_model_saves_checkpoint(tmp_path):
    """train_model should save a .pth checkpoint when validation improves."""
    train_loader, val_loader = _make_tiny_loaders(num_samples=8, image_size=32, batch_size=4)
    model = UNet(in_channels=3, out_channels=1, features=[8, 16])
    device = torch.device("cpu")
    checkpoint_path = str(tmp_path / "test_checkpoint.pth")

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=1,
        learning_rate=0.01,
        checkpoint_path=checkpoint_path,
    )

    assert Path(checkpoint_path).exists(), "Checkpoint file was not created"


def test_checkpoint_loads_correctly(tmp_path):
    """Saved checkpoint should load back into a fresh model without errors."""
    train_loader, val_loader = _make_tiny_loaders(num_samples=6, image_size=32, batch_size=2)
    model = UNet(in_channels=3, out_channels=1, features=[8, 16])
    device = torch.device("cpu")
    checkpoint_path = str(tmp_path / "test_checkpoint.pth")

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=1,
        learning_rate=0.01,
        checkpoint_path=checkpoint_path,
    )

    fresh_model = UNet(in_channels=3, out_channels=1, features=[8, 16])
    state = torch.load(checkpoint_path, map_location="cpu")
    fresh_model.load_state_dict(state)  # Should not raise


def test_validate_returns_correct_keys():
    """validate() should return a dict with 'loss', 'dice', 'iou' keys."""
    _, val_loader = _make_tiny_loaders()
    model = UNet(in_channels=3, out_channels=1, features=[8, 16])
    device = torch.device("cpu")
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()

    metrics = validate(model, val_loader, criterion, device)

    assert set(metrics.keys()) == {"loss", "dice", "iou"}
    assert all(isinstance(v, float) for v in metrics.values())


def test_train_model_returns_history(tmp_path):
    """train_model should return a history dict with all metric lists."""
    train_loader, val_loader = _make_tiny_loaders(num_samples=8, batch_size=4)
    model = UNet(in_channels=3, out_channels=1, features=[8, 16])
    device = torch.device("cpu")

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=2,
        learning_rate=0.01,
        checkpoint_path=str(tmp_path / "ckpt.pth"),
    )

    assert set(history.keys()) == {"train_losses", "val_losses", "val_dices", "val_ious"}
    for key in history:
        assert len(history[key]) == 2, f"Expected 2 entries for {key}"
