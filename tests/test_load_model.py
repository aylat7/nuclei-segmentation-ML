"""Tests for load_model architecture dispatch."""

import torch
import pytest

from src.attention_unet import AttentionUNet
from src.evaluate import load_model
from src.model import UNet


def _save_checkpoint(model: torch.nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def test_load_model_unet(tmp_path):
    """load_model with model_type='unet' should return a UNet instance."""
    ckpt = str(tmp_path / "unet.pth")
    _save_checkpoint(UNet(3, 1, [8, 16]), ckpt)
    model = load_model(ckpt, features=[8, 16], model_type="unet")
    assert isinstance(model, UNet)


def test_load_model_attention_unet(tmp_path):
    """load_model with model_type='attention_unet' should return an AttentionUNet."""
    ckpt = str(tmp_path / "attn.pth")
    _save_checkpoint(AttentionUNet(3, 1, [8, 16]), ckpt)
    model = load_model(ckpt, features=[8, 16], model_type="attention_unet")
    assert isinstance(model, AttentionUNet)


def test_load_model_default_is_unet(tmp_path):
    """load_model without model_type should default to UNet."""
    ckpt = str(tmp_path / "unet_default.pth")
    _save_checkpoint(UNet(3, 1, [8, 16]), ckpt)
    model = load_model(ckpt, features=[8, 16])
    assert isinstance(model, UNet)


def test_load_model_wrong_type_raises(tmp_path):
    """Loading an AttentionUNet checkpoint into UNet should raise a RuntimeError."""
    ckpt = str(tmp_path / "attn.pth")
    _save_checkpoint(AttentionUNet(3, 1, [8, 16]), ckpt)
    with pytest.raises(RuntimeError):
        load_model(ckpt, features=[8, 16], model_type="unet")


def test_load_model_returns_eval_mode(tmp_path):
    """Loaded model should be in eval mode."""
    ckpt = str(tmp_path / "unet.pth")
    _save_checkpoint(UNet(3, 1, [8, 16]), ckpt)
    model = load_model(ckpt, features=[8, 16], model_type="unet")
    assert not model.training
