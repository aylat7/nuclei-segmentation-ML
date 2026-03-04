"""Unit tests for the Attention U-Net and AttentionGate modules."""

import torch
import pytest

from src.attention_unet import AttentionGate, AttentionUNet
from src.model import UNet


@pytest.fixture
def attention_model():
    return AttentionUNet(in_channels=3, out_channels=1)


def test_attention_unet_output_shape(attention_model):
    """Attention U-Net should produce the same output shape as standard U-Net."""
    x = torch.randn(2, 3, 128, 128)
    out = attention_model(x)
    assert out.shape == (2, 1, 128, 128)


def test_attention_unet_different_sizes(attention_model):
    """Attention U-Net should handle 64x64 and 256x256 inputs."""
    for size in [64, 256]:
        x = torch.randn(1, 3, size, size)
        out = attention_model(x)
        assert out.shape == (1, 1, size, size), f"Failed for size {size}"


def test_attention_unet_single_image(attention_model):
    """Batch size of 1 should work."""
    x = torch.randn(1, 3, 128, 128)
    out = attention_model(x)
    assert out.shape == (1, 1, 128, 128)


def test_attention_gate_output_shape():
    """AttentionGate should return a tensor with the same shape as the skip connection."""
    gate = AttentionGate(gate_channels=256, skip_channels=256)
    g = torch.randn(2, 256, 16, 16)
    x = torch.randn(2, 256, 16, 16)
    out = gate(g, x)
    assert out.shape == x.shape


def test_attention_gate_output_shape_mismatched_spatial():
    """AttentionGate should handle gating signal with smaller spatial dims."""
    gate = AttentionGate(gate_channels=512, skip_channels=256)
    # Gating signal is 8x8 (smaller), skip is 16x16
    g = torch.randn(2, 512, 8, 8)
    x = torch.randn(2, 256, 16, 16)
    out = gate(g, x)
    assert out.shape == x.shape


def test_attention_vs_standard_different_weights():
    """Attention U-Net and standard U-Net should have distinct parameter sets."""
    std = UNet(in_channels=3, out_channels=1)
    attn = AttentionUNet(in_channels=3, out_channels=1)

    std_params = sum(p.numel() for p in std.parameters())
    attn_params = sum(p.numel() for p in attn.parameters())

    # Attention gates add parameters, so attn should have more
    assert attn_params > std_params, (
        f"Attention U-Net ({attn_params:,}) should have more params than "
        f"standard U-Net ({std_params:,})"
    )


def test_attention_unet_no_nan():
    """Forward pass should not produce NaN or Inf values."""
    model = AttentionUNet(in_channels=3, out_channels=1)
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    assert torch.isfinite(out).all()
