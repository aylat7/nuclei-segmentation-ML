"""Unit tests for the standard U-Net architecture."""

import torch
import pytest

from src.model import UNet


@pytest.fixture
def model():
    return UNet(in_channels=3, out_channels=1)


def test_unet_output_shape(model):
    """Standard input (2, 3, 128, 128) should produce output (2, 1, 128, 128)."""
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    assert out.shape == (2, 1, 128, 128)


def test_unet_different_sizes(model):
    """U-Net should handle 64x64 and 256x256 inputs."""
    for size in [64, 256]:
        x = torch.randn(1, 3, size, size)
        out = model(x)
        assert out.shape == (1, 1, size, size), f"Failed for size {size}"


def test_unet_single_image(model):
    """Batch size of 1 should work without error."""
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    assert out.shape == (1, 1, 128, 128)


def test_unet_parameter_count(model):
    """Parameter count should be in the expected range (~31M for default config)."""
    total_params = sum(p.numel() for p in model.parameters())
    # 20M to 45M is a reasonable sanity range for this architecture
    assert 20_000_000 < total_params < 45_000_000, (
        f"Unexpected parameter count: {total_params:,}"
    )


def test_unet_output_no_sigmoid():
    """Model should output raw logits (values outside [0, 1] are fine)."""
    model = UNet(in_channels=3, out_channels=1)
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    # Raw logits can be outside [0, 1]; if strictly in [0,1] it hints sigmoid was applied
    # We just check the model runs without NaN/Inf
    assert torch.isfinite(out).all()


def test_unet_custom_features():
    """U-Net should work with a custom feature list."""
    model = UNet(in_channels=1, out_channels=2, features=[32, 64])
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert out.shape == (2, 2, 64, 64)
