"""Attention U-Net with attention gates on skip connections."""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import DoubleConv


class AttentionGate(nn.Module):
    """Soft attention gate for gating encoder skip connections.

    Computes a spatial attention map from the gating signal (decoder) and
    the skip connection (encoder), then weights the skip connection.

    Args:
        gate_channels: Number of channels in the gating signal (from decoder).
        skip_channels: Number of channels in the skip connection (from encoder).
        inter_channels: Number of intermediate channels in the attention layer.

    Example:
        >>> gate = AttentionGate(gate_channels=256, skip_channels=256)
        >>> g = torch.randn(2, 256, 16, 16)
        >>> x = torch.randn(2, 256, 16, 16)
        >>> gate(g, x).shape
        torch.Size([2, 256, 16, 16])
    """

    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        inter_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        inter_channels = inter_channels or skip_channels // 2

        # Project gating signal
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )

        # Project skip connection
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )

        # Compute attention coefficient
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply attention gate.

        Args:
            g: Gating signal (decoder feature map), shape (B, C_g, H, W).
            x: Skip connection (encoder feature map), shape (B, C_x, H, W).

        Returns:
            Attention-weighted skip connection, shape (B, C_x, H, W).
        """
        # Align spatial dimensions: g may be smaller than x after transposed conv
        g_proj = self.W_g(g)
        x_proj = self.W_x(x)

        if g_proj.shape[2:] != x_proj.shape[2:]:
            g_proj = F.interpolate(g_proj, size=x_proj.shape[2:], mode="bilinear", align_corners=False)

        attention = self.psi(F.relu(g_proj + x_proj, inplace=True))
        return x * attention


class AttentionUNet(nn.Module):
    """U-Net with attention gates on every skip connection in the decoder.

    The attention gates learn to suppress irrelevant background activations in
    the skip connections, focusing the decoder on salient regions.

    Args:
        in_channels: Number of input image channels.
        out_channels: Number of output segmentation channels.
        features: Encoder feature map sizes at each level.

    Example:
        >>> model = AttentionUNet(in_channels=3, out_channels=1)
        >>> x = torch.randn(2, 3, 128, 128)
        >>> model(x).shape
        torch.Size([2, 1, 128, 128])
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: List[int] = [64, 128, 256, 512],
    ) -> None:
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder_upsample = nn.ModuleList()
        self.decoder_conv = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        current_channels = in_channels
        for feature in features:
            self.encoder.append(DoubleConv(current_channels, feature))
            current_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder: for each level we have an upsample, attention gate, double conv
        for feature in reversed(features):
            self.decoder_upsample.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.attention_gates.append(
                AttentionGate(gate_channels=feature, skip_channels=feature)
            )
            self.decoder_conv.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention-gated skip connections.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Raw logits of shape (B, out_channels, H, W).
        """
        skip_connections = []

        for enc_block in self.encoder:
            x = enc_block(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for i in range(len(self.decoder_upsample)):
            x = self.decoder_upsample[i](x)
            skip = skip_connections[i]

            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])

            # Gate the skip connection using the decoder feature map as signal
            gated_skip = self.attention_gates[i](g=x, x=skip)
            x = torch.cat([gated_skip, x], dim=1)
            x = self.decoder_conv[i](x)

        return self.final_conv(x)
