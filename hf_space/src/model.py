"""Standard U-Net architecture for binary image segmentation."""

from typing import List

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two consecutive Conv2d -> BatchNorm2d -> ReLU blocks.

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.

    Example:
        >>> block = DoubleConv(3, 64)
        >>> x = torch.randn(1, 3, 128, 128)
        >>> block(x).shape
        torch.Size([1, 64, 128, 128])
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """Standard U-Net for semantic segmentation.

    Encoder-decoder architecture with skip connections. Outputs raw logits;
    apply sigmoid externally for inference or use BCEWithLogitsLoss for training.

    Args:
        in_channels: Number of input image channels (e.g., 3 for RGB).
        out_channels: Number of output segmentation channels (e.g., 1 for binary).
        features: List of feature map sizes at each encoder level.

    Example:
        >>> model = UNet(in_channels=3, out_channels=1)
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
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder: progressively double feature maps
        current_channels = in_channels
        for feature in features:
            self.encoder.append(DoubleConv(current_channels, feature))
            current_channels = feature

        # Bottleneck at deepest resolution
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder: upsample then concatenate skip connection then double conv
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder, bottleneck, and decoder.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Raw logits of shape (B, out_channels, H, W).
        """
        skip_connections = []

        # Encode: save feature maps before pooling for skip connections
        for enc_block in self.encoder:
            x = enc_block(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Reverse so index 0 is the deepest skip (closest to bottleneck)
        skip_connections = skip_connections[::-1]

        # Decode: upsample, concatenate skip, apply double conv
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            skip = skip_connections[i // 2]

            # Handle odd spatial dimensions from pooling
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])

            x = torch.cat([skip, x], dim=1)
            x = self.decoder[i + 1](x)

        return self.final_conv(x)
