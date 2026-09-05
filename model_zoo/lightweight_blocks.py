"""Reusable blocks for the lightweight RAW UNet variants."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """A kxk depthwise convolution followed by a 1x1 pointwise convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class LiteDownsample2D(nn.Module):
    """Original UNet downsample topology with only the 3x3 conv factorized."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.down = nn.Sequential(
            DepthwiseSeparableConv(channels, channels),
            nn.Conv2d(channels, channels, kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class LiteUpsample2D(nn.Module):
    """Original nearest-neighbour upsample with a factorized 3x3 projection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            DepthwiseSeparableConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


def _double_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        DepthwiseSeparableConv(in_channels, out_channels),
        nn.ReLU(),
        DepthwiseSeparableConv(out_channels, out_channels),
        nn.ReLU(),
    )


class LightweightUNet(nn.Module):
    """UNet topology shared by LiteUNet-v1/v2.

    The residual RAW-to-RAW contract and padding behaviour intentionally match
    the repository's original ``Unet`` implementation for a fair ablation.
    """

    def __init__(
        self,
        channels: tuple[int, int, int, int, int],
        in_channels: int = 4,
        out_channels: int = 4,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = channels

        self.conv_1 = _double_conv(in_channels, c1)
        self.pool1 = LiteDownsample2D(c1)
        self.conv_2 = _double_conv(c1, c2)
        self.pool2 = LiteDownsample2D(c2)
        self.conv_3 = _double_conv(c2, c3)
        self.pool3 = LiteDownsample2D(c3)
        self.conv_4 = _double_conv(c3, c4)
        self.pool4 = LiteDownsample2D(c4)
        self.conv_5 = _double_conv(c4, c5)

        self.upv6 = LiteUpsample2D(c5, c4)
        self.conv_6 = _double_conv(c4 * 2, c4)
        self.upv7 = LiteUpsample2D(c4, c3)
        self.conv_7 = _double_conv(c3 * 2, c3)
        self.upv8 = LiteUpsample2D(c3, c2)
        self.conv_8 = _double_conv(c2 * 2, c2)
        self.upv9 = LiteUpsample2D(c2, c1)
        self.conv_9 = _double_conv(c1 * 2, c1)
        self.conv_10 = nn.Conv2d(c1, out_channels, kernel_size=1)

    @staticmethod
    def forward_features(inputs: torch.Tensor) -> torch.Tensor:
        _, _, height, width = inputs.shape
        height_pad = 32 - height % 32
        width_pad = 32 - width % 32
        return F.pad(inputs, (0, width_pad, 0, height_pad), "constant")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, _, height, width = inputs.shape
        inp = self.forward_features(inputs)

        conv1 = self.conv_1(inp)
        conv2 = self.conv_2(self.pool1(conv1))
        conv3 = self.conv_3(self.pool2(conv2))
        conv4 = self.conv_4(self.pool3(conv3))
        conv5 = self.conv_5(self.pool4(conv4))

        conv6 = self.conv_6(torch.cat([self.upv6(conv5), conv4], dim=1))
        conv7 = self.conv_7(torch.cat([self.upv7(conv6), conv3], dim=1))
        conv8 = self.conv_8(torch.cat([self.upv8(conv7), conv2], dim=1))
        conv9 = self.conv_9(torch.cat([self.upv9(conv8), conv1], dim=1))

        output = inp + self.conv_10(conv9)
        return output[..., :height, :width]
