"""LiteUNet-v1: depthwise separable convolutions at original channel widths."""

from model_zoo.lightweight_blocks import LightweightUNet


class LiteUNet_v1(LightweightUNet):
    def __init__(self, in_channels: int = 4, out_channels: int = 4) -> None:
        super().__init__((32, 64, 128, 256, 512), in_channels, out_channels)
