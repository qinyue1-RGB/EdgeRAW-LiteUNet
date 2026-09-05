"""LiteUNet-v2: LiteUNet-v1 with every main channel width halved."""

from model_zoo.lightweight_blocks import LightweightUNet


class LiteUNet_v2(LightweightUNet):
    def __init__(self, in_channels: int = 4, out_channels: int = 4) -> None:
        super().__init__((16, 32, 64, 128, 256), in_channels, out_channels)
