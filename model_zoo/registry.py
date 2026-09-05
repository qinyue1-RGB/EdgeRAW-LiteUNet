"""Single model factory used by training, profiling and benchmarking."""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable

from model_zoo.Unet import Unet
from model_zoo.LiteUNet_v1 import LiteUNet_v1
from model_zoo.LiteUNet_v2 import LiteUNet_v2


MODEL_REGISTRY: OrderedDict[str, Callable[..., object]] = OrderedDict(
    [
        ("Original UNet", Unet),
        ("LiteUNet-v1", LiteUNet_v1),
        ("LiteUNet-v2", LiteUNet_v2),
    ]
)


def _normalise(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


_ALIASES = {
    "unet": "Original UNet",
    "originalunet": "Original UNet",
    "liteunetv1": "LiteUNet-v1",
    "liteunetv2": "LiteUNet-v2",
}


def canonical_model_name(name: str) -> str:
    try:
        return _ALIASES[_normalise(name)]
    except KeyError as error:
        choices = ", ".join(MODEL_REGISTRY)
        raise ValueError(f"Unknown model '{name}'. Available models: {choices}") from error


def create_model(name: str, in_channels: int = 4, out_channels: int = 4):
    canonical_name = canonical_model_name(name)
    return MODEL_REGISTRY[canonical_name](in_channels=in_channels, out_channels=out_channels)
