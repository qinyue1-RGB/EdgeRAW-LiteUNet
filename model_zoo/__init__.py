from model_zoo.Unet import Unet
from model_zoo.LiteUNet_v1 import LiteUNet_v1
from model_zoo.LiteUNet_v2 import LiteUNet_v2
from model_zoo.lightweight_blocks import DepthwiseSeparableConv

__all__ = ["Unet", "LiteUNet_v1", "LiteUNet_v2", "DepthwiseSeparableConv"]


def __getattr__(name):
    """Keep legacy models importable without forcing all optional dependencies."""
    legacy = {
        "cycleisp": ("model_zoo.Cycleisp", "cycleisp"),
        "MPRNet": ("model_zoo.MPRNet", "MPRNet"),
        "PMRID": ("model_zoo.PMRID", "PMRID"),
        "Restormer": ("model_zoo.Restormer", "Restormer"),
    }
    if name not in legacy:
        raise AttributeError(name)
    import importlib

    module_name, attribute = legacy[name]
    return getattr(importlib.import_module(module_name), attribute)
