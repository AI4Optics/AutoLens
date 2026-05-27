import torch


def init_device():
    """Initialize and return the default compute device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"Using CUDA: {device_name} for DeepLens")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple MPS"
        print("Using MPS (Apple Silicon) for DeepLens")
    else:
        device = torch.device("cpu")
        device_name = "CPU"
        print("Using CPU for DeepLens")
    return device


# optics
from .base import DeepObj
from .material import Material
from .ray import Ray

# Lens classes
from .lens import Lens
from .geolens import GeoLens

# geolens
from .geolens_pkg import *

# utilities
from .utils_deeplens import *
