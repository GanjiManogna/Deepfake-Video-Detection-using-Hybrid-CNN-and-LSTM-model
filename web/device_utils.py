#!/usr/bin/env python3
"""
device_utils.py

Selects the best available PyTorch device in this order:
1) CUDA (NVIDIA)
2) DirectML (Intel/AMD on Windows via torch-directml)
3) CPU

Usage:
    from .device_utils import get_torch_device
    device, device_name = get_torch_device()
    model.to(device)
    tensor = tensor.to(device)
"""

from __future__ import annotations

import torch


def get_torch_device():
    """Return (device, human_readable_name)."""
    # Prefer CUDA if available
    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return torch.device("cuda"), f"CUDA: {name}"
    except Exception:
        pass

    # Try DirectML (Intel/AMD on Windows)
    try:
        import torch_directml as dml  # type: ignore
        dml_device = dml.device()
        return dml_device, "DirectML"
    except Exception:
        pass

    # CPU fallback
    return torch.device("cpu"), "CPU"









