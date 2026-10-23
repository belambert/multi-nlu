"""Picking a device and dtype, and timing work on it, across cpu, mps and cuda."""

import time
from collections.abc import Callable

import torch


def pick_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device, train: bool = False) -> torch.dtype:
    """Half precision where it is safe; cpu and mps train in fp32 to stay stable."""
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps" and not train:
        return torch.float16
    return torch.float32


def timed[T](dev: torch.device, fn: Callable[[], T]) -> tuple[T, float]:
    """Call fn, returning its result and the wall-clock seconds it took on dev."""
    _sync(dev)
    start = time.perf_counter()
    result = fn()
    _sync(dev)
    return result, time.perf_counter() - start


def _sync(dev: torch.device) -> None:
    # gpu kernels run asynchronously, so wait for them before reading the clock
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps":
        torch.mps.synchronize()
