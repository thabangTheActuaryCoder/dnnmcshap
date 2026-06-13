"""Type aliases used across the package."""

from __future__ import annotations

from typing import Union

import numpy as np
import numpy.typing as npt
import torch

ArrayLike = Union[np.ndarray, list[float], list[list[float]]]
FloatArray = npt.NDArray[np.float32]
TensorLike = Union[torch.Tensor, np.ndarray]
