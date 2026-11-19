import numbers
import warnings

import numpy as np


def check_args(x_shape, sub_j, sup_i, slope):
    """Validate unshear arguments and return the output shape. Shared by the CPU and GPU implementations."""
    if len(x_shape) != 3:
        raise ValueError(f"Input must be 3D (planes x cameraY x cameraX), got shape {tuple(x_shape)}")
    if isinstance(sub_j, bool) or not isinstance(sub_j, numbers.Integral):
        raise TypeError(f"sub_j must be an integer, got {sub_j!r} of type {type(sub_j)}")
    if sub_j < 1:
        raise ValueError(f"sub_j must be >= 1, got {sub_j}")
    if not np.isfinite(sup_i) or sup_i <= 0:
        raise ValueError(f"sup_i must be > 0, got {sup_i}")
    if not np.isfinite(slope) or slope == 0:
        raise ValueError(f"slope must be finite and non-zero, got {slope}")
    if sub_j > abs(slope):
        warnings.warn(f"sub_j > abs(slope) leads to subsampling! sub_j: {sub_j}, slope: {slope}")
    return (int(x_shape[0] * sup_i), x_shape[1] // int(sub_j), x_shape[2])
