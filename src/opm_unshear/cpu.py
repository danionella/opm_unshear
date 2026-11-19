import math

import numpy as np
from numba import njit, prange

from ._common import check_args


@njit(parallel=True)
def _unshear_numba(x, sub_j, sup_i, slope, fill_value, out):
    x_sh0, x_sh1, x_sh2 = x.shape
    y_sh0, y_sh1, y_sh2 = out.shape
    abs_slope = abs(slope)
    width_i = max(1.0 / sup_i, 1.0 / abs_slope)
    slope_int = int(abs_slope)
    # integer centre, identical to the GPU kernel (x_sh[1]/2 in C integer arithmetic)
    center = float(x_sh1 // 2)

    for i in prange(y_sh0):
        for j in range(y_sh1):
            for k in range(y_sh2):
                integrator = 0.0
                norm = 0.0
                for jj in range(-slope_int, slope_int + 1):
                    s = j * sub_j + jj
                    r_float = i / sup_i + (s - center) / slope
                    r = int(math.floor(r_float))
                    ratio = r_float - r

                    if (0 <= r < x_sh0 - 1) and (0 <= s < x_sh1):
                        # left weight = triangle(ratio, width_i) * triangle(jj + ratio*slope, abs_slope)
                        wl = width_i - abs(ratio)
                        if wl > 0.0:
                            t1 = abs_slope - abs(jj + ratio * slope)
                            weight_left = wl * t1 if t1 > 0.0 else 0.0
                        else:
                            weight_left = 0.0
                        # right weight = triangle(ratio-1, width_i) * triangle(jj + (ratio-1)*slope, abs_slope)
                        wr = width_i - abs(ratio - 1.0)
                        if wr > 0.0:
                            t2 = abs_slope - abs(jj + (ratio - 1.0) * slope)
                            weight_right = wr * t2 if t2 > 0.0 else 0.0
                        else:
                            weight_right = 0.0
                        integrator += x[r, s, k] * weight_left + x[r + 1, s, k] * weight_right
                        norm += weight_left + weight_right
                if norm > 0.0:
                    out[i, j, k] = integrator / norm
                else:
                    out[i, j, k] = fill_value


def unshear(x, sub_j, sup_i, slope, out=None, fill_value=0.0):
    r"""
    Unshear a 3D array using anisotropic linear interpolation (CPU implementation).

    Args:
        x (array): 3D input (planes × cameraY × cameraX). Converted to float32.
        sub_j (int): subsampling factor along axis 1. Set to 1 (no subsampling) or larger.
        sup_i (float): upsampling factor along axis 0. Set to 1 (no upsampling) or larger.
        slope (float): shear slope (px along dim 1) / (px along dim 0). You can use `get_slope` to calculate this value.
        out (np.ndarray, optional): pre-allocated float32 output of shape (int(x.shape[0]*sup_i), x.shape[1]//sub_j, x.shape[2]).
        fill_value (float): value to use where no valid input voxels contribute.

    Returns:
        np.ndarray: the unsheared volume, dtype float32.

    Example with sub_j=2, sup_i=2, slope=3 (showing a slice along axis 2):

               input                                      output
            o-----o-----o                             o--o--o--o--o
             \     \     \                            |  |  |  |  |
              o-----o-----o                           |  |  |  |  |
               \     \     \                          |  |  |  |  |
        axis 1  o-----o-----o       -->       axis 1  o--o--o--o--o
                 \     \     \                        |  |  |  |  |
                  o-----o-----o                       |  |  |  |  |
                   \     \     \                      |  |  |  |  |
                    o-----o-----o                     o--o--o--o--o
                        axis 0                            axis 0
    """
    x = np.asarray(x, dtype=np.float32, order="C")
    out_shape = check_args(x.shape, sub_j, sup_i, slope)
    if out is None:
        out = np.empty(out_shape, dtype=np.float32)
    else:
        if not isinstance(out, np.ndarray):
            raise TypeError(f"out must be a numpy array, got {type(out)}")
        if out.shape != out_shape:
            raise ValueError(f"out has shape {out.shape}, expected {out_shape}")
        if out.dtype != np.float32:
            raise ValueError(f"out must be float32, got {out.dtype}")

    _unshear_numba(x, int(sub_j), float(sup_i), float(slope), float(fill_value), out)
    return out
