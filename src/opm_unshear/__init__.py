"""
.. include:: ../../README.md

---
# API Documentation

## `opm_unshear.unshear`

Depending on GPU presence, `opm_unshear.unshear` maps to one of the following:

- `opm_unshear.gpu.unshear` (when GPU is available)
- `opm_unshear.cpu.unshear` (otherwise)

## `opm_unshear.get_slope`

`get_slope` calculates the slope parameter for `unshear`.

"""

import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

try:
    import cupy as cp
    _ = cp.cuda.runtime.getDeviceCount()  # Check if any GPU devices are available
    gpu_available = True
    from .gpu import unshear
    logging.info("GPU found. Using GPU implementation.")
except (ImportError, cp.cuda.runtime.CUDARuntimeError):
    gpu_available = False
    from .cpu import unshear
    logging.info("GPU not available. Using CPU implementation.")

def get_slope(n1, n2, M12, M23, dv, dp, polarity=1, theta_iip=None, theta_sample=None):
    """Calculate the slope of the shear.

    Args:
        n1 (float): refractive index at the sample (Obj1)
        n2 (float): refractive index at the intermediate image plane (Obj2)
        M12 (float): magnification from Obj1 to Obj2
        M23 (float): magnification from Obj2 to Obj3
        dv (float): vertical pixel size (on the camera)
        dp (float): plane separation along the plane-scanning axis.
        polarity (int): scanning polarity (depends on scanning and plane tilt directiona). 1 for positive, -1 for negative.
        theta_iip (float): angle of the intermedia image plane (vs optical axis of Obj2), in radians. Either this or theta_sample must be provided.
        theta_sample (float): angle of the sample plane (vs optical axis of Obj1), in radians. Either this or theta_iip must be provided.

    Returns:
        slope (float): slope of the shear in (px along axis 1) / (px along axis 0). Note: this value is always positive. You may still have to fliip the sign depending on scanning polarity.
        theta_sample (float): angle of the sample plane, in radians
        theta_iip (float): angle of the intermedia image plane, in radians
    """
    if polarity not in [1, -1]:
        raise ValueError("Polarity must be 1 or -1.")
    if theta_iip is None and theta_sample is None:
        raise ValueError("Either theta_iip or theta_sample must be provided.")
    if theta_iip is not None and theta_sample is not None:
        raise ValueError("Only one of theta_iip or theta_sample must be provided.")
    if theta_iip is not None:
        theta_sample = np.arctan(np.tan(theta_iip) * (M12 / n1 * n2))
    if theta_sample is not None:
        theta_iip = np.arctan(np.tan(theta_sample) / (M12 / n1 * n2))
    dz_sample = dv / M23 * np.sin(theta_iip) / (M12**2) * n1 / n2
    slope = (dp / np.tan(theta_sample)) / dz_sample
    slope = slope * polarity
    return slope, theta_sample, theta_iip


# Explicitly define the public API
# __all__ = ["unshear", "get_slope"]
