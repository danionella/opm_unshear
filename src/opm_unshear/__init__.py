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

logger = logging.getLogger(__name__)

try:
    import cupy as cp

    cp.cuda.runtime.getDeviceCount()  # raises if no driver / no device
    gpu_available = True
except Exception:  # ImportError, CUDARuntimeError, missing CUDA libraries, ...
    gpu_available = False

if gpu_available:
    from .gpu import unshear

    logger.debug("GPU found. Using GPU implementation.")
else:
    from .cpu import unshear

    logger.debug("GPU not available. Using CPU implementation.")


def get_slope(n1, n2, M12, M23, dv, dp, polarity=1, theta_iip=None, theta_sample=None):
    """Calculate the slope of the shear.

    All angles are measured between the (oblique) plane and the optical axis of the respective objective.

    Args:
        n1 (float): refractive index at the sample (Obj1)
        n2 (float): refractive index at the intermediate image plane (Obj2)
        M12 (float): lateral magnification from Obj1 to Obj2
        M23 (float): magnification from Obj2 to Obj3
        dv (float): vertical pixel size (on the camera)
        dp (float): plane separation along the plane-scanning axis (in the sample, same unit as dv).
        polarity (int): scanning polarity (depends on scanning and plane tilt direction). 1 for positive, -1 for negative.
        theta_iip (float): angle of the intermediate image plane (vs optical axis of Obj2), in radians. Either this or theta_sample must be provided.
        theta_sample (float): angle of the sample plane (vs optical axis of Obj1), in radians. Either this or theta_iip must be provided.

    Returns:
        slope (float): slope of the shear in (px along axis 1) / (px along axis 0), with the sign given by `polarity`.
        theta_sample (float): angle of the sample plane, in radians
        theta_iip (float): angle of the intermediate image plane, in radians
    """
    if polarity not in [1, -1]:
        raise ValueError("Polarity must be 1 or -1.")
    if theta_iip is None and theta_sample is None:
        raise ValueError("Either theta_iip or theta_sample must be provided.")
    if theta_iip is not None and theta_sample is not None:
        raise ValueError("Only one of theta_iip or theta_sample must be provided.")
    if theta_iip is not None:
        theta_sample = np.arctan(np.tan(theta_iip) * (M12 / n1 * n2))
    else:
        theta_iip = np.arctan(np.tan(theta_sample) / (M12 / n1 * n2))
    # axial (z) extent of one camera pixel in the sample: pixel length along the intermediate image plane,
    # projected onto the optical axis (cos, as the angle is measured from the axis), divided by the axial magnification
    dz_sample = dv / M23 * np.cos(theta_iip) / (M12**2) * n1 / n2
    # axial shift of the plane intersection with a line parallel to the optical axis, per scan step
    dz_per_plane = dp / np.tan(theta_sample)
    slope = polarity * dz_per_plane / dz_sample
    return slope, theta_sample, theta_iip


__all__ = ["unshear", "get_slope", "gpu_available"]
