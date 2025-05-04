"""
.. include:: ../../README.md
"""

import logging

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

def get_slope(n1, n2, M1_2, M2_3, dv, dp, theta_iip=None, theta_sample=None):
    """Calculate the slope of the shear.

    Args:
        n1 (float): refractive index at the sample (Obj1)
        n2 (float): refractive index at the intermediate image plane (Obj2)
        M1_2 (float): magnification from Obj1 to Obj2
        M2_3 (float): magnification from Obj2 to Obj3
        dv (float): vertical pixel size (on the camera)
        dp (float): plane separation along the plane-scanning axis.
        theta_iip (float): angle of the intermedia image plane, in radians. Either this or theta_sample must be provided.
        theta_sample (float): angle of the sample plane, in radians. Either this or theta_iip must be provided.

    Returns:
        slope (float): slope of the shear in (px along axis 1) / (px along axis 0). Note: this value is always positive. You may still have to fliip the sign depending on scanning polarity.
        theta_sample (float): angle of the sample plane, in radians
        theta_iip (float): angle of the intermedia image plane, in radians
    """

    if theta_iip is None and theta_sample is None:
        raise ValueError("Either theta_iip or theta_sample must be provided.")
    if theta_iip is not None and theta_sample is not None:
        raise ValueError("Only one of theta_iip or theta_sample must be provided.")
    if theta_iip is not None:
        theta_sample = np.arctan(np.tan(theta_iip) * (M1_2 / n1 * n2))
    if theta_sample is not None:
        theta_iip = np.arctan(np.tan(theta_sample) / (M1_2 / n1 * n2))
    dz_sample = dv / M2_3 * np.sin(theta_iip) / (M1_2**2) * n1 / n2
    slope = (dp / np.tan(theta_sample)) / dz_sample
    return slope, theta_sample, theta_iip
