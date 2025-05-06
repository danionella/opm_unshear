# Run: pytest tests/test_all.py
# All functions to be tested should start with test_ prefix

import warnings
import os
import subprocess
import pytest

import numpy as np

import opm_unshear.gpu
import opm_unshear.cpu
from opm_unshear import gpu_available

if not gpu_available:
    warnings.warn("No GPU detected. Skipping GPU tests.")

sample_data = np.random.rand(20, 30, 40).astype(np.float32)


@pytest.mark.skipif(not gpu_available, reason="No GPU detected.")
def test_unshear_gpu():
    """Test basic functionality of the unshear function."""
    slope = 5
    sub_j = 2
    sup_i = 2
    result = opm_unshear.gpu.unshear(sample_data, sub_j=sub_j, sup_i=sup_i, slope=slope)


def test_unshear_cpu():
    """Test basic functionality of the unshear function."""
    slope = 5
    sub_j = 2
    sup_i = 2
    result = opm_unshear.cpu.unshear(sample_data, sub_j=sub_j, sup_i=sup_i, slope=slope)


@pytest.mark.skipif(not gpu_available, reason="No GPU detected.")
def test_cli_basic(tmp_path):
    """Test the CLI with basic arguments."""
    input_file = tmp_path / "input.npy"
    np.save(input_file, sample_data)
    output_file = tmp_path / "output.npy"
    slope = 1.0
    command = ["python", "-m", "opm_unshear", str(input_file), str(output_file), "--slope", str(slope)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, f"CLI failed with error: {result.stderr}"
    assert os.path.exists(output_file), "Output file should be created by the CLI."


def test_get_slope():
    """Test the get_slope function with known parameters."""
    n1 = 1.33  # refractive index at the sample
    n2 = 1.0  # refractive index at the intermediate image plane
    M12 = 1.6  # magnification from Obj1 to Obj2
    M23 = 2.0  # magnification from Obj2 to Obj3
    dv = 5.0  # vertical pixel size (on the camera)
    dp = 2.0  # plane separation along the plane-scanning axis
    theta_sample = np.radians(30)  # sample plane angle in radians
    polarity = 1  # scanning polarity

    # Call the function
    slope, theta_sample_out, theta_iip_out = opm_unshear.get_slope(
        n1=n1, n2=n2, M12=M12, M23=M23, dv=dv, dp=dp, polarity=polarity, theta_sample=theta_sample
    )