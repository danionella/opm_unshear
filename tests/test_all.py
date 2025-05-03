# Run: pytest tests/test_all.py
# All functions to be tested should start with test_ prefix

import warnings
import os
import subprocess
import pytest

import numpy as np

from opm_unshear import unshear

try:
    import cupy as cp

    _ = cp.cuda.runtime.getDeviceCount()  # Check if any GPU devices are available
    gpu_available = True
except (ImportError, cp.cuda.runtime.CUDARuntimeError):
    gpu_available = False
    warnings.warn("No GPU detected. Skipping GPU tests.")


def test_trivial():
    assert True == True


def test_trivial2():
    assert False == False


@pytest.mark.skipif(not gpu_available, reason="No GPU detected.")
def test_unshear_basic():
    """Test basic functionality of the unshear function."""
    sample_data = cp.random.rand(20, 30, 40, dtype="float32")
    slope = 2.5
    sub_j = 2
    sup_i = 2
    result = unshear(sample_data, sub_j=sub_j, sup_i=sup_i, slope=slope)


@pytest.mark.skipif(not gpu_available, reason="No GPU detected.")
def test_cli_basic(tmp_path):
    """Test the CLI with basic arguments."""
    data = np.random.rand(20, 30, 40).astype(np.float32)
    input_file = tmp_path / "input.npy"
    np.save(input_file, data)
    output_file = tmp_path / "output.npy"
    slope = 1.0
    command = ["python", "-m", "opm_unshear", str(input_file), str(output_file), "--slope", str(slope)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, f"CLI failed with error: {result.stderr}"
    assert os.path.exists(output_file), "Output file should be created by the CLI."
