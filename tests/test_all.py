# Run: pytest tests/
import math
import os
import subprocess
import sys

import numpy as np
import pytest

import opm_unshear
import opm_unshear.cpu
from opm_unshear import gpu_available

if gpu_available:
    import cupy as cp
    import opm_unshear.gpu

requires_gpu = pytest.mark.skipif(not gpu_available, reason="No GPU detected.")

rng = np.random.default_rng(0)
sample_data = rng.random((20, 30, 40)).astype(np.float32)

# (shape, sub_j, sup_i, slope): even/odd heights, negative slopes, sup_i > |slope|, non-integer slopes
PARAMS = [
    ((16, 30, 3), 2, 2, 5.0),
    ((12, 31, 3), 1, 2, 3.3),
    ((12, 30, 3), 1, 4, -3.0),
    ((12, 29, 3), 1, 4, -2.0),
    ((12, 30, 3), 1, 4, 2.0),
    ((12, 30, 3), 3, 1.5, -6.6),
    ((12, 25, 3), 1, 3, 0.8),
]


def reference_unshear(x, sub_j, sup_i, slope, fill_value=np.nan):
    """Slow, readable reference of the interpolation kernel (float64)."""
    X0, X1, X2 = x.shape
    out = np.full((int(X0 * sup_i), X1 // sub_j, X2), fill_value, dtype=np.float64)
    tri = lambda v, w: max(w - abs(v), 0.0)
    a = abs(slope)
    width_i = max(1 / sup_i, 1 / a)
    center = X1 // 2
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            acc = np.zeros(X2)
            norm = 0.0
            for jj in range(-int(a), int(a) + 1):
                s = j * sub_j + jj
                r_float = i / sup_i + (s - center) / slope
                r = math.floor(r_float)
                ratio = r_float - r
                if 0 <= r < X0 - 1 and 0 <= s < X1:
                    wl = tri(ratio, width_i) * tri(jj + ratio * slope, a)
                    wr = tri(ratio - 1, width_i) * tri(jj + (ratio - 1) * slope, a)
                    acc += x[r, s] * wl + x[r + 1, s] * wr
                    norm += wl + wr
            if norm > 0:
                out[i, j] = acc / norm
    return out


def assert_close_away_from_edges(a, b, x_shape, sub_j, sup_i, slope, rtol=1e-4, atol=1e-5):
    """Compare two outputs (computed with fill_value=nan) away from the first/last planes of the input.

    Within ~1 plane of the data edge, floating point rounding (float32 vs float64, fastmath) decides whether a
    sample pair falls inside the data, so values and filled/empty status may legitimately differ there."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape
    i, j = np.meshgrid(np.arange(a.shape[0]), np.arange(a.shape[1]), indexing="ij")
    r_float = i / sup_i + (j * sub_j - x_shape[1] // 2) / slope
    keep = (r_float > 2.5) & (r_float < x_shape[0] - 3.5)  # kernel spans ~ +-1 plane around r_float
    assert keep.sum() > 10
    np.testing.assert_array_equal(np.isfinite(a[keep]), np.isfinite(b[keep]))
    np.testing.assert_allclose(a[keep], b[keep], rtol=rtol, atol=atol, equal_nan=True)


def interior_mask(out_shape, x_shape, sub_j, sup_i, slope, margin=None):
    """Output voxels whose full kernel support lies inside the input."""
    margin = abs(slope) + 2 if margin is None else margin
    i, j = np.meshgrid(np.arange(out_shape[0]), np.arange(out_shape[1]), indexing="ij")
    s = j * sub_j
    r_float = i / sup_i + (s - x_shape[1] // 2) / slope
    return (r_float >= 2) & (r_float < x_shape[0] - 3) & (s >= margin) & (s < x_shape[1] - margin)


# ---------------------------------------------------------------- import / API


def test_import_without_cupy():
    """The CPU fallback must work when cupy is not installed."""
    code = "import sys; sys.modules['cupy'] = None; import opm_unshear; assert not opm_unshear.gpu_available; print(opm_unshear.unshear.__module__)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "opm_unshear.cpu"


def test_unshear_cpu_basic():
    result = opm_unshear.cpu.unshear(sample_data, sub_j=2, sup_i=2, slope=5)
    assert result.shape == (40, 15, 40)
    assert result.dtype == np.float32


@requires_gpu
def test_unshear_gpu_basic():
    result = opm_unshear.gpu.unshear(sample_data, sub_j=2, sup_i=2, slope=5)
    assert isinstance(result, np.ndarray)
    assert result.shape == (40, 15, 40)
    result_cp = opm_unshear.gpu.unshear(cp.asarray(sample_data), sub_j=2, sup_i=2, slope=5)
    assert isinstance(result_cp, cp.ndarray)


@requires_gpu
def test_gpu_accepts_non_float32_cupy_input():
    """uint16 camera data already on the GPU (CuPy >= 14 raises on cp.array(..., copy=False) if a copy is needed)."""
    x = cp.asarray((sample_data * 1000).astype(np.uint16))
    result = opm_unshear.gpu.unshear(x, sub_j=2, sup_i=2, slope=5)
    assert result.dtype == cp.float32


def _implementations():
    impls = [opm_unshear.cpu.unshear]
    if gpu_available:
        impls.append(opm_unshear.gpu.unshear)
    return impls


@pytest.mark.parametrize("unshear", _implementations())
def test_argument_validation(unshear):
    x = sample_data
    unshear(x, sub_j=np.int64(2), sup_i=2, slope=5)  # numpy integers are fine
    with pytest.raises(TypeError):
        unshear(x, sub_j=2.0, sup_i=2, slope=5)
    with pytest.raises(ValueError):
        unshear(x, sub_j=0, sup_i=2, slope=5)
    with pytest.raises(ValueError):
        unshear(x, sub_j=1, sup_i=0, slope=5)
    with pytest.raises(ValueError):
        unshear(x, sub_j=1, sup_i=2, slope=0)
    with pytest.raises(ValueError):
        unshear(x[0], sub_j=1, sup_i=2, slope=5)
    with pytest.warns(UserWarning):
        unshear(x, sub_j=6, sup_i=2, slope=5)


def test_cpu_out_argument():
    out = np.zeros((40, 15, 40), np.float32)
    result = opm_unshear.cpu.unshear(sample_data, sub_j=2, sup_i=2, slope=5, out=out)
    assert result is out
    with pytest.raises(ValueError):
        opm_unshear.cpu.unshear(sample_data, sub_j=2, sup_i=2, slope=5, out=np.zeros((40, 14, 40), np.float32))


@requires_gpu
def test_gpu_out_argument():
    out = cp.zeros((40, 15, 40), cp.float32)
    result = opm_unshear.gpu.unshear(cp.asarray(sample_data), sub_j=2, sup_i=2, slope=5, out=out)
    assert result is out
    with pytest.raises(ValueError):
        opm_unshear.gpu.unshear(sample_data, sub_j=2, sup_i=2, slope=5, out=cp.zeros((40, 14, 40), cp.float32))
    with pytest.raises(ValueError):
        opm_unshear.gpu.unshear(sample_data, sub_j=2, sup_i=2, slope=5, out=cp.zeros((40, 40, 15), cp.float32).transpose(0, 2, 1))


# ---------------------------------------------------------------- correctness


@pytest.mark.parametrize("shape, sub_j, sup_i, slope", PARAMS)
def test_cpu_matches_reference(shape, sub_j, sup_i, slope):
    x = rng.random(shape).astype(np.float32)
    ref = reference_unshear(x.astype(np.float64), sub_j, sup_i, slope)
    out = opm_unshear.cpu.unshear(x, sub_j=sub_j, sup_i=sup_i, slope=slope, fill_value=np.nan)
    assert_close_away_from_edges(out, ref, shape, sub_j, sup_i, slope)


@requires_gpu
@pytest.mark.parametrize("shape, sub_j, sup_i, slope", PARAMS)
def test_gpu_matches_cpu(shape, sub_j, sup_i, slope):
    x = rng.random(shape).astype(np.float32)
    out_cpu = opm_unshear.cpu.unshear(x, sub_j=sub_j, sup_i=sup_i, slope=slope, fill_value=np.nan)
    out_gpu = opm_unshear.gpu.unshear(x, sub_j=sub_j, sup_i=sup_i, slope=slope, fill_value=np.nan)
    assert_close_away_from_edges(out_gpu, out_cpu, shape, sub_j, sup_i, slope)


@pytest.mark.parametrize("unshear", _implementations())
@pytest.mark.parametrize("X1", [64, 65])
@pytest.mark.parametrize(
    "sub_j, sup_i, slope", [(1, 2, 4.0), (2, 2, 4.0), (1, 3, -3.0), (2, 1, 5.0), (1, 4, 2.0), (2, 2, 3.3), (1, 2, -6.6)]
)
def test_linear_ramp_is_reproduced(unshear, X1, sub_j, sup_i, slope):
    """The unsheared coordinates must be reproduced at the right position.

    The finite kernel (|jj| <= floor(|slope|)) and non-integer slopes shift positions by up to ~0.015 planes along
    axis 0 and ~0.2 camera rows along axis 1. The tolerances (0.03 planes, 0.3 rows) are below e.g. a half-pixel
    centre offset (0.5/|slope| planes)."""
    X0 = 30
    r, s = np.meshgrid(np.arange(X0), np.arange(X1), indexing="ij")
    a = r - (s - X1 // 2) / slope  # unsheared axis-0 coordinate, in input planes
    for field, expected_fcn, atol in [
        (a, lambda i, j: i / sup_i, 0.03),  # axis 0, in planes
        (s.astype(float), lambda i, j: j * sub_j, 0.3),  # axis 1, in camera rows
    ]:
        x = np.repeat(field[:, :, None], 2, axis=2).astype(np.float32)
        out = np.asarray(unshear(x, sub_j=sub_j, sup_i=sup_i, slope=slope, fill_value=np.nan))
        i, j = np.meshgrid(np.arange(out.shape[0]), np.arange(out.shape[1]), indexing="ij")
        m = interior_mask(out.shape, x.shape, sub_j, sup_i, slope)
        assert m.sum() > 20
        np.testing.assert_allclose(out[m][:, 0], expected_fcn(i, j)[m], rtol=0, atol=atol)


@pytest.mark.parametrize("unshear", _implementations())
@pytest.mark.parametrize("sup_i, slope", [(4, 2.0), (4, -2.0), (2, 1.18), (3, -1.5)])
def test_no_holes_when_sup_i_exceeds_slope(unshear, sup_i, slope):
    """Constant input must give a constant output (no fill_value holes) inside the data, also for sup_i > |slope|."""
    x = np.full((30, 120, 2), 100.0, np.float32)
    out = np.asarray(unshear(x, sub_j=1, sup_i=sup_i, slope=slope, fill_value=np.nan))
    m = interior_mask(out.shape, x.shape, 1, sup_i, slope)
    assert m.sum() > 20
    np.testing.assert_allclose(out[m], 100.0, rtol=1e-5)


# ---------------------------------------------------------------- get_slope


def _geometric_slope(n1, n2, M12, M23, dv, dp, theta_iip):
    """Independent geometric construction: follow a line parallel to the optical axis through consecutive planes."""
    u_iip = np.array([np.sin(theta_iip), np.cos(theta_iip)])  # (lateral, axial) direction of the IIP, angle from optical axis
    step_sample = u_iip * dv / M23 / np.array([M12, M12**2 * n2 / n1])  # one camera pixel, mapped back to the sample
    v = lambda r, x0=0.37: (x0 - r * dp) / step_sample[0]  # camera pixel at which plane r crosses the line x = x0
    return abs(v(1) - v(0))


@pytest.mark.parametrize("theta_deg", [20, 30, 45, 60, 70])
@pytest.mark.parametrize("n1, n2, M12", [(1.33, 1.33, 1.0), (1.33, 1.0, 1.33), (1.33, 1.0, 1.6)])
def test_get_slope_matches_geometry(theta_deg, n1, n2, M12):
    theta = np.radians(theta_deg)
    slope, theta_sample, theta_iip = opm_unshear.get_slope(n1=n1, n2=n2, M12=M12, M23=2.0, dv=5.0, dp=2.0, theta_iip=theta)
    assert slope == pytest.approx(_geometric_slope(n1, n2, M12, 2.0, 5.0, 2.0, theta), rel=1e-9)
    # round trip via theta_sample
    slope2, _, theta_iip2 = opm_unshear.get_slope(n1=n1, n2=n2, M12=M12, M23=2.0, dv=5.0, dp=2.0, theta_sample=theta_sample)
    assert slope2 == pytest.approx(slope, rel=1e-9)
    assert theta_iip2 == pytest.approx(theta_iip, rel=1e-9)


def test_get_slope_ideal_remote_focus():
    """With unit angular magnification (M12 = n1/n2) the slope is dp / (pixel size in sample * sin(theta))."""
    n1, n2, M23, dv, dp, theta = 1.33, 1.0, 2.0, 5.0, 2.0, np.radians(33)
    M12 = n1 / n2
    slope, theta_sample, theta_iip = opm_unshear.get_slope(n1, n2, M12, M23, dv, dp, theta_sample=theta)
    assert theta_iip == pytest.approx(theta)
    px_sample = dv / M23 / M12
    assert slope == pytest.approx(dp / (px_sample * np.sin(theta)))


def test_get_slope_polarity_and_errors():
    kw = dict(n1=1.33, n2=1.0, M12=1.6, M23=2.0, dv=5.0, dp=2.0, theta_sample=np.radians(30))
    s_pos, _, _ = opm_unshear.get_slope(polarity=1, **kw)
    s_neg, _, _ = opm_unshear.get_slope(polarity=-1, **kw)
    assert s_pos > 0 and s_neg == -s_pos
    with pytest.raises(ValueError):
        opm_unshear.get_slope(polarity=0, **kw)
    with pytest.raises(ValueError):
        opm_unshear.get_slope(n1=1.33, n2=1.0, M12=1.6, M23=2.0, dv=5.0, dp=2.0)
    with pytest.raises(ValueError):
        opm_unshear.get_slope(theta_iip=0.5, **kw)


# ---------------------------------------------------------------- CLI


def test_split_key_windows_path():
    from opm_unshear.__main__ import _split_key

    assert _split_key(r"C:\data\x.mat:vol") == (r"C:\data\x.mat", "vol")
    assert _split_key("/data/x.h5:/group/key") == ("/data/x.h5", "/group/key")


@pytest.mark.parametrize("fmt", ["npy", "h5", "mat", "tif"])
def test_cli_formats(tmp_path, fmt):
    h5py = pytest.importorskip("h5py")
    data = (sample_data * 1000).astype(np.uint16)
    if fmt == "npy":
        fn = tmp_path / "input.npy"
        np.save(fn, data)
        arg = str(fn)
    elif fmt == "h5":
        fn = tmp_path / "input.h5"
        with h5py.File(fn, "w") as f:
            f["vol"] = data
        arg = f"{fn}:vol"
    elif fmt == "mat":
        scipy_io = pytest.importorskip("scipy.io")
        fn = tmp_path / "input.mat"
        scipy_io.savemat(fn, {"vol": data})
        arg = f"{fn}:vol"
    else:
        tifffile = pytest.importorskip("tifffile")
        fn = tmp_path / "input.tif"
        tifffile.imwrite(fn, data)
        arg = str(fn)
    output_file = tmp_path / "output.h5"
    command = [sys.executable, "-m", "opm_unshear", arg, str(output_file), "--slope", "3", "--sub_j", "2", "--out_dtype", "uint16"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, f"CLI failed with error: {result.stderr}"
    with h5py.File(output_file, "r") as f:
        out = f["unsheared_data"]
        assert out.shape == (40, 15, 40)
        assert out.dtype == np.uint16
