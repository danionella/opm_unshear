![Python Version](https://img.shields.io/badge/python-3.9+-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![tests](https://github.com/danionella/opm_unshear/actions/workflows/test.yml/badge.svg)
[![PyPI - Version](https://img.shields.io/pypi/v/opm_unshear)](https://pypi.org/project/opm_unshear/)
![GitHub last commit](https://img.shields.io/github/last-commit/danionella/opm_unshear)

# Oblique Interpolation for Oblique Plane Microscopy
A GPU-accelerated oblique interpolation library for oblique plane microscopy (OPM) volume reconstruction. Avoids aliasing and interpolation artifacts caused by rectilinear interpolation approaches.

Links: [API documentation](http://danionella.github.io/opm_unshear), [GitHub repository](https://github.com/danionella/opm_unshear)

### Background

Oblique Plane Microscopy (OPM) acquires 3D volumes by scanning an inclined light sheet through the sample. Because the imaging plane is tilted relative to the camera axes, standard rectilinear interpolation during volume reconstruction can introduce aliasing artifacts and loss of resolution – problems that are ameliorated by oversampling, Fourier stitching or deconvolution (see [McFadden et al. 2025](https://doi.org/10.1364/BOE.555473) and [Lamb et al. 2025](https://www.biorxiv.org/content/10.1101/2025.04.30.651458)).

`opm_unshear` performs true oblique interpolation in the sample’s native coordinate frame, mapping voxels along the tilted plane directly onto an isotropic grid (see [Hoffmann et al. 2023](https://www.nature.com/articles/s41467-023-43741-x#Sec13), Equation 2). We achieve both artifact-free reconstructions and GPU-accelerated throughput.

### Features

- Oblique interpolation in oblique space, avoiding aliasing from misaligned sampling axes  
- CUDA-accelerated compute kernel for real-time GPU processing (`opm_unshear.gpu.unshear`, ~1 gigavoxel/s on a NVIDIA RTX 3090)
- Numba-accelerated multi-threaded CPU fallback for non-GPU systems (`opm_unshear.cpu.unshear`, ~0.1 gigavoxel/s on a 16-core CPU)

## Installation
### Hardware Requirements
- Linux or Windows PC
- For GPU acceleration (recommended): CUDA-capable NVIDIA GPU with sufficient memory (≥ 10 bytes per voxel of the volumes to be processed) and [drivers](https://www.nvidia.com/en-us/drivers/) supporting CUDA version ≥ 11.2 (check with `nvidia-smi`)

### Install `opm_unshear`

1. Install dependencies using [conda/mamba](https://github.com/conda-forge/miniforge) (recommended, installs CuPy and the CUDA libraries for GPU support):
- clone or download [environment.yml](https://raw.githubusercontent.com/danionella/opm_unshear/refs/heads/main/environment.yml) and type `conda env create -n opm_unshear_env -f environment.yml` (or if you already have a conda environment: `conda env update -f environment.yml`)

2. Install opm_unshear:
- `conda activate opm_unshear_env` (or the name of your conda environment)
- `pip install "opm_unshear[cli]"` (or, for development, clone this repository, change to the directory containing `pyproject.toml` and run: `pip install -e ".[test]"`)

Without conda, `pip install opm_unshear` installs the CPU version (NumPy + Numba). For GPU support without conda, use `pip install "opm_unshear[gpu]"` (installs `cupy-cuda12x`; do not combine with a conda-installed CuPy). The extra `[cli]` adds the packages needed by the command line interface (h5py, hdf5plugin, tifffile, scipy).

`opm_unshear.unshear` uses the GPU implementation if an NVIDIA driver, CuPy and a GPU are found, and the CPU implementation otherwise (`opm_unshear.gpu_available` tells which). Set the environment variable `OPM_UNSHEAR_FORCE_CPU=1` to always use the CPU.

## Use

### Geometry and parameters

The oblique interpolation function requires the following parameters:
- `data`: the input data, a 3D array of shape `(p, v, h)`, where `p` is the number of oblique planes (camera frames), `v` is the number of vertical camera pixels, and `h` is the number of horizontal camera pixels. (note that we are using numpy-standard C-order indexing, where the axes are ordered from slowest-varying to fastest-varying)
- `slope(float)`: Imagining an ideal pencil parallel to the optical axis, this is the shift of the pencil image, in `v` camera pixels, between consecutive camera frames. It can be estimated in practice by imaging any structured sample and inspecting the apparent drift between camera frames. This parameter can be positive or negative, depending on the direction of plane scanning or oblique plane tilt. You can also use `opm_unshear.get_slope` to calculate this value based on setup parameters (angles are measured between the oblique plane and the optical axis). For a sheet at angle θ to the optical axis, a scan step `dp` and a camera pixel size `p` (in the sample, along the sheet), `slope = dp / (p · sin θ)`.
- `sub_j(int)`: subsampling factor along the vertical direction (axis 1, along the optical axis) of the output dataset. Values should be between 1 (no subsampling)  and `abs(slope)`.
- `sup_i(float)`: supersampling factor along the plane-scanning direction (axis 0 of the output). Values should be between 1 and `abs(slope)` (higher values are possible but this just wastes memory).
- `fill_value(float, optional)`: value of output voxels that receive no input data (default 0). Note that the output covers the input's plane range at mid-height (axis 1), so two corners of the output volume are always empty.

Example geometry with slope=3, sub_j=2, sup_i=2 (showing a slice along axis 2, or "h"):
```
               input                                      output
            o-----o-----o                             o--o--o--o--o
             \     \     \                            |  |  |  |  |
              o-----o-----o                           |  |  |  |  |
               \     \     \                          |  |  |  |  |
        axis 1  o-----o-----o       -->       axis 1  o--o--o--o--o
         ("v")   \     \     \                        |  |  |  |  |
                  o-----o-----o                       |  |  |  |  |
                   \     \     \                      |  |  |  |  |
                    o-----o-----o                     o--o--o--o--o
                        axis 0 ("p")                      axis 0
```

### Python API
```python
import numpy as np
from opm_unshear import unshear, get_slope

slope, _, _ = get_slope(n1=1.33, n2=1, M12=1.6, M23=2, dv=5, dp=2, theta_sample=np.radians(30), polarity=1)
#polarity = 1 or -1, depending on the direction of plane scanning

data = np.random.rand(20, 30, 40).astype(np.float32)
result = unshear(data, sub_j=2, sup_i=2, slope=slope)
```
### Command line interface (CLI)
```bash
python -m opm_unshear data.h5:dataset_name result.h5 --sub_j 2 --sup_i 2 --slope 5
```

- first argument: path to the input file containing a 3D dataset to be deskewed (.npy, .tif/.tiff, .h5/.hdf5, .mat). For hierarchical file formats (.h5 or .mat), specify the dataset name using the format `filename.h5:dataset_name`. Note that the dataset's axes have to be ordered as described above.
- second argument (optional): path to the output file (.h5, default: `unshear_output.h5`). The result is stored in the dataset `unsheared_data`.
- `--slope` (required), `--sub_j`, `--sup_i`, `--fill_value`: as described above
- `--out_dtype`: output data type (default: float32; integer types are rounded and clipped)
- the GPU implementation is used if available, otherwise the CPU implementation

## Feedback and contributions

We value your feedback! If you encounter  issues, have suggestions for improvements, or simply want to let us know that `opm_unshear` worked well for you, please reach out:

- **Bug Reports**: Open an issue on our [GitHub Issues page](https://github.com/danionella/opm_unshear/issues).
- **Feature Requests**: If you have ideas for new features, feel free to suggest them via a GitHub issue.

We welcome contributions! Please:
- Follow the existing coding style and include tests for new features or fixes.
- Fork the repo, create a branch, and submit a pull request (PR) with a clear description.


## Citing our work
If you use `opm_unshear` in your research, please cite the paper that first described our interpolation approach:

Hoffmann, M., Henninger, J. et al. Blazed oblique plane microscopy reveals scale-invariant inference of brain-wide population activity. Nature Communications 14, 8019 (2023). [https://doi.org/10.1038/s41467-023-43741-x](https://doi.org/10.1038/s41467-023-43741-x)

```bibtex
@article{Hoffmann2023,
  title={Blazed oblique plane microscopy reveals scale-invariant inference of brain-wide population activity},
  author={Hoffmann, Maximilian and Henninger, Jorg and Veith, Johannes and Richter, Lars and Judkewitz, Benjamin},
  journal={Nature Communications},
  volume={14},
  number={1},
  pages={8019},
  year={2023},
  publisher={Nature Publishing Group},
  doi={10.1038/s41467-023-43741-x}
}
```

## See also
- [dexp](https://github.com/royerlab/dexp)
- [PetaKit5D](https://github.com/abcucberkeley/PetaKit5D)
- [LiveDeskew](https://github.com/Jrl-98/Live-Deskewing)
