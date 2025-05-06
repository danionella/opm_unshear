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
- For GPU acceleration (recommended): CUDA-capable NVIDIA GPU with sufficient memory (≥ 10 bytes per voxel of the volumes to be processed)

### Install `opm_unshear`

1. Install CUDA Toolkit
- Using [conda/mamba](https://github.com/conda-forge/miniforge) (recommended): `conda create -n opm_unshear_env -f environment.yml` (or if you already have a conda environment: `conda env update -f environment.yml`)
- Using [NVIDIA's installer](https://developer.nvidia.com/cuda-downloads) (not recommended)

2. Install opm_unshear:
- `conda activate opm_unshear_env` (or the name of your conda environment)
- `pip install opm_unshear` (or, for development, clone this repository, change to the directory containing `pyproject.toml` and run: `pip install -e .`)

## Use

### Geometry and parameters

The oblique interpolation function requires the following parameters:
- `data`: the input data, a 3D array of shape (p, v, h), where p is the number of oblique planes (camera frames), v is the number of vertical camera pixels, and h is the number of horizontal camera pixels.
- `slope(float)`: Imagining an ideal pencil parallel to the optical axis, this is the shift of the pencil image, in `v` camera pixels, between consecutive camera frames. This parameter can be positive or negative, depending on the direction of plane scanning or oblique plane tilt. You can also use `opm_unshear.get_slope` to calculate this value.
- `sub_j(int)`: subsampling factor along the vertical direction of the output dataset (along the optical axis). Values should be between 1 (no subsampling)  and `abs(slope)`. Larger values reducethe memory footprint of the output dataset (and are often justified in OPM, depending on axial resolution).
- `sup_i(float)`: supersampling factor along the plane-scanning direction. Values should be between 1 and `abs(slope)` (higher values are possible but this just wastes memory). Optimal values depend on the plane spacing, optical resolution and desired memory footprint.

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
python -m opm_unshear --input data.tif --output result.h5 --sub_j 2 --sup_i 2 --slope 5
```

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
- [DaXi](https://github.com/royerlab/daxi)
- [PetaKit5D](https://github.com/abcucberkeley/PetaKit5D)
- [LiveDeskew](https://github.com/Jrl-98/Live-Deskewing)
