import argparse
import logging

import numpy as np

from . import gpu_available, unshear

logger = logging.getLogger("opm_unshear")


def _split_key(file_path: str):
    """Split '/path/to/file.ext:key' at the last colon (safe for Windows drive letters like 'C:\\...')."""
    split_index = file_path.rfind(":")
    return file_path[:split_index], file_path[split_index + 1 :]


def import_data(file_path: str):
    """
    Import a 3D volume from .npy, HDF5 (.h5/.hdf5), MATLAB (.mat) or TIFF (.tif/.tiff) files.

    Args:
        file_path (str): Path to the file. For HDF5 and MATLAB files, specify the dataset / variable
                         using the format '/path/to/file.h5:/group/key' or '/path/to/file.mat:variable_name'.

    Returns:
        tuple: (data, metadata)
            - data (np.ndarray): Loaded data as a NumPy array.
            - metadata (dict): file type, path, key and (where available) dataset attributes.
    """
    lower = file_path.lower()

    if lower.endswith(".npy"):
        data = np.load(file_path)
        return data, dict(filetype="npy", path=file_path, meta={})

    elif ".h5:" in lower or ".hdf5:" in lower:
        import h5py
        import hdf5plugin  # noqa: F401  (registers compression filters)

        file_path, key = _split_key(file_path)
        with h5py.File(file_path, "r") as f:
            data = np.array(f[key])
            attrs = dict(f[key].attrs)
        return data, dict(filetype="hdf5", path=file_path, key=key, meta=attrs)

    elif lower.endswith(".h5") or lower.endswith(".hdf5"):
        raise ValueError("HDF5 file path was provided without dataset key (example: /path/to/file.h5:dataset_name)")

    elif ".mat:" in lower:
        try:
            from scipy.io import loadmat
        except ImportError:
            raise ImportError("The 'scipy' package is required to load MATLAB files. Please install it.")
        file_path, variable_name = _split_key(file_path)
        mat_data = loadmat(file_path)
        if variable_name not in mat_data:
            raise ValueError(f"Variable '{variable_name}' not found in MATLAB file '{file_path}'")
        data = mat_data[variable_name]
        return data, dict(filetype="matlab", path=file_path, key=variable_name, meta={})

    elif lower.endswith(".mat"):
        raise ValueError("MATLAB file path was provided without variable name (example: /path/to/file.mat:variable_name)")

    elif lower.endswith(".tiff") or lower.endswith(".tif"):
        try:
            import tifffile
        except ImportError:
            raise ImportError("The 'tifffile' package is required to load TIFF files. Please install it.")
        data = tifffile.imread(file_path)
        return data, dict(filetype="tiff", path=file_path, meta={})

    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Perform oblique interpolation for OPM volume unshearing/deskewing.")
    parser.add_argument("input_file", type=str, help="Path to the input 3D volume file (*.npy, *.tif, *.h5:key, *.mat:key).")
    parser.add_argument(
        "output_file", type=str, nargs="?", default="unshear_output.h5", help="Path to the output HDF5 file (default: unshear_output.h5)."
    )
    parser.add_argument("--sub_j", type=int, default=1, help="Subsampling factor along axis 1 (default: 1).")
    parser.add_argument("--sup_i", type=float, default=2, help="Upsampling factor along axis 0 (default: 2).")
    parser.add_argument("--slope", type=float, required=True, help="Shear slope in (px along dim 1)/(px along dim 0).")
    parser.add_argument("--fill_value", type=float, default=0.0, help="Value for output voxels without input data (default: 0).")
    parser.add_argument("--out_dtype", type=str, default="float32", help="Output data type (default: float32).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    try:
        import h5py
    except ImportError:
        raise ImportError("The 'h5py' package is required to write the output. Install with: pip install opm_unshear[cli]")

    out_dtype = np.dtype(args.out_dtype)

    logger.info(f"Loading input volume from {args.input_file}...")
    data, metadata = import_data(args.input_file)
    if data.ndim != 3:
        raise ValueError(f"Input data must be 3D, but got {data.ndim}D data.")

    logger.info(f"Using {'GPU' if gpu_available else 'CPU'} implementation.")
    logger.info(f"Unshearing data of shape {data.shape} with sub_j={args.sub_j}, sup_i={args.sup_i}, slope={args.slope}...")
    result = unshear(data, sub_j=args.sub_j, sup_i=args.sup_i, slope=args.slope, fill_value=args.fill_value)

    if np.issubdtype(out_dtype, np.integer):
        info = np.iinfo(out_dtype)
        result = np.clip(np.rint(result), info.min, info.max)
    result = result.astype(out_dtype, copy=False)

    logger.info(f"Saving output data with shape {result.shape} and dtype {result.dtype} to {args.output_file}...")
    with h5py.File(args.output_file, "w") as f:
        f.create_dataset("unsheared_data", data=result, compression="gzip")
        f.create_dataset("args", data=str(vars(args)))

    logger.info("Done!")


if __name__ == "__main__":
    main()
