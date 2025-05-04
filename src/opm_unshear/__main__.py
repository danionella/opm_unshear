import argparse
import logging

import cupy as cp
import numpy as np
import h5py
import hdf5plugin
from tqdm.auto import tqdm

from . import unshear


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def import_data(file_path: str):
    """
    Import data from various file types, including .npy, .nii, .h5, .mat, DICOM, and TIFF.

    Args:
        file_path (str): Path to the file. For HDF5 and MATLAB files, you can specify the group/key or variable
                         using the format '/path/to/file.h5:/group/key' or '/path/to/file.mat:variable_name'.

    Returns:
        tuple: (data, metadata)
            - data (np.ndarray): Loaded data as a NumPy array.
            - metadata (dict): Dictionary containing metadata (e.g., scale, orientation, origin).
    """

    if file_path.endswith(".npy"):
        data = np.load(file_path).copy()
        return data, dict(filetype="npy", path=file_path, meta={})

    elif ".h5:" in file_path or ".hdf5:" in file_path:
        split_index = file_path.rfind(":")  # Find the last colon
        file_path, key = file_path[:split_index], file_path[split_index + 1 :]
        with h5py.File(file_path, "r") as f:
            data = np.array(f[key])
            attrs = dict(f[key].attrs)  # Extract attributes as metadata
        return data, dict(filetype="hdf5", path=file_path, key=key, meta=attrs)

    elif file_path.endswith(".h5") or file_path.endswith(".hdf5"):
        raise ValueError(f"HDF5 file path was provided without dataset key (example: /path/to/file.h5:dataset_name)")

    elif ".mat:" in file_path:
        try:
            from scipy.io import loadmat
        except ImportError:
            raise ImportError("The 'scipy' package is required to load MATLAB files. Please install it.")
        file_path, variable_name = file_path.split(":", 1)
        mat_data = loadmat(file_path)
        if variable_name not in mat_data:
            raise ValueError(f"Variable '{variable_name}' not found in MATLAB file '{file_path}'")
        data = mat_data[variable_name]
        return data, dict(filetype="matlab", path=file_path, key=variable_name, meta=mat_data[variable_name].attrs)

    elif file_path.endswith(".tiff") or file_path.endswith(".tif"):
        try:
            import tifffile
        except ImportError:
            raise ImportError("The 'tifffile' package is required to load TIFF files. Please install it.")
        data = tifffile.imread(file_path)
        tiff_meta = tifffile.TiffFile(file_path).pages[0].tags._dict  # Extract TIFF metadata
        return data, dict(filetype="tiff", path=file_path, meta=tiff_meta)

    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def main():
    parser = argparse.ArgumentParser(description="Perform oblique interpolation for OPM volume unshearing/deskewing.")
    parser.add_argument("input_file", type=str, help="Path to the input 3D volume file (*.tif, *.h5:key *.mat:key).")
    parser.add_argument("output_file", type=str, default='unshear_output.h5', help="Path to save the output HDF5 file (.h5).")
    parser.add_argument("--sub_j", type=int, default=1, help="Subsampling factor along axis 1 (default: 1).")
    parser.add_argument("--sup_i", type=int, default=2, help="Upsampling factor along axis 0 (default: 2).")
    parser.add_argument("--slope", type=float, required=True, help="Shear slope in (px along dim 1)/(px along dim 0).")
    parser.add_argument("--out_dtype", type=str, default="float32", help="Output data type (default: float32).")
    args = parser.parse_args()

    logging.info(f"Loading input volume from {args.input_file}...")
    data, metadata = import_data(args.input_file)
    assert data.ndim == 3, f"Input data must be 3D, but got {data.ndim}D data."

    logging.info(f"Unshearing data of shape ({data.shape}) with sub_j={args.sub_j}, sup_i={args.sup_i}, slope={args.slope}...")
    result = unshear(data, sub_j=args.sub_j, sup_i=args.sup_i, slope=args.slope)

    logging.info(f"Saving output data with shape {result.shape} to {args.output_file}...")
    with h5py.File(args.output_file, "w") as f:
        f.create_dataset("unsheared_data", data=result, compression="gzip")
        f.create_dataset("args", data=str(args))

    logging.info("Done!")

if __name__ == "__main__":
    main()
