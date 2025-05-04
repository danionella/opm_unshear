import cupy as cp
import numpy as np
import warnings

_unshear_kernel = cp.RawKernel(
    r"""//cpp highlighting

// Compute a triangular shape (used for linear interpolation).
//
// x:           The input value.
// width:       The base width of the triangular weighting function.
//
// Returns:     The computed triangular weight.
//
__device__ float triangle(float x, float width){
    return max(width-abs(x),0.0f);
    // TODO: consider dividing result by width. Would make interpolation a bit easier to understand, but would add an operation and might scale the output of unshear_kernel.
}

// Unshear a 3D input array into the output array using anisotropic linear interpolation.
//
// Iterates over each voxel in y (indexed by i, j, k) and computes its value based on the input array x
// and, using linear interpolation, computes its value from voxels in the skewed input array x (indexed by r, s, k).
//
// Example with sub_j=2, sup_i=2, slope=3:
//
//            input (x)                                 output (y)   
//          o-----o-----o                             o--o--o--o--o
//           \     \     \                            |  |  |  |  |
//            o-----o-----o                           |  |  |  |  |                         
//             \     \     \                          |  |  |  |  |                  
//   axis 1 (s) o-----o-----o       -->    axis 1 (j) o--o--o--o--o               
//               \     \     \                        |  |  |  |  |               
//                o-----o-----o                       |  |  |  |  |               
//                 \     \     \                      |  |  |  |  |
//                  o-----o-----o                     o--o--o--o--o
//                    axis 0 (r)                        axis 0 (i)
//
// y:           Pointer to the output array (float*).
// y_sh:        Pointer to a 3-element integer array containing the shape of y.
// x:           Pointer to the input array (float*).
// x_sh:        Pointer to a 3-element integer array containing the shape of x.
// sup_i:       Upsampling factor along axis 0.
// sub_j:       Subsampling factor along axis 1.
// slope:       Input shear in (px along axis 1) / (px along axis 0).
// fill_value:  Value used to fill voxels without valid contributions.
//
extern "C" __global__ void unshear_kernel(float * y, const int * y_sh, const float * x, const int *x_sh, const int sub_j, const float sup_i, const float slope, const float fill_value) {
    int slope_int, r, s, x_ravel_idx, x_sh12;
    float integrator, norm, r_float, ratio, weight_left, weight_right, width_i, abs_slope;
    abs_slope = fabsf(slope);
    width_i = fmaxf(1.0f/sup_i, 1.0f/slope);
    slope_int = __float2int_rd(abs_slope);
    x_sh12 = x_sh[1]*x_sh[2];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < y_sh[0]; i += blockDim.x * gridDim.x) {
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < y_sh[1]; j += blockDim.y * gridDim.y) {
            for (int k = blockIdx.z * blockDim.z + threadIdx.z; k < y_sh[2]; k += blockDim.z * gridDim.z) {
                integrator = 0.0f;
                norm = 0.0f;
                for (int jj = -slope_int; jj < slope_int+1; jj++){
                    s = j*sub_j+jj;
                    r_float = (float)i/sup_i+(s-x_sh[1]/2)/slope;
                    r = __float2int_rd(r_float);
                    ratio = r_float - r;
                    if ((r<(x_sh[0]-1)) & (r>=0) & (s<x_sh[1]) & (s>=0)){
                        weight_left = triangle(ratio, width_i) * triangle(jj+ratio*slope, abs_slope);
                        weight_right = triangle(ratio-1, width_i) * triangle(jj+(ratio-1)*slope, abs_slope);
                        x_ravel_idx = (long long)r * x_sh12 + s * x_sh[2] + k;
                        integrator += x[x_ravel_idx]*weight_left + x[x_ravel_idx + x_sh12]*weight_right;
                        norm += weight_left+weight_right;
                    }
                }
                if (norm > 0.0f){
                    y[(long long)i * y_sh[1] * y_sh[2] + j * y_sh[2] + k] = integrator/norm;
                } else {
                    y[(long long)i * y_sh[1] * y_sh[2] + j * y_sh[2] + k] = fill_value;
                }    
            }
        }
    }
}
""",
    "unshear_kernel",
)


def unshear(x, sub_j, sup_i, slope, out=None, fill_value=0.0, tpb=[8, 8, 8]):
    r"""Unshear a volume using anisotropic linear interpolation.

    Args:
        x (array): 3D-Volume: planes (axis 0) by cameraY (axis 1) by cameraX (axis 2). Can be a numpy or cupy array.
        sup_i (float): upsampling factor along axis 0. Set to 1 (no upsampling) or larger.
        sub_j (int): subsampling factor along axis 1. Set to 1 (no subsampling) or larger. This should not be larger than the slope to avoid undersampling.
        slope (float): shear slope in (px along dim 1)/(px along dim 0). You can use `get_slope` to calculate this value.
        out (array): 3D-Volume where output is saved. Defaults to None (creates a new array).
        fill_value (float32): empty voxels will be filled with this value. Default: 0.0
        tpb (list): CUDA-Threads per Block. Default: [8, 8, 8]

    Returns
        array: Unsheared array (cupy or numpy array, depending on the input).

    Example with sub_j=2, sup_i=2, slope=3 (showing a slice along axis 2):

               input                                      output
            o-----o-----o                             o--o--o--o--o
             \     \     \                            |  |  |  |  |
              o-----o-----o                           |  |  |  |  |
               \     \     \                          |  |  |  |  |
        axis 1  o-----o-----o       -->       axis 1  o--o--o--o--o
                 \     \     \                        |  |  |  |  |
                  o-----o-----o                       |  |  |  |  |
                   \     \     \                      |  |  |  |  |
                    o-----o-----o                     o--o--o--o--o
                        axis 0                            axis 0
    """
    if not isinstance(sub_j, int):
        raise ValueError(f"sub_j must be an integer, got {sub_j} of type {type(sub_j)}")
    was_numpy = isinstance(x, np.ndarray)
    x = cp.array(x, dtype="float32", copy=False, order="C")
    if out is None:
        out = cp.zeros((int(x.shape[0] * sup_i), x.shape[1] // sub_j, x.shape[2]), dtype=x.dtype)
    assert out.dtype == cp.dtype("float32"), "Output array must be float32"
    if sub_j > np.abs(slope):
        warnings.warn(f"sub_j > abs(slope) leads to subsampling! sub_j: {sub_j}, slope: {slope}")
    bpg = np.ceil(np.r_[out.shape] / tpb).astype("int")  # blocks per grid
    _unshear_kernel(
        tuple(bpg),
        tuple(tpb),
        (
            out,
            cp.array(out.shape).astype("int32"),
            x,
            cp.array(x.shape).astype("int32"),
            cp.int32(sub_j),
            cp.float32(sup_i),
            cp.float32(slope),
            cp.float32(fill_value),
        ),
    )
    if was_numpy:
        out = cp.asnumpy(out)
    return out
