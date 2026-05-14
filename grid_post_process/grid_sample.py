"""3D grid sampling (tri-linear interpolation) via C backend."""

import ctypes
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_sample_lib():
    """Load libgrid3d.so and set up sampling function signature."""
    lib_path = os.path.join(PROJECT_ROOT, "src_c", "libgrid3d.so")
    if not os.path.exists(lib_path):
        print(f"Error: C library not found at {lib_path}")
        print("Please run: cd src_c && bash build.sh")
        sys.exit(1)

    lib = ctypes.CDLL(lib_path)
    float_p = np.ctypeslib.ndpointer(dtype=np.float32)
    c_int = ctypes.c_int

    lib.sample_interp_c.argtypes = [
        float_p, float_p, float_p,  # x3d, y3d, z3d (old)
        float_p, float_p, float_p,  # x3d_new, y3d_new, z3d_new
        c_int, c_int, c_int,        # nx, ny, nz (old)
        c_int, c_int, c_int,        # nx_new, ny_new, nz_new
    ]
    lib.sample_interp_c.restype = None

    return lib


def grid_sample(gd, sample_factor):
    """Upsample grid using tri-linear interpolation.

    Args:
        gd: SimpleGridData with x3d, y3d, z3d arrays.
        sample_factor: [fx, fy, fz] upsample factors (int, >= 1).

    Returns:
        New SimpleGridData with upsampled grid.
    """
    fx, fy, fz = sample_factor
    if fx < 1 or fy < 1 or fz < 1:
        raise ValueError("sample_factor must be >= 1")

    nx_new = (gd.nx - 1) * fx + 1
    ny_new = (gd.ny - 1) * fy + 1
    nz_new = (gd.nz - 1) * fz + 1

    if nx_new == gd.nx and ny_new == gd.ny and nz_new == gd.nz:
        return gd  # No sampling needed

    from common.grid_data import SimpleGridData
    gd_new = SimpleGridData(nx_new, ny_new, nz_new)

    lib = _load_sample_lib()

    lib.sample_interp_c(
        np.ascontiguousarray(gd.x3d),
        np.ascontiguousarray(gd.y3d),
        np.ascontiguousarray(gd.z3d),
        np.ascontiguousarray(gd_new.x3d),
        np.ascontiguousarray(gd_new.y3d),
        np.ascontiguousarray(gd_new.z3d),
        gd.nx, gd.ny, gd.nz,
        nx_new, ny_new, nz_new,
    )

    return gd_new
