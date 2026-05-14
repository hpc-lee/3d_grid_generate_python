"""Grid quality metrics for 3D curvilinear grids.

All quality computations are delegated to the C backend (libgrid3d.so).
"""

import numpy as np
import ctypes
import os
import sys

from .io_operations import quality_export, quality_export_mpi


def _load_quality_lib():
    """Load the C library and set up quality function signatures."""
    lib_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lib_path = os.path.join(lib_dir, "src_c", "libgrid3d.so")
    if not os.path.exists(lib_path):
        print(f"Error: C library not found at {lib_path}")
        print("Please run: cd src_c && bash build.sh")
        sys.exit(1)
    lib = ctypes.CDLL(lib_path)

    float_p = np.ctypeslib.ndpointer(dtype=np.float32)
    int_p = np.ctypeslib.ndpointer(dtype=np.int32)
    c_int = ctypes.c_int

    # Set up all quality function signatures
    funcs = [
        'cal_orth_xiet_c', 'cal_orth_xizt_c', 'cal_orth_etzt_c',
        'cal_jacobi_c',
        'cal_step_xi_c', 'cal_step_et_c', 'cal_step_zt_c',
        'cal_smooth_xi_c', 'cal_smooth_et_c', 'cal_smooth_zt_c',
        'cal_ratio_c',
    ]
    for fname in funcs:
        fn = getattr(lib, fname)
        fn.argtypes = [float_p, float_p, float_p, float_p, c_int, c_int, c_int]
        fn.restype = None

    # cal_min_dist_c
    lib.cal_min_dist_c.argtypes = [
        float_p, float_p, float_p, c_int, c_int, c_int,
        int_p, int_p, int_p, float_p
    ]
    lib.cal_min_dist_c.restype = None

    return lib


_lib_cache = None


def _get_lib():
    global _lib_cache
    if _lib_cache is None:
        _lib_cache = _load_quality_lib()
    return _lib_cache


def grid_quality_check(gd, cfgs, export_mode='single'):
    """Perform grid quality checks.

    export_mode: 'single' - single partition (parabolic/hyperbolic)
                 'mpi'    - MPI-partitioned (grid_post_process with PML)
    """
    lib = _get_lib()
    x3d = gd.x3d
    y3d = gd.y3d
    z3d = gd.z3d
    ni = gd.ni
    nj = gd.nj
    nk = gd.nk

    def _export(var, quality_name):
        if export_mode == 'mpi':
            quality_export_mpi(var, gd, cfgs, quality_name)
        else:
            quality_export(gd, var, cfgs['grid_export_dir'], quality_name)

    var = np.zeros((nk, nj, ni), dtype=np.float32)

    if cfgs.get('check_orth') == 1:
        var[:] = 0
        lib.cal_orth_xiet_c(x3d, y3d, z3d, var, ni, nj, nk)
        _export(var, "orth_xiet")

        var[:] = 0
        lib.cal_orth_xizt_c(x3d, y3d, z3d, var, ni, nj, nk)
        _export(var, "orth_xizt")

        var[:] = 0
        lib.cal_orth_etzt_c(x3d, y3d, z3d, var, ni, nj, nk)
        _export(var, "orth_etzt")

    if cfgs.get('check_jac') == 1:
        var[:] = 0
        lib.cal_jacobi_c(x3d, y3d, z3d, var, ni, nj, nk)
        _export(var, "jacobi")

    if cfgs.get('check_ratio') == 1:
        var[:] = 0
        lib.cal_ratio_c(x3d, y3d, z3d, var, ni, nj, nk)
        _export(var, "ratio")

    if cfgs.get('check_step_xi') == 1:
        var[:] = 0
        lib.cal_step_xi_c(x3d, y3d, z3d, var, ni, nj, nk)
        _export(var, "step_xi")

    if cfgs.get('check_step_et') == 1:
        var[:] = 0
        lib.cal_step_et_c(x3d, y3d, z3d, var, ni, nj, nk)
        _export(var, "step_et")

    if cfgs.get('check_step_zt') == 1:
        var[:] = 0
        lib.cal_step_zt_c(x3d, y3d, z3d, var, ni, nj, nk)
        _export(var, "step_zt")

    if cfgs.get('check_smooth_xi') == 1:
        var[:] = 0
        lib.cal_smooth_xi_c(x3d, y3d, z3d, var, ni, nj, nk)
        _export(var, "smooth_xi")

    if cfgs.get('check_smooth_et') == 1:
        var[:] = 0
        lib.cal_smooth_et_c(x3d, y3d, z3d, var, ni, nj, nk)
        _export(var, "smooth_et")

    if cfgs.get('check_smooth_zt') == 1:
        var[:] = 0
        lib.cal_smooth_zt_c(x3d, y3d, z3d, var, ni, nj, nk)
        _export(var, "smooth_zt")


def cal_min_dist(gd):
    """Calculate minimum distance from any grid point to 8 surrounding planes.
    Returns: (indx_i, indx_j, indx_k, dL_min)
    """
    lib = _get_lib()
    x3d = gd.x3d
    y3d = gd.y3d
    z3d = gd.z3d
    ni = gd.ni
    nj = gd.nj
    nk = gd.nk

    indx_i = np.zeros(1, dtype=np.int32)
    indx_j = np.zeros(1, dtype=np.int32)
    indx_k = np.zeros(1, dtype=np.int32)
    dL_min = np.zeros(1, dtype=np.float32)

    lib.cal_min_dist_c(x3d, y3d, z3d, ni, nj, nk,
                       indx_i, indx_j, indx_k, dL_min)

    return int(indx_i[0]), int(indx_j[0]), int(indx_k[0]), float(dL_min[0])
