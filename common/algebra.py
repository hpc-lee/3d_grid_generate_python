"""Arc-length stretching functions for 3D grid post-processing.

All stretching computations are delegated to the C backend (libgrid3d.so).
"""

import numpy as np
import ctypes
import os
import sys


def _load_stretch_lib():
    """Load the C library and set up stretch function signatures."""
    lib_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lib_path = os.path.join(lib_dir, "src_c", "libgrid3d.so")
    if not os.path.exists(lib_path):
        print(f"Error: C library not found at {lib_path}")
        print("Please run: cd src_c && bash build.sh")
        sys.exit(1)
    lib = ctypes.CDLL(lib_path)

    float_p = np.ctypeslib.ndpointer(dtype=np.float32)
    c_int = ctypes.c_int

    for fname in ['xi_arc_stretch_c', 'et_arc_stretch_c', 'zt_arc_stretch_c']:
        fn = getattr(lib, fname)
        fn.argtypes = [float_p, float_p, float_p, c_int, c_int, c_int,
                       float_p, c_int]
        fn.restype = None

    return lib


_lib_cache = None


def _get_lib():
    global _lib_cache
    if _lib_cache is None:
        _lib_cache = _load_stretch_lib()
    return _lib_cache


def xi_arc_stretch(gd, arc_len):
    """Arc-length stretching in xi (x) direction."""
    lib = _get_lib()
    n_arc = len(arc_len)
    arc = np.ascontiguousarray(arc_len, dtype=np.float32)
    lib.xi_arc_stretch_c(gd.x3d, gd.y3d, gd.z3d, gd.nx, gd.ny, gd.nz, arc, n_arc)


def et_arc_stretch(gd, arc_len):
    """Arc-length stretching in eta (y) direction."""
    lib = _get_lib()
    n_arc = len(arc_len)
    arc = np.ascontiguousarray(arc_len, dtype=np.float32)
    lib.et_arc_stretch_c(gd.x3d, gd.y3d, gd.z3d, gd.nx, gd.ny, gd.nz, arc, n_arc)


def zt_arc_stretch(gd, arc_len):
    """Arc-length stretching in zeta (z) direction."""
    lib = _get_lib()
    n_arc = len(arc_len)
    arc = np.ascontiguousarray(arc_len, dtype=np.float32)
    lib.zt_arc_stretch_c(gd.x3d, gd.y3d, gd.z3d, gd.nx, gd.ny, gd.nz, arc, n_arc)


def read_stretch_file(filepath):
    """Read arc-length stretch parameters from text file."""
    arc_len = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            arc_len.append(float(line))
    return np.array(arc_len, dtype=np.float32)
