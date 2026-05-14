"""Dirichlet method for 3D elliptic grid generation.

SOR iteration with Dirichlet boundary conditions and source term control.
Matches C diri_gene in elliptic.c.
"""

import ctypes
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _make_contiguous(arr):
    """Ensure array is C-contiguous float32, in-place if possible."""
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def diri_gene(lib, gdinfo, x3d, y3d, z3d, bdry, mympi, cfgs):
    """Run Dirichlet elliptic grid generation.

    Matches C diri_gene (elliptic.c lines 80-263):
      Before loop: ghost_cal → set_src_diri → interp_source → copy x3d to x3d_tmp
      Loop: update_SOR → exchange_ghost(x3d_tmp) → residual → allreduce → swap
            → set_src_diri → interp_source → check convergence
    """
    nx = gdinfo['nx']
    ny = gdinfo['ny']
    nz = gdinfo['nz']
    ni = gdinfo['ni']
    nj = gdinfo['nj']
    nk = gdinfo['nk']
    gni1 = gdinfo['gni1']
    gnj1 = gdinfo['gnj1']
    gnk1 = gdinfo['gnk1']
    total_nx = gdinfo['total_nx']
    total_ny = gdinfo['total_ny']
    total_nz = gdinfo['total_nz']

    coef = cfgs.get('coef', [1.0]*6)
    if isinstance(coef, list):
        coef_arr = np.array(coef[:6], dtype=np.float32)
    else:
        coef_arr = np.ones(6, dtype=np.float32) * coef
    n_coef = len(coef_arr)
    iter_err = float(cfgs.get('iter_err', 1e-4))
    max_iter = int(cfgs.get('max_iter', 1000))
    w = 1.0  # Gauss-Seidel (SOR with w=1)
    verbose = cfgs.get('_verbose', 0)

    # Allocate coordinate arrays (contiguous, reused across iterations)
    x3d_c = _make_contiguous(x3d.copy())
    y3d_c = _make_contiguous(y3d.copy())
    z3d_c = _make_contiguous(z3d.copy())
    x3d_tmp_c = np.zeros((nz, ny, nx), dtype=np.float32)
    y3d_tmp_c = np.zeros((nz, ny, nx), dtype=np.float32)
    z3d_tmp_c = np.zeros((nz, ny, nx), dtype=np.float32)

    # Source term arrays (3D, per partition)
    P = np.zeros((nz, ny, nx), dtype=np.float32)
    Q = np.zeros((nz, ny, nx), dtype=np.float32)
    R = np.zeros((nz, ny, nx), dtype=np.float32)

    # Boundary source term sizes (interior only, matching C init_src)
    siz_bx = (total_ny - 2) * (total_nz - 2)
    siz_by = (total_nx - 2) * (total_nz - 2)
    siz_bz = (total_nx - 2) * (total_ny - 2)

    # Global boundary source terms (1D arrays, contiguous)
    src_global = {}
    for face, siz in [('x1', siz_bx), ('x2', siz_bx),
                       ('y1', siz_by), ('y2', siz_by),
                       ('z1', siz_bz), ('z2', siz_bz)]:
        src_global[face] = {
            'P': np.zeros(siz, dtype=np.float32),
            'Q': np.zeros(siz, dtype=np.float32),
            'R': np.zeros(siz, dtype=np.float32),
        }

    # Local boundary source terms (1D, matching C init_src)
    src_local = {}
    for face, siz in [('x1', siz_bx), ('x2', siz_bx),
                       ('y1', siz_by), ('y2', siz_by),
                       ('z1', siz_bz), ('z2', siz_bz)]:
        src_local[face] = {
            'P': np.zeros(siz, dtype=np.float32),
            'Q': np.zeros(siz, dtype=np.float32),
            'R': np.zeros(siz, dtype=np.float32),
        }

    # Ghost calculation arrays (1D, double-buffered, matching C ghost_cal)
    # C allocates: p_x = calloc(nz*ny*3*2), g11_x = calloc(nz*ny*2), etc.
    p_x = np.zeros(nz * ny * 3 * 2, dtype=np.float32)
    p_y = np.zeros(nz * nx * 3 * 2, dtype=np.float32)
    p_z = np.zeros(ny * nx * 3 * 2, dtype=np.float32)
    g11_x = np.zeros(nz * ny * 2, dtype=np.float32)
    g22_y = np.zeros(nz * nx * 2, dtype=np.float32)
    g33_z = np.zeros(ny * nx * 2, dtype=np.float32)

    neighid_c = np.ascontiguousarray(mympi.neighid)

    # Ghost calculation — called ONCE before loop (matching C line 125)
    lib.ghost_cal_c(x3d_c, y3d_c, z3d_c, nx, ny, nz,
                    p_x, p_y, p_z, g11_x, g22_y, g33_z,
                    neighid_c)

    # Exchange ghost cells
    mympi.exchange_ghost(gdinfo, x3d_c, y3d_c, z3d_c)

    # Set boundary source terms
    _set_src_diri_c(lib, gdinfo, x3d_c, y3d_c, z3d_c,
                    P, Q, R, src_global, src_local,
                    p_x, p_y, p_z, g11_x, g22_y, g33_z,
                    neighid_c, mympi)

    # Interpolate source terms to interior
    _interp_source(lib, gdinfo, P, Q, R, src_global, coef_arr, n_coef)

    # Copy current coordinates to tmp (so boundary values are preserved in x3d_tmp_c)
    # Matching C lines 131-141
    x3d_tmp_c[:] = x3d_c
    y3d_tmp_c[:] = y3d_c
    z3d_tmp_c[:] = z3d_c

    # Main iteration loop
    for it in range(max_iter):
        # SOR update: x3d_tmp = SOR(x3d)
        lib.update_SOR_c(x3d_c, y3d_c, z3d_c,
                         x3d_tmp_c, y3d_tmp_c, z3d_tmp_c,
                         nx, ny, nz,
                         P, Q, R,
                         ctypes.c_float(w))

        # Exchange ghost for x3d_tmp
        mympi.exchange_ghost(gdinfo, x3d_tmp_c, y3d_tmp_c, z3d_tmp_c)

        # Compute residual
        local_max = np.zeros(3, dtype=np.float32)
        lib.compute_residual_c(x3d_c, y3d_c, z3d_c,
                               x3d_tmp_c, y3d_tmp_c, z3d_tmp_c,
                               nx, ny, nz,
                               local_max)

        # Global max residual
        global_max = mympi.allreduce_max(local_max)

        # Swap x3d and x3d_tmp (pointer swap)
        x3d_c, x3d_tmp_c = x3d_tmp_c, x3d_c
        y3d_c, y3d_tmp_c = y3d_tmp_c, y3d_c
        z3d_c, z3d_tmp_c = z3d_tmp_c, z3d_c

        # Recompute source terms (after swap, x3d_c has the new values)
        # Matching C lines 233-234
        P[:] = 0; Q[:] = 0; R[:] = 0
        _set_src_diri_c(lib, gdinfo, x3d_c, y3d_c, z3d_c,
                        P, Q, R, src_global, src_local,
                        p_x, p_y, p_z, g11_x, g22_y, g33_z,
                        neighid_c, mympi)
        _interp_source(lib, gdinfo, P, Q, R, src_global, coef_arr, n_coef)

        if mympi.is_root() and verbose > 0:
            print(f"iter {it+1}: res_i={global_max[0]:.6e} "
                  f"res_j={global_max[1]:.6e} res_k={global_max[2]:.6e}")

        # Check convergence
        if (global_max[0] < iter_err and
            global_max[1] < iter_err and
            global_max[2] < iter_err):
            if mympi.is_root():
                print(f"Dirichlet method converged at iteration {it+1}")
            break
    else:
        if mympi.is_root():
            print(f"Dirichlet method reached max iterations ({max_iter})")

    # Copy back from whichever pointer holds the final result
    x3d[:] = x3d_c
    y3d[:] = y3d_c
    z3d[:] = z3d_c


def _set_src_diri_c(lib, gdinfo, x3d_c, y3d_c, z3d_c,
                    P, Q, R, src_global, src_local,
                    p_x, p_y, p_z, g11_x, g22_y, g33_z,
                    neighid_c, mympi):
    """Call set_src_diri_c to compute boundary source terms."""
    nx = gdinfo['nx']
    ny = gdinfo['ny']
    nz = gdinfo['nz']
    ni = gdinfo['ni']
    nj = gdinfo['nj']
    nk = gdinfo['nk']
    gni1 = gdinfo['gni1']
    gnj1 = gdinfo['gnj1']
    gnk1 = gdinfo['gnk1']
    total_nx = gdinfo['total_nx']
    total_ny = gdinfo['total_ny']
    total_nz = gdinfo['total_nz']

    lib.set_src_diri_c(
        x3d_c, y3d_c, z3d_c,
        nx, ny, nz, ni, nj, nk,
        P, Q, R,
        # Global source terms for 6 faces
        src_global['x1']['P'], src_global['x1']['Q'], src_global['x1']['R'],
        src_global['x2']['P'], src_global['x2']['Q'], src_global['x2']['R'],
        src_global['y1']['P'], src_global['y1']['Q'], src_global['y1']['R'],
        src_global['y2']['P'], src_global['y2']['Q'], src_global['y2']['R'],
        src_global['z1']['P'], src_global['z1']['Q'], src_global['z1']['R'],
        src_global['z2']['P'], src_global['z2']['Q'], src_global['z2']['R'],
        # Local source terms for 6 faces
        src_local['x1']['P'], src_local['x1']['Q'], src_local['x1']['R'],
        src_local['x2']['P'], src_local['x2']['Q'], src_local['x2']['R'],
        src_local['y1']['P'], src_local['y1']['Q'], src_local['y1']['R'],
        src_local['y2']['P'], src_local['y2']['Q'], src_local['y2']['R'],
        src_local['z1']['P'], src_local['z1']['Q'], src_local['z1']['R'],
        src_local['z2']['P'], src_local['z2']['Q'], src_local['z2']['R'],
        # Ghost arrays
        p_x, p_y, p_z, g11_x, g22_y, g33_z,
        neighid_c,
        gni1, gnj1, gnk1,
        total_nx, total_ny, total_nz)


def _interp_source(lib, gdinfo, P, Q, R, src_global, coef_arr, n_coef):
    """Call interp_inner_source_c to interpolate source terms from boundaries to interior."""
    nx = gdinfo['nx']
    ny = gdinfo['ny']
    nz = gdinfo['nz']
    gni1 = gdinfo['gni1']
    gnj1 = gdinfo['gnj1']
    gnk1 = gdinfo['gnk1']
    total_nx = gdinfo['total_nx']
    total_ny = gdinfo['total_ny']
    total_nz = gdinfo['total_nz']

    lib.interp_inner_source_c(
        P, Q, R,
        nx, ny, nz,
        src_global['x1']['P'], src_global['x1']['Q'], src_global['x1']['R'],
        src_global['x2']['P'], src_global['x2']['Q'], src_global['x2']['R'],
        src_global['y1']['P'], src_global['y1']['Q'], src_global['y1']['R'],
        src_global['y2']['P'], src_global['y2']['Q'], src_global['y2']['R'],
        src_global['z1']['P'], src_global['z1']['Q'], src_global['z1']['R'],
        src_global['z2']['P'], src_global['z2']['Q'], src_global['z2']['R'],
        gni1, gnj1, gnk1,
        total_nx, total_ny, total_nz,
        coef_arr, n_coef)
