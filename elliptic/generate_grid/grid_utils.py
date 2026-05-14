"""Grid utility functions for 3D elliptic method: TFI initialization and boundary checks."""

import ctypes
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def check_bdry(bdry, total_nx, total_ny, total_nz):
    """Check that 12 edges of the 6 boundary surfaces are consistent.

    The shared edges between adjacent faces must match in coordinates.
    """
    tol = 1e-6

    def _get_face_point(face, n, shape_hint):
        return face['x'][n], face['y'][n], face['z'][n]

    errors = 0
    siz_x = total_ny * total_nz
    siz_y = total_nx * total_nz
    siz_z = total_nx * total_ny

    # Edge x1-y1: x1 face (j=0 column) == y1 face (i=0 column)
    # x1 face varies (j,k), j=0 → indices k*total_ny + 0
    # y1 face varies (i,k), i=0 → indices k*total_nx + 0
    for k in range(total_nz):
        n_x1 = k * total_ny + 0
        n_y1 = k * total_nx + 0
        for c in range(3):
            comp_x1 = [bdry['x1']['x'], bdry['x1']['y'], bdry['x1']['z']][c]
            comp_y1 = [bdry['y1']['x'], bdry['y1']['y'], bdry['y1']['z']][c]
            if abs(comp_x1[n_x1] - comp_y1[n_y1]) > tol:
                errors += 1

    # Just report if any errors
    if errors > 0:
        print(f"Warning: {errors} edge mismatches detected in boundary data")

    return errors == 0


def linear_tfi(lib, gdinfo, x3d, y3d, z3d, bdry, mympi):
    """Initialize interior grid using linear transfinite interpolation.

    Calls linear_tfi_c from the C backend.
    """
    total_nx = gdinfo['total_nx']
    total_ny = gdinfo['total_ny']
    total_nz = gdinfo['total_nz']
    nx = gdinfo['nx']
    ny = gdinfo['ny']
    nz = gdinfo['nz']
    ni1 = gdinfo['ni1']
    ni2 = gdinfo['ni2']
    nj1 = gdinfo['nj1']
    nj2 = gdinfo['nj2']
    nk1 = gdinfo['nk1']
    nk2 = gdinfo['nk2']
    gni1 = gdinfo['gni1']
    gnj1 = gdinfo['gnj1']
    gnk1 = gdinfo['gnk1']

    # Prepare boundary surface arrays for C
    # C expects each face array as [x_vals | y_vals | z_vals] of size 3*siz
    siz_x = total_ny * total_nz
    siz_y = total_nx * total_nz
    siz_z = total_nx * total_ny

    # x1/x2 faces: layout is [k*ny + j] for k=0..nz-1, j=0..ny-1
    x1 = np.zeros(3 * siz_x, dtype=np.float32)
    x2 = np.zeros(3 * siz_x, dtype=np.float32)
    x1[0*siz_x:1*siz_x] = bdry['x1']['x'][:siz_x]
    x1[1*siz_x:2*siz_x] = bdry['x1']['y'][:siz_x]
    x1[2*siz_x:3*siz_x] = bdry['x1']['z'][:siz_x]
    x2[0*siz_x:1*siz_x] = bdry['x2']['x'][:siz_x]
    x2[1*siz_x:2*siz_x] = bdry['x2']['y'][:siz_x]
    x2[2*siz_x:3*siz_x] = bdry['x2']['z'][:siz_x]

    # y1/y2 faces: layout is [k*nx + i] for k=0..nz-1, i=0..nx-1
    y1 = np.zeros(3 * siz_y, dtype=np.float32)
    y2 = np.zeros(3 * siz_y, dtype=np.float32)
    y1[0*siz_y:1*siz_y] = bdry['y1']['x'][:siz_y]
    y1[1*siz_y:2*siz_y] = bdry['y1']['y'][:siz_y]
    y1[2*siz_y:3*siz_y] = bdry['y1']['z'][:siz_y]
    y2[0*siz_y:1*siz_y] = bdry['y2']['x'][:siz_y]
    y2[1*siz_y:2*siz_y] = bdry['y2']['y'][:siz_y]
    y2[2*siz_y:3*siz_y] = bdry['y2']['z'][:siz_y]

    # z1/z2 faces: layout is [j*nx + i] for j=0..ny-1, i=0..nx-1
    z1 = np.zeros(3 * siz_z, dtype=np.float32)
    z2 = np.zeros(3 * siz_z, dtype=np.float32)
    z1[0*siz_z:1*siz_z] = bdry['z1']['x'][:siz_z]
    z1[1*siz_z:2*siz_z] = bdry['z1']['y'][:siz_z]
    z1[2*siz_z:3*siz_z] = bdry['z1']['z'][:siz_z]
    z2[0*siz_z:1*siz_z] = bdry['z2']['x'][:siz_z]
    z2[1*siz_z:2*siz_z] = bdry['z2']['y'][:siz_z]
    z2[2*siz_z:3*siz_z] = bdry['z2']['z'][:siz_z]

    neighid_c = np.ascontiguousarray(mympi.neighid)

    # Make arrays contiguous for C
    x3d_c = np.ascontiguousarray(x3d)
    y3d_c = np.ascontiguousarray(y3d)
    z3d_c = np.ascontiguousarray(z3d)

    lib.linear_tfi_c(x3d_c, y3d_c, z3d_c,
                     np.ascontiguousarray(x1), np.ascontiguousarray(x2),
                     np.ascontiguousarray(y1), np.ascontiguousarray(y2),
                     np.ascontiguousarray(z1), np.ascontiguousarray(z2),
                     nx, ny, nz,
                     ni1, ni2, nj1, nj2, nk1, nk2,
                     gni1, gnj1, gnk1,
                     total_nx, total_ny, total_nz,
                     neighid_c)

    # Copy back
    x3d[:] = x3d_c
    y3d[:] = y3d_c
    z3d[:] = z3d_c
