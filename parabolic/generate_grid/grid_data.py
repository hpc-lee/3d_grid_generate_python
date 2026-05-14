"""Parabolic grid data setup and configuration parsing for 3D parabolic method."""

import json
import sys
import numpy as np

from common.constants import X_DIRE, Y_DIRE, Z_DIRE, DIRE_MAP
from common.utils import remove_comment_keys, print_quality_checks


def cfgs_read(config_file):
    """Read JSON config file and return parameter dict.

    Supports both nested ("parabolic": {...}) and flat config formats,
    matching the C original code's JSON structure.
    """
    with open(config_file, 'r') as f:
        cfgs = json.load(f)
    cfgs = remove_comment_keys(cfgs)

    # Flatten nested "parabolic" sub-object (C original format)
    if 'parabolic' in cfgs:
        para_cfgs = cfgs.pop('parabolic')
        for key, val in para_cfgs.items():
            if key not in cfgs:
                cfgs[key] = val

    return cfgs


def cfgs_print(cfgs):
    """Print configuration parameters."""
    dire_itype = DIRE_MAP.get(cfgs.get('direction', 'z'), Z_DIRE)
    print("-------------------------------------------------------")
    print("--> parabolic grid generation parameters:")
    print("-------------------------------------------------------")
    print(f" number_of_grid_points_x  = {cfgs['number_of_grid_points_x']}")
    print(f" number_of_grid_points_y  = {cfgs['number_of_grid_points_y']}")
    print(f" number_of_grid_points_z  = {cfgs['number_of_grid_points_z']}")
    print(f" number_of_mpiprocs       = {cfgs.get('number_of_mpiprocs', 1)}")
    print(f" direction                = {cfgs.get('direction', 'z')}")
    print(f" coef                     = {cfgs.get('coef', 40)}")
    print(f" t2b                      = {cfgs.get('t2b', 0)}")
    print(f" geometry_input_file      = {cfgs['geometry_input_file']}")
    print(f" step_input_file          = {cfgs['step_input_file']}")
    print(f" grid_export_dir          = {cfgs['grid_export_dir']}")
    if cfgs.get('grid_check', 0) == 1:
        print_quality_checks(cfgs)
    print()


def grid_info_set(cfgs, mympi=None):
    """Calculate local grid dimensions with ghost cells for MPI decomposition.

    For parabolic, MPI decomposes only the y-direction (1D Cartesian).
    Ghost cells: 1 layer on each side in y, 1 layer on x and z boundaries.

    Returns a dict with all grid info.
    """
    dire_itype = DIRE_MAP.get(cfgs.get('direction', 'z'), Z_DIRE)

    # Determine total grid size (after direction permutation)
    ngx = cfgs['number_of_grid_points_x']
    ngy = cfgs['number_of_grid_points_y']
    ngz = cfgs['number_of_grid_points_z']

    if dire_itype == Z_DIRE:
        total_nx, total_ny, total_nz = ngx, ngy, ngz
    elif dire_itype == Y_DIRE:
        total_nx, total_ny, total_nz = ngx, ngz, ngy
    elif dire_itype == X_DIRE:
        total_nx, total_ny, total_nz = ngz, ngy, ngx

    nproc = cfgs.get('number_of_mpiprocs', 1)

    # Y-direction partitioning (interior points: total_ny - 2)
    ny_et = total_ny - 2
    ny_avg = ny_et // nproc
    ny_left = ny_et % nproc

    # Get my MPI rank position
    if mympi is not None:
        topoid = mympi.topoid[0]
    else:
        topoid = 0

    nj = ny_avg
    if topoid < ny_left:
        nj += 1

    # Global y-index of first interior point
    if topoid == 0:
        gnj1 = 0
    else:
        gnj1 = topoid * ny_avg
    if ny_left != 0:
        gnj1 += topoid if topoid < ny_left else ny_left

    # Interior dimensions
    ni = total_nx - 2
    nk = total_nz - 2

    # Add ghost cells
    nx = ni + 2
    ny = nj + 2
    nz = nk + 2

    info = {
        'total_nx': total_nx, 'total_ny': total_ny, 'total_nz': total_nz,
        'nx': nx, 'ny': ny, 'nz': nz,
        'ni': ni, 'nj': nj, 'nk': nk,
        'ni1': 1, 'ni2': ni,
        'nj1': 1, 'nj2': nj,
        'nk1': 1, 'nk2': nk,
        'gni1': 0, 'gni2': ni - 1,
        'gnj1': gnj1, 'gnj2': gnj1 + nj - 1,
        'gnk1': 0, 'gnk2': nk - 1,
        'dire_itype': dire_itype,
    }
    return info


def read_bdry_file(gdinfo, cfgs):
    """Read boundary surfaces from geometry and step files.

    Returns: x3d, y3d, z3d arrays of shape (nz, ny, nx) with boundary data
             filled in at k=0 (bottom) and k=nz-1 (top).
             Also returns step array of shape (nz-2,).
    """
    nx = gdinfo['nx']
    ny = gdinfo['ny']
    nz = gdinfo['nz']
    gnj1 = gdinfo['gnj1']
    dire_itype = gdinfo['dire_itype']
    total_nx = gdinfo['total_nx']
    total_ny = gdinfo['total_ny']
    total_nz = gdinfo['total_nz']

    # Read step file
    step = []
    num_step = 0
    with open(cfgs['step_input_file'], 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if num_step == 0:
                num_step = int(line)
            else:
                step.append(float(line))

    if total_nz - 1 != num_step:
        print(f"Error: num_step={num_step} but total_nz-1={total_nz-1}")
        sys.exit(1)

    step_arr = np.array(step[:num_step], dtype=np.float32)

    # Read geometry file
    x3d = np.zeros((nz, ny, nx), dtype=np.float32)
    y3d = np.zeros((nz, ny, nx), dtype=np.float32)
    z3d = np.zeros((nz, ny, nx), dtype=np.float32)

    # Parse boundary surfaces
    lines = []
    with open(cfgs['geometry_input_file'], 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            lines.append(line)

    idx = 0
    # First line: num_point_x num_point_y
    if dire_itype == Z_DIRE or dire_itype == Y_DIRE:
        num_point_x, num_point_y = map(int, lines[idx].split())
    else:  # X_DIRE
        num_point_y, num_point_x = map(int, lines[idx].split())
    idx += 1

    # Read 2 boundary surfaces (bottom and top)
    bz1 = np.zeros((total_nx, total_ny, 3), dtype=np.float32)
    bz2 = np.zeros((total_nx, total_ny, 3), dtype=np.float32)

    for surf_idx, bz in enumerate([bz1, bz2]):
        if dire_itype == Z_DIRE:
            for j in range(num_point_y):
                for i in range(num_point_x):
                    vals = list(map(float, lines[idx].split()))
                    bz[i, j, 0] = vals[0]
                    bz[i, j, 1] = vals[1]
                    bz[i, j, 2] = vals[2]
                    idx += 1
        elif dire_itype == Y_DIRE:
            for j in range(num_point_y):
                for i in range(num_point_x):
                    vals = list(map(float, lines[idx].split()))
                    bz[i, j, 0] = vals[0]
                    bz[i, j, 2] = vals[1]
                    bz[i, j, 1] = vals[2]
                    idx += 1
        elif dire_itype == X_DIRE:
            for i in range(num_point_x):
                for j in range(num_point_y):
                    vals = list(map(float, lines[idx].split()))
                    bz[i, j, 2] = vals[0]
                    bz[i, j, 1] = vals[1]
                    bz[i, j, 0] = vals[2]
                    idx += 1

    # Fill boundary data into x3d, y3d, z3d
    if dire_itype == Z_DIRE or dire_itype == Y_DIRE:
        for j in range(ny):
            for i in range(nx):
                x3d[0, j, i] = bz1[i, gnj1 + j, 0]
                y3d[0, j, i] = bz1[i, gnj1 + j, 1]
                z3d[0, j, i] = bz1[i, gnj1 + j, 2]

                x3d[nz-1, j, i] = bz2[i, gnj1 + j, 0]
                y3d[nz-1, j, i] = bz2[i, gnj1 + j, 1]
                z3d[nz-1, j, i] = bz2[i, gnj1 + j, 2]
    elif dire_itype == X_DIRE:
        for j in range(ny):
            for i in range(nx):
                gj = gnj1 + j
                x3d[0, j, i] = bz1[i, gj, 0]
                y3d[0, j, i] = bz1[i, gj, 1]
                z3d[0, j, i] = bz1[i, gj, 2]

                x3d[nz-1, j, i] = bz2[i, gj, 0]
                y3d[nz-1, j, i] = bz2[i, gj, 1]
                z3d[nz-1, j, i] = bz2[i, gj, 2]

    return x3d, y3d, z3d, step_arr
