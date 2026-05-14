"""Elliptic grid data setup and configuration parsing for 3D elliptic method."""

import json
import sys
import numpy as np

from common.constants import (X_DIRE, Y_DIRE, Z_DIRE, DIRE_MAP,
                              LINEAR_TFI, ELLI_DIRI, ELLI_HIGEN,
                              FACE_X1, FACE_X2, FACE_Y1, FACE_Y2,
                              FACE_Z1, FACE_Z2)
from common.utils import remove_comment_keys, print_quality_checks

METHOD_MAP = {
    "linear_tfi": LINEAR_TFI,
    "elli_diri": ELLI_DIRI,
    "elli_higen": ELLI_HIGEN,
}


def cfgs_read(config_file):
    """Read JSON config file and return parameter dict.

    Flattens nested method sub-objects into top-level keys.
    """
    with open(config_file, 'r') as f:
        cfgs = json.load(f)
    cfgs = remove_comment_keys(cfgs)

    # Flatten nested method config (e.g. "grid_method": {"elli_diri": {...}})
    if 'grid_method' in cfgs and isinstance(cfgs['grid_method'], dict):
        method_cfg = cfgs.pop('grid_method')
        for method_name, method_params in method_cfg.items():
            cfgs['method_itype'] = METHOD_MAP.get(method_name, LINEAR_TFI)
            if isinstance(method_params, dict):
                for key, val in method_params.items():
                    cfgs[key] = val
            break  # Only one method allowed

    if 'method_itype' not in cfgs:
        cfgs['method_itype'] = LINEAR_TFI

    return cfgs


def cfgs_print(cfgs):
    """Print configuration parameters."""
    method_name = {v: k for k, v in METHOD_MAP.items()}.get(
        cfgs.get('method_itype', LINEAR_TFI), 'linear_tfi')

    print("-------------------------------------------------------")
    print("--> elliptic grid generation parameters:")
    print("-------------------------------------------------------")
    print(f" number_of_grid_points_x  = {cfgs['number_of_grid_points_x']}")
    print(f" number_of_grid_points_y  = {cfgs['number_of_grid_points_y']}")
    print(f" number_of_grid_points_z  = {cfgs['number_of_grid_points_z']}")
    print(f" number_of_mpiprocs_x     = {cfgs.get('number_of_mpiprocs_x', 1)}")
    print(f" number_of_mpiprocs_y     = {cfgs.get('number_of_mpiprocs_y', 1)}")
    print(f" number_of_mpiprocs_z     = {cfgs.get('number_of_mpiprocs_z', 1)}")
    print(f" method                   = {method_name}")
    if cfgs.get('method_itype', LINEAR_TFI) == ELLI_DIRI:
        print(f" coef                     = {cfgs.get('coef', [1]*6)}")
        print(f" iter_err                 = {cfgs.get('iter_err', 1e-4)}")
        print(f" max_iter                 = {cfgs.get('max_iter', 1000)}")
    elif cfgs.get('method_itype', LINEAR_TFI) == ELLI_HIGEN:
        print(f" a_diri                   = {cfgs.get('a_diri', 0.5)}")
        print(f" a_higen                  = {cfgs.get('a_higen', 0.5)}")
        print(f" iter_err                 = {cfgs.get('iter_err', 1e-4)}")
        print(f" max_iter                 = {cfgs.get('max_iter', 1000)}")
    print(f" geometry_input_file      = {cfgs['geometry_input_file']}")
    print(f" grid_export_dir          = {cfgs['grid_export_dir']}")
    if cfgs.get('grid_check', 0) == 1:
        print_quality_checks(cfgs)
    print()


def grid_info_set(cfgs, mympi=None):
    """Calculate local grid dimensions with ghost cells for 3D MPI decomposition.

    Ghost cells: 1 layer on each side in x, y, z directions.

    Returns a dict with all grid info.
    """
    total_nx = cfgs['number_of_grid_points_x']
    total_ny = cfgs['number_of_grid_points_y']
    total_nz = cfgs['number_of_grid_points_z']

    nproc_x = cfgs.get('number_of_mpiprocs_x', 1)
    nproc_y = cfgs.get('number_of_mpiprocs_y', 1)
    nproc_z = cfgs.get('number_of_mpiprocs_z', 1)

    # Get MPI rank position
    if mympi is not None:
        topoid = mympi.topoid
    else:
        topoid = [0, 0, 0]

    # Partition interior points (total - 2 boundary points per direction)
    ni_et = total_nx - 2
    nj_et = total_ny - 2
    nk_et = total_nz - 2

    # X-direction partition
    ni = _partition_size(ni_et, nproc_x, topoid[0])
    gni1 = _partition_start(ni_et, nproc_x, topoid[0])

    # Y-direction partition
    nj = _partition_size(nj_et, nproc_y, topoid[1])
    gnj1 = _partition_start(nj_et, nproc_y, topoid[1])

    # Z-direction partition
    nk = _partition_size(nk_et, nproc_z, topoid[2])
    gnk1 = _partition_start(nk_et, nproc_z, topoid[2])

    # Add ghost cells (1 on each side)
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
        'gni1': gni1, 'gni2': gni1 + ni - 1,
        'gnj1': gnj1, 'gnj2': gnj1 + nj - 1,
        'gnk1': gnk1, 'gnk2': gnk1 + nk - 1,
        'nproc_x': nproc_x, 'nproc_y': nproc_y, 'nproc_z': nproc_z,
    }
    return info


def _partition_size(total_interior, nproc, rank):
    """Calculate local interior size for a given rank."""
    avg = total_interior // nproc
    leftover = total_interior % nproc
    size = avg + (1 if rank < leftover else 0)
    return size


def _partition_start(total_interior, nproc, rank):
    """Calculate global start index for a given rank."""
    avg = total_interior // nproc
    leftover = total_interior % nproc
    if rank == 0:
        return 0
    start = rank * avg
    if leftover != 0:
        start += rank if rank < leftover else leftover
    return start


def read_bdry_file(gdinfo, cfgs):
    """Read 6 boundary surfaces from geometry file.

    File format:
        nx
        ny
        nz
        [x1 face: ny*nz lines of x y z] (i=0, varies j,k)
        [x2 face: ny*nz lines of x y z] (i=nx-1)
        [y1 face: nx*nz lines of x y z] (j=0, varies i,k)
        [y2 face: nx*nz lines of x y z] (j=ny-1)
        [z1 face: nx*ny lines of x y z] (k=0, varies i,j)
        [z2 face: nx*ny lines of x y z] (k=nz-1)

    Returns dict with keys: x1, x2, y1, y2, z1, z2, each with x,y,z arrays.
    """
    total_nx = gdinfo['total_nx']
    total_ny = gdinfo['total_ny']
    total_nz = gdinfo['total_nz']

    # Parse all lines
    lines = []
    with open(cfgs['geometry_input_file'], 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            lines.append(line)

    idx = 0
    # First 3 lines: nx, ny, nz
    file_nx = int(lines[idx]); idx += 1
    file_ny = int(lines[idx]); idx += 1
    file_nz = int(lines[idx]); idx += 1

    if file_nx != total_nx or file_ny != total_ny or file_nz != total_nz:
        print(f"Warning: boundary file dimensions ({file_nx},{file_ny},{file_nz}) "
              f"!= config ({total_nx},{total_ny},{total_nz})")

    # Size of each face
    siz_x = total_ny * total_nz  # x1, x2 faces
    siz_y = total_nx * total_nz  # y1, y2 faces
    siz_z = total_nx * total_ny  # z1, z2 faces

    bdry = {}
    for face_name, face_size in [('x1', siz_x), ('x2', siz_x),
                                  ('y1', siz_y), ('y2', siz_y),
                                  ('z1', siz_z), ('z2', siz_z)]:
        x_arr = np.zeros(face_size, dtype=np.float32)
        y_arr = np.zeros(face_size, dtype=np.float32)
        z_arr = np.zeros(face_size, dtype=np.float32)
        for n in range(face_size):
            vals = list(map(float, lines[idx].split()))
            x_arr[n] = vals[0]
            y_arr[n] = vals[1]
            z_arr[n] = vals[2]
            idx += 1
        bdry[face_name] = {'x': x_arr, 'y': y_arr, 'z': z_arr}

    return bdry
