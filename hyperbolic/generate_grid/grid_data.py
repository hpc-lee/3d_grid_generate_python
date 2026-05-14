"""Hyperbolic grid data setup and configuration parsing for 3D hyperbolic method."""

import json
import sys
import numpy as np

from common.constants import X_DIRE, Y_DIRE, Z_DIRE, DIRE_MAP
from common.utils import remove_comment_keys, print_quality_checks


def cfgs_read(config_file):
    """Read JSON config file and return parameter dict.

    Supports nested format with "hyperbolic" sub-object, flattening it
    into the top-level dict for uniform access.
    """
    with open(config_file, 'r') as f:
        cfgs = json.load(f)
    cfgs = remove_comment_keys(cfgs)

    # Flatten nested "hyperbolic" sub-object
    if 'hyperbolic' in cfgs and isinstance(cfgs['hyperbolic'], dict):
        hyper_cfgs = cfgs.pop('hyperbolic')
        for key, val in hyper_cfgs.items():
            if key not in cfgs:
                cfgs[key] = val

    return cfgs


def cfgs_print(cfgs):
    """Print configuration parameters."""
    dire_itype = DIRE_MAP.get(cfgs.get('direction', 'z'), Z_DIRE)
    print("-------------------------------------------------------")
    print("--> hyperbolic grid generation parameters:")
    print("-------------------------------------------------------")
    print(f" number_of_grid_points_x  = {cfgs['number_of_grid_points_x']}")
    print(f" number_of_grid_points_y  = {cfgs['number_of_grid_points_y']}")
    print(f" number_of_grid_points_z  = {cfgs['number_of_grid_points_z']}")
    print(f" direction                = {cfgs.get('direction', 'z')}")
    print(f" coef                     = {cfgs.get('coef', 40)}")
    print(f" t2b                      = {cfgs.get('t2b', 0)}")
    print(f" flag_stretch             = {cfgs.get('flag_stretch', 0)}")
    print(f" bdry_x_itype             = {cfgs.get('bdry_x_itype', cfgs.get('bdry_x_type', 0))}")
    print(f" bdry_y_itype             = {cfgs.get('bdry_y_itype', cfgs.get('bdry_y_type', 0))}")
    print(f" epsilon_x                = {cfgs.get('epsilon_x', 0.0)}")
    print(f" epsilon_y                = {cfgs.get('epsilon_y', 0.0)}")
    print(f" geometry_input_file      = {cfgs['geometry_input_file']}")
    print(f" step_input_file          = {cfgs['step_input_file']}")
    print(f" grid_export_dir          = {cfgs['grid_export_dir']}")
    if cfgs.get('grid_check', 0) == 1:
        print_quality_checks(cfgs)
    print()


def grid_info_set(cfgs):
    """Calculate grid dimensions for hyperbolic method (serial only).

    For the marching direction, nz is determined from the step file
    (nz = num_step + 1), matching the C original code.
    nx and ny come from the geometry file header.

    Returns a dict with all grid info.
    """
    dire_itype = DIRE_MAP.get(cfgs.get('direction', 'z'), Z_DIRE)

    # Determine total grid size (after direction permutation)
    ngx = cfgs['number_of_grid_points_x']
    ngy = cfgs['number_of_grid_points_y']
    ngz = cfgs['number_of_grid_points_z']

    # Read step file to determine marching direction dimension
    num_step = 0
    with open(cfgs['step_input_file'], 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            num_step = int(line)
            break

    # Read geometry file header for nx, ny
    with open(cfgs['geometry_input_file'], 'r') as f:
        geo_lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

    # nx and ny can be on the same line ("51 41") or separate lines
    first_parts = geo_lines[0].split()
    if len(first_parts) >= 2:
        geo_nx = int(first_parts[0])
        geo_ny = int(first_parts[1])
    else:
        geo_nx = int(geo_lines[0])
        geo_ny = int(geo_lines[1])

    # Set dimensions based on marching direction
    if dire_itype == Z_DIRE:
        total_nx, total_ny, total_nz = geo_nx, geo_ny, num_step + 1
    elif dire_itype == Y_DIRE:
        total_nx, total_ny, total_nz = geo_nx, num_step + 1, geo_ny
    elif dire_itype == X_DIRE:
        total_nx, total_ny, total_nz = num_step + 1, geo_ny, geo_nx

    # Hyperbolic doesn't use ghost cells or MPI - full domain
    nx = total_nx
    ny = total_ny
    nz = total_nz

    info = {
        'total_nx': total_nx, 'total_ny': total_ny, 'total_nz': total_nz,
        'nx': nx, 'ny': ny, 'nz': nz,
        'dire_itype': dire_itype,
    }
    return info


def read_bdry_file(gdinfo, cfgs):
    """Read boundary surfaces from geometry and step files.

    Returns: x3d, y3d, z3d arrays of shape (nz, ny, nx) with k=0 boundary filled.
             Also returns step array of shape (nz-1,).
    """
    nx = gdinfo['nx']
    ny = gdinfo['ny']
    nz = gdinfo['nz']
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

    if nz - num_step != 1:
        print(f"Error: num_step={num_step} but nz-1={nz-1}")
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
    # Parse header: nx ny can be on the same line or separate lines
    first_parts = lines[idx].split()
    if len(first_parts) >= 2:
        if dire_itype == Z_DIRE or dire_itype == Y_DIRE:
            num_point_x = int(first_parts[0])
            num_point_y = int(first_parts[1])
        else:
            num_point_y = int(first_parts[0])
            num_point_x = int(first_parts[1])
        idx += 1
    else:
        if dire_itype == Z_DIRE or dire_itype == Y_DIRE:
            num_point_x = int(lines[idx]); idx += 1
            num_point_y = int(lines[idx]); idx += 1
        else:
            num_point_y = int(lines[idx]); idx += 1
            num_point_x = int(lines[idx]); idx += 1

    # Read only 1 boundary surface for hyperbolic (initial surface at k=0)
    bz1 = np.zeros((total_nx, total_ny, 3), dtype=np.float32)

    if dire_itype == Z_DIRE:
        for j in range(num_point_y):
            for i in range(num_point_x):
                vals = list(map(float, lines[idx].split()))
                bz1[i, j, 0] = vals[0]
                bz1[i, j, 1] = vals[1]
                bz1[i, j, 2] = vals[2]
                idx += 1
    elif dire_itype == Y_DIRE:
        for j in range(num_point_y):
            for i in range(num_point_x):
                vals = list(map(float, lines[idx].split()))
                bz1[i, j, 0] = vals[0]
                bz1[i, j, 2] = vals[1]
                bz1[i, j, 1] = vals[2]
                idx += 1
    elif dire_itype == X_DIRE:
        for i in range(num_point_x):
            for j in range(num_point_y):
                vals = list(map(float, lines[idx].split()))
                bz1[i, j, 2] = vals[0]
                bz1[i, j, 1] = vals[1]
                bz1[i, j, 0] = vals[2]
                idx += 1

    # Fill boundary data into x3d, y3d, z3d at k=0
    if dire_itype == Z_DIRE or dire_itype == Y_DIRE:
        for j in range(ny):
            for i in range(nx):
                x3d[0, j, i] = bz1[i, j, 0]
                y3d[0, j, i] = bz1[i, j, 1]
                z3d[0, j, i] = bz1[i, j, 2]
    elif dire_itype == X_DIRE:
        for j in range(ny):
            for i in range(nx):
                x3d[0, j, i] = bz1[i, j, 0]
                y3d[0, j, i] = bz1[i, j, 1]
                z3d[0, j, i] = bz1[i, j, 2]

    return x3d, y3d, z3d, step_arr
