"""Utility functions for 3D grid generation."""

import numpy as np


def print_quality_checks(cfgs: dict) -> None:
    """Print enabled grid quality check flags."""
    if cfgs.get('check_orth') == 1:
        print("------- check grid orthogonality -------")
    if cfgs.get('check_jac') == 1:
        print("------- check grid jacobi -------")
    if cfgs.get('check_step_xi') == 1:
        print("------- check grid step xi direction -------")
    if cfgs.get('check_step_et') == 1:
        print("------- check grid step et direction -------")
    if cfgs.get('check_step_zt') == 1:
        print("------- check grid step zt direction -------")
    if cfgs.get('check_smooth_xi') == 1:
        print("------- check grid smooth xi direction -------")
    if cfgs.get('check_smooth_et') == 1:
        print("------- check grid smooth et direction -------")
    if cfgs.get('check_smooth_zt') == 1:
        print("------- check grid smooth zt direction -------")


def remove_comment_keys(data):
    """Filter out keys starting with '#' from nested dictionaries (JSON-loaded data)."""
    if isinstance(data, dict):
        new_dict = data.copy()
        for key in list(new_dict.keys()):
            if key.startswith('#'):
                del new_dict[key]
            else:
                new_dict[key] = remove_comment_keys(new_dict[key])
        return new_dict
    elif isinstance(data, list):
        return [remove_comment_keys(item) for item in data]
    else:
        return data


def permute_coord_x(gd):
    """Permute coordinates so x becomes the marching direction (z).
    Before: x3d(k,j,i), y3d(k,j,i), z3d(k,j,i)
    After:  x3d(k,j,i)←original z, y3d(k,j,i)←original y, z3d(k,j,i)←original x
    Swap nx↔nz, total_nx↔total_nz.
    """
    x_old = gd.x3d.copy()
    y_old = gd.y3d.copy()
    z_old = gd.z3d.copy()
    # z_new = x_old transposed: original (nz,ny,nx) → new (nx,ny,nz)
    gd.x3d = np.transpose(z_old, (2, 1, 0)).copy()
    gd.y3d = np.transpose(y_old, (2, 1, 0)).copy()
    gd.z3d = np.transpose(x_old, (2, 1, 0)).copy()
    gd.nx, gd.nz = gd.nz, gd.nx
    gd.ni, gd.nk = gd.nk, gd.ni
    if hasattr(gd, 'total_nx') and hasattr(gd, 'total_nz'):
        gd.total_nx, gd.total_nz = gd.total_nz, gd.total_nx


def permute_coord_y(gd):
    """Permute coordinates so y becomes the marching direction (z).
    Before: x3d(k,j,i), y3d(k,j,i), z3d(k,j,i)
    After:  x3d(k,j,i)←original x, y3d(k,j,i)←original z, z3d(k,j,i)←original y
    Swap ny↔nz, total_ny↔total_nz.
    """
    x_old = gd.x3d.copy()
    y_old = gd.y3d.copy()
    z_old = gd.z3d.copy()
    # new(k,j,i): x_new = x_old(i,j,k) transposed: original (nz,ny,nx) → new (ny,nz,nx)
    gd.x3d = np.transpose(x_old, (1, 0, 2)).copy()
    gd.y3d = np.transpose(z_old, (1, 0, 2)).copy()
    gd.z3d = np.transpose(y_old, (1, 0, 2)).copy()
    gd.ny, gd.nz = gd.nz, gd.ny
    gd.nj, gd.nk = gd.nk, gd.nj
    if hasattr(gd, 'total_ny') and hasattr(gd, 'total_nz'):
        gd.total_ny, gd.total_nz = gd.total_nz, gd.total_ny


def flip_coord_z(gd):
    """Flip grid in z (marching) direction."""
    gd.x3d = gd.x3d[::-1, :, :].copy()
    gd.y3d = gd.y3d[::-1, :, :].copy()
    gd.z3d = gd.z3d[::-1, :, :].copy()


def flip_step_z(step):
    """Flip step array in z direction."""
    step[:] = step[::-1]


def extend_abs_layer_3d(bdry, dx, dy, nx, ny, num_pml):
    """Extend 3D boundary surface with PML absorbing layers.

    Args:
        bdry: Surface array, shape (nx, ny, 3) with x,y,z components.
        dx: Grid spacing in x-direction.
        dy: Grid spacing in y-direction.
        nx: Total number of x-points (including PML).
        ny: Total number of y-points (including PML).
        num_pml: Number of PML layers on each side.

    Returns:
        Extended bdry array.
    """
    coef = 0.7

    # Extend in x-direction (i-axis)
    for j in range(ny):
        for i in range(num_pml - 1, -1, -1):
            dz = bdry[i + 2, j, 2] - bdry[i + 1, j, 2]
            slope = coef * dz / dx
            bdry[i, j, 0] = bdry[i + 1, j, 0] - dx
            bdry[i, j, 1] = bdry[i + 1, j, 1]
            bdry[i, j, 2] = bdry[i + 1, j, 2] - slope * dx

        for i in range(nx - num_pml, nx):
            dz = bdry[i - 1, j, 2] - bdry[i - 2, j, 2]
            slope = coef * dz / dx
            bdry[i, j, 0] = bdry[i - 1, j, 0] + dx
            bdry[i, j, 1] = bdry[i - 1, j, 1]
            bdry[i, j, 2] = bdry[i - 1, j, 2] + slope * dx

    # Extend in y-direction (j-axis)
    for i in range(nx):
        for j in range(num_pml - 1, -1, -1):
            dz = bdry[i, j + 2, 2] - bdry[i, j + 1, 2]
            slope = coef * dz / dy
            bdry[i, j, 0] = bdry[i, j + 1, 0]
            bdry[i, j, 1] = bdry[i, j + 1, 1] - dy
            bdry[i, j, 2] = bdry[i, j + 1, 2] - slope * dy

        for j in range(ny - num_pml, ny):
            dz = bdry[i, j - 1, 2] - bdry[i, j - 2, 2]
            slope = coef * dz / dy
            bdry[i, j, 0] = bdry[i, j - 1, 0]
            bdry[i, j, 1] = bdry[i, j - 1, 1] + dy
            bdry[i, j, 2] = bdry[i, j - 1, 2] + slope * dy

    return bdry


def read_bdry_value(line):
    """Parse a non-comment line from boundary file, returning float values."""
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    return [float(v) for v in line.split()]


def read_bdry_file(filepath, total_nx, total_ny, total_nz):
    """Read 6 boundary surfaces from text file.

    Returns dict with keys: x1, x2, y1, y2, z1, z2
    Each value is a dict with 'x', 'y', 'z' arrays.
    """
    bdry = {}
    current_face = None
    with open(filepath, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('# boundary'):
                # e.g. "# boundary_x1" → "x1"
                name = stripped.split('_')[1]
                current_face = name
                if name == 'x1' or name == 'x2':
                    ny_b = total_ny - 2
                    nz_b = total_nz - 2
                    bdry[name] = {
                        'x': np.zeros((nz_b, ny_b), dtype=np.float32),
                        'y': np.zeros((nz_b, ny_b), dtype=np.float32),
                        'z': np.zeros((nz_b, ny_b), dtype=np.float32),
                    }
                elif name == 'y1' or name == 'y2':
                    nx_b = total_nx - 2
                    nz_b = total_nz - 2
                    bdry[name] = {
                        'x': np.zeros((nz_b, nx_b), dtype=np.float32),
                        'y': np.zeros((nz_b, nx_b), dtype=np.float32),
                        'z': np.zeros((nz_b, nx_b), dtype=np.float32),
                    }
                elif name == 'z1' or name == 'z2':
                    nx_b = total_nx - 2
                    ny_b = total_ny - 2
                    bdry[name] = {
                        'x': np.zeros((ny_b, nx_b), dtype=np.float32),
                        'y': np.zeros((ny_b, nx_b), dtype=np.float32),
                        'z': np.zeros((ny_b, nx_b), dtype=np.float32),
                    }
                idx = 0
                continue
            if stripped.startswith('#'):
                continue
            if current_face is None:
                continue
            vals = [float(v) for v in stripped.split()]
            if len(vals) < 3:
                continue
            arr = bdry[current_face]
            nz_b = arr['x'].shape[0]
            nx_b = arr['x'].shape[1]
            j = idx // nx_b
            i = idx % nx_b
            if j < nz_b:
                arr['x'][j, i] = vals[0]
                arr['y'][j, i] = vals[1]
                arr['z'][j, i] = vals[2]
                idx += 1
    return bdry
