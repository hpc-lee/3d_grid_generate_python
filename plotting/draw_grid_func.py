"""Core 3D visualization functions for reading and assembling grid data.

Handles MPI-partitioned NetCDF files with 3D processor decomposition
(px, py, pz) and provides functions to locate, gather, and subset
coordinate and quality data.
"""

import os
import numpy as np
from netCDF4 import Dataset
import glob


def locate_coord(cfgs: dict, subs: list, subc: list, subt: list) -> list:
    """Locate coordinate NetCDF files based on parameters and spatial selection.

    Args:
        cfgs: JSON parameter dict with 'number_of_grid_points_x/y/z'
              and 'grid_export_dir'.
        subs: Starting indices for x, y, z dimensions (0-based).
        subc: Count of points to select for x, y, z (-1 for all).
        subt: Stride for x, y, z dimensions.

    Returns:
        List of dicts containing file info for each relevant NetCDF partition.
    """
    ngijk = [cfgs['number_of_grid_points_x'],
             cfgs['number_of_grid_points_y'],
             cfgs['number_of_grid_points_z']]

    gsubs = subs.copy()
    gsubt = subt.copy()
    gsubc = subc.copy()

    for idx, val in enumerate(gsubc):
        if val == -1:
            gsubc[idx] = int(np.ceil((ngijk[idx] - gsubs[idx]) / gsubt[idx]))

    gsube = [gsubs[0] + (gsubc[0] - 1) * gsubt[0],
             gsubs[1] + (gsubc[1] - 1) * gsubt[1],
             gsubs[2] + (gsubc[2] - 1) * gsubt[2]]

    coordprefix = 'coord'
    coord_pattern = os.path.join(cfgs['grid_export_dir'], f"{coordprefix}*.nc")
    coord_files = glob.glob(coord_pattern)

    px, py, pz = [], [], []

    for coordnm in coord_files:
        with Dataset(coordnm, 'r') as nc_file:
            xyzs = nc_file.getncattr('global_index_of_first_physical_points')
            xs, ys, zs = int(xyzs[0]), int(xyzs[1]), int(xyzs[2])
            xyzc = nc_file.getncattr('count_of_physical_points')
            xc, yc, zc = int(xyzc[0]), int(xyzc[1]), int(xyzc[2])

        xarray = np.arange(xs, xs + xc - 1)
        yarray = np.arange(ys, ys + yc - 1)
        zarray = np.arange(zs, zs + zc - 1)

        x_in_range = np.any((xarray >= gsubs[0]) & (xarray <= gsube[0]))
        y_in_range = np.any((yarray >= gsubs[1]) & (yarray <= gsube[1]))
        z_in_range = np.any((zarray >= gsubs[2]) & (zarray <= gsube[2]))
        if x_in_range and y_in_range and z_in_range:
            filename = os.path.basename(coordnm)
            px_match = filename.find('px') + 2
            py_match = filename.find('_py')
            px_val = int(filename[px_match:py_match])

            py_match_start = py_match + 3
            pz_match = filename.find('_pz')
            py_val = int(filename[py_match_start:pz_match])

            pz_match_start = pz_match + 3
            nc_match = filename.find('.nc')
            pz_val = int(filename[pz_match_start:nc_match])

            px.append(px_val)
            py.append(py_val)
            pz.append(pz_val)

    coordinfo = []
    for ip in range(len(px)):
        coordnm = os.path.join(cfgs['grid_export_dir'],
                               f"{coordprefix}_px{px[ip]}_py{py[ip]}_pz{pz[ip]}.nc")

        with Dataset(coordnm, 'r') as nc_file:
            xyzs = nc_file.getncattr('global_index_of_first_physical_points')
            xs, ys, zs = int(xyzs[0]), int(xyzs[1]), int(xyzs[2])
            xyzc = nc_file.getncattr('count_of_physical_points')
            xc, yc, zc = int(xyzc[0]), int(xyzc[1]), int(xyzc[2])

        xe = xs + xc - 1
        ye = ys + yc - 1
        ze = zs + zc - 1

        gxarray = np.arange(gsubs[0], gsube[0] + 1, gsubt[0])
        gyarray = np.arange(gsubs[1], gsube[1] + 1, gsubt[1])
        gzarray = np.arange(gsubs[2], gsube[2] + 1, gsubt[2])

        i_mask = (gxarray >= xs) & (gxarray <= xe)
        j_mask = (gyarray >= ys) & (gyarray <= ye)
        k_mask = (gzarray >= zs) & (gzarray <= ze)
        i_indices = np.where(i_mask)[0]
        j_indices = np.where(j_mask)[0]
        k_indices = np.where(k_mask)[0]

        if len(i_indices) > 0 and len(j_indices) > 0 and len(k_indices) > 0:
            coord_dict = {}
            coord_dict['thisid'] = [px[ip], py[ip], pz[ip]]
            coord_dict['indxs'] = [int(i_indices[0]), int(j_indices[0]), int(k_indices[0])]
            coord_dict['indxe'] = [int(i_indices[-1]), int(j_indices[-1]), int(k_indices[-1])]
            coord_dict['indxc'] = [coord_dict['indxe'][0] - coord_dict['indxs'][0] + 1,
                                   coord_dict['indxe'][1] - coord_dict['indxs'][1] + 1,
                                   coord_dict['indxe'][2] - coord_dict['indxs'][2] + 1]

            coord_dict['subs'] = [int(gxarray[i_indices[0]] - xs),
                                  int(gyarray[j_indices[0]] - ys),
                                  int(gzarray[k_indices[0]] - zs)]
            coord_dict['subc'] = coord_dict['indxc'].copy()
            coord_dict['subt'] = gsubt.copy()

            coord_dict['fnmprefix'] = coordprefix

            coordinfo.append(coord_dict)

    return coordinfo


def gather_coord(coordinfo: list, output_dir: str):
    """Gather coordinate data from multiple NetCDF files.

    Args:
        coordinfo: List of dicts from locate_coord.
        output_dir: Directory containing the NetCDF files.

    Returns:
        Tuple of (x, y, z) numpy arrays.
    """
    max_i = 0
    max_j = 0
    max_k = 0
    for info in coordinfo:
        max_i = max(max_i, info['indxe'][0])
        max_j = max(max_j, info['indxe'][1])
        max_k = max(max_k, info['indxe'][2])

    x = np.zeros((max_i + 1, max_j + 1, max_k + 1))
    y = np.zeros((max_i + 1, max_j + 1, max_k + 1))
    z = np.zeros((max_i + 1, max_j + 1, max_k + 1))

    for info in coordinfo:
        n_i, n_j, n_k = info['thisid']
        i1, j1, k1 = info['indxs']
        i2, j2, k2 = info['indxe']
        subs = info['subs']
        subc = info['subc']
        subt = info['subt']

        fnm_coord = os.path.join(output_dir,
                                 f"{info['fnmprefix']}_px{n_i}_py{n_j}_pz{n_k}.nc")

        if not os.path.exists(fnm_coord):
            raise FileNotFoundError(f"gather_coord: file {fnm_coord} does not exist")

        with Dataset(fnm_coord, 'r') as nc_file:
            start_x = subs[0]
            start_y = subs[1]
            start_z = subs[2]
            stop_x_py = start_x + (subc[0] - 1) * subt[0] + 1
            stop_y_py = start_y + (subc[1] - 1) * subt[1] + 1
            stop_z_py = start_z + (subc[2] - 1) * subt[2] + 1
            step_x_py = subt[0]
            step_y_py = subt[1]
            step_z_py = subt[2]

            x_data = nc_file['x'][start_z:stop_z_py:step_z_py,
                                  start_y:stop_y_py:step_y_py,
                                  start_x:stop_x_py:step_x_py]
            y_data = nc_file['y'][start_z:stop_z_py:step_z_py,
                                  start_y:stop_y_py:step_y_py,
                                  start_x:stop_x_py:step_x_py]
            z_data = nc_file['z'][start_z:stop_z_py:step_z_py,
                                  start_y:stop_y_py:step_y_py,
                                  start_x:stop_x_py:step_x_py]

            x[i1:i2+1, j1:j2+1, k1:k2+1] = x_data
            y[i1:i2+1, j1:j2+1, k1:k2+1] = y_data
            z[i1:i2+1, j1:j2+1, k1:k2+1] = z_data

    x = np.squeeze(x)
    y = np.squeeze(y)
    z = np.squeeze(z)

    return x, y, z


def gather_quality(coordinfo: list, output_dir: str, varnm: str) -> np.ndarray:
    """Gather quality metric data from multiple NetCDF files.

    Args:
        coordinfo: List of dicts from locate_coord.
        output_dir: Directory containing the NetCDF files.
        varnm: Quality variable name.

    Returns:
        Quality metric data array.
    """
    max_i = 0
    max_j = 0
    max_k = 0
    for info in coordinfo:
        max_i = max(max_i, info['indxe'][0])
        max_j = max(max_j, info['indxe'][1])
        max_k = max(max_k, info['indxe'][2])

    var_data = np.zeros((max_i + 1, max_j + 1, max_k + 1))

    for info in coordinfo:
        n_i, n_j, n_k = info['thisid']
        i1, j1, k1 = info['indxs']
        i2, j2, k2 = info['indxe']
        subs = info['subs']
        subc = info['subc']
        subt = info['subt']

        fnm_var = os.path.join(output_dir,
                               f"{varnm}_px{n_i}_py{n_j}_pz{n_k}.nc")

        if not os.path.exists(fnm_var):
            raise FileNotFoundError(f"gather_quality: file {fnm_var} does not exist")

        with Dataset(fnm_var, 'r') as nc_file:
            start_x = subs[0]
            start_y = subs[1]
            start_z = subs[2]
            stop_x_py = start_x + (subc[0] - 1) * subt[0] + 1
            stop_y_py = start_y + (subc[1] - 1) * subt[1] + 1
            stop_z_py = start_z + (subc[2] - 1) * subt[2] + 1
            step_x_py = subt[0]
            step_y_py = subt[1]
            step_z_py = subt[2]

            var = nc_file[varnm][start_z:stop_z_py:step_z_py,
                                 start_y:stop_y_py:step_y_py,
                                 start_x:stop_x_py:step_x_py]

            var_data[i1:i2+1, j1:j2+1, k1:k2+1] = var

    var_data = np.squeeze(var_data)

    return var_data
