"""3D grid I/O operations: import, merge, and export for post-processing."""

import sys
import os
import gc
import numpy as np
import netCDF4 as nc
from common.grid_data import SimpleGridData
from common.algebra import xi_arc_stretch, et_arc_stretch, zt_arc_stretch
from common.utils import print_quality_checks


def read_import_coord(cfgs):
    """Read input grid coordinates from MPI-partitioned NetCDF files.

    Handles multiple input grids, arc-length stretching, and merging.
    Returns a SimpleGridData with the assembled global grid.
    """
    grid_count = cfgs['input_grid_number']
    if grid_count < 1:
        raise ValueError(f"input_grid_number must be >= 1, got {grid_count}")

    if grid_count >= 2:
        if 'merge_direction' not in cfgs:
            raise KeyError("Missing 'merge_direction' for multi-grid merge")
        merge_dir = cfgs['merge_direction']
        if merge_dir not in ['x', 'y', 'z']:
            raise ValueError(f"merge_direction must be 'x'/'y'/'z', got {merge_dir}")

    gdcurv_dict = {}
    for i in range(grid_count):
        grid_key = f"input_grid_info_{i}"
        nx = cfgs[grid_key]['number_of_grid_points'][0]
        ny = cfgs[grid_key]['number_of_grid_points'][1]
        nz = cfgs[grid_key]['number_of_grid_points'][2]
        nprocx_in = cfgs[grid_key]['number_of_mpiprocs_in'][0]
        nprocy_in = cfgs[grid_key]['number_of_mpiprocs_in'][1]
        nprocz_in = cfgs[grid_key]['number_of_mpiprocs_in'][2]
        import_dir = cfgs[grid_key]['grid_import_dir']

        gdcurv = SimpleGridData(nx, ny, nz)
        gdcurv_dict[f"gdcurv_{i}"] = gdcurv

        att_global = "global_index_of_first_physical_points"
        att_count = "count_of_physical_points"

        for kk in range(nprocz_in):
            for jj in range(nprocy_in):
                for ii in range(nprocx_in):
                    fname_coords = f"px{ii}_py{jj}_pz{kk}"
                    in_file = f"{import_dir}/coord_{fname_coords}.nc"
                    try:
                        with nc.Dataset(in_file, 'r') as ds:
                            global_index = ds.getncattr(att_global)
                            count_points = ds.getncattr(att_count)

                            gni1 = global_index[0]
                            gnj1 = global_index[1]
                            gnk1 = global_index[2]
                            ni = count_points[0]
                            nj = count_points[1]
                            nk = count_points[2]

                            coord_x = ds.variables['x'][:].astype(np.float32)
                            coord_y = ds.variables['y'][:].astype(np.float32)
                            coord_z = ds.variables['z'][:].astype(np.float32)

                            global_k = slice(gnk1, gnk1 + nk)
                            global_j = slice(gnj1, gnj1 + nj)
                            global_i = slice(gni1, gni1 + ni)

                            gdcurv.x3d[global_k, global_j, global_i] = coord_x
                            gdcurv.y3d[global_k, global_j, global_i] = coord_y
                            gdcurv.z3d[global_k, global_j, global_i] = coord_z

                    except FileNotFoundError:
                        raise FileNotFoundError(f"NC file not found: {in_file}")
                    except Exception as e:
                        raise RuntimeError(f"Failed to read NC file {in_file}: {e}")

        # Arc-length stretching
        if cfgs[grid_key].get('flag_stretch', 0) == 1:
            stretch_file = cfgs[grid_key]['stretch_file']
            stretch_dire = cfgs[grid_key]['stretch_direction']

            with open(stretch_file, 'r') as fp:
                lines = [line.strip() for line in fp
                         if line.strip() and not line.strip().startswith('#')]

            first_val = int(float(lines[0]))
            if first_val == len(lines) - 1:
                npoints = first_val
                arc_len = np.array([float(v) for v in lines[1:npoints+1]], dtype=np.float32)
            else:
                arc_len = np.array([float(v) for v in lines], dtype=np.float32)

            if stretch_dire == 'z':
                zt_arc_stretch(gdcurv, arc_len)
            elif stretch_dire == 'x':
                xi_arc_stretch(gdcurv, arc_len)
            elif stretch_dire == 'y':
                et_arc_stretch(gdcurv, arc_len)

    if grid_count == 1:
        return gdcurv_dict["gdcurv_0"]

    # Merge grids
    total_nx = 0
    total_ny = 0
    total_nz = 0
    if cfgs['merge_direction'] == 'x':
        for i in range(grid_count):
            grid_key = f"input_grid_info_{i}"
            total_nx += cfgs[grid_key]['number_of_grid_points'][0]
            total_ny = cfgs[grid_key]['number_of_grid_points'][1]
            total_nz = cfgs[grid_key]['number_of_grid_points'][2]
        total_nx = total_nx - grid_count + 1
    elif cfgs['merge_direction'] == 'y':
        for i in range(grid_count):
            grid_key = f"input_grid_info_{i}"
            total_nx = cfgs[grid_key]['number_of_grid_points'][0]
            total_ny += cfgs[grid_key]['number_of_grid_points'][1]
            total_nz = cfgs[grid_key]['number_of_grid_points'][2]
        total_ny = total_ny - grid_count + 1
    elif cfgs['merge_direction'] == 'z':
        for i in range(grid_count):
            grid_key = f"input_grid_info_{i}"
            total_nx = cfgs[grid_key]['number_of_grid_points'][0]
            total_ny = cfgs[grid_key]['number_of_grid_points'][1]
            total_nz += cfgs[grid_key]['number_of_grid_points'][2]
        total_nz = total_nz - grid_count + 1

    merge_gd = SimpleGridData(total_nx, total_ny, total_nz)
    gni1 = 0
    gnj1 = 0
    gnk1 = 0

    for i in range(grid_count):
        gd_in = gdcurv_dict[f"gdcurv_{i}"]
        nx_in = gd_in.nx
        ny_in = gd_in.ny
        nz_in = gd_in.nz

        if cfgs['merge_direction'] == 'x' and i > 0:
            gni1 -= 1
        elif cfgs['merge_direction'] == 'y' and i > 0:
            gnj1 -= 1
        elif cfgs['merge_direction'] == 'z' and i > 0:
            gnk1 -= 1

        k_slice = slice(gnk1, gnk1 + nz_in)
        j_slice = slice(gnj1, gnj1 + ny_in)
        i_slice = slice(gni1, gni1 + nx_in)

        merge_gd.x3d[k_slice, j_slice, i_slice] = gd_in.x3d
        merge_gd.y3d[k_slice, j_slice, i_slice] = gd_in.y3d
        merge_gd.z3d[k_slice, j_slice, i_slice] = gd_in.z3d

        if cfgs['merge_direction'] == 'x':
            gni1 += nx_in
        elif cfgs['merge_direction'] == 'y':
            gnj1 += ny_in
        elif cfgs['merge_direction'] == 'z':
            gnk1 += nz_in

        # Free memory
        gd_in.x3d = None
        gd_in.y3d = None
        gd_in.z3d = None
        del gdcurv_dict[f"gdcurv_{i}"]
        gc.collect()

    return merge_gd


def cfgs_print(cfgs):
    """Print post-processing configuration."""
    print("-" * 20 + " input grid info " + "-" * 20)
    print(f"input_grid_number is {cfgs['input_grid_number']}")
    for i in range(cfgs['input_grid_number']):
        grid_key = f"input_grid_info_{i}"
        print(f"Grid {i}:")
        print(f"  number_of_grid_points = {cfgs[grid_key]['number_of_grid_points']}")
        print(f"  number_of_mpiprocs_in = {cfgs[grid_key]['number_of_mpiprocs_in']}")
        print(f"  grid_import_dir = {cfgs[grid_key]['grid_import_dir']}")
        if cfgs[grid_key].get('flag_stretch', 0) == 1:
            print(f"  stretch_direction = {cfgs[grid_key]['stretch_direction']}")

    print("-" * 20 + " output grid info " + "-" * 20)
    print(f"number_of_mpiprocs_out = {cfgs.get('number_of_mpiprocs_out', [1,1,1])}")
    if cfgs['input_grid_number'] >= 2:
        print(f"merge_direction = {cfgs['merge_direction']}")
    if cfgs.get('flag_sample', 0) == 1:
        print(f"sample_factor = {cfgs.get('sample_factor', [1,1,1])}")
    print(f"grid_export_dir = {cfgs['grid_export_dir']}")

    print_quality_checks(cfgs)

    if cfgs.get('pml_weight_2x', 0) == 1:
        for d in ['x1', 'x2', 'y1', 'y2', 'z1', 'z2']:
            key = f'number_of_pml_{d}'
            if key in cfgs:
                print(f"{key} = {cfgs[key]}")
