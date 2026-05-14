"""NetCDF I/O operations for 3D grid generation."""

from netCDF4 import Dataset
import os
import sys
import numpy as np


def coord_export(gd, output_dir):
    """Export grid coordinates to single NetCDF file (single-partition mode)."""
    ni = gd.ni
    nj = gd.nj
    nk = gd.nk
    gni1 = gd.gni1
    gnj1 = gd.gnj1
    gnk1 = gd.gnk1
    fname_part = gd.fname_part

    ou_file = os.path.join(output_dir, f"coord_{fname_part}.nc")

    with Dataset(ou_file, 'w', format='NETCDF4') as nc:
        nc.createDimension('k', nk)
        nc.createDimension('j', nj)
        nc.createDimension('i', ni)

        x_var = nc.createVariable('x', 'f4', ('k', 'j', 'i'))
        y_var = nc.createVariable('y', 'f4', ('k', 'j', 'i'))
        z_var = nc.createVariable('z', 'f4', ('k', 'j', 'i'))

        nc.global_index_of_first_physical_points = [gni1, gnj1, gnk1]
        nc.count_of_physical_points = [ni, nj, nk]

        x_var[:, :, :] = gd.x3d
        y_var[:, :, :] = gd.y3d
        z_var[:, :, :] = gd.z3d


def quality_export(gd, var, output_dir, var_name):
    """Export quality data to single NetCDF file (single-partition mode)."""
    ni = gd.ni
    nj = gd.nj
    nk = gd.nk
    gni1 = gd.gni1
    gnj1 = gd.gnj1
    gnk1 = gd.gnk1
    fname_part = gd.fname_part

    ou_file = os.path.join(output_dir, f"{var_name}_{fname_part}.nc")

    with Dataset(ou_file, 'w', format='NETCDF4') as nc:
        nc.createDimension('k', nk)
        nc.createDimension('j', nj)
        nc.createDimension('i', ni)

        var_out = nc.createVariable(var_name, 'f4', ('k', 'j', 'i'))

        nc.global_index_of_first_physical_points = [gni1, gnj1, gnk1]
        nc.count_of_physical_points = [ni, nj, nk]

        var_out[:, :, :] = var


def coord_export_mpi(gd, cfgs):
    """Export grid coordinates to MPI-partitioned NetCDF files."""
    nprocx_out = cfgs['number_of_mpiprocs_out'][0]
    nprocy_out = cfgs['number_of_mpiprocs_out'][1]
    nprocz_out = cfgs['number_of_mpiprocs_out'][2]
    total_nx = gd.nx
    total_ny = gd.ny
    total_nz = gd.nz

    os.makedirs(cfgs['grid_export_dir'], exist_ok=True)

    global_index = [0, 0, 0]
    count = [0, 0, 0]
    for kk in range(nprocz_out):
        for jj in range(nprocy_out):
            for ii in range(nprocx_out):
                fname_coords = f"px{ii}_py{jj}_pz{kk}"
                ou_file = os.path.join(cfgs['grid_export_dir'], f"coord_{fname_coords}.nc")

                gd_info_set(cfgs, total_nx, total_ny, total_nz,
                            ii, jj, kk, global_index, count)
                gni1, gnj1, gnk1 = global_index
                ni, nj, nk = count

                coord_x = gd.x3d[gnk1:gnk1+nk, gnj1:gnj1+nj, gni1:gni1+ni].astype(np.float32)
                coord_y = gd.y3d[gnk1:gnk1+nk, gnj1:gnj1+nj, gni1:gni1+ni].astype(np.float32)
                coord_z = gd.z3d[gnk1:gnk1+nk, gnj1:gnj1+nj, gni1:gni1+ni].astype(np.float32)

                with Dataset(ou_file, 'w', format='NETCDF4') as ncid:
                    ncid.createDimension('k', nk)
                    ncid.createDimension('j', nj)
                    ncid.createDimension('i', ni)

                    x_var = ncid.createVariable('x', np.float32, ('k', 'j', 'i'))
                    y_var = ncid.createVariable('y', np.float32, ('k', 'j', 'i'))
                    z_var = ncid.createVariable('z', np.float32, ('k', 'j', 'i'))

                    ncid.setncattr("global_index_of_first_physical_points",
                                   np.array(global_index, dtype=np.int32))
                    ncid.setncattr("count_of_physical_points",
                                   np.array(count, dtype=np.int32))

                    x_var[:] = coord_x
                    y_var[:] = coord_y
                    z_var[:] = coord_z


def quality_export_mpi(var_in, gd, cfgs, var_name):
    """Export quality data to MPI-partitioned NetCDF files."""
    nprocx_out = cfgs['number_of_mpiprocs_out'][0]
    nprocy_out = cfgs['number_of_mpiprocs_out'][1]
    nprocz_out = cfgs['number_of_mpiprocs_out'][2]
    total_nx = gd.nx
    total_ny = gd.ny
    total_nz = gd.nz

    os.makedirs(cfgs['grid_export_dir'], exist_ok=True)

    global_index = [0, 0, 0]
    count = [0, 0, 0]
    for kk in range(nprocz_out):
        for jj in range(nprocy_out):
            for ii in range(nprocx_out):
                fname_coords = f"px{ii}_py{jj}_pz{kk}"
                ou_file = os.path.join(cfgs['grid_export_dir'],
                                       f"{var_name}_{fname_coords}.nc")

                gd_info_set(cfgs, total_nx, total_ny, total_nz,
                            ii, jj, kk, global_index, count)
                gni1, gnj1, gnk1 = global_index
                ni, nj, nk = count

                var = var_in[gnk1:gnk1+nk, gnj1:gnj1+nj, gni1:gni1+ni].astype(np.float32)

                with Dataset(ou_file, 'w', format='NETCDF4') as ncid:
                    ncid.createDimension('k', nk)
                    ncid.createDimension('j', nj)
                    ncid.createDimension('i', ni)

                    var_out = ncid.createVariable(var_name, np.float32, ('k', 'j', 'i'))

                    ncid.setncattr("global_index_of_first_physical_points",
                                   np.array(global_index, dtype=np.int32))
                    ncid.setncattr("count_of_physical_points",
                                   np.array(count, dtype=np.int32))

                    var_out[:] = var


def gd_info_set(cfgs, total_nx, total_ny, total_nz,
                iprocx, iprocy, iprocz, global_index, count):
    """Calculate global starting indices and data dimensions for a specific process."""
    nprocx_out = cfgs['number_of_mpiprocs_out'][0]
    nprocy_out = cfgs['number_of_mpiprocs_out'][1]
    nprocz_out = cfgs['number_of_mpiprocs_out'][2]

    if cfgs.get('pml_weight_2x', 0) == 1:
        pml_x1 = cfgs['number_of_pml_x1']
        pml_x2 = cfgs['number_of_pml_x2']
        pml_y1 = cfgs['number_of_pml_y1']
        pml_y2 = cfgs['number_of_pml_y2']
        pml_z1 = cfgs['number_of_pml_z1']
        pml_z2 = cfgs['number_of_pml_z2']
    else:
        pml_x1 = pml_x2 = pml_y1 = pml_y2 = pml_z1 = pml_z2 = 0

    # X-direction partitioning
    ni = _calc_partition_size(total_nx, pml_x1, pml_x2, nprocx_out, iprocx)
    gni1 = _calc_partition_start(total_nx, pml_x1, pml_x2, nprocx_out, iprocx)

    # Y-direction partitioning
    nj = _calc_partition_size(total_ny, pml_y1, pml_y2, nprocy_out, iprocy)
    gnj1 = _calc_partition_start(total_ny, pml_y1, pml_y2, nprocy_out, iprocy)

    # Z-direction partitioning
    nk = _calc_partition_size(total_nz, pml_z1, pml_z2, nprocz_out, iprocz)
    gnk1 = _calc_partition_start(total_nz, pml_z1, pml_z2, nprocz_out, iprocz)

    global_index[0] = gni1
    global_index[1] = gnj1
    global_index[2] = gnk1
    count[0] = ni
    count[1] = nj
    count[2] = nk


def _calc_partition_size(total_n, pml_lo, pml_hi, nproc, iproc):
    n_et = total_n + pml_lo + pml_hi
    n_avg = n_et // nproc
    if n_avg <= pml_lo or n_avg <= pml_hi:
        print(f"Error: n_avg={n_avg} must be larger than PML ({pml_lo}, {pml_hi})")
        sys.exit(1)
    ni = n_avg
    if iproc == 0:
        ni -= pml_lo
    if iproc == nproc - 1:
        ni -= pml_hi
    n_left = n_et % nproc
    if iproc < n_left:
        ni += 1
    return ni


def _calc_partition_start(total_n, pml_lo, pml_hi, nproc, iproc):
    n_et = total_n + pml_lo + pml_hi
    n_avg = n_et // nproc
    n_left = n_et % nproc
    if iproc == 0:
        return 0
    gni1 = iproc * n_avg - pml_lo
    if n_left != 0:
        gni1 += iproc if iproc < n_left else n_left
    return gni1


def coord_import(import_dir, nx, ny, nz, nproc_x, nproc_y, nproc_z):
    """Read MPI-partitioned NetCDF files and assemble a global grid.

    Returns: x3d, y3d, z3d arrays of shape (nz, ny, nx).
    """
    x3d = np.zeros((nz, ny, nx), dtype=np.float32)
    y3d = np.zeros((nz, ny, nx), dtype=np.float32)
    z3d = np.zeros((nz, ny, nx), dtype=np.float32)

    for ipz in range(nproc_z):
        for ipy in range(nproc_y):
            for ipx in range(nproc_x):
                fname = f"coord_px{ipx}_py{ipy}_pz{ipz}.nc"
                fpath = os.path.join(import_dir, fname)
                if not os.path.exists(fpath):
                    print(f"Warning: {fpath} not found, skipping")
                    continue
                with Dataset(fpath, 'r') as nc:
                    gni1 = int(nc.global_index_of_first_physical_points[0])
                    gnj1 = int(nc.global_index_of_first_physical_points[1])
                    gnk1 = int(nc.global_index_of_first_physical_points[2])
                    ni = int(nc.count_of_physical_points[0])
                    nj = int(nc.count_of_physical_points[1])
                    nk = int(nc.count_of_physical_points[2])
                    x3d[gnk1:gnk1+nk, gnj1:gnj1+nj, gni1:gni1+ni] = nc.variables['x'][:]
                    y3d[gnk1:gnk1+nk, gnj1:gnj1+nj, gni1:gni1+ni] = nc.variables['y'][:]
                    z3d[gnk1:gnk1+nk, gnj1:gnj1+nj, gni1:gni1+ni] = nc.variables['z'][:]
    return x3d, y3d, z3d
