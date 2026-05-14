"""3D Parabolic grid generation main driver.

Usage (MPI):
    mpiexec -np N python para.py --config-file config.json --verbose 10
"""

import argparse
import ctypes
import os
import sys
import time
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from common.constants import X_DIRE, Y_DIRE, Z_DIRE, DIRE_MAP
from common.io_operations import coord_export
from common.grid_quality import grid_quality_check, cal_min_dist
from common.utils import permute_coord_x, permute_coord_y, flip_coord_z, flip_step_z
from parabolic.generate_grid.grid_data import cfgs_read, cfgs_print, grid_info_set, read_bdry_file
from parabolic.generate_grid.mympi import MyMPI1D


def _load_c_lib():
    """Load libgrid3d.so and set up parabolic function signatures."""
    lib_path = os.path.join(PROJECT_ROOT, "src_c", "libgrid3d.so")
    if not os.path.exists(lib_path):
        print(f"Error: C library not found at {lib_path}")
        print("Please run: cd src_c && bash build.sh")
        sys.exit(1)

    lib = ctypes.CDLL(lib_path)
    float_p = np.ctypeslib.ndpointer(dtype=np.float32)
    c_int = ctypes.c_int
    c_float = ctypes.c_float

    # predict_point_c
    lib.predict_point_c.argtypes = [
        float_p, float_p, float_p,  # x3d, y3d, z3d
        c_int, c_int, c_int,        # nx, ny, nz
        c_int, c_int, c_float,      # k, t2b, coef
        float_p,                     # step_len
    ]
    lib.predict_point_c.restype = None

    # update_point_c
    lib.update_point_c.argtypes = [
        float_p, float_p, float_p,  # x3d, y3d, z3d
        c_int, c_int, c_int,        # nx, ny, nz
        c_int,                       # k
    ]
    lib.update_point_c.restype = None

    return lib


def para_gene(lib, gdinfo, x3d, y3d, z3d, step, mympi, cfgs):
    """Main parabolic grid generation algorithm.

    Marches from k=1 to k=nz-2, predicting and correcting each layer.
    """
    nx = gdinfo['nx']
    ny = gdinfo['ny']
    nz = gdinfo['nz']
    coef = float(cfgs.get('coef', 40))
    t2b = cfgs.get('t2b', 0)
    verbose = cfgs.get('_verbose', 0)

    if mympi.is_root() and t2b == 1:
        print("march direction from top bdry(k=nz-1) to bottom bdry(k=1)")
        print("so we flip bdry, set top bdry to bottom bdry")

    if t2b == 1:
        # Flip in z-direction (use numpy views for efficiency)
        x3d[:] = x3d[::-1, :, :]
        y3d[:] = y3d[::-1, :, :]
        z3d[:] = z3d[::-1, :, :]
        flip_step_z(step)

    # Compute cumulative step length
    step_len = np.zeros(nz, dtype=np.float32)
    for k in range(1, nz):
        step_len[k] = step_len[k-1] + step[k-1]

    # Make arrays contiguous for C
    x3d_c = np.ascontiguousarray(x3d)
    y3d_c = np.ascontiguousarray(y3d)
    z3d_c = np.ascontiguousarray(z3d)

    for k in range(1, nz - 1):
        # Predict points at layers k and k+1
        lib.predict_point_c(x3d_c, y3d_c, z3d_c,
                            nx, ny, nz,
                            k, t2b, coef,
                            step_len)

        # Exchange ghost coordinates
        mympi.exchange_coord(gdinfo, x3d_c, y3d_c, z3d_c, k)

        # Update (correct) points at layer k
        lib.update_point_c(x3d_c, y3d_c, z3d_c,
                           nx, ny, nz,
                           k)

        # Exchange ghost coordinates after update
        mympi.exchange_coord(gdinfo, x3d_c, y3d_c, z3d_c, k)

        if mympi.is_root() and verbose > 0:
            print(f"number of layer is {k+1}")
            sys.stdout.flush()

    if t2b == 1:
        # Flip back
        x3d_c[:] = x3d_c[::-1, :, :]
        y3d_c[:] = y3d_c[::-1, :, :]
        z3d_c[:] = z3d_c[::-1, :, :]

    # Copy back
    x3d[:] = x3d_c
    y3d[:] = y3d_c
    z3d[:] = z3d_c


def main():
    parser = argparse.ArgumentParser(description='3D Parabolic Grid Generation')
    parser.add_argument('--config-file', required=True, help='JSON config file')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level')
    args = parser.parse_args()

    # Read config
    cfgs = cfgs_read(args.config_file)
    cfgs['_verbose'] = args.verbose

    # Setup MPI
    nproc_y = cfgs.get('number_of_mpiprocs', 1)
    mympi = MyMPI1D(nproc_y)

    if mympi.is_root():
        cfgs_print(cfgs)

    # Setup grid info
    gdinfo = grid_info_set(cfgs, mympi)

    if args.verbose > 0:
        print(f"[rank {mympi.myid}] nx={gdinfo['nx']} ny={gdinfo['ny']} nz={gdinfo['nz']} "
              f"ni={gdinfo['ni']} nj={gdinfo['nj']} nk={gdinfo['nk']} "
              f"gnj1={gdinfo['gnj1']}")

    # Read boundary data
    x3d, y3d, z3d, step = read_bdry_file(gdinfo, cfgs)

    # Set output filename
    dire_itype = gdinfo['dire_itype']
    if dire_itype == Z_DIRE:
        fname_part = f"px0_py{mympi.topoid[0]}_pz0"
    elif dire_itype == Y_DIRE:
        fname_part = f"px0_py0_pz{mympi.topoid[0]}"
    elif dire_itype == X_DIRE:
        fname_part = f"px0_py{mympi.topoid[0]}_pz0"

    # Load C library
    lib = _load_c_lib()

    # Generate grid
    t_start = time.time()
    para_gene(lib, gdinfo, x3d, y3d, z3d, step, mympi, cfgs)
    t_end = time.time()

    if mympi.is_root():
        print(f"\n************************************")
        print(f"grid generate running time: {t_end - t_start:.2f} s")
        print(f"************************************\n")

    # Permute coordinates back if direction was x or y
    if dire_itype == X_DIRE:
        # Create a temporary object with the arrays for permute
        _permute_x(x3d, y3d, z3d, gdinfo)
    elif dire_itype == Y_DIRE:
        _permute_y(x3d, y3d, z3d, gdinfo)

    # Export coordinates — match C version: include boundary ghost points
    # C code extends ni/nj/nk by 2 in x/z, and adds 1 ghost on each y side
    # only at MPI domain boundaries (neighid == MPI_PROC_NULL)
    ni_phys = gdinfo['ni']
    nj_phys = gdinfo['nj']
    nk_phys = gdinfo['nk']
    gni1_phys = gdinfo['gni1']
    gnj1_phys = gdinfo['gnj1']
    gnk1_phys = gdinfo['gnk1']

    ni = ni_phys
    nj = nj_phys
    nk = nk_phys
    gni1 = gni1_phys
    gnj1 = gnj1_phys
    gnk1 = gnk1_phys

    # Extend indices to include boundary points (matching C gd_curv_coord_export)
    if dire_itype == Z_DIRE or dire_itype == X_DIRE:
        ni = ni + 2
        nk = nk + 2
        if mympi.neighid[0] == -2:  # MPI_PROC_NULL = -2
            nj = nj + 1
        if mympi.neighid[1] == -2:
            nj = nj + 1
        if mympi.topoid[0] != 0:
            gnj1 = gnj1 + 1

    elif dire_itype == Y_DIRE:
        ni = ni + 2
        nj = nj + 2
        if mympi.neighid[0] == -2:
            nk = nk + 1
        if mympi.neighid[1] == -2:
            nk = nk + 1
        if mympi.topoid[0] != 0:
            gnk1 = gnk1 + 1

    output_dir = cfgs['grid_export_dir']
    os.makedirs(output_dir, exist_ok=True)
    ou_file = os.path.join(output_dir, f"coord_{fname_part}.nc")

    # Extract the export region (including boundary ghost points)
    # For Z_DIRE: i in [0:ni], j in [nj1-1:nj2+1], k in [0:nk]
    ni1_exp = 0
    ni2_exp = ni
    nj1_exp = 1 if mympi.neighid[0] != -2 else 0
    nj2_exp = gdinfo['nj'] + 1 if mympi.neighid[1] != -2 else gdinfo['nj'] + 2
    nk1_exp = 0
    nk2_exp = nk

    coord_x = x3d[nk1_exp:nk1_exp+nk, nj1_exp:nj1_exp+nj, ni1_exp:ni1_exp+ni].copy()
    coord_y = y3d[nk1_exp:nk1_exp+nk, nj1_exp:nj1_exp+nj, ni1_exp:ni1_exp+ni].copy()
    coord_z = z3d[nk1_exp:nk1_exp+nk, nj1_exp:nj1_exp+nj, ni1_exp:ni1_exp+ni].copy()

    from netCDF4 import Dataset
    with Dataset(ou_file, 'w', format='NETCDF4') as nc:
        nc.createDimension('k', nk)
        nc.createDimension('j', nj)
        nc.createDimension('i', ni)
        nc.createVariable('x', 'f4', ('k', 'j', 'i'))[:] = coord_x
        nc.createVariable('y', 'f4', ('k', 'j', 'i'))[:] = coord_y
        nc.createVariable('z', 'f4', ('k', 'j', 'i'))[:] = coord_z
        nc.global_index_of_first_physical_points = [gni1, gnj1, gnk1]
        nc.count_of_physical_points = [ni, nj, nk]

    if mympi.is_root():
        print(f"export coord to file: {ou_file}")

    # Calculate minimum distance
    # (simplified - using physical points only)
    print(f"[rank {mympi.myid}] Computing min dist...")

    # Grid quality check
    if cfgs.get('grid_check', 0) == 1:
        if mympi.is_root():
            print("******************************************************")
            print("***** grid quality check and export quality data *****")
            print("******************************************************")
        # Quality check on physical points
        from common.grid_data import SimpleGridData
        gd = SimpleGridData(ni_phys, nj_phys, nk_phys)
        gd.x3d = x3d[1:nk_phys+1, 1:nj_phys+1, 1:ni_phys+1].copy()
        gd.y3d = y3d[1:nk_phys+1, 1:nj_phys+1, 1:ni_phys+1].copy()
        gd.z3d = z3d[1:nk_phys+1, 1:nj_phys+1, 1:ni_phys+1].copy()
        gd.gni1 = gni1_phys
        gd.gnj1 = gnj1_phys
        gd.gnk1 = gnk1_phys
        gd.fname_part = fname_part
        grid_quality_check(gd, cfgs)


def _permute_x(x3d, y3d, z3d, gdinfo):
    """Permute coordinates back from x-marching to original layout."""
    nx, ny, nz = gdinfo['nx'], gdinfo['ny'], gdinfo['nz']
    x_old = x3d.copy()
    y_old = y3d.copy()
    z_old = z3d.copy()
    # After generation, marching was along z (which was originally x)
    # x_new = original z, z_new = original x
    x3d[:] = np.transpose(z_old, (2, 1, 0))
    y3d[:] = np.transpose(y_old, (2, 1, 0))
    z3d[:] = np.transpose(x_old, (2, 1, 0))
    gdinfo['nx'], gdinfo['nz'] = nz, nx
    gdinfo['ni'], gdinfo['nk'] = gdinfo['nk'], gdinfo['ni']
    gdinfo['gni1'], gdinfo['gnk1'] = gdinfo['gnk1'], gdinfo['gni1']


def _permute_y(x3d, y3d, z3d, gdinfo):
    """Permute coordinates back from y-marching to original layout."""
    nx, ny, nz = gdinfo['nx'], gdinfo['ny'], gdinfo['nz']
    x_old = x3d.copy()
    y_old = y3d.copy()
    z_old = z3d.copy()
    # After generation, marching was along z (which was originally y)
    x3d[:] = np.transpose(x_old, (1, 0, 2))
    y3d[:] = np.transpose(z_old, (1, 0, 2))
    z3d[:] = np.transpose(y_old, (1, 0, 2))
    gdinfo['ny'], gdinfo['nz'] = nz, ny
    gdinfo['nj'], gdinfo['nk'] = gdinfo['nk'], gdinfo['nj']
    gdinfo['gnj1'], gdinfo['gnk1'] = gdinfo['gnk1'], gdinfo['gnj1']


if __name__ == '__main__':
    main()
