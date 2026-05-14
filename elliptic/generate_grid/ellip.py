"""3D Elliptic grid generation main driver.

Usage (MPI):
    mpiexec -np N python ellip.py --config-file config.json --verbose 10
"""

import argparse
import ctypes
import os
import sys
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from common.constants import LINEAR_TFI, ELLI_DIRI, ELLI_HIGEN
from common.grid_quality import grid_quality_check, cal_min_dist
from common.io_operations import coord_export_mpi
from elliptic.generate_grid.grid_data import cfgs_read, cfgs_print, grid_info_set, read_bdry_file
from elliptic.generate_grid.mympi import MyMPI3D
from elliptic.generate_grid.grid_utils import linear_tfi, check_bdry
from elliptic.generate_grid.dirichlet import diri_gene
from elliptic.generate_grid.higenstock import higen_gene


def _load_c_lib():
    """Load libgrid3d.so and set up elliptic function signatures."""
    lib_path = os.path.join(PROJECT_ROOT, "src_c", "libgrid3d.so")
    if not os.path.exists(lib_path):
        print(f"Error: C library not found at {lib_path}")
        print("Please run: cd src_c && bash build.sh")
        sys.exit(1)

    lib = ctypes.CDLL(lib_path)
    float_p = np.ctypeslib.ndpointer(dtype=np.float32)
    int_p = np.ctypeslib.ndpointer(dtype=np.int32)
    c_int = ctypes.c_int
    c_float = ctypes.c_float

    # update_SOR_c
    lib.update_SOR_c.argtypes = [
        float_p, float_p, float_p,  # x3d, y3d, z3d
        float_p, float_p, float_p,  # x3d_tmp, y3d_tmp, z3d_tmp
        c_int, c_int, c_int,        # nx, ny, nz
        float_p, float_p, float_p,  # P, Q, R
        c_float,                     # w
    ]
    lib.update_SOR_c.restype = None

    # linear_tfi_c
    lib.linear_tfi_c.argtypes = [
        float_p, float_p, float_p,  # x3d, y3d, z3d
        float_p, float_p,           # x1, x2
        float_p, float_p,           # y1, y2
        float_p, float_p,           # z1, z2
        c_int, c_int, c_int,        # nx, ny, nz
        c_int, c_int, c_int, c_int,  # ni1, ni2, nj1, nj2
        c_int, c_int,               # nk1, nk2
        c_int, c_int, c_int,        # gni1, gnj1, gnk1
        c_int, c_int, c_int,        # total_nx, total_ny, total_nz
        int_p,                       # neighid
    ]
    lib.linear_tfi_c.restype = None

    # ghost_cal_c
    lib.ghost_cal_c.argtypes = [
        float_p, float_p, float_p,  # x3d, y3d, z3d
        c_int, c_int, c_int,        # nx, ny, nz
        float_p, float_p, float_p,  # p_x, p_y, p_z
        float_p, float_p, float_p,  # g11_x, g22_y, g33_z
        int_p,                       # neighid
    ]
    lib.ghost_cal_c.restype = None

    # interp_inner_source_c
    lib.interp_inner_source_c.argtypes = [
        float_p, float_p, float_p,  # P, Q, R
        c_int, c_int, c_int,        # nx, ny, nz
        float_p, float_p, float_p,  # P_x1, Q_x1, R_x1
        float_p, float_p, float_p,  # P_x2, Q_x2, R_x2
        float_p, float_p, float_p,  # P_y1, Q_y1, R_y1
        float_p, float_p, float_p,  # P_y2, Q_y2, R_y2
        float_p, float_p, float_p,  # P_z1, Q_z1, R_z1
        float_p, float_p, float_p,  # P_z2, Q_z2, R_z2
        c_int, c_int, c_int,        # gni1, gnj1, gnk1
        c_int, c_int, c_int,        # total_nx, total_ny, total_nz
        float_p,                     # coef
        c_int,                       # n_coef
    ]
    lib.interp_inner_source_c.restype = None

    # compute_residual_c
    lib.compute_residual_c.argtypes = [
        float_p, float_p, float_p,  # x3d, y3d, z3d
        float_p, float_p, float_p,  # x3d_tmp, y3d_tmp, z3d_tmp
        c_int, c_int, c_int,        # nx, ny, nz
        float_p,                     # local_max
    ]
    lib.compute_residual_c.restype = None

    # set_src_diri_c
    lib.set_src_diri_c.argtypes = [
        float_p, float_p, float_p,  # x3d, y3d, z3d
        c_int, c_int, c_int,        # nx, ny, nz
        c_int, c_int, c_int,        # ni, nj, nk
        float_p, float_p, float_p,  # P, Q, R
        float_p, float_p, float_p,  # P_x1, Q_x1, R_x1
        float_p, float_p, float_p,  # P_x2, Q_x2, R_x2
        float_p, float_p, float_p,  # P_y1, Q_y1, R_y1
        float_p, float_p, float_p,  # P_y2, Q_y2, R_y2
        float_p, float_p, float_p,  # P_z1, Q_z1, R_z1
        float_p, float_p, float_p,  # P_z2, Q_z2, R_z2
        float_p, float_p, float_p,  # P_x1_loc, Q_x1_loc, R_x1_loc
        float_p, float_p, float_p,  # P_x2_loc, Q_x2_loc, R_x2_loc
        float_p, float_p, float_p,  # P_y1_loc, Q_y1_loc, R_y1_loc
        float_p, float_p, float_p,  # P_y2_loc, Q_y2_loc, R_y2_loc
        float_p, float_p, float_p,  # P_z1_loc, Q_z1_loc, R_z1_loc
        float_p, float_p, float_p,  # P_z2_loc, Q_z2_loc, R_z2_loc
        float_p, float_p, float_p,  # p_x, p_y, p_z
        float_p, float_p, float_p,  # g11_x, g22_y, g33_z
        int_p,                       # neighid
        c_int, c_int, c_int,        # gni1, gnj1, gnk1
        c_int, c_int, c_int,        # total_nx, total_ny, total_nz
    ]
    lib.set_src_diri_c.restype = None

    # dist_cal_c
    lib.dist_cal_c.argtypes = [
        float_p, float_p, float_p,  # x3d, y3d, z3d
        c_int, c_int, c_int,        # nx, ny, nz
        float_p, float_p,           # dx1, dx2
        float_p, float_p,           # dy1, dy2
        float_p, float_p,           # dz1, dz2
        int_p,                       # neighid
    ]
    lib.dist_cal_c.restype = None

    # set_src_higen_c
    lib.set_src_higen_c.argtypes = [
        float_p, float_p, float_p,  # x3d, y3d, z3d
        c_int, c_int, c_int,        # nx, ny, nz
        c_int, c_int, c_int,        # ni, nj, nk
        float_p, float_p, float_p,  # P, Q, R
        float_p, float_p, float_p,  # P_x1, Q_x1, R_x1
        float_p, float_p, float_p,  # P_x2, Q_x2, R_x2
        float_p, float_p, float_p,  # P_y1, Q_y1, R_y1
        float_p, float_p, float_p,  # P_y2, Q_y2, R_y2
        float_p, float_p, float_p,  # P_z1, Q_z1, R_z1
        float_p, float_p, float_p,  # P_z2, Q_z2, R_z2
        float_p, float_p, float_p,  # P_x1_loc, Q_x1_loc, R_x1_loc
        float_p, float_p, float_p,  # P_x2_loc, Q_x2_loc, R_x2_loc
        float_p, float_p, float_p,  # P_y1_loc, Q_y1_loc, R_y1_loc
        float_p, float_p, float_p,  # P_y2_loc, Q_y2_loc, R_y2_loc
        float_p, float_p, float_p,  # P_z1_loc, Q_z1_loc, R_z1_loc
        float_p, float_p, float_p,  # P_z2_loc, Q_z2_loc, R_z2_loc
        float_p, float_p,           # dx1, dx2
        float_p, float_p,           # dy1, dy2
        float_p, float_p,           # dz1, dz2
        int_p,                       # neighid
        c_int, c_int, c_int,        # gni1, gnj1, gnk1
        c_int, c_int, c_int,        # total_nx, total_ny, total_nz
    ]
    lib.set_src_higen_c.restype = None

    return lib


def main():
    parser = argparse.ArgumentParser(description='3D Elliptic Grid Generation')
    parser.add_argument('--config-file', required=True, help='JSON config file')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level')
    args = parser.parse_args()

    # Read config
    cfgs = cfgs_read(args.config_file)
    cfgs['_verbose'] = args.verbose

    # Setup MPI
    nproc_x = cfgs.get('number_of_mpiprocs_x', 1)
    nproc_y = cfgs.get('number_of_mpiprocs_y', 1)
    nproc_z = cfgs.get('number_of_mpiprocs_z', 1)
    mympi = MyMPI3D(nproc_x, nproc_y, nproc_z)

    if mympi.is_root():
        cfgs_print(cfgs)

    # Setup grid info
    gdinfo = grid_info_set(cfgs, mympi)

    if args.verbose > 0:
        print(f"[rank {mympi.myid}] nx={gdinfo['nx']} ny={gdinfo['ny']} nz={gdinfo['nz']} "
              f"ni={gdinfo['ni']} nj={gdinfo['nj']} nk={gdinfo['nk']}")

    # Read boundary data
    bdry = read_bdry_file(gdinfo, cfgs)

    # Check boundary consistency
    if mympi.is_root():
        check_bdry(bdry, gdinfo['total_nx'], gdinfo['total_ny'], gdinfo['total_nz'])

    # Allocate coordinate arrays
    nx = gdinfo['nx']
    ny = gdinfo['ny']
    nz = gdinfo['nz']
    x3d = np.zeros((nz, ny, nx), dtype=np.float32)
    y3d = np.zeros((nz, ny, nx), dtype=np.float32)
    z3d = np.zeros((nz, ny, nx), dtype=np.float32)

    # Load C library
    lib = _load_c_lib()

    method_itype = cfgs.get('method_itype', LINEAR_TFI)

    # TFI initialization
    if mympi.is_root():
        print("Initializing grid with linear TFI...")
    linear_tfi(lib, gdinfo, x3d, y3d, z3d, bdry, mympi)
    mympi.exchange_ghost(gdinfo, x3d, y3d, z3d)

    # Run selected method
    t_start = time.time()

    if method_itype == ELLI_DIRI:
        if mympi.is_root():
            print("Running Dirichlet elliptic method...")
        diri_gene(lib, gdinfo, x3d, y3d, z3d, bdry, mympi, cfgs)
    elif method_itype == ELLI_HIGEN:
        if mympi.is_root():
            print("Running Higenstock elliptic method...")
        higen_gene(lib, gdinfo, x3d, y3d, z3d, bdry, mympi, cfgs)
    else:
        if mympi.is_root():
            print("Using TFI only (no elliptic iteration)")

    t_end = time.time()

    if mympi.is_root():
        print(f"\n************************************")
        print(f"grid generate running time: {t_end - t_start:.2f} s")
        print(f"************************************\n")

    # Set output filename
    fname_part = f"px{mympi.topoid[0]}_py{mympi.topoid[1]}_pz{mympi.topoid[2]}"

    # Export coordinates
    ni = gdinfo['ni']
    nj = gdinfo['nj']
    nk = gdinfo['nk']
    gni1 = gdinfo['gni1']
    gnj1 = gdinfo['gnj1']
    gnk1 = gdinfo['gnk1']

    output_dir = cfgs['grid_export_dir']
    os.makedirs(output_dir, exist_ok=True)
    ou_file = os.path.join(output_dir, f"coord_{fname_part}.nc")

    from netCDF4 import Dataset
    with Dataset(ou_file, 'w', format='NETCDF4') as nc:
        nc.createDimension('k', nk)
        nc.createDimension('j', nj)
        nc.createDimension('i', ni)
        nc.createVariable('x', 'f4', ('k', 'j', 'i'))[:] = x3d[1:nk+1, 1:nj+1, 1:ni+1]
        nc.createVariable('y', 'f4', ('k', 'j', 'i'))[:] = y3d[1:nk+1, 1:nj+1, 1:ni+1]
        nc.createVariable('z', 'f4', ('k', 'j', 'i'))[:] = z3d[1:nk+1, 1:nj+1, 1:ni+1]
        nc.global_index_of_first_physical_points = [gni1, gnj1, gnk1]
        nc.count_of_physical_points = [ni, nj, nk]

    if mympi.is_root():
        print(f"export coord to file: {ou_file}")

    # Grid quality check
    if cfgs.get('grid_check', 0) == 1:
        if mympi.is_root():
            print("******************************************************")
            print("***** grid quality check and export quality data *****")
            print("******************************************************")
        from common.grid_data import SimpleGridData
        gd = SimpleGridData(ni, nj, nk)
        gd.x3d = x3d[1:nk+1, 1:nj+1, 1:ni+1].copy()
        gd.y3d = y3d[1:nk+1, 1:nj+1, 1:ni+1].copy()
        gd.z3d = z3d[1:nk+1, 1:nj+1, 1:ni+1].copy()
        gd.gni1 = gni1
        gd.gnj1 = gnj1
        gd.gnk1 = gnk1
        gd.fname_part = fname_part
        grid_quality_check(gd, cfgs)


if __name__ == '__main__':
    main()
