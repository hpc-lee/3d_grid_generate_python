"""3D Hyperbolic grid generation main driver.

Usage:
    python hyper.py --config-file config.json --verbose 10
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
from hyperbolic.generate_grid.grid_data import cfgs_read, cfgs_print, grid_info_set, read_bdry_file


def _load_c_lib():
    """Load libgrid3d.so and set up hyperbolic function signature."""
    lib_path = os.path.join(PROJECT_ROOT, "src_c", "libgrid3d.so")
    if not os.path.exists(lib_path):
        print(f"Error: C library not found at {lib_path}")
        print("Please run: cd src_c && bash build.sh")
        sys.exit(1)

    lib = ctypes.CDLL(lib_path)
    float_p = np.ctypeslib.ndpointer(dtype=np.float32)
    c_int = ctypes.c_int
    c_float = ctypes.c_float

    # hyper_gene_c
    lib.hyper_gene_c.argtypes = [
        float_p, float_p, float_p,  # x3d, y3d, z3d
        float_p,                     # step
        c_int, c_int, c_int,        # nx, ny, nz
        c_float,                     # coef
        c_int,                       # t2b
        c_int,                       # flag_stretch
        c_int,                       # bdry_x_itype
        c_int,                       # bdry_y_itype
        c_float,                     # epsilon_x
        c_float,                     # epsilon_y
    ]
    lib.hyper_gene_c.restype = None

    return lib


def hyper_gene(lib, gdinfo, x3d, y3d, z3d, step, cfgs):
    """Main hyperbolic grid generation algorithm.

    Calls hyper_gene_c which marches from k=1 to k=nz-1.
    """
    nx = gdinfo['nx']
    ny = gdinfo['ny']
    nz = gdinfo['nz']
    coef = float(cfgs.get('coef', 40))
    t2b = cfgs.get('t2b', 0)
    flag_stretch = cfgs.get('flag_stretch', 0)
    bdry_x_itype = cfgs.get('bdry_x_itype', cfgs.get('bdry_x_type', 0))
    bdry_y_itype = cfgs.get('bdry_y_itype', cfgs.get('bdry_y_type', 0))
    epsilon_x = float(cfgs.get('epsilon_x', 0.0))
    epsilon_y = float(cfgs.get('epsilon_y', 0.0))
    verbose = cfgs.get('_verbose', 0)

    if verbose > 0 and t2b == 1:
        print("march direction from top bdry(k=nz-1) to bottom bdry(k=1)")
        print("so we flip bdry, set top bdry to bottom bdry")

    if t2b == 1:
        # We'll let hyper_gene_c handle flipping, but we need to pass the step correctly
        # Wait, let's see what hyper_gene_c does - it calls flip_coord_z internally
        pass

    # Make arrays contiguous for C
    x3d_c = np.ascontiguousarray(x3d)
    y3d_c = np.ascontiguousarray(y3d)
    z3d_c = np.ascontiguousarray(z3d)
    step_c = np.ascontiguousarray(step)

    lib.hyper_gene_c(x3d_c, y3d_c, z3d_c, step_c,
                     nx, ny, nz,
                     coef, t2b, flag_stretch,
                     bdry_x_itype, bdry_y_itype,
                     epsilon_x, epsilon_y)

    # Copy back
    x3d[:] = x3d_c
    y3d[:] = y3d_c
    z3d[:] = z3d_c


def main():
    parser = argparse.ArgumentParser(description='3D Hyperbolic Grid Generation')
    parser.add_argument('--config-file', required=True, help='JSON config file')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level')
    args = parser.parse_args()

    # Read config
    cfgs = cfgs_read(args.config_file)
    cfgs['_verbose'] = args.verbose

    # Print config
    cfgs_print(cfgs)

    # Setup grid info
    gdinfo = grid_info_set(cfgs)

    if args.verbose > 0:
        print(f"nx={gdinfo['nx']} ny={gdinfo['ny']} nz={gdinfo['nz']}")

    # Read boundary data
    x3d, y3d, z3d, step = read_bdry_file(gdinfo, cfgs)

    # Set output filename
    dire_itype = gdinfo['dire_itype']
    fname_part = "px0_py0_pz0"

    # Load C library
    lib = _load_c_lib()

    # Generate grid
    t_start = time.time()
    hyper_gene(lib, gdinfo, x3d, y3d, z3d, step, cfgs)
    t_end = time.time()

    print(f"\n************************************")
    print(f"grid generate running time: {t_end - t_start:.2f} s")
    print(f"************************************\n")

    # Permute coordinates back if direction was x or y
    if dire_itype == X_DIRE:
        _permute_x(x3d, y3d, z3d, gdinfo)
    elif dire_itype == Y_DIRE:
        _permute_y(x3d, y3d, z3d, gdinfo)

    # Export coordinates
    nx = gdinfo['total_nx']
    ny = gdinfo['total_ny']
    nz = gdinfo['total_nz']

    output_dir = cfgs['grid_export_dir']
    os.makedirs(output_dir, exist_ok=True)
    ou_file = os.path.join(output_dir, f"coord_{fname_part}.nc")

    from netCDF4 import Dataset
    with Dataset(ou_file, 'w', format='NETCDF4') as nc:
        nc.createDimension('k', nz)
        nc.createDimension('j', ny)
        nc.createDimension('i', nx)
        nc.createVariable('x', 'f4', ('k', 'j', 'i'))[:] = x3d
        nc.createVariable('y', 'f4', ('k', 'j', 'i'))[:] = y3d
        nc.createVariable('z', 'f4', ('k', 'j', 'i'))[:] = z3d
        nc.global_index_of_first_physical_points = [0, 0, 0]
        nc.count_of_physical_points = [nx, ny, nz]

    print(f"export coord to file: {ou_file}")

    # Grid quality check
    if cfgs.get('grid_check', 0) == 1:
        print("******************************************************")
        print("***** grid quality check and export quality data *****")
        print("******************************************************")
        from common.grid_data import SimpleGridData
        gd = SimpleGridData(nx, ny, nz)
        gd.x3d = x3d.copy()
        gd.y3d = y3d.copy()
        gd.z3d = z3d.copy()
        gd.gni1 = 0
        gd.gnj1 = 0
        gd.gnk1 = 0
        gd.fname_part = fname_part
        grid_quality_check(gd, cfgs)


def _permute_x(x3d, y3d, z3d, gdinfo):
    """Permute coordinates back from x-marching to original layout."""
    nx, ny, nz = gdinfo['nx'], gdinfo['ny'], gdinfo['nz']
    x_old = x3d.copy()
    y_old = y3d.copy()
    z_old = z3d.copy()
    x3d[:] = np.transpose(z_old, (2, 1, 0))
    y3d[:] = np.transpose(y_old, (2, 1, 0))
    z3d[:] = np.transpose(x_old, (2, 1, 0))
    gdinfo['nx'], gdinfo['nz'] = nz, nx


def _permute_y(x3d, y3d, z3d, gdinfo):
    """Permute coordinates back from y-marching to original layout."""
    nx, ny, nz = gdinfo['nx'], gdinfo['ny'], gdinfo['nz']
    x_old = x3d.copy()
    y_old = y3d.copy()
    z_old = z3d.copy()
    x3d[:] = np.transpose(x_old, (1, 0, 2))
    y3d[:] = np.transpose(z_old, (1, 0, 2))
    z3d[:] = np.transpose(y_old, (1, 0, 2))
    gdinfo['ny'], gdinfo['nz'] = nz, ny


if __name__ == '__main__':
    main()

