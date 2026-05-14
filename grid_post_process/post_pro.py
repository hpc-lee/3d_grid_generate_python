"""3D grid post-processing main driver.

Usage:
    python post_pro.py --config-file config.json --verbose 10
"""

import argparse
import json
import os
import sys
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from common.grid_quality import grid_quality_check, cal_min_dist
from common.io_operations import coord_export
from common.utils import remove_comment_keys
from grid_post_process.grid_io import read_import_coord, cfgs_print
from grid_post_process.grid_sample import grid_sample


def main():
    parser = argparse.ArgumentParser(description='3D Grid Post-Processing')
    parser.add_argument('--config-file', required=True, help='JSON config file')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level')
    args = parser.parse_args()

    # Read config
    with open(args.config_file, 'r') as f:
        cfgs = json.load(f)
    cfgs = remove_comment_keys(cfgs)
    cfgs['_verbose'] = args.verbose

    cfgs_print(cfgs)

    # Import coordinates
    t_start = time.time()
    gd = read_import_coord(cfgs)
    t_import = time.time()

    print(f"Import completed: {gd.nx} x {gd.ny} x {gd.nz} grid points")
    print(f"Import time: {t_import - t_start:.2f} s")

    # Sampling
    if cfgs.get('flag_sample', 0) == 1:
        sample_factor = cfgs.get('sample_factor', [1, 1, 1])
        print(f"Sampling with factor {sample_factor}...")
        gd = grid_sample(gd, sample_factor)
        print(f"After sampling: {gd.nx} x {gd.ny} x {gd.nz} grid points")

    # Export
    output_dir = cfgs['grid_export_dir']
    os.makedirs(output_dir, exist_ok=True)

    nproc_out = cfgs.get('number_of_mpiprocs_out', [1, 1, 1])

    if nproc_out == [1, 1, 1]:
        # Single process export
        coord_export(gd, output_dir)
    else:
        # MPI partitioned export
        from common.io_operations import coord_export_mpi
        coord_export_mpi(gd, cfgs)

    # Minimum distance calculation
    if cfgs.get('flag_min_dist', 0) == 1:
        print("Calculating minimum distance...")
        cal_min_dist(gd, cfgs)

    # Grid quality check
    if cfgs.get('grid_check', 0) == 1:
        print("******************************************************")
        print("***** grid quality check and export quality data *****")
        print("******************************************************")
        gd.fname_part = "px0_py0_pz0"
        grid_quality_check(gd, cfgs)

    t_end = time.time()
    print(f"\n************************************")
    print(f"post-processing total time: {t_end - t_start:.2f} s")
    print(f"************************************")


if __name__ == '__main__':
    main()
