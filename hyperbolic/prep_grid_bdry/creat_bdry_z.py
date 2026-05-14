"""Create 3D boundary surfaces for z-direction hyperbolic grid generation.

Generates bottom (bz1) and top (bz2) surfaces with optional Gaussian
topography on the top surface. Supports PML extension and arc stretching.
"""

import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from common.utils import extend_abs_layer_3d
from hyperbolic.prep_grid_bdry.export import export_bdry

flag_printf = 1
flag_topo = 1
flag_extend_abs = 0
flag_arc_stretch = 0

file_dir = "./data_file_3d.txt"

if flag_extend_abs:
    num_pml = 20
else:
    num_pml = 0

nx1 = 51
ny1 = 41
nx = nx1 + 2 * num_pml
ny = ny1 + 2 * num_pml
nz = 21

dx = 50.0
dy = 50.0
dz = 50.0
origin_x = 0.0
origin_y = 0.0
origin_z = 0.0

bz1 = np.zeros((nx, ny, 3), dtype=np.float64)
bz2 = np.zeros((nx, ny, 3), dtype=np.float64)

for i in range(nx1):
    for j in range(ny1):
        ii = i + num_pml
        jj = j + num_pml
        bz1[ii, jj, 0] = origin_x + i * dx
        bz1[ii, jj, 1] = origin_y + j * dy
        bz1[ii, jj, 2] = origin_z - (nz - 1) * dz

        bz2[ii, jj, 0] = origin_x + i * dx
        bz2[ii, jj, 1] = origin_y + j * dy
        bz2[ii, jj, 2] = origin_z

if flag_topo:
    x0 = 10.0 * 1e3
    x1 = 30.0 * 1e3
    ax = 3.0 * 1e3
    H = 6.0 * 1e3

    for i in range(nx1):
        for j in range(ny1):
            ii = i + num_pml
            jj = j + num_pml
            x = ii * dx
            topo = (H * np.exp(-((x - x0)**2) / (ax**2)) -
                    H * np.exp(-((x - x1)**2) / (ax**2)))
            bz2[ii, jj, 2] = bz2[ii, jj, 2] + topo

if flag_extend_abs:
    bz1 = extend_abs_layer_3d(bz1, dx, dy, nx, ny, num_pml)
    bz2 = extend_abs_layer_3d(bz2, dx, dy, nx, ny, num_pml)

export_bdry(bz1, bz2, nx, ny, file_dir)

print(f"nx = {nx}, ny = {ny}, nz = {nz}")
print("Boundary file created:", file_dir)
