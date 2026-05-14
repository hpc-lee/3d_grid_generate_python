"""Create 3D boundary surface for y-direction hyperbolic grid generation.

Generates the y-direction boundary surface (by) with optional topography.
Supports arc stretching.
"""

import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from hyperbolic.prep_grid_bdry.export import export_bdry

flag_printf = 1
flag_topo_y = 1
flag_arc_stretch = 0

file_dir = "./data_file_3d_by.txt"

nx = 400
nz = 300

dx = 10.0
dz = 10.0
origin_x = 0.0
origin_y = 0.0
origin_z = 0.0

# by shape is (nz, nx, 3)
by = np.zeros((nz, nx, 3), dtype=np.float64)

for k in range(nz):
    for i in range(nx):
        by[k, i, 0] = origin_x + i * dx
        by[k, i, 1] = origin_y
        by[k, i, 2] = origin_z + k * dz

if flag_topo_y:
    point_x = origin_x + (nx // 2) * dx
    point_z = origin_z + (nz // 2) * dz
    L = 0.3 * nx * dx
    H = 0.2 * nx * dx
    for k in range(nz):
        for i in range(nx):
            r1 = np.sqrt(
                (by[k, i, 0] - point_x)**2 +
                (by[k, i, 2] - point_z)**2
            )
            topo = 0.0
            if r1 < L:
                topo = 0.5 * H * (1 + np.cos(np.pi * r1 / L))
            by[k, i, 1] = by[k, i, 1] + topo

if flag_arc_stretch:
    from common.bdry_operations import arc_strech
    A = 0.00001
    for k in range(nz):
        row = by[k, :, :].copy()
        by[k, :, :] = arc_strech(A, row)

export_bdry(by, by, nx, nz, file_dir)

print(f"nx = {nx}, nz = {nz}")
print("Boundary file created:", file_dir)
