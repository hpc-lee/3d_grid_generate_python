"""Create 3D boundary surface for x-direction hyperbolic grid generation.

Generates the x-direction boundary surface (bx) with optional topography.
Supports arc stretching.
"""

import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from hyperbolic.prep_grid_bdry.export import export_bdry

flag_printf = 1
flag_topo_x = 1
flag_arc_stretch = 0

file_dir = "./data_file_3d_bx.txt"

ny = 400
nz = 300

dy = 10.0
dz = 10.0
origin_x = 0.0
origin_y = 0.0
origin_z = 0.0

# bx shape is (nz, ny, 3)
bx = np.zeros((nz, ny, 3), dtype=np.float64)

for k in range(nz):
    for j in range(ny):
        bx[k, j, 0] = origin_x
        bx[k, j, 1] = origin_y + j * dy
        bx[k, j, 2] = origin_z + k * dz

if flag_topo_x:
    point_y = origin_y + (ny // 2) * dy
    point_z = origin_z + (nz // 2) * dz
    L = 0.3 * ny * dy
    H = 0.2 * ny * dy
    for k in range(nz):
        for j in range(ny):
            r1 = np.sqrt(
                (bx[k, j, 1] - point_y)**2 +
                (bx[k, j, 2] - point_z)**2
            )
            topo = 0.0
            if r1 < L:
                topo = 0.5 * H * (1 + np.cos(np.pi * r1 / L))
            bx[k, j, 0] = bx[k, j, 0] + topo

if flag_arc_stretch:
    from common.bdry_operations import arc_strech
    A = 0.00001
    for k in range(nz):
        row = bx[k, :, :].copy()
        bx[k, :, :] = arc_strech(A, row)

export_bdry(bx, bx, ny, nz, file_dir)

print(f"ny = {ny}, nz = {nz}")
print("Boundary file created:", file_dir)
