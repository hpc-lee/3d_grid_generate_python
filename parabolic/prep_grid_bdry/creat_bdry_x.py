"""Create 3D boundary surfaces for x-direction parabolic grid generation.

Generates x1 (left) and x2 (right) boundary surfaces with optional
topography on the x2 surface. Supports PML extension and arc stretching.
"""

import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from common.utils import extend_abs_layer_3d
from parabolic.prep_grid_bdry.export import export_bdry

flag_printf = 1
flag_topo_x = 1
flag_extend_abs = 0
flag_arc_stretch = 0

file_dir = "./data_file_3d_bx.txt"

if flag_extend_abs:
    num_pml = 20
else:
    num_pml = 0

ny1 = 300
nz1 = 200
ny = ny1 + 2 * num_pml
nz = nz1 + 2 * num_pml

dy = 100.0
dz = 100.0
nx = 400
dx = 100.0
origin_x = 0.0
origin_y = 0.0
origin_z = 0.0

# bx1 and bx2 are (nz, ny, 3)
bx1 = np.zeros((nz, ny, 3), dtype=np.float64)
bx2 = np.zeros((nz, ny, 3), dtype=np.float64)

for k in range(nz1):
    for j in range(ny1):
        bx2[k + num_pml, j + num_pml, 0] = origin_x
        bx2[k + num_pml, j + num_pml, 1] = origin_y + j * dy
        bx2[k + num_pml, j + num_pml, 2] = origin_z + k * dz

if flag_topo_x:
    point_y = origin_y + (ny1 // 2) * dy
    point_z = origin_z + (nz1 // 2) * dz
    L = 0.2 * ny * dy
    H = 0.15 * nx * dx
    for k in range(nz1):
        for j in range(ny1):
            r1 = np.sqrt(
                (bx2[k + num_pml, j + num_pml, 1] - point_y)**2 +
                (bx2[k + num_pml, j + num_pml, 2] - point_z)**2
            )
            topo = 0.0
            if r1 < L:
                topo = 0.5 * H * (1 + np.cos(np.pi * r1 / L))
            bx2[k + num_pml, j + num_pml, 0] -= topo

if flag_extend_abs:
    bx2 = extend_abs_layer_3d(bx2, dz, dy, nz, ny, num_pml)

for k in range(nz):
    for j in range(ny):
        bx1[k, j, 0] = origin_x - (nx - 1) * dx
        bx1[k, j, 1] = bx2[k, j, 1]
        bx1[k, j, 2] = bx2[k, j, 2]

if flag_arc_stretch:
    from common.bdry_operations import arc_strech
    A = 0.000001
    # Reshape for arc_strech: treat each row as a boundary curve
    for k in range(nz):
        row = bx2[k, :, :].copy()
        bx2[k, :, :] = arc_strech(A, row)

export_bdry(bx1, bx2, ny, nz, file_dir)

print(f"nx = {nx}, ny = {ny}, nz = {nz}")
print("Boundary file created:", file_dir)
