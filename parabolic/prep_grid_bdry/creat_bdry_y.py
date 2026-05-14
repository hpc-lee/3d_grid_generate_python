"""Create 3D boundary surfaces for y-direction parabolic grid generation.

Generates y1 (front) and y2 (back) boundary surfaces with optional
topography on the y2 surface. Supports PML extension and arc stretching.
"""

import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from common.utils import extend_abs_layer_3d
from parabolic.prep_grid_bdry.export import export_bdry

flag_printf = 1
flag_topo_y = 1
flag_extend_abs = 0
flag_arc_stretch = 0

file_dir = "./data_file_3d_by.txt"

if flag_extend_abs:
    num_pml = 20
else:
    num_pml = 0

nx1 = 400
nz1 = 200
nx = nx1 + 2 * num_pml
nz = nz1 + 2 * num_pml

dx = 100.0
dz = 100.0
ny = 300
dy = 100.0
origin_x = 0.0
origin_y = 0.0
origin_z = 0.0

# by1 and by2 are (nz, nx, 3)
by1 = np.zeros((nz, nx, 3), dtype=np.float64)
by2 = np.zeros((nz, nx, 3), dtype=np.float64)

for k in range(nz1):
    for i in range(nx1):
        by2[k + num_pml, i + num_pml, 0] = origin_x + i * dx
        by2[k + num_pml, i + num_pml, 1] = origin_y
        by2[k + num_pml, i + num_pml, 2] = origin_z + k * dz

if flag_topo_y:
    point_x = origin_x + (nx1 // 2) * dx
    point_z = origin_z + (nz1 // 2) * dz
    L = 0.2 * nx * dx
    H = 0.2 * ny * dy
    for k in range(nz1):
        for i in range(nx1):
            r1 = np.sqrt(
                (by2[k + num_pml, i + num_pml, 0] - point_x)**2 +
                (by2[k + num_pml, i + num_pml, 2] - point_z)**2
            )
            topo = 0.0
            if r1 < L:
                topo = 0.5 * H * (1 + np.cos(np.pi * r1 / L))
            by2[k + num_pml, i + num_pml, 1] -= topo

if flag_extend_abs:
    by2 = extend_abs_layer_3d(by2, dz, dx, nz, nx, num_pml)

for k in range(nz):
    for i in range(nx):
        by1[k, i, 0] = by2[k, i, 0]
        by1[k, i, 1] = origin_y - (ny - 1) * dy
        by1[k, i, 2] = by2[k, i, 2]

if flag_arc_stretch:
    from common.bdry_operations import arc_strech
    A = 0.000001
    for k in range(nz):
        row = by2[k, :, :].copy()
        by2[k, :, :] = arc_strech(A, row)

export_bdry(by1, by2, nx, nz, file_dir)

print(f"nx = {nx}, ny = {ny}, nz = {nz}")
print("Boundary file created:", file_dir)
