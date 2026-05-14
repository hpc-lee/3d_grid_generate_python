"""Create 6 boundary surfaces for 3D elliptic grid generation.

Generates a rectangular hexahedral domain with optional Gaussian
topography on the top (z2) surface. The x and y boundary faces
are derived from the z faces to ensure edge consistency.
"""

import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from elliptic.prep_grid_bdry.export import export_bdry_6

flag_topo = 1
flag_extend_abs = 0

file_dir = "./bdry_file_3d.txt"

if flag_extend_abs:
    num_pml = 5
else:
    num_pml = 0

nx1 = 51
ny1 = 41
nz1 = 21
nx = nx1 + 2 * num_pml
ny = ny1 + 2 * num_pml
nz = nz1 + 2 * num_pml

dx = 50.0
dy = 50.0
dz = 50.0
origin_x = 0.0
origin_y = 0.0
origin_z = 0.0

# z1 face (bottom, k=0): varies i,j
z1 = np.zeros((nx, ny, 3), dtype=np.float64)
# z2 face (top, k=nz-1): varies i,j
z2 = np.zeros((nx, ny, 3), dtype=np.float64)

for j in range(ny):
    for i in range(nx):
        z1[i, j, 0] = origin_x + i * dx
        z1[i, j, 1] = origin_y + j * dy
        z1[i, j, 2] = origin_z - (nz - 1) * dz

        z2[i, j, 0] = origin_x + i * dx
        z2[i, j, 1] = origin_y + j * dy
        z2[i, j, 2] = origin_z

if flag_topo:
    x0 = 10.0 * 1e3
    x1_pos = 30.0 * 1e3
    ax = 3.0 * 1e3
    H = 6.0 * 1e3
    for j in range(ny):
        for i in range(nx):
            x = i * dx
            topo = (H * np.exp(-((x - x0)**2) / (ax**2)) -
                    H * np.exp(-((x - x1_pos)**2) / (ax**2)))
            z2[i, j, 2] += topo

# Derive x boundary faces from z faces (ensures edge consistency)
bx1 = np.zeros((ny, nz, 3), dtype=np.float64)
bx2 = np.zeros((ny, nz, 3), dtype=np.float64)

for j in range(ny):
    dz1 = (z2[0, j, 2] - z1[0, j, 2]) / (nz - 1)
    dz2 = (z2[nx-1, j, 2] - z1[nx-1, j, 2]) / (nz - 1)
    for k in range(nz):
        bx1[j, k, 0] = z1[0, j, 0]
        bx1[j, k, 1] = z1[0, j, 1]
        bx1[j, k, 2] = z1[0, j, 2] + k * dz1

        bx2[j, k, 0] = z1[nx-1, j, 0]
        bx2[j, k, 1] = z1[nx-1, j, 1]
        bx2[j, k, 2] = z1[nx-1, j, 2] + k * dz2

# Derive y boundary faces from z faces (ensures edge consistency)
by1 = np.zeros((nx, nz, 3), dtype=np.float64)
by2 = np.zeros((nx, nz, 3), dtype=np.float64)

for i in range(nx):
    dz1 = (z2[i, 0, 2] - z1[i, 0, 2]) / (nz - 1)
    dz2 = (z2[i, ny-1, 2] - z1[i, ny-1, 2]) / (nz - 1)
    for k in range(nz):
        by1[i, k, 0] = z1[i, 0, 0]
        by1[i, k, 1] = z1[i, 0, 1]
        by1[i, k, 2] = z1[i, 0, 2] + k * dz1

        by2[i, k, 0] = z1[i, ny-1, 0]
        by2[i, k, 1] = z1[i, ny-1, 1]
        by2[i, k, 2] = z1[i, ny-1, 2] + k * dz2

export_bdry_6(bx1, bx2, by1, by2, z1, z2, nx, ny, nz, file_dir)

print(f"nx={nx}, ny={ny}, nz={nz}")
print(f"Boundary file created: {file_dir}")
