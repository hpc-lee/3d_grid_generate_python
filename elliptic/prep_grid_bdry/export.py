"""Export 3D boundary surface data for elliptic grid generation.

Exports 6 boundary faces (x1, x2, y1, y2, z1, z2) to a text file
in the format expected by the elliptic grid generator.
"""

import numpy as np


def export_bdry_6(x1, x2, y1, y2, z1, z2, nx, ny, nz, file_dir):
    """Export 6 boundary faces to file.

    Args:
        x1: x1 face, shape (ny, nz, 3).
        x2: x2 face, shape (ny, nz, 3).
        y1: y1 face, shape (nx, nz, 3).
        y2: y2 face, shape (nx, nz, 3).
        z1: z1 face (bottom), shape (nx, ny, 3).
        z2: z2 face (top), shape (nx, ny, 3).
        nx: Number of grid points in x-direction.
        ny: Number of grid points in y-direction.
        nz: Number of grid points in z-direction.
        file_dir: Output file path.
    """
    with open(file_dir, 'w') as fd:
        fd.write(f"{nx}\n")
        fd.write(f"{ny}\n")
        fd.write(f"{nz}\n")

        # x1 face (i=0): shape (ny, nz)
        for k in range(nz):
            for j in range(ny):
                fd.write(f"{x1[j, k, 0]:.9e} {x1[j, k, 1]:.9e} {x1[j, k, 2]:.9e}\n")

        # x2 face (i=nx-1): shape (ny, nz)
        for k in range(nz):
            for j in range(ny):
                fd.write(f"{x2[j, k, 0]:.9e} {x2[j, k, 1]:.9e} {x2[j, k, 2]:.9e}\n")

        # y1 face (j=0): shape (nx, nz)
        for k in range(nz):
            for i in range(nx):
                fd.write(f"{y1[i, k, 0]:.9e} {y1[i, k, 1]:.9e} {y1[i, k, 2]:.9e}\n")

        # y2 face (j=ny-1): shape (nx, nz)
        for k in range(nz):
            for i in range(nx):
                fd.write(f"{y2[i, k, 0]:.9e} {y2[i, k, 1]:.9e} {y2[i, k, 2]:.9e}\n")

        # z1 face (k=0, bottom): shape (nx, ny)
        for j in range(ny):
            for i in range(nx):
                fd.write(f"{z1[i, j, 0]:.9e} {z1[i, j, 1]:.9e} {z1[i, j, 2]:.9e}\n")

        # z2 face (k=nz-1, top): shape (nx, ny)
        for j in range(ny):
            for i in range(nx):
                fd.write(f"{z2[i, j, 0]:.9e} {z2[i, j, 1]:.9e} {z2[i, j, 2]:.9e}\n")
