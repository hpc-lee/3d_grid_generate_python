"""Export 3D boundary surface data to text file."""


def export_bdry(bz1, bz2, nx, ny, file_dir):
    """Export two boundary surfaces to file.

    Args:
        bz1: Bottom surface, shape (nx, ny, 3) with x,y,z components.
        bz2: Top surface, shape (nx, ny, 3) with x,y,z components.
        nx: Number of points in x-direction.
        ny: Number of points in y-direction.
        file_dir: Output file path.
    """
    with open(file_dir, 'w') as fd:
        fd.write(f"# nx ny number\n")
        fd.write(f"{nx} {ny}\n")
        fd.write(f"# bz1 coords\n")
        for j in range(ny):
            for i in range(nx):
                fd.write(f"{bz1[i,j,0]:.9e} {bz1[i,j,1]:.9e} {bz1[i,j,2]:.9e}\n")
        fd.write(f"# bz2 coords\n")
        for j in range(ny):
            for i in range(nx):
                fd.write(f"{bz2[i,j,0]:.9e} {bz2[i,j,1]:.9e} {bz2[i,j,2]:.9e}\n")
