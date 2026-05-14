"""Grid data containers for 3D curvilinear grid generation."""

import numpy as np


class SimpleGridData:
    """Non-MPI grid data container (parabolic, hyperbolic, post-process)."""

    def __init__(self, nx: int = 0, ny: int = 0, nz: int = 0):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.ni = nx
        self.nj = ny
        self.nk = nz

        self.x3d = np.zeros((nz, ny, nx), dtype=np.float32)
        self.y3d = np.zeros((nz, ny, nx), dtype=np.float32)
        self.z3d = np.zeros((nz, ny, nx), dtype=np.float32)

        # File output name part (for MPI-partitioned re-export)
        self.fname_part = "px0_py0_pz0"
        # Global index of first physical points (for unified I/O)
        self.gni1 = 0
        self.gnj1 = 0
        self.gnk1 = 0

        # Step lengths for marching methods (parabolic/hyperbolic only)
        self.step = np.zeros(max(nz - 1, 0), dtype=np.float32)
