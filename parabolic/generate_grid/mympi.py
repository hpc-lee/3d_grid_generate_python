"""MPI 1D Cartesian topology for 3D parabolic grid generation.

Decomposes only the y-direction across MPI processes.
"""

import numpy as np

try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False


class MyMPI1D:
    """1D Cartesian MPI topology for parabolic method (y-direction decomposition)."""

    def __init__(self, nproc_y):
        if not HAS_MPI:
            self.comm = None
            self.myid = 0
            self.nproc = 1
            self.topocomm = None
            self.topoid = [0]
            self.neighid = [MPI.PROC_NULL, MPI.PROC_NULL] if HAS_MPI else [-2, -2]
            return

        self.comm = MPI.COMM_WORLD
        self.myid = self.comm.Get_rank()
        self.nproc = self.comm.Get_size()
        self.nproc_y = nproc_y

        # Create 1D Cartesian topology
        self.topocomm = self.comm.Create_cart([nproc_y])
        self.topoid = list(self.topocomm.Get_coords(self.myid))

        # Get y-direction neighbors (lower and upper)
        self.neighid = list(self.topocomm.Shift(0, 1))
        # MPI_PROC_NULL = -2 in our C convention
        self.neighid = [
            n if n != MPI.PROC_NULL else -2
            for n in self.neighid
        ]

    def exchange_coord(self, gdinfo, x3d, y3d, z3d, k):
        """Exchange ghost cell coordinates in y-direction for layers k and k+1."""
        if not HAS_MPI or self.nproc == 1:
            return

        nx = gdinfo['nx']
        nj1 = gdinfo['nj1']
        nj2 = gdinfo['nj2']

        # Pack and send boundary data
        # y1 boundary: layer k and k+1 at j = nj1
        send_y1 = np.zeros(6 * nx, dtype=np.float32)
        send_y1[0*nx:1*nx] = x3d[k, nj1, :]
        send_y1[1*nx:2*nx] = y3d[k, nj1, :]
        send_y1[2*nx:3*nx] = z3d[k, nj1, :]
        send_y1[3*nx:4*nx] = x3d[k+1, nj1, :]
        send_y1[4*nx:5*nx] = y3d[k+1, nj1, :]
        send_y1[5*nx:6*nx] = z3d[k+1, nj1, :]

        # y2 boundary: layer k and k+1 at j = nj2
        send_y2 = np.zeros(6 * nx, dtype=np.float32)
        send_y2[0*nx:1*nx] = x3d[k, nj2, :]
        send_y2[1*nx:2*nx] = y3d[k, nj2, :]
        send_y2[2*nx:3*nx] = z3d[k, nj2, :]
        send_y2[3*nx:4*nx] = x3d[k+1, nj2, :]
        send_y2[4*nx:5*nx] = y3d[k+1, nj2, :]
        send_y2[5*nx:6*nx] = z3d[k+1, nj2, :]

        # Exchange: send y1 to lower, y2 to upper; recv from both
        recv_y1 = np.zeros(6 * nx, dtype=np.float32)
        recv_y2 = np.zeros(6 * nx, dtype=np.float32)

        neigh_lo = self.neighid[0]
        neigh_hi = self.neighid[1]

        if neigh_lo != -2 and neigh_hi != -2:
            # Both neighbors exist
            self.topocomm.Sendrecv(send_y1, neigh_lo, 0,
                                   recv_y2, neigh_hi, 0)
            self.topocomm.Sendrecv(send_y2, neigh_hi, 1,
                                   recv_y1, neigh_lo, 1)
        elif neigh_lo != -2:
            # Only lower neighbor
            self.topocomm.Sendrecv(send_y1, neigh_lo, 0,
                                   recv_y1, neigh_lo, 1)
        elif neigh_hi != -2:
            # Only upper neighbor
            self.topocomm.Sendrecv(send_y2, neigh_hi, 1,
                                   recv_y2, neigh_hi, 0)

        # Unpack received data into ghost cells
        if neigh_lo != -2:
            # Received from lower neighbor -> ghost at j = nj1 - 1
            x3d[k, nj1-1, :] = recv_y1[0*nx:1*nx]
            y3d[k, nj1-1, :] = recv_y1[1*nx:2*nx]
            z3d[k, nj1-1, :] = recv_y1[2*nx:3*nx]
            x3d[k+1, nj1-1, :] = recv_y1[3*nx:4*nx]
            y3d[k+1, nj1-1, :] = recv_y1[4*nx:5*nx]
            z3d[k+1, nj1-1, :] = recv_y1[5*nx:6*nx]

        if neigh_hi != -2:
            # Received from upper neighbor -> ghost at j = nj2 + 1
            x3d[k, nj2+1, :] = recv_y2[0*nx:1*nx]
            y3d[k, nj2+1, :] = recv_y2[1*nx:2*nx]
            z3d[k, nj2+1, :] = recv_y2[2*nx:3*nx]
            x3d[k+1, nj2+1, :] = recv_y2[3*nx:4*nx]
            y3d[k+1, nj2+1, :] = recv_y2[4*nx:5*nx]
            z3d[k+1, nj2+1, :] = recv_y2[5*nx:6*nx]

    def is_root(self):
        return self.myid == 0
