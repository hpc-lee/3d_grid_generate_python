"""MPI 3D Cartesian topology for 3D elliptic grid generation.

Decomposes x, y, z directions across MPI processes.
"""

import numpy as np

try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False


class MyMPI3D:
    """3D Cartesian MPI topology for elliptic method."""

    def __init__(self, nproc_x, nproc_y, nproc_z):
        if not HAS_MPI:
            self.comm = None
            self.myid = 0
            self.nproc = 1
            self.topocomm = None
            self.topoid = [0, 0, 0]
            self.neighid = np.array([-2]*6, dtype=np.int32)
            return

        self.comm = MPI.COMM_WORLD
        self.myid = self.comm.Get_rank()
        self.nproc = self.comm.Get_size()
        self.nproc_x = nproc_x
        self.nproc_y = nproc_y
        self.nproc_z = nproc_z

        # Create 3D Cartesian topology
        self.topocomm = self.comm.Create_cart([nproc_x, nproc_y, nproc_z])
        self.topoid = list(self.topocomm.Get_coords(self.myid))

        # Get 6 neighbors: x1, x2, y1, y2, z1, z2
        self.neighid = np.zeros(6, dtype=np.int32)
        self.neighid[0], self.neighid[1] = self.topocomm.Shift(0, 1)  # x
        self.neighid[2], self.neighid[3] = self.topocomm.Shift(1, 1)  # y
        self.neighid[4], self.neighid[5] = self.topocomm.Shift(2, 1)  # z

        # Replace MPI_PROC_NULL with -2
        for i in range(6):
            if self.neighid[i] == MPI.PROC_NULL:
                self.neighid[i] = -2

    def exchange_ghost(self, gdinfo, x3d, y3d, z3d):
        """Exchange ghost cells in all 6 directions.

        Uses non-blocking sends/receives for overlap.
        """
        if not HAS_MPI or self.nproc == 1:
            return

        nx = gdinfo['nx']
        ny = gdinfo['ny']
        nz = gdinfo['nz']
        ni = gdinfo['ni']
        nj = gdinfo['nj']
        nk = gdinfo['nk']

        # X-direction exchange (send i=1..ni slice, recv into ghost)
        self._exchange_face_x(x3d, y3d, z3d, nx, ny, nz, ni, nj, nk)
        # Y-direction exchange
        self._exchange_face_y(x3d, y3d, z3d, nx, ny, nz, ni, nj, nk)
        # Z-direction exchange
        self._exchange_face_z(x3d, y3d, z3d, nx, ny, nz, ni, nj, nk)

    def _exchange_face_x(self, x3d, y3d, z3d, nx, ny, nz, ni, nj, nk):
        """Exchange ghost cells in x-direction."""
        # x1: send i=1 to lower neighbor, recv i=0 (ghost)
        # x2: send i=ni to upper neighbor, recv i=ni+1 (ghost)
        size = nj * nk
        float_p = np.float32

        send_lo = np.zeros(3 * size, dtype=float_p)
        send_hi = np.zeros(3 * size, dtype=float_p)
        recv_lo = np.zeros(3 * size, dtype=float_p)
        recv_hi = np.zeros(3 * size, dtype=float_p)

        # Pack i=1 (lower boundary)
        idx = 0
        for k in range(1, nk + 1):
            for j in range(1, nj + 1):
                send_lo[idx] = x3d[k, j, 1]
                send_lo[size + idx] = y3d[k, j, 1]
                send_lo[2*size + idx] = z3d[k, j, 1]
                idx += 1

        # Pack i=ni (upper boundary)
        idx = 0
        for k in range(1, nk + 1):
            for j in range(1, nj + 1):
                send_hi[idx] = x3d[k, j, ni]
                send_hi[size + idx] = y3d[k, j, ni]
                send_hi[2*size + idx] = z3d[k, j, ni]
                idx += 1

        # Exchange
        reqs = []
        if self.neighid[0] != -2:
            self.topocomm.Isend(send_lo, self.neighid[0], tag=0)
            req = self.topocomm.Irecv(recv_hi, self.neighid[0], tag=1)
            reqs.append(req)
        if self.neighid[1] != -2:
            self.topocomm.Isend(send_hi, self.neighid[1], tag=1)
            req = self.topocomm.Irecv(recv_lo, self.neighid[1], tag=0)
            reqs.append(req)

        for req in reqs:
            req.Wait()

        # Unpack i=0 (ghost from lower)
        if self.neighid[0] != -2:
            idx = 0
            for k in range(1, nk + 1):
                for j in range(1, nj + 1):
                    x3d[k, j, 0] = recv_lo[idx]
                    y3d[k, j, 0] = recv_lo[size + idx]
                    z3d[k, j, 0] = recv_lo[2*size + idx]
                    idx += 1

        # Unpack i=ni+1 (ghost from upper)
        if self.neighid[1] != -2:
            idx = 0
            for k in range(1, nk + 1):
                for j in range(1, nj + 1):
                    x3d[k, j, ni+1] = recv_hi[idx]
                    y3d[k, j, ni+1] = recv_hi[size + idx]
                    z3d[k, j, ni+1] = recv_hi[2*size + idx]
                    idx += 1

    def _exchange_face_y(self, x3d, y3d, z3d, nx, ny, nz, ni, nj, nk):
        """Exchange ghost cells in y-direction."""
        size = ni * nk
        float_p = np.float32

        send_lo = np.zeros(3 * size, dtype=float_p)
        send_hi = np.zeros(3 * size, dtype=float_p)
        recv_lo = np.zeros(3 * size, dtype=float_p)
        recv_hi = np.zeros(3 * size, dtype=float_p)

        # Pack j=1
        idx = 0
        for k in range(1, nk + 1):
            for i in range(1, ni + 1):
                send_lo[idx] = x3d[k, 1, i]
                send_lo[size + idx] = y3d[k, 1, i]
                send_lo[2*size + idx] = z3d[k, 1, i]
                idx += 1

        # Pack j=nj
        idx = 0
        for k in range(1, nk + 1):
            for i in range(1, ni + 1):
                send_hi[idx] = x3d[k, nj, i]
                send_hi[size + idx] = y3d[k, nj, i]
                send_hi[2*size + idx] = z3d[k, nj, i]
                idx += 1

        reqs = []
        if self.neighid[2] != -2:
            self.topocomm.Isend(send_lo, self.neighid[2], tag=2)
            req = self.topocomm.Irecv(recv_hi, self.neighid[2], tag=3)
            reqs.append(req)
        if self.neighid[3] != -2:
            self.topocomm.Isend(send_hi, self.neighid[3], tag=3)
            req = self.topocomm.Irecv(recv_lo, self.neighid[3], tag=2)
            reqs.append(req)

        for req in reqs:
            req.Wait()

        if self.neighid[2] != -2:
            idx = 0
            for k in range(1, nk + 1):
                for i in range(1, ni + 1):
                    x3d[k, 0, i] = recv_lo[idx]
                    y3d[k, 0, i] = recv_lo[size + idx]
                    z3d[k, 0, i] = recv_lo[2*size + idx]
                    idx += 1

        if self.neighid[3] != -2:
            idx = 0
            for k in range(1, nk + 1):
                for i in range(1, ni + 1):
                    x3d[k, nj+1, i] = recv_hi[idx]
                    y3d[k, nj+1, i] = recv_hi[size + idx]
                    z3d[k, nj+1, i] = recv_hi[2*size + idx]
                    idx += 1

    def _exchange_face_z(self, x3d, y3d, z3d, nx, ny, nz, ni, nj, nk):
        """Exchange ghost cells in z-direction."""
        size = ni * nj
        float_p = np.float32

        send_lo = np.zeros(3 * size, dtype=float_p)
        send_hi = np.zeros(3 * size, dtype=float_p)
        recv_lo = np.zeros(3 * size, dtype=float_p)
        recv_hi = np.zeros(3 * size, dtype=float_p)

        # Pack k=1
        idx = 0
        for j in range(1, nj + 1):
            for i in range(1, ni + 1):
                send_lo[idx] = x3d[1, j, i]
                send_lo[size + idx] = y3d[1, j, i]
                send_lo[2*size + idx] = z3d[1, j, i]
                idx += 1

        # Pack k=nk
        idx = 0
        for j in range(1, nj + 1):
            for i in range(1, ni + 1):
                send_hi[idx] = x3d[nk, j, i]
                send_hi[size + idx] = y3d[nk, j, i]
                send_hi[2*size + idx] = z3d[nk, j, i]
                idx += 1

        reqs = []
        if self.neighid[4] != -2:
            self.topocomm.Isend(send_lo, self.neighid[4], tag=4)
            req = self.topocomm.Irecv(recv_hi, self.neighid[4], tag=5)
            reqs.append(req)
        if self.neighid[5] != -2:
            self.topocomm.Isend(send_hi, self.neighid[5], tag=5)
            req = self.topocomm.Irecv(recv_lo, self.neighid[5], tag=4)
            reqs.append(req)

        for req in reqs:
            req.Wait()

        if self.neighid[4] != -2:
            idx = 0
            for j in range(1, nj + 1):
                for i in range(1, ni + 1):
                    x3d[0, j, i] = recv_lo[idx]
                    y3d[0, j, i] = recv_lo[size + idx]
                    z3d[0, j, i] = recv_lo[2*size + idx]
                    idx += 1

        if self.neighid[5] != -2:
            idx = 0
            for j in range(1, nj + 1):
                for i in range(1, ni + 1):
                    x3d[nk+1, j, i] = recv_hi[idx]
                    y3d[nk+1, j, i] = recv_hi[size + idx]
                    z3d[nk+1, j, i] = recv_hi[2*size + idx]
                    idx += 1

    def allreduce_max(self, values):
        """Global maximum reduction."""
        if not HAS_MPI or self.nproc == 1:
            return values
        result = np.zeros_like(values)
        self.comm.Allreduce(values, result, MPI.MAX)
        return result

    def is_root(self):
        return self.myid == 0
