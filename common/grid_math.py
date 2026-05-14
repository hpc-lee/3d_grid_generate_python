"""3x3 matrix and 3-vector operations for 3D grid generation."""

import numpy as np
import numba


@numba.jit(nopython=True, nogil=True, cache=True)
def mat_invert3x3(mat):
    """Invert a 3x3 matrix."""
    inv = np.empty((3, 3), dtype=mat.dtype)
    det = (mat[0, 0] * (mat[1, 1] * mat[2, 2] - mat[1, 2] * mat[2, 1])
         - mat[0, 1] * (mat[1, 0] * mat[2, 2] - mat[1, 2] * mat[2, 0])
         + mat[0, 2] * (mat[1, 0] * mat[2, 1] - mat[1, 1] * mat[2, 0]))
    if abs(det) < 1e-10:
        det = 1e-10
    inv_det = 1.0 / det
    inv[0, 0] = (mat[1, 1] * mat[2, 2] - mat[1, 2] * mat[2, 1]) * inv_det
    inv[0, 1] = (mat[0, 2] * mat[2, 1] - mat[0, 1] * mat[2, 2]) * inv_det
    inv[0, 2] = (mat[0, 1] * mat[1, 2] - mat[0, 2] * mat[1, 1]) * inv_det
    inv[1, 0] = (mat[1, 2] * mat[2, 0] - mat[1, 0] * mat[2, 2]) * inv_det
    inv[1, 1] = (mat[0, 0] * mat[2, 2] - mat[0, 2] * mat[2, 0]) * inv_det
    inv[1, 2] = (mat[0, 2] * mat[1, 0] - mat[0, 0] * mat[1, 2]) * inv_det
    inv[2, 0] = (mat[1, 0] * mat[2, 1] - mat[1, 1] * mat[2, 0]) * inv_det
    inv[2, 1] = (mat[0, 1] * mat[2, 0] - mat[0, 0] * mat[2, 1]) * inv_det
    inv[2, 2] = (mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]) * inv_det
    return inv


@numba.jit(nopython=True, nogil=True, cache=True)
def mat_mul3x3(a, b):
    """Multiply two 3x3 matrices."""
    res = np.empty((3, 3), dtype=a.dtype)
    for i in range(3):
        for j in range(3):
            res[i, j] = 0.0
            for k in range(3):
                res[i, j] += a[i, k] * b[k, j]
    return res


@numba.jit(nopython=True, nogil=True, cache=True)
def mat_add3x3(a, b):
    res = np.empty((3, 3), dtype=a.dtype)
    for i in range(3):
        for j in range(3):
            res[i, j] = a[i, j] + b[i, j]
    return res


@numba.jit(nopython=True, nogil=True, cache=True)
def mat_sub3x3(a, b):
    res = np.empty((3, 3), dtype=a.dtype)
    for i in range(3):
        for j in range(3):
            res[i, j] = a[i, j] - b[i, j]
    return res


@numba.jit(nopython=True, nogil=True, cache=True)
def mat_mul3x1(mat, vec):
    res = np.empty(3, dtype=mat.dtype)
    for i in range(3):
        res[i] = 0.0
        for j in range(3):
            res[i] += mat[i, j] * vec[j]
    return res


@numba.jit(nopython=True, nogil=True, cache=True)
def vec_add3x1(a, b):
    return np.array([a[0]+b[0], a[1]+b[1], a[2]+b[2]], dtype=a.dtype)


@numba.jit(nopython=True, nogil=True, cache=True)
def vec_sub3x1(a, b):
    return np.array([a[0]-b[0], a[1]-b[1], a[2]-b[2]], dtype=a.dtype)


@numba.jit(nopython=True, nogil=True, cache=True)
def cross_product(a, b):
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ], dtype=a.dtype)


@numba.jit(nopython=True, nogil=True, cache=True)
def dot_product(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


@numba.jit(nopython=True, nogil=True, cache=True)
def mat_iden3x3():
    res = np.zeros((3, 3), dtype=np.float64)
    res[0, 0] = 1.0
    res[1, 1] = 1.0
    res[2, 2] = 1.0
    return res


@numba.jit(nopython=True, nogil=True, cache=True)
def mat_copy3x3(src):
    dst = np.empty((3, 3), dtype=src.dtype)
    for i in range(3):
        for j in range(3):
            dst[i, j] = src[i, j]
    return dst


@numba.jit(nopython=True, nogil=True, cache=True)
def dist_point2plane(px, py, pz, x1, y1, z1, x2, y2, z2, x3, y3, z3):
    """Distance from point (px,py,pz) to plane defined by 3 points."""
    v1 = np.array([x2-x1, y2-y1, z2-z1], dtype=np.float64)
    v2 = np.array([x3-x1, y3-y1, z3-z1], dtype=np.float64)
    n = np.array([
        v1[1]*v2[2] - v1[2]*v2[1],
        v1[2]*v2[0] - v1[0]*v2[2],
        v1[0]*v2[1] - v1[1]*v2[0]
    ], dtype=np.float64)
    denom = np.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
    if denom < 1e-10:
        return 1e10
    return abs(n[0]*(px-x1) + n[1]*(py-y1) + n[2]*(pz-z1)) / denom
