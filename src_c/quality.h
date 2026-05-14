#ifndef QUALITY_3D_H
#define QUALITY_3D_H

/*
 * 3D grid quality metrics.
 * Arrays in row-major: index = k*ny*nx + j*nx + i  (NumPy shape (nz,ny,nx)).
 *
 * All quality arrays `var` are pre-allocated and zeroed by the caller
 * with the same shape (nz,ny,nx).  Results are computed for interior
 * points and boundary values are extrapolated from nearest interior.
 */

/* Orthogonality angle between xi and eta tangent vectors (degrees).
 * orth = 90 - |angle - 90|, range [0, 90], 90 = perfectly orthogonal. */
void cal_orth_xiet_c(float *x3d, float *y3d, float *z3d, float *var, int nx, int ny, int nz);

/* Orthogonality angle between xi and zt tangent vectors (degrees). */
void cal_orth_xizt_c(float *x3d, float *y3d, float *z3d, float *var, int nx, int ny, int nz);

/* Orthogonality angle between eta and zt tangent vectors (degrees). */
void cal_orth_etzt_c(float *x3d, float *y3d, float *z3d, float *var, int nx, int ny, int nz);

/* Jacobian (scalar triple product): |r_xi . (r_eta x r_zt)|
 * = volume of the parallelepiped spanned by the three tangent vectors. */
void cal_jacobi_c(float *x3d, float *y3d, float *z3d, float *var, int nx, int ny, int nz);

/* Physical step size in xi direction: |r(i+1) - r(i)|. */
void cal_step_xi_c(float *x3d, float *y3d, float *z3d, float *var, int nx, int ny, int nz);

/* Physical step size in eta direction: |r(j+1) - r(j)|. */
void cal_step_et_c(float *x3d, float *y3d, float *z3d, float *var, int nx, int ny, int nz);

/* Physical step size in zt direction: |r(k+1) - r(k)|. */
void cal_step_zt_c(float *x3d, float *y3d, float *z3d, float *var, int nx, int ny, int nz);

/* Smoothness in xi direction: max(step_i / step_{i-1}, step_{i-1} / step_i).
 * 1.0 = perfectly uniform spacing. */
void cal_smooth_xi_c(float *x3d, float *y3d, float *z3d, float *var, int nx, int ny, int nz);

/* Smoothness in eta direction. */
void cal_smooth_et_c(float *x3d, float *y3d, float *z3d, float *var, int nx, int ny, int nz);

/* Smoothness in zt direction. */
void cal_smooth_zt_c(float *x3d, float *y3d, float *z3d, float *var, int nx, int ny, int nz);

/* Aspect ratio: max(|r_xi|/|r_et|, |r_et|/|r_xi|) for each cell.
 * 1.0 = perfectly uniform cell shape. */
void cal_ratio_c(float *x3d, float *y3d, float *z3d, float *var, int nx, int ny, int nz);

#endif /* QUALITY_3D_H */
