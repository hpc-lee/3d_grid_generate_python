#ifndef POST_PROCESS_3D_H
#define POST_PROCESS_3D_H

/*
 * 3D grid post-processing: resampling, minimum distance, arc-length stretch.
 * Arrays in row-major: index = k*ny*nx + j*nx + i  (NumPy shape (nz,ny,nx)).
 */

/* Tri-linear resampling from (nx,ny,nz) to (nx_new,ny_new,nz_new).
 * Interpolation order: z -> y -> x.
 * Output arrays (x3d_new, y3d_new, z3d_new) must be pre-allocated
 * with size nx_new * ny_new * nz_new. */
void sample_interp_c(float *x3d, float *y3d, float *z3d,
                     float *x3d_new, float *y3d_new, float *z3d_new,
                     int nx, int ny, int nz,
                     int nx_new, int ny_new, int nz_new);

/* Find the minimum distance from any interior grid point to the 8
 * surrounding planes defined by its 6 face-adjacent neighbours.
 * Returns the distance in *dL_min and the grid indices in
 * *indx_i, *indx_j, *indx_k. */
void cal_min_dist_c(float *x3d, float *y3d, float *z3d,
                    int nx, int ny, int nz,
                    int *indx_i, int *indx_j, int *indx_k,
                    float *dL_min);

/* Redistribute grid points along the xi (i) direction using
 * arc-length parameterization.  arc_len[n_arc] holds the target
 * normalized arc-length for each index; n_arc should equal nx. */
void xi_arc_stretch_c(float *x3d, float *y3d, float *z3d,
                      int nx, int ny, int nz, float *arc_len, int n_arc);

/* Redistribute grid points along the eta (j) direction.
 * n_arc should equal ny. */
void et_arc_stretch_c(float *x3d, float *y3d, float *z3d,
                      int nx, int ny, int nz, float *arc_len, int n_arc);

/* Redistribute grid points along the zt (k) direction.
 * n_arc should equal nz. */
void zt_arc_stretch_c(float *x3d, float *y3d, float *z3d,
                      int nx, int ny, int nz, float *arc_len, int n_arc);

#endif /* POST_PROCESS_3D_H */
