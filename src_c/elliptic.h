#ifndef ELLIPTIC_3D_H
#define ELLIPTIC_3D_H

/*
 * 3D Elliptic grid generation kernels.
 * Arrays in row-major: index = k*ny*nx + j*nx + i (NumPy shape (nz,ny,nx)).
 */

/* SOR iteration update for one sweep */
void update_SOR_c(float *x3d, float *y3d, float *z3d,
                  float *x3d_tmp, float *y3d_tmp, float *z3d_tmp,
                  int nx, int ny, int nz,
                  float *P, float *Q, float *R, float w);

/* Linear transfinite interpolation from 6 boundary surfaces */
void linear_tfi_c(float *x3d, float *y3d, float *z3d,
                  float *x1, float *x2,
                  float *y1, float *y2,
                  float *z1, float *z2,
                  int nx, int ny, int nz,
                  int ni1, int ni2, int nj1, int nj2, int nk1, int nk2,
                  int gni1, int gnj1, int gnk1,
                  int total_nx, int total_ny, int total_nz,
                  int *neighid);

/* Ghost point calculation for Dirichlet method */
void ghost_cal_c(float *x3d, float *y3d, float *z3d,
                 int nx, int ny, int nz,
                 float *p_x, float *p_y, float *p_z,
                 float *g11_x, float *g22_y, float *g33_z,
                 int *neighid);

/* Source term interpolation from 6 boundary faces to interior */
void interp_inner_source_c(float *P, float *Q, float *R,
                           int nx, int ny, int nz,
                           float *P_x1, float *Q_x1, float *R_x1,
                           float *P_x2, float *Q_x2, float *R_x2,
                           float *P_y1, float *Q_y1, float *R_y1,
                           float *P_y2, float *Q_y2, float *R_y2,
                           float *P_z1, float *Q_z1, float *R_z1,
                           float *P_z2, float *Q_z2, float *R_z2,
                           int gni1, int gnj1, int gnk1,
                           int total_nx, int total_ny, int total_nz,
                           float *coef, int n_coef);

/* Compute maximum residual across local domain */
void compute_residual_c(float *x3d, float *y3d, float *z3d,
                        float *x3d_tmp, float *y3d_tmp, float *z3d_tmp,
                        int nx, int ny, int nz,
                        float *local_max);

/* Set source terms for Dirichlet method on all 6 boundaries */
void set_src_diri_c(float *x3d, float *y3d, float *z3d,
                    int nx, int ny, int nz,
                    int ni, int nj, int nk,
                    float *P, float *Q, float *R,
                    float *P_x1, float *Q_x1, float *R_x1,
                    float *P_x2, float *Q_x2, float *R_x2,
                    float *P_y1, float *Q_y1, float *R_y1,
                    float *P_y2, float *Q_y2, float *R_y2,
                    float *P_z1, float *Q_z1, float *R_z1,
                    float *P_z2, float *Q_z2, float *R_z2,
                    float *P_x1_loc, float *Q_x1_loc, float *R_x1_loc,
                    float *P_x2_loc, float *Q_x2_loc, float *R_x2_loc,
                    float *P_y1_loc, float *Q_y1_loc, float *R_y1_loc,
                    float *P_y2_loc, float *Q_y2_loc, float *R_y2_loc,
                    float *P_z1_loc, float *Q_z1_loc, float *R_z1_loc,
                    float *P_z2_loc, float *Q_z2_loc, float *R_z2_loc,
                    float *p_x, float *p_y, float *p_z,
                    float *g11_x, float *g22_y, float *g33_z,
                    int *neighid,
                    int gni1, int gnj1, int gnk1,
                    int total_nx, int total_ny, int total_nz);

/* Distance calculation for Higenstock method */
void dist_cal_c(float *x3d, float *y3d, float *z3d,
                int nx, int ny, int nz,
                float *dx1, float *dx2,
                float *dy1, float *dy2,
                float *dz1, float *dz2,
                int *neighid);

/* Set source terms for Higenstock method on all 6 boundaries */
void set_src_higen_c(float *x3d, float *y3d, float *z3d,
                     int nx, int ny, int nz,
                     int ni, int nj, int nk,
                     float *P, float *Q, float *R,
                     float *P_x1, float *Q_x1, float *R_x1,
                     float *P_x2, float *Q_x2, float *R_x2,
                     float *P_y1, float *Q_y1, float *R_y1,
                     float *P_y2, float *Q_y2, float *R_y2,
                     float *P_z1, float *Q_z1, float *R_z1,
                     float *P_z2, float *Q_z2, float *R_z2,
                     float *P_x1_loc, float *Q_x1_loc, float *R_x1_loc,
                     float *P_x2_loc, float *Q_x2_loc, float *R_x2_loc,
                     float *P_y1_loc, float *Q_y1_loc, float *R_y1_loc,
                     float *P_y2_loc, float *Q_y2_loc, float *R_y2_loc,
                     float *P_z1_loc, float *Q_z1_loc, float *R_z1_loc,
                     float *P_z2_loc, float *Q_z2_loc, float *R_z2_loc,
                     float *dx1, float *dx2,
                     float *dy1, float *dy2,
                     float *dz1, float *dz2,
                     int *neighid,
                     int gni1, int gnj1, int gnk1,
                     int total_nx, int total_ny, int total_nz);

#endif /* ELLIPTIC_3D_H */
