#ifndef PARABOLIC_3D_H
#define PARABOLIC_3D_H

/*
 * 3D Parabolic grid generation kernel.
 * Arrays in row-major: index = k*ny*nx + j*nx + i (NumPy shape (nz,ny,nx)).
 */

/* Predict points at layers k and k+1 from layer k-1 */
void predict_point_c(float *x3d, float *y3d, float *z3d,
                     int nx, int ny, int nz,
                     int k, int t2b, float coef,
                     float *step_len);

/* Update (correct) points at layer k using elliptic equation + Thomas solver */
void update_point_c(float *x3d, float *y3d, float *z3d,
                    int nx, int ny, int nz,
                    int k);

#endif /* PARABOLIC_3D_H */
