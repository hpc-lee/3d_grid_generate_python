#ifndef HYPERBOLIC_3D_H
#define HYPERBOLIC_3D_H

/*
 * Hyperbolic 3D grid generation kernel.
 *
 * Arrays are in row-major order: index = k*ny*nx + j*nx + i
 * (matches NumPy shape (nz, ny, nx)).
 *
 * x3d, y3d, z3d : coordinate arrays of size nx*ny*nz
 *                  On input, k=0 layer must be initialised.
 *                  On output, the full grid is filled.
 * step          : step sizes for each layer, size nz-1
 * nx, ny, nz    : grid dimensions
 * coef          : smoothing coefficient
 * t2b           : if 1, flip grid in z after generation
 * flag_stretch  : if 1, apply arc-length stretching in z
 * bdry_x_itype  : x-boundary type (0=none, 1=floating, 2=cartesian dx=0)
 * bdry_y_itype  : y-boundary type (0=none, 1=floating, 2=cartesian dy=0)
 * epsilon_x     : x-boundary floating parameter
 * epsilon_y     : y-boundary floating parameter
 */
void hyper_gene_c(float *x3d, float *y3d, float *z3d, float *step,
                  int nx, int ny, int nz,
                  float coef, int t2b, int flag_stretch,
                  int bdry_x_itype, int bdry_y_itype,
                  float epsilon_x, float epsilon_y);

#endif /* HYPERBOLIC_3D_H */
