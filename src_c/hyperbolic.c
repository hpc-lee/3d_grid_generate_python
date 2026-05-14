/*
 * hyperbolic.c - 3D hyperbolic grid generation kernel
 *
 * Marches layer by layer from k=1 to k=nz-1, solving block tridiagonal
 * systems in eta and xi directions at each step.
 *
 * All arrays use row-major indexing: index = k*ny*nx + j*nx + i
 * siz_iy = nx, siz_iz = nx*ny (computed locally)
 *
 * No struct usage - all parameters are raw pointers and integers.
 * No MPI calls - hyperbolic is serial.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "hyperbolic.h"
#include "lib_math.h"

/* ------------------------------------------------------------------ */
/*  static helper: thomas_block                                        */
/*  Block tridiagonal solver (Thomas algorithm) for 3x3 block system   */
/* ------------------------------------------------------------------ */
static int
thomas_block(int n, float *a, float *b, float *c, float *d,
             float *x, float *D, float *y)
{
  float mat_a[3][3], mat_b[3][3], mat_c[3][3], vec_d[3];
  float mat_G[3][3], mat_D[3][3], vec_y[3], vec_x[3];
  float mat1[3][3], vec[3], vec1[3], vec2[3], vec3[3];
  float jac;
  size_t iptr1, iptr2, iptr3, iptr4;

  /* i=0 */
  for (int ii = 0; ii < 3; ii++) {
    for (int jj = 0; jj < 3; jj++) {
      mat_G[ii][jj] = b[ii*3+jj];
      mat_c[ii][jj] = c[ii*3+jj];
    }
    vec_d[ii] = d[ii];
  }

  vec1[0] = mat_G[0][0]; vec1[1] = mat_G[0][1]; vec1[2] = mat_G[0][2];
  vec2[0] = mat_G[1][0]; vec2[1] = mat_G[1][1]; vec2[2] = mat_G[1][2];
  vec3[0] = mat_G[2][0]; vec3[1] = mat_G[2][1]; vec3[2] = mat_G[2][2];
  cross_product(vec1, vec2, vec);
  jac = dot_product(vec, vec3);
  if (fabs(jac) < 0.000001f) {
    fprintf(stderr, "b{1} is singular, stop calculate\n");
    exit(1);
  }

  mat_invert3x3(mat_G);
  mat_mul3x3(mat_G, mat_c, mat_D);
  mat_mul3x1(mat_G, vec_d, vec_y);
  for (int ii = 0; ii < 3; ii++) {
    for (int jj = 0; jj < 3; jj++) {
      D[ii*3+jj] = mat_D[ii][jj];
    }
    y[ii] = vec_y[ii];
  }

  /* i=1 ~ n-1 */
  for (int i = 1; i < n; i++) {
    iptr1 = i*3*3;
    iptr2 = i*3;
    iptr3 = (i-1)*3*3;
    iptr4 = (i-1)*3;
    for (int ii = 0; ii < 3; ii++) {
      for (int jj = 0; jj < 3; jj++) {
        mat_a[ii][jj] = a[iptr1+ii*3+jj];
        mat_b[ii][jj] = b[iptr1+ii*3+jj];
        mat_c[ii][jj] = c[iptr1+ii*3+jj];
        mat_D[ii][jj] = D[iptr3+ii*3+jj];
      }
      vec_d[ii] = d[iptr2+ii];
      vec_y[ii] = y[iptr4+ii];
    }
    mat_mul3x3(mat_a, mat_D, mat1);   /* a(i)*D(i-1)           */
    mat_sub3x3(mat_b, mat1, mat_G);   /* G(i) = b(i)-a(i)*D(i-1) */
    mat_invert3x3(mat_G);             /* inv(G(i))             */
    mat_mul3x3(mat_G, mat_c, mat_D);  /* D(i) = inv(G(i))*c(i) */
    mat_mul3x1(mat_a, vec_y, vec1);   /* a(i) * y(i-1)        */
    vec_sub3x1(vec_d, vec1, vec2);    /* d(i) - a(i)*y(i-1)   */
    mat_mul3x1(mat_G, vec2, vec_y);   /* y(i) = inv(G(i))*(d(i)-a(i)*y(i-1)) */

    for (int ii = 0; ii < 3; ii++) {
      for (int jj = 0; jj < 3; jj++) {
        D[iptr1+ii*3+jj] = mat_D[ii][jj];
      }
      y[iptr2+ii] = vec_y[ii];
    }
  }

  /* back substitution: i=n-1 */
  iptr2 = (n-1)*3;
  for (int ii = 0; ii < 3; ii++) {
    x[iptr2+ii] = y[iptr2+ii];
  }

  for (int i = n-2; i >= 0; i--) {
    iptr1 = i*3*3;
    iptr2 = i*3;
    iptr4 = (i+1)*3;

    for (int ii = 0; ii < 3; ii++) {
      for (int jj = 0; jj < 3; jj++) {
        mat_D[ii][jj] = D[iptr1+ii*3+jj];
      }
      vec_x[ii] = x[iptr4+ii];
    }
    mat_mul3x1(mat_D, vec_x, vec1);   /* D(i)*x(i+1) */
    for (int ii = 0; ii < 3; ii++) {
      x[iptr2+ii] = y[iptr2+ii] - vec1[ii];  /* u(i) = y(i) - D(i)*x(i+1) */
    }
  }

  return 0;
}

/* ------------------------------------------------------------------ */
/*  static helper: cal_smooth_coef                                     */
/*  Calculate adaptive dissipation (smooth) coefficients               */
/* ------------------------------------------------------------------ */
static int
cal_smooth_coef(const float *x3d, const float *y3d, const float *z3d,
                int nx, int ny, int nz,
                float coef, int k,
                float *coef_e_xi, float *coef_e_et)
{
  const size_t siz_iy = (size_t)nx;
  const size_t siz_iz = (size_t)nx * (size_t)ny;

  float S;
  size_t iptr1, iptr2, iptr3;
  float x_xi, y_xi, z_xi;
  float x_et, y_et, z_et;
  float x_zt, y_zt, z_zt;
  float xi_len, et_len, zt_len, N_xi, N_et;
  float x_xi_plus, y_xi_plus, z_xi_plus;
  float x_xi_minus, y_xi_minus, z_xi_minus;
  float x_et_plus, y_et_plus, z_et_plus;
  float x_et_minus, y_et_minus, z_et_minus;
  float xi_plus1, xi_minus1, xi_plus2, xi_minus2;
  float et_plus1, et_minus1, et_plus2, et_minus2;
  float d_xi1, d_xi2, d_et1, d_et2;
  float delta_xi, delta_xi_mdfy;
  float delta_et, delta_et_mdfy;
  float len_vec;
  float r_xi[3], r_et[3], vec_n[3];
  float cos_alpha_xi1, cos_alpha_xi2, cos_alpha_xi;
  float cos_alpha_et1, cos_alpha_et2, cos_alpha_et;
  float alpha_xi, alpha_et;

  S = sqrtf((1.0f*k) / (nz-1));

  if (k == 1) {
    int k1 = 2;
    for (int j = 1; j < ny-1; j++) {
      for (int i = 1; i < nx-1; i++) {
        iptr1 = (k1-1)*siz_iz + j*siz_iy + i+1;   /* (i+1,j,k-1) */
        iptr2 = (k1-1)*siz_iz + j*siz_iy + i-1;   /* (i-1,j,k-1) */
        x_xi = 0.5f*(x3d[iptr1] - x3d[iptr2]);
        y_xi = 0.5f*(y3d[iptr1] - y3d[iptr2]);
        z_xi = 0.5f*(z3d[iptr1] - z3d[iptr2]);

        iptr1 = (k1-1)*siz_iz + (j+1)*siz_iy + i; /* (i,j+1,k-1) */
        iptr2 = (k1-1)*siz_iz + (j-1)*siz_iy + i; /* (i,j-1,k-1) */
        x_et = 0.5f*(x3d[iptr1] - x3d[iptr2]);
        y_et = 0.5f*(y3d[iptr1] - y3d[iptr2]);
        z_et = 0.5f*(z3d[iptr1] - z3d[iptr2]);

        iptr1 = (k1-1)*siz_iz + j*siz_iy + i;     /* (i,j,k-1)   */
        iptr2 = (k1-2)*siz_iz + j*siz_iy + i;     /* (i,j,k-2)   */
        x_zt = x3d[iptr1] - x3d[iptr2];
        y_zt = y3d[iptr1] - y3d[iptr2];
        z_zt = z3d[iptr1] - z3d[iptr2];

        xi_len = sqrtf(x_xi*x_xi + y_xi*y_xi + z_xi*z_xi);
        et_len = sqrtf(x_et*x_et + y_et*y_et + z_et*z_et);
        zt_len = sqrtf(x_zt*x_zt + y_zt*y_zt + z_zt*z_zt);
        N_xi = zt_len / xi_len;
        N_et = zt_len / et_len;

        /* xi spacing at k-2 */
        iptr1 = (k1-2)*siz_iz + j*siz_iy + i+1;
        iptr2 = (k1-2)*siz_iz + j*siz_iy + i;
        iptr3 = (k1-2)*siz_iz + j*siz_iy + i-1;
        x_xi_plus = x3d[iptr1] - x3d[iptr2];
        y_xi_plus = y3d[iptr1] - y3d[iptr2];
        z_xi_plus = z3d[iptr1] - z3d[iptr2];
        xi_plus1 = sqrtf(x_xi_plus*x_xi_plus + y_xi_plus*y_xi_plus + z_xi_plus*z_xi_plus);
        x_xi_minus = x3d[iptr2] - x3d[iptr3];
        y_xi_minus = y3d[iptr2] - y3d[iptr3];
        z_xi_minus = z3d[iptr2] - z3d[iptr3];
        xi_minus1 = sqrtf(x_xi_minus*x_xi_minus + y_xi_minus*y_xi_minus + z_xi_minus*z_xi_minus);

        /* et spacing at k-2 */
        iptr1 = (k1-2)*siz_iz + (j+1)*siz_iy + i;
        iptr2 = (k1-2)*siz_iz + j*siz_iy + i;
        iptr3 = (k1-2)*siz_iz + (j-1)*siz_iy + i;
        x_et_plus = x3d[iptr1] - x3d[iptr2];
        y_et_plus = y3d[iptr1] - y3d[iptr2];
        z_et_plus = z3d[iptr1] - z3d[iptr2];
        et_plus1 = sqrtf(x_et_plus*x_et_plus + y_et_plus*y_et_plus + z_et_plus*z_et_plus);
        x_et_minus = x3d[iptr2] - x3d[iptr3];
        y_et_minus = y3d[iptr2] - y3d[iptr3];
        z_et_minus = z3d[iptr2] - z3d[iptr3];
        et_minus1 = sqrtf(x_et_minus*x_et_minus + y_et_minus*y_et_minus + z_et_minus*z_et_minus);

        /* xi spacing at k-1 */
        iptr1 = (k1-1)*siz_iz + j*siz_iy + i+1;
        iptr2 = (k1-1)*siz_iz + j*siz_iy + i;
        iptr3 = (k1-1)*siz_iz + j*siz_iy + i-1;
        x_xi_plus = x3d[iptr1] - x3d[iptr2];
        y_xi_plus = y3d[iptr1] - y3d[iptr2];
        z_xi_plus = z3d[iptr1] - z3d[iptr2];
        xi_plus2 = sqrtf(x_xi_plus*x_xi_plus + y_xi_plus*y_xi_plus + z_xi_plus*z_xi_plus);
        x_xi_minus = x3d[iptr2] - x3d[iptr3];
        y_xi_minus = y3d[iptr2] - y3d[iptr3];
        z_xi_minus = z3d[iptr2] - z3d[iptr3];
        xi_minus2 = sqrtf(x_xi_minus*x_xi_minus + y_xi_minus*y_xi_minus + z_xi_minus*z_xi_minus);

        /* et spacing at k-1 */
        iptr1 = (k1-1)*siz_iz + (j+1)*siz_iy + i;
        iptr2 = (k1-1)*siz_iz + j*siz_iy + i;
        iptr3 = (k1-1)*siz_iz + (j-1)*siz_iy + i;
        x_et_plus = x3d[iptr1] - x3d[iptr2];
        y_et_plus = y3d[iptr1] - y3d[iptr2];
        z_et_plus = z3d[iptr1] - z3d[iptr2];
        et_plus2 = sqrtf(x_et_plus*x_et_plus + y_et_plus*y_et_plus + z_et_plus*z_et_plus);
        x_et_minus = x3d[iptr2] - x3d[iptr3];
        y_et_minus = y3d[iptr2] - y3d[iptr3];
        z_et_minus = z3d[iptr2] - z3d[iptr3];
        et_minus2 = sqrtf(x_et_minus*x_et_minus + y_et_minus*y_et_minus + z_et_minus*z_et_minus);

        d_xi1 = xi_plus1 + xi_minus1;
        d_xi2 = xi_plus2 + xi_minus2;
        d_et1 = et_plus1 + et_minus1;
        d_et2 = et_plus2 + et_minus2;

        delta_xi = d_xi1 / d_xi2;
        delta_et = d_et1 / d_et2;
        delta_xi_mdfy = fmaxf(powf(delta_xi, 2.0f/S), 0.1f);
        delta_et_mdfy = fmaxf(powf(delta_et, 2.0f/S), 0.1f);

        /* normalization */
        iptr1 = (k1-1)*siz_iz + j*siz_iy + i+1;
        iptr2 = (k1-1)*siz_iz + j*siz_iy + i;
        iptr3 = (k1-1)*siz_iz + j*siz_iy + i-1;
        x_xi_plus = (x3d[iptr1]-x3d[iptr2]) / xi_plus2;
        y_xi_plus = (y3d[iptr1]-y3d[iptr2]) / xi_plus2;
        z_xi_plus = (z3d[iptr1]-z3d[iptr2]) / xi_plus2;
        x_xi_minus = (x3d[iptr3]-x3d[iptr2]) / xi_minus2;
        y_xi_minus = (y3d[iptr3]-y3d[iptr2]) / xi_minus2;
        z_xi_minus = (z3d[iptr3]-z3d[iptr2]) / xi_minus2;

        iptr1 = (k1-1)*siz_iz + (j+1)*siz_iy + i;
        iptr2 = (k1-1)*siz_iz + j*siz_iy + i;
        iptr3 = (k1-1)*siz_iz + (j-1)*siz_iy + i;
        x_et_plus = (x3d[iptr1]-x3d[iptr2]) / et_plus2;
        y_et_plus = (y3d[iptr1]-y3d[iptr2]) / et_plus2;
        z_et_plus = (z3d[iptr1]-z3d[iptr2]) / et_plus2;
        x_et_minus = (x3d[iptr3]-x3d[iptr2]) / et_minus2;
        y_et_minus = (y3d[iptr3]-y3d[iptr2]) / et_minus2;
        z_et_minus = (z3d[iptr3]-z3d[iptr2]) / et_minus2;

        r_xi[0] = x_xi_plus - x_xi_minus;
        r_xi[1] = y_xi_plus - y_xi_minus;
        r_xi[2] = z_xi_plus - z_xi_minus;
        r_et[0] = x_et_plus - x_et_minus;
        r_et[1] = y_et_plus - y_et_minus;
        r_et[2] = z_et_plus - z_et_minus;

        /* vec_n = r_xi X r_et */
        cross_product(r_xi, r_et, vec_n);
        len_vec = sqrtf(vec_n[0]*vec_n[0] + vec_n[1]*vec_n[1] + vec_n[2]*vec_n[2]);
        vec_n[0] /= len_vec;
        vec_n[1] /= len_vec;
        vec_n[2] /= len_vec;

        cos_alpha_xi1 = vec_n[0]*x_xi_plus  + vec_n[1]*y_xi_plus  + vec_n[2]*z_xi_plus;
        cos_alpha_xi2 = vec_n[0]*x_xi_minus + vec_n[1]*y_xi_minus + vec_n[2]*z_xi_minus;
        cos_alpha_xi = 0.5f*(cos_alpha_xi1 + cos_alpha_xi2);

        cos_alpha_et1 = vec_n[0]*x_et_plus  + vec_n[1]*y_et_plus  + vec_n[2]*z_et_plus;
        cos_alpha_et2 = vec_n[0]*x_et_minus + vec_n[1]*y_et_minus + vec_n[2]*z_et_minus;
        cos_alpha_et = 0.5f*(cos_alpha_et1 + cos_alpha_et2);

        if (cos_alpha_xi >= 0) {
          alpha_xi = 1.0f / (1.0f - cos_alpha_xi*cos_alpha_xi);
        } else {
          alpha_xi = 1.0f;
        }
        if (cos_alpha_et >= 0) {
          alpha_et = 1.0f / (1.0f - cos_alpha_et*cos_alpha_et);
        } else {
          alpha_et = 1.0f;
        }
        if (cos_alpha_xi > 1 || cos_alpha_xi < (-1) ||
            cos_alpha_et > 1 || cos_alpha_et < (-1)) {
          fprintf(stdout, "angle calculation is wrong\n");
          fflush(stdout);
          exit(1);
        }

        iptr1 = (j-1)*(nx-2) + i-1;
        coef_e_xi[iptr1] = coef * N_xi * S * delta_xi_mdfy * alpha_xi;
        coef_e_et[iptr1] = coef * N_et * S * delta_et_mdfy * alpha_et;
      }
    }
  }

  if (k > 1) {
    for (int j = 1; j < ny-1; j++) {
      for (int i = 1; i < nx-1; i++) {
        iptr1 = (k-1)*siz_iz + j*siz_iy + i+1;
        iptr2 = (k-1)*siz_iz + j*siz_iy + i-1;
        x_xi = 0.5f*(x3d[iptr1] - x3d[iptr2]);
        y_xi = 0.5f*(y3d[iptr1] - y3d[iptr2]);
        z_xi = 0.5f*(z3d[iptr1] - z3d[iptr2]);

        iptr1 = (k-1)*siz_iz + (j+1)*siz_iy + i;
        iptr2 = (k-1)*siz_iz + (j-1)*siz_iy + i;
        x_et = 0.5f*(x3d[iptr1] - x3d[iptr2]);
        y_et = 0.5f*(y3d[iptr1] - y3d[iptr2]);
        z_et = 0.5f*(z3d[iptr1] - z3d[iptr2]);

        iptr1 = (k-1)*siz_iz + j*siz_iy + i;
        iptr2 = (k-2)*siz_iz + j*siz_iy + i;
        x_zt = x3d[iptr1] - x3d[iptr2];
        y_zt = y3d[iptr1] - y3d[iptr2];
        z_zt = z3d[iptr1] - z3d[iptr2];

        xi_len = sqrtf(x_xi*x_xi + y_xi*y_xi + z_xi*z_xi);
        et_len = sqrtf(x_et*x_et + y_et*y_et + z_et*z_et);
        zt_len = sqrtf(x_zt*x_zt + y_zt*y_zt + z_zt*z_zt);
        N_xi = zt_len / xi_len;
        N_et = zt_len / et_len;

        /* xi spacing at k-2 */
        iptr1 = (k-2)*siz_iz + j*siz_iy + i+1;
        iptr2 = (k-2)*siz_iz + j*siz_iy + i;
        iptr3 = (k-2)*siz_iz + j*siz_iy + i-1;
        x_xi_plus = x3d[iptr1] - x3d[iptr2];
        y_xi_plus = y3d[iptr1] - y3d[iptr2];
        z_xi_plus = z3d[iptr1] - z3d[iptr2];
        xi_plus1 = sqrtf(x_xi_plus*x_xi_plus + y_xi_plus*y_xi_plus + z_xi_plus*z_xi_plus);
        x_xi_minus = x3d[iptr2] - x3d[iptr3];
        y_xi_minus = y3d[iptr2] - y3d[iptr3];
        z_xi_minus = z3d[iptr2] - z3d[iptr3];
        xi_minus1 = sqrtf(x_xi_minus*x_xi_minus + y_xi_minus*y_xi_minus + z_xi_minus*z_xi_minus);

        /* et spacing at k-2 */
        iptr1 = (k-2)*siz_iz + (j+1)*siz_iy + i;
        iptr2 = (k-2)*siz_iz + j*siz_iy + i;
        iptr3 = (k-2)*siz_iz + (j-1)*siz_iy + i;
        x_et_plus = x3d[iptr1] - x3d[iptr2];
        y_et_plus = y3d[iptr1] - y3d[iptr2];
        z_et_plus = z3d[iptr1] - z3d[iptr2];
        et_plus1 = sqrtf(x_et_plus*x_et_plus + y_et_plus*y_et_plus + z_et_plus*z_et_plus);
        x_et_minus = x3d[iptr2] - x3d[iptr3];
        y_et_minus = y3d[iptr2] - y3d[iptr3];
        z_et_minus = z3d[iptr2] - z3d[iptr3];
        et_minus1 = sqrtf(x_et_minus*x_et_minus + y_et_minus*y_et_minus + z_et_minus*z_et_minus);

        /* xi spacing at k-1 */
        iptr1 = (k-1)*siz_iz + j*siz_iy + i+1;
        iptr2 = (k-1)*siz_iz + j*siz_iy + i;
        iptr3 = (k-1)*siz_iz + j*siz_iy + i-1;
        x_xi_plus = x3d[iptr1] - x3d[iptr2];
        y_xi_plus = y3d[iptr1] - y3d[iptr2];
        z_xi_plus = z3d[iptr1] - z3d[iptr2];
        xi_plus2 = sqrtf(x_xi_plus*x_xi_plus + y_xi_plus*y_xi_plus + z_xi_plus*z_xi_plus);
        x_xi_minus = x3d[iptr2] - x3d[iptr3];
        y_xi_minus = y3d[iptr2] - y3d[iptr3];
        z_xi_minus = z3d[iptr2] - z3d[iptr3];
        xi_minus2 = sqrtf(x_xi_minus*x_xi_minus + y_xi_minus*y_xi_minus + z_xi_minus*z_xi_minus);

        /* et spacing at k-1 */
        iptr1 = (k-1)*siz_iz + (j+1)*siz_iy + i;
        iptr2 = (k-1)*siz_iz + j*siz_iy + i;
        iptr3 = (k-1)*siz_iz + (j-1)*siz_iy + i;
        x_et_plus = x3d[iptr1] - x3d[iptr2];
        y_et_plus = y3d[iptr1] - y3d[iptr2];
        z_et_plus = z3d[iptr1] - z3d[iptr2];
        et_plus2 = sqrtf(x_et_plus*x_et_plus + y_et_plus*y_et_plus + z_et_plus*z_et_plus);
        x_et_minus = x3d[iptr2] - x3d[iptr3];
        y_et_minus = y3d[iptr2] - y3d[iptr3];
        z_et_minus = z3d[iptr2] - z3d[iptr3];
        et_minus2 = sqrtf(x_et_minus*x_et_minus + y_et_minus*y_et_minus + z_et_minus*z_et_minus);

        d_xi1 = xi_plus1 + xi_minus1;
        d_xi2 = xi_plus2 + xi_minus2;
        d_et1 = et_plus1 + et_minus1;
        d_et2 = et_plus2 + et_minus2;

        delta_xi = d_xi1 / d_xi2;
        delta_et = d_et1 / d_et2;
        delta_xi_mdfy = fmaxf(powf(delta_xi, 2.0f/S), 0.1f);
        delta_et_mdfy = fmaxf(powf(delta_et, 2.0f/S), 0.1f);

        /* normalization */
        iptr1 = (k-1)*siz_iz + j*siz_iy + i+1;
        iptr2 = (k-1)*siz_iz + j*siz_iy + i;
        iptr3 = (k-1)*siz_iz + j*siz_iy + i-1;
        x_xi_plus = (x3d[iptr1]-x3d[iptr2]) / xi_plus2;
        y_xi_plus = (y3d[iptr1]-y3d[iptr2]) / xi_plus2;
        z_xi_plus = (z3d[iptr1]-z3d[iptr2]) / xi_plus2;
        x_xi_minus = (x3d[iptr3]-x3d[iptr2]) / xi_minus2;
        y_xi_minus = (y3d[iptr3]-y3d[iptr2]) / xi_minus2;
        z_xi_minus = (z3d[iptr3]-z3d[iptr2]) / xi_minus2;

        iptr1 = (k-1)*siz_iz + (j+1)*siz_iy + i;
        iptr2 = (k-1)*siz_iz + j*siz_iy + i;
        iptr3 = (k-1)*siz_iz + (j-1)*siz_iy + i;
        x_et_plus = (x3d[iptr1]-x3d[iptr2]) / et_plus2;
        y_et_plus = (y3d[iptr1]-y3d[iptr2]) / et_plus2;
        z_et_plus = (z3d[iptr1]-z3d[iptr2]) / et_plus2;
        x_et_minus = (x3d[iptr3]-x3d[iptr2]) / et_minus2;
        y_et_minus = (y3d[iptr3]-y3d[iptr2]) / et_minus2;
        z_et_minus = (z3d[iptr3]-z3d[iptr2]) / et_minus2;

        r_xi[0] = x_xi_plus - x_xi_minus;
        r_xi[1] = y_xi_plus - y_xi_minus;
        r_xi[2] = z_xi_plus - z_xi_minus;
        r_et[0] = x_et_plus - x_et_minus;
        r_et[1] = y_et_plus - y_et_minus;
        r_et[2] = z_et_plus - z_et_minus;

        /* vec_n = r_xi X r_et */
        cross_product(r_xi, r_et, vec_n);
        len_vec = sqrtf(vec_n[0]*vec_n[0] + vec_n[1]*vec_n[1] + vec_n[2]*vec_n[2]);
        vec_n[0] /= len_vec;
        vec_n[1] /= len_vec;
        vec_n[2] /= len_vec;

        cos_alpha_xi1 = vec_n[0]*x_xi_plus  + vec_n[1]*y_xi_plus  + vec_n[2]*z_xi_plus;
        cos_alpha_xi2 = vec_n[0]*x_xi_minus + vec_n[1]*y_xi_minus + vec_n[2]*z_xi_minus;
        cos_alpha_xi = 0.5f*(cos_alpha_xi1 + cos_alpha_xi2);

        cos_alpha_et1 = vec_n[0]*x_et_plus  + vec_n[1]*y_et_plus  + vec_n[2]*z_et_plus;
        cos_alpha_et2 = vec_n[0]*x_et_minus + vec_n[1]*y_et_minus + vec_n[2]*z_et_minus;
        cos_alpha_et = 0.5f*(cos_alpha_et1 + cos_alpha_et2);

        if (cos_alpha_xi >= 0) {
          alpha_xi = 1.0f / (1.0f - cos_alpha_xi*cos_alpha_xi);
        } else {
          alpha_xi = 1.0f;
        }
        if (cos_alpha_et >= 0) {
          alpha_et = 1.0f / (1.0f - cos_alpha_et*cos_alpha_et);
        } else {
          alpha_et = 1.0f;
        }
        if (cos_alpha_xi > 1 || cos_alpha_xi < (-1) ||
            cos_alpha_et > 1 || cos_alpha_et < (-1)) {
          fprintf(stdout, "angle calculation is wrong\n");
          fflush(stdout);
          exit(1);
        }

        iptr1 = (j-1)*(nx-2) + i-1;
        coef_e_xi[iptr1] = coef * N_xi * S * delta_xi_mdfy * alpha_xi;
        coef_e_et[iptr1] = coef * N_et * S * delta_et_mdfy * alpha_et;
      }
    }
  }

  return 0;
}

/* ------------------------------------------------------------------ */
/*  static helper: cal_matrix                                          */
/*  Calculate coefficient matrices A, B, C, D                          */
/* ------------------------------------------------------------------ */
static int
cal_matrix(const float *x3d, const float *y3d, const float *z3d,
           const float *step,
           int nx, int ny, int k,
           float *a_xi, float *b_xi, float *c_xi,
           float *a_et, float *b_et, float *c_et, float *d_et,
           float *volume)
{
  const size_t siz_iy = (size_t)nx;
  const size_t siz_iz = (size_t)nx * (size_t)ny;

  float A[3][3], B[3][3], C[3][3];
  float mat1[3][3], mat2[3][3], mat3[3][3];
  float vec_d[3], vec_g[3];
  size_t iptr, iptr1, iptr2;
  float r_xi0[3], r_et0[3], r_zt0[3];
  float sigma[3], omega[3], tau[3];
  float area;

  if (k == 1) {
    for (int j = 1; j < ny-1; j++) {
      for (int i = 1; i < nx-1; i++) {
        iptr1 = (k-1)*siz_iz + j*siz_iy + i+1;
        iptr2 = (k-1)*siz_iz + j*siz_iy + i-1;
        r_xi0[0] = 0.5f*(x3d[iptr1] - x3d[iptr2]);
        r_xi0[1] = 0.5f*(y3d[iptr1] - y3d[iptr2]);
        r_xi0[2] = 0.5f*(z3d[iptr1] - z3d[iptr2]);

        iptr1 = (k-1)*siz_iz + (j+1)*siz_iy + i;
        iptr2 = (k-1)*siz_iz + (j-1)*siz_iy + i;
        r_et0[0] = 0.5f*(x3d[iptr1] - x3d[iptr2]);
        r_et0[1] = 0.5f*(y3d[iptr1] - y3d[iptr2]);
        r_et0[2] = 0.5f*(z3d[iptr1] - z3d[iptr2]);

        /* sigma = r_xi0 X r_et0 */
        cross_product(r_xi0, r_et0, sigma);

        C[0][0] = r_xi0[0]+1e-7f; C[0][1] = r_xi0[1];       C[0][2] = r_xi0[2];
        C[1][0] = r_et0[0];       C[1][1] = r_et0[1]+1e-7f;  C[1][2] = r_et0[2];
        C[2][0] = sigma[0];       C[2][1] = sigma[1];        C[2][2] = sigma[2]+1e-7f;

        area = sqrtf(sigma[0]*sigma[0] + sigma[1]*sigma[1] + sigma[2]*sigma[2]);
        /* area->volume */
        /* volume(j,i,1) = V1   volume(j,i,2) = V0 */
        iptr = j*nx + i;
        volume[iptr] = area * step[k-1];
        volume[iptr+siz_iz] = volume[iptr];

        vec_g[0] = 0; vec_g[1] = 0; vec_g[2] = volume[iptr+siz_iz];

        /* r_zt0 = inv(C) * g */
        mat_invert3x3(C);
        mat_mul3x1(C, vec_g, r_zt0);

        /* omega = r_zt0 X r_xi0 */
        cross_product(r_zt0, r_xi0, omega);

        /* tau = r_et0 X r_zt0 */
        cross_product(r_et0, r_zt0, tau);

        A[0][0] = r_zt0[0]; A[0][1] = r_zt0[1]; A[0][2] = r_zt0[2];
        A[1][0] = 0;        A[1][1] = 0;        A[1][2] = 0;
        A[2][0] = tau[0];   A[2][1] = tau[1];   A[2][2] = tau[2];

        B[0][0] = 0;        B[0][1] = 0;        B[0][2] = 0;
        B[1][0] = r_zt0[0]; B[1][1] = r_zt0[1]; B[1][2] = r_zt0[2];
        B[2][0] = omega[0]; B[2][1] = omega[1]; B[2][2] = omega[2];

        mat_mul3x3(C, A, mat1);  /* mat1 = inv(C) * A */
        mat_mul3x3(C, B, mat2);  /* mat2 = inv(C) * B */
        mat_iden3x3(mat3);
        /* NOTE: now volume is volume(j,i,1) = V1 */
        vec_g[0] = 0; vec_g[1] = 0; vec_g[2] = volume[iptr];
        mat_mul3x1(C, vec_g, vec_d);

        iptr1 = ((j-1)*(nx-2) + (i-1))*3*3;
        iptr2 = ((j-1)*(nx-2) + (i-1))*3;
        for (int ii = 0; ii < 3; ii++) {
          for (int jj = 0; jj < 3; jj++) {
            a_xi[iptr1+3*ii+jj] = -0.5f*mat1[ii][jj];
            b_xi[iptr1+3*ii+jj] = mat3[ii][jj];
            c_xi[iptr1+3*ii+jj] = 0.5f*mat1[ii][jj];

            a_et[iptr1+3*ii+jj] = -0.5f*mat2[ii][jj];
            b_et[iptr1+3*ii+jj] = mat3[ii][jj];
            c_et[iptr1+3*ii+jj] = 0.5f*mat2[ii][jj];
          }
          d_et[iptr2+ii] = vec_d[ii];
        }
      }
    }
  } else {
    for (int j = 1; j < ny-1; j++) {
      for (int i = 1; i < nx-1; i++) {
        iptr1 = (k-1)*siz_iz + j*siz_iy + i+1;
        iptr2 = (k-1)*siz_iz + j*siz_iy + i-1;
        r_xi0[0] = 0.5f*(x3d[iptr1] - x3d[iptr2]);
        r_xi0[1] = 0.5f*(y3d[iptr1] - y3d[iptr2]);
        r_xi0[2] = 0.5f*(z3d[iptr1] - z3d[iptr2]);

        iptr1 = (k-1)*siz_iz + (j+1)*siz_iy + i;
        iptr2 = (k-1)*siz_iz + (j-1)*siz_iy + i;
        r_et0[0] = 0.5f*(x3d[iptr1] - x3d[iptr2]);
        r_et0[1] = 0.5f*(y3d[iptr1] - y3d[iptr2]);
        r_et0[2] = 0.5f*(z3d[iptr1] - z3d[iptr2]);

        /* sigma = r_xi0 X r_et0 */
        cross_product(r_xi0, r_et0, sigma);

        /* add damping factor, maybe inv(C) is singular */
        C[0][0] = r_xi0[0]+1e-7f; C[0][1] = r_xi0[1];       C[0][2] = r_xi0[2];
        C[1][0] = r_et0[0];       C[1][1] = r_et0[1]+1e-7f;  C[1][2] = r_et0[2];
        C[2][0] = sigma[0];       C[2][1] = sigma[1];        C[2][2] = sigma[2]+1e-7f;

        area = sqrtf(sigma[0]*sigma[0] + sigma[1]*sigma[1] + sigma[2]*sigma[2]);
        /* area->volume */
        /* volume(j,i,1) = V1   volume(j,i,2) = V0 */
        iptr = j*nx + i;
        volume[iptr+siz_iz] = volume[iptr];
        volume[iptr] = area * step[k-1];

        vec_g[0] = 0; vec_g[1] = 0; vec_g[2] = volume[iptr+siz_iz];

        /* r_zt0 = inv(C) * g */
        mat_invert3x3(C);
        mat_mul3x1(C, vec_g, r_zt0);

        /* omega = r_zt0 X r_xi0 */
        cross_product(r_zt0, r_xi0, omega);

        /* tau = r_et0 X r_zt0 */
        cross_product(r_et0, r_zt0, tau);

        A[0][0] = r_zt0[0]; A[0][1] = r_zt0[1]; A[0][2] = r_zt0[2];
        A[1][0] = 0;        A[1][1] = 0;        A[1][2] = 0;
        A[2][0] = tau[0];   A[2][1] = tau[1];   A[2][2] = tau[2];

        B[0][0] = 0;        B[0][1] = 0;        B[0][2] = 0;
        B[1][0] = r_zt0[0]; B[1][1] = r_zt0[1]; B[1][2] = r_zt0[2];
        B[2][0] = omega[0]; B[2][1] = omega[1]; B[2][2] = omega[2];

        mat_mul3x3(C, A, mat1);  /* mat1 = inv(C) * A */
        mat_mul3x3(C, B, mat2);  /* mat2 = inv(C) * B */
        mat_iden3x3(mat3);
        /* NOTE: now volume is volume(j,i,1) = V1 */
        vec_g[0] = 0; vec_g[1] = 0; vec_g[2] = volume[iptr];
        mat_mul3x1(C, vec_g, vec_d);

        iptr1 = ((j-1)*(nx-2) + (i-1))*3*3;
        iptr2 = ((j-1)*(nx-2) + (i-1))*3;
        for (int ii = 0; ii < 3; ii++) {
          for (int jj = 0; jj < 3; jj++) {
            a_xi[iptr1+3*ii+jj] = -0.5f*mat1[ii][jj];
            b_xi[iptr1+3*ii+jj] = mat3[ii][jj];
            c_xi[iptr1+3*ii+jj] = 0.5f*mat1[ii][jj];

            a_et[iptr1+3*ii+jj] = -0.5f*mat2[ii][jj];
            b_et[iptr1+3*ii+jj] = mat3[ii][jj];
            c_et[iptr1+3*ii+jj] = 0.5f*mat2[ii][jj];
          }
          d_et[iptr2+ii] = vec_d[ii];
        }
      }
    }
  }

  return 0;
}

/* ------------------------------------------------------------------ */
/*  static helper: modify_smooth                                       */
/*  Modify matrices for smoothness (dissipation operator)              */
/* ------------------------------------------------------------------ */
static int
modify_smooth(const float *x3d, const float *y3d, const float *z3d,
              int nx, int ny, int k,
              const float *coef_e_xi, const float *coef_e_et,
              float *a_xi, float *b_xi, float *c_xi,
              float *a_et, float *b_et, float *c_et, float *d_et)
{
  const size_t siz_iy = (size_t)nx;
  const size_t siz_iz = (size_t)nx * (size_t)ny;

  float vec1[3], vec2[3], vec3[3], vec4[3], vec5[3];
  float coef_i_xi, coef_i_et;
  size_t iptr1, iptr2, iptr3, iptr4, iptr5;
  float mat[3][3];
  mat_iden3x3(mat);

  for (int j = 1; j < ny-1; j++) {
    for (int i = 1; i < nx-1; i++) {
      iptr1 = (k-1)*siz_iz + j*siz_iy + i;      /* (i,j,k-1)   */
      iptr2 = (k-1)*siz_iz + j*siz_iy + i+1;    /* (i+1,j,k-1) */
      iptr3 = (k-1)*siz_iz + j*siz_iy + i-1;    /* (i-1,j,k-1) */
      iptr4 = (k-1)*siz_iz + (j+1)*siz_iy + i;  /* (i,j+1,k-1) */
      iptr5 = (k-1)*siz_iz + (j-1)*siz_iy + i;  /* (i,j-1,k-1) */
      vec1[0] = x3d[iptr1]; vec1[1] = y3d[iptr1]; vec1[2] = z3d[iptr1];
      vec2[0] = x3d[iptr2]; vec2[1] = y3d[iptr2]; vec2[2] = z3d[iptr2];
      vec3[0] = x3d[iptr3]; vec3[1] = y3d[iptr3]; vec3[2] = z3d[iptr3];
      vec4[0] = x3d[iptr4]; vec4[1] = y3d[iptr4]; vec4[2] = z3d[iptr4];
      vec5[0] = x3d[iptr5]; vec5[1] = y3d[iptr5]; vec5[2] = z3d[iptr5];

      iptr1 = ((j-1)*(nx-2)+(i-1))*3*3;
      iptr2 = ((j-1)*(nx-2)+(i-1))*3;
      iptr3 = (j-1)*(nx-2)+(i-1);

      coef_i_xi = 2*coef_e_xi[iptr3];
      coef_i_et = 2*coef_e_et[iptr3];

      for (int ii = 0; ii < 3; ii++) {
        for (int jj = 0; jj < 3; jj++) {
          a_xi[iptr1+3*ii+jj] = a_xi[iptr1+3*ii+jj] - coef_i_xi*mat[ii][jj];
          b_xi[iptr1+3*ii+jj] = b_xi[iptr1+3*ii+jj] + 2*coef_i_xi*mat[ii][jj];
          c_xi[iptr1+3*ii+jj] = c_xi[iptr1+3*ii+jj] - coef_i_xi*mat[ii][jj];

          a_et[iptr1+3*ii+jj] = a_et[iptr1+3*ii+jj] - coef_i_et*mat[ii][jj];
          b_et[iptr1+3*ii+jj] = b_et[iptr1+3*ii+jj] + 2*coef_i_et*mat[ii][jj];
          c_et[iptr1+3*ii+jj] = c_et[iptr1+3*ii+jj] - coef_i_et*mat[ii][jj];
        }
        d_et[iptr2+ii] = d_et[iptr2+ii]
                        + coef_e_xi[iptr3]*(vec2[ii]+vec3[ii]-2*vec1[ii])
                        + coef_e_et[iptr3]*(vec4[ii]+vec5[ii]-2*vec1[ii]);
      }
    }
  }

  return 0;
}

/* ------------------------------------------------------------------ */
/*  static helper: modify_bdry                                         */
/*  Apply boundary conditions (floating or Cartesian)                  */
/* ------------------------------------------------------------------ */
static int
modify_bdry(float *a_xi, float *b_xi, float *c_xi,
            float *a_et, float *b_et, float *c_et, float *d_et,
            int nx, int ny,
            float epsilon_x, int bdry_x_itype,
            float epsilon_y, int bdry_y_itype)
{
  size_t iptr1, iptr2;

  /* floating boundary in x */
  if (bdry_x_itype == 1) {
    for (int j = 1; j < ny-1; j++) {
      iptr1 = ((j-1)*(nx-2)+0)*3*3;
      iptr2 = ((j-1)*(nx-2)+(nx-3))*3*3;
      for (int ii = 0; ii < 3; ii++) {
        for (int jj = 0; jj < 3; jj++) {
          /* modify i=0 */
          b_xi[iptr1+ii*3+jj] = b_xi[iptr1+ii*3+jj] + (1+epsilon_x)*a_xi[iptr1+ii*3+jj];
          c_xi[iptr1+ii*3+jj] = c_xi[iptr1+ii*3+jj] - epsilon_x*a_xi[iptr1+ii*3+jj];
          /* modify i=nx-3 */
          b_xi[iptr2+ii*3+jj] = b_xi[iptr2+ii*3+jj] + (1+epsilon_x)*c_xi[iptr2+ii*3+jj];
          a_xi[iptr2+ii*3+jj] = a_xi[iptr2+ii*3+jj] - epsilon_x*c_xi[iptr2+ii*3+jj];
        }
      }
    }
  }
  /* cartesian boundary dx=0 */
  if (bdry_x_itype == 2) {
    for (int j = 1; j < ny-1; j++) {
      iptr1 = ((j-1)*(nx-2)+0)*3*3;
      iptr2 = ((j-1)*(nx-2)+(nx-3))*3*3;
      for (int ii = 0; ii < 3; ii++) {
        for (int jj = 0; jj < 3; jj++) {
          /* only modify second and third column
             due to dx=0, so first column equal to 0 */
          if (jj == 0) continue;
          /* modify i=0 */
          b_xi[iptr1+ii*3+jj] = b_xi[iptr1+ii*3+jj] + a_xi[iptr1+ii*3+jj];
          /* modify i=nx-3 */
          b_xi[iptr2+ii*3+jj] = b_xi[iptr2+ii*3+jj] + c_xi[iptr2+ii*3+jj];
        }
      }
    }
  }

  /* floating boundary in y */
  if (bdry_y_itype == 1) {
    for (int i = 1; i < nx-1; i++) {
      iptr1 = (0*(nx-2)+(i-1))*3*3;
      iptr2 = ((ny-3)*(nx-2)+(i-1))*3*3;
      for (int ii = 0; ii < 3; ii++) {
        for (int jj = 0; jj < 3; jj++) {
          /* modify j=0 */
          b_et[iptr1+ii*3+jj] = b_et[iptr1+ii*3+jj] + (1+epsilon_y)*a_et[iptr1+ii*3+jj];
          c_et[iptr1+ii*3+jj] = c_et[iptr1+ii*3+jj] - epsilon_y*a_et[iptr1+ii*3+jj];
          /* modify j=ny-3 */
          b_et[iptr2+ii*3+jj] = b_et[iptr2+ii*3+jj] + (1+epsilon_y)*c_et[iptr2+ii*3+jj];
          a_et[iptr2+ii*3+jj] = a_et[iptr2+ii*3+jj] - epsilon_y*c_et[iptr2+ii*3+jj];
        }
      }
    }
  }

  /* cartesian boundary dy=0 */
  if (bdry_y_itype == 2) {
    for (int i = 1; i < nx-1; i++) {
      iptr1 = (0*(nx-2)+i-1)*3*3;
      iptr2 = ((ny-3)*(nx-2)+i-1)*3*3;
      for (int ii = 0; ii < 3; ii++) {
        for (int jj = 0; jj < 3; jj++) {
          /* only modify first and third column
             due to dy=0, so second column equal to 0 */
          if (jj == 1) continue;
          /* modify j=0 */
          b_et[iptr1+ii*3+jj] = b_et[iptr1+ii*3+jj] + a_et[iptr1+ii*3+jj];
          /* modify j=ny-3 */
          b_et[iptr2+ii*3+jj] = b_et[iptr2+ii*3+jj] + c_et[iptr2+ii*3+jj];
        }
      }
    }
  }

  return 0;
}

/* ------------------------------------------------------------------ */
/*  static helper: solve_et_block                                      */
/*  Solve eta-direction block tridiagonal system                       */
/* ------------------------------------------------------------------ */
static int
solve_et_block(const float *a_et, const float *b_et, const float *c_et,
               const float *d_et, float *g_xi, int nx, int ny)
{
  size_t iptr1, iptr2;
  int n = ny - 2;
  float *a = NULL;
  float *b = NULL;
  float *c = NULL;
  float *d = NULL;
  float *g = NULL;
  a = (float *)calloc((size_t)n*3*3, sizeof(float));
  b = (float *)calloc((size_t)n*3*3, sizeof(float));
  c = (float *)calloc((size_t)n*3*3, sizeof(float));
  d = (float *)calloc((size_t)n*3,   sizeof(float));
  g = (float *)calloc((size_t)n*3,   sizeof(float));
  /* temp var */
  float *D = NULL;
  float *y = NULL;
  D = (float *)calloc((size_t)n*3*3, sizeof(float));
  y = (float *)calloc((size_t)n*3,   sizeof(float));

  for (int i = 0; i < nx-2; i++) {
    /* get i=const matrix */
    for (int j = 0; j < ny-2; j++) {
      iptr1 = (j*(nx-2) + i)*3*3;
      iptr2 = (j*(nx-2) + i)*3;
      for (int ii = 0; ii < 3; ii++) {
        for (int jj = 0; jj < 3; jj++) {
          a[j*3*3+3*ii+jj] = a_et[iptr1+3*ii+jj];
          b[j*3*3+3*ii+jj] = b_et[iptr1+3*ii+jj];
          c[j*3*3+3*ii+jj] = c_et[iptr1+3*ii+jj];
        }
        d[j*3+ii] = d_et[iptr2+ii];
      }
    }

    thomas_block(n, a, b, c, d, g, D, y);

    for (int j = 0; j < ny-2; j++) {
      iptr2 = (j*(nx-2) + i)*3;
      for (int ii = 0; ii < 3; ii++) {
        g_xi[iptr2+ii] = g[j*3+ii];
      }
    }
  }

  free(a);
  free(b);
  free(c);
  free(d);
  free(g);
  free(D);
  free(y);

  return 0;
}

/* ------------------------------------------------------------------ */
/*  static helper: solve_xi_block                                      */
/*  Solve xi-direction block tridiagonal system                        */
/* ------------------------------------------------------------------ */
static int
solve_xi_block(const float *a_xi, const float *b_xi, const float *c_xi,
               const float *g_xi, float *xyz, int nx, int ny)
{
  size_t iptr1, iptr2;
  int n = nx - 2;
  float *a, *b, *c, *g;
  float *cord = NULL;
  cord = (float *)calloc((size_t)n*3, sizeof(float));
  /* temp var */
  float *D = NULL;
  float *y = NULL;
  D = (float *)calloc((size_t)n*3*3, sizeof(float));
  y = (float *)calloc((size_t)n*3,   sizeof(float));

  for (int j = 0; j < ny-2; j++) {
    /* get j=const matrix */
    iptr1 = (j*(nx-2)+0)*3*3;
    iptr2 = (j*(nx-2)+0)*3;
    a = (float *)(a_xi + iptr1);
    b = (float *)(b_xi + iptr1);
    c = (float *)(c_xi + iptr1);
    g = (float *)(g_xi + iptr2);

    thomas_block(n, a, b, c, g, cord, D, y);

    for (int i = 0; i < nx-2; i++) {
      iptr2 = (j*(nx-2) + i)*3;
      for (int ii = 0; ii < 3; ii++) {
        xyz[iptr2+ii] = cord[i*3+ii];
      }
    }
  }

  free(cord);
  free(D);
  free(y);

  return 0;
}

/* ------------------------------------------------------------------ */
/*  static helper: assign_coords                                       */
/*  Assign new coordinates from the solved increment                   */
/* ------------------------------------------------------------------ */
static int
assign_coords(float *x3d, float *y3d, float *z3d,
              int nx, int ny, int k,
              const float *xyz,
              float epsilon_x, int bdry_x_itype,
              float epsilon_y, int bdry_y_itype)
{
  const size_t siz_iy = (size_t)nx;
  const size_t siz_iz = (size_t)nx * (size_t)ny;

  size_t iptr, iptr1, iptr2;
  size_t iptr3, iptr4, iptr5;

  /* interior points */
  for (int j = 1; j < ny-1; j++) {
    for (int i = 1; i < nx-1; i++) {
      iptr  =  k*siz_iz + j*siz_iy + i;
      iptr1 = (k-1)*siz_iz + j*siz_iy + i;
      iptr2 = ((j-1)*(nx-2) + (i-1))*3;
      x3d[iptr] = x3d[iptr1] + xyz[iptr2];
      y3d[iptr] = y3d[iptr1] + xyz[iptr2+1];
      z3d[iptr] = z3d[iptr1] + xyz[iptr2+2];
    }
  }

  /* floating boundary in x */
  if (bdry_x_itype == 1) {
    for (int j = 0; j < ny; j++) {
      /* i=0 */
      iptr  = k*siz_iz + j*siz_iy + 0;
      iptr1 = (k-1)*siz_iz + j*siz_iy + 0;
      iptr2 = k*siz_iz + j*siz_iy + 1;
      iptr3 = (k-1)*siz_iz + j*siz_iy + 1;
      iptr4 = k*siz_iz + j*siz_iy + 2;
      iptr5 = (k-1)*siz_iz + j*siz_iy + 2;

      x3d[iptr] = x3d[iptr1] + (1+epsilon_x)*(x3d[iptr2]-x3d[iptr3])
                 - epsilon_x*(x3d[iptr4]-x3d[iptr5]);
      y3d[iptr] = y3d[iptr1] + (1+epsilon_x)*(y3d[iptr2]-y3d[iptr3])
                 - epsilon_x*(y3d[iptr4]-y3d[iptr5]);
      z3d[iptr] = z3d[iptr1] + (1+epsilon_x)*(z3d[iptr2]-z3d[iptr3])
                 - epsilon_x*(z3d[iptr4]-z3d[iptr5]);

      /* i=nx-1 */
      iptr  = k*siz_iz + j*siz_iy + nx-1;
      iptr1 = (k-1)*siz_iz + j*siz_iy + nx-1;
      iptr2 = k*siz_iz + j*siz_iy + nx-2;
      iptr3 = (k-1)*siz_iz + j*siz_iy + nx-2;
      iptr4 = k*siz_iz + j*siz_iy + nx-3;
      iptr5 = (k-1)*siz_iz + j*siz_iy + nx-3;

      x3d[iptr] = x3d[iptr1] + (1+epsilon_x)*(x3d[iptr2]-x3d[iptr3])
                 - epsilon_x*(x3d[iptr4]-x3d[iptr5]);
      y3d[iptr] = y3d[iptr1] + (1+epsilon_x)*(y3d[iptr2]-y3d[iptr3])
                 - epsilon_x*(y3d[iptr4]-y3d[iptr5]);
      z3d[iptr] = z3d[iptr1] + (1+epsilon_x)*(z3d[iptr2]-z3d[iptr3])
                 - epsilon_x*(z3d[iptr4]-z3d[iptr5]);
    }
  }

  /* cartesian boundary dx=0 */
  if (bdry_x_itype == 2) {
    for (int j = 0; j < ny; j++) {
      /* i=0 */
      iptr  = k*siz_iz + j*siz_iy + 0;
      iptr1 = (k-1)*siz_iz + j*siz_iy + 0;
      iptr2 = k*siz_iz + j*siz_iy + 1;
      iptr3 = (k-1)*siz_iz + j*siz_iy + 1;

      x3d[iptr] = x3d[iptr1];
      y3d[iptr] = y3d[iptr1] + (y3d[iptr2]-y3d[iptr3]);
      z3d[iptr] = z3d[iptr1] + (z3d[iptr2]-z3d[iptr3]);

      /* i=nx-1 */
      iptr  = k*siz_iz + j*siz_iy + nx-1;
      iptr1 = (k-1)*siz_iz + j*siz_iy + nx-1;
      iptr2 = k*siz_iz + j*siz_iy + nx-2;
      iptr3 = (k-1)*siz_iz + j*siz_iy + nx-2;

      x3d[iptr] = x3d[iptr1];
      y3d[iptr] = y3d[iptr1] + (y3d[iptr2]-y3d[iptr3]);
      z3d[iptr] = z3d[iptr1] + (z3d[iptr2]-z3d[iptr3]);
    }
  }

  /* floating boundary in y */
  if (bdry_y_itype == 1) {
    for (int i = 0; i < nx; i++) {
      /* j=0 */
      iptr  = k*siz_iz + 0*siz_iy + i;
      iptr1 = (k-1)*siz_iz + 0*siz_iy + i;
      iptr2 = k*siz_iz + 1*siz_iy + i;
      iptr3 = (k-1)*siz_iz + 1*siz_iy + i;
      iptr4 = k*siz_iz + 2*siz_iy + i;
      iptr5 = (k-1)*siz_iz + 2*siz_iy + i;

      x3d[iptr] = x3d[iptr1] + (1+epsilon_y)*(x3d[iptr2]-x3d[iptr3])
                 - epsilon_y*(x3d[iptr4]-x3d[iptr5]);
      y3d[iptr] = y3d[iptr1] + (1+epsilon_y)*(y3d[iptr2]-y3d[iptr3])
                 - epsilon_y*(y3d[iptr4]-y3d[iptr5]);
      z3d[iptr] = z3d[iptr1] + (1+epsilon_y)*(z3d[iptr2]-z3d[iptr3])
                 - epsilon_y*(z3d[iptr4]-z3d[iptr5]);

      /* j=ny-1 */
      iptr  = k*siz_iz + (ny-1)*siz_iy + i;
      iptr1 = (k-1)*siz_iz + (ny-1)*siz_iy + i;
      iptr2 = k*siz_iz + (ny-2)*siz_iy + i;
      iptr3 = (k-1)*siz_iz + (ny-2)*siz_iy + i;
      iptr4 = k*siz_iz + (ny-3)*siz_iy + i;
      iptr5 = (k-1)*siz_iz + (ny-3)*siz_iy + i;

      x3d[iptr] = x3d[iptr1] + (1+epsilon_y)*(x3d[iptr2]-x3d[iptr3])
                 - epsilon_y*(x3d[iptr4]-x3d[iptr5]);
      y3d[iptr] = y3d[iptr1] + (1+epsilon_y)*(y3d[iptr2]-y3d[iptr3])
                 - epsilon_y*(y3d[iptr4]-y3d[iptr5]);
      z3d[iptr] = z3d[iptr1] + (1+epsilon_y)*(z3d[iptr2]-z3d[iptr3])
                 - epsilon_y*(z3d[iptr4]-z3d[iptr5]);
    }
  }

  /* cartesian boundary dy=0 */
  if (bdry_y_itype == 2) {
    for (int i = 0; i < nx; i++) {
      /* j=0 */
      iptr  = k*siz_iz + 0*siz_iy + i;
      iptr1 = (k-1)*siz_iz + 0*siz_iy + i;
      iptr2 = k*siz_iz + 1*siz_iy + i;
      iptr3 = (k-1)*siz_iz + 1*siz_iy + i;

      x3d[iptr] = x3d[iptr1] + (x3d[iptr2]-x3d[iptr3]);
      y3d[iptr] = y3d[iptr1];
      z3d[iptr] = z3d[iptr1] + (z3d[iptr2]-z3d[iptr3]);

      /* j=ny-1 */
      iptr  = k*siz_iz + (ny-1)*siz_iy + i;
      iptr1 = (k-1)*siz_iz + (ny-1)*siz_iy + i;
      iptr2 = k*siz_iz + (ny-2)*siz_iy + i;
      iptr3 = (k-1)*siz_iz + (ny-2)*siz_iy + i;

      x3d[iptr] = x3d[iptr1] + (x3d[iptr2]-x3d[iptr3]);
      y3d[iptr] = y3d[iptr1];
      z3d[iptr] = z3d[iptr1] + (z3d[iptr2]-z3d[iptr3]);
    }
  }

  return 0;
}

/* ================================================================== */
/*  Main entry point: hyper_gene_c                                     */
/* ================================================================== */
void
hyper_gene_c(float *x3d, float *y3d, float *z3d, float *step,
             int nx, int ny, int nz,
             float coef, int t2b, int flag_stretch,
             int bdry_x_itype, int bdry_y_itype,
             float epsilon_x, float epsilon_y)
{
  const size_t siz_iy = (size_t)nx;
  const size_t siz_iz = (size_t)nx * (size_t)ny;
  size_t size = (size_t)(nx-2) * (size_t)(ny-2);  /* not include bdry 2 points */

  float *coef_e_xi = NULL;
  float *coef_e_et = NULL;
  float *volume    = NULL;

  coef_e_xi = (float *)calloc(size, sizeof(float));
  coef_e_et = (float *)calloc(size, sizeof(float));
  volume    = (float *)calloc((size_t)nx * (size_t)ny * 2, sizeof(float));

  /* malloc space for thomas_block method
     each element is 3x3 matrix */
  /* xi */
  float *a_xi = NULL;
  float *b_xi = NULL;
  float *c_xi = NULL;
  float *g_xi = NULL;

  a_xi = (float *)calloc(size*3*3, sizeof(float));
  b_xi = (float *)calloc(size*3*3, sizeof(float));
  c_xi = (float *)calloc(size*3*3, sizeof(float));
  g_xi = (float *)calloc(size*3,   sizeof(float));

  /* et */
  float *a_et = NULL;
  float *b_et = NULL;
  float *c_et = NULL;
  float *d_et = NULL;

  a_et = (float *)calloc(size*3*3, sizeof(float));
  b_et = (float *)calloc(size*3*3, sizeof(float));
  c_et = (float *)calloc(size*3*3, sizeof(float));
  d_et = (float *)calloc(size*3,   sizeof(float));

  float *xyz = NULL;
  xyz = (float *)calloc(size*3, sizeof(float));

  /* solve first layer (k=1) without smooth coefficients */
  {
    int k = 1;
    cal_matrix(x3d, y3d, z3d, step, nx, ny, k,
               a_xi, b_xi, c_xi, a_et, b_et, c_et, d_et, volume);
    modify_smooth(x3d, y3d, z3d, nx, ny, k,
                  coef_e_xi, coef_e_et,
                  a_xi, b_xi, c_xi, a_et, b_et, c_et, d_et);
    modify_bdry(a_xi, b_xi, c_xi, a_et, b_et, c_et, d_et,
                nx, ny, epsilon_x, bdry_x_itype, epsilon_y, bdry_y_itype);
    solve_et_block(a_et, b_et, c_et, d_et, g_xi, nx, ny);
    solve_xi_block(a_xi, b_xi, c_xi, g_xi, xyz, nx, ny);
    assign_coords(x3d, y3d, z3d, nx, ny, k, xyz,
                  epsilon_x, bdry_x_itype, epsilon_y, bdry_y_itype);
  }

  /* march remaining layers */
  for (int k = 1; k < nz; k++) {
    cal_smooth_coef(x3d, y3d, z3d, nx, ny, nz, coef, k,
                    coef_e_xi, coef_e_et);
    cal_matrix(x3d, y3d, z3d, step, nx, ny, k,
               a_xi, b_xi, c_xi, a_et, b_et, c_et, d_et, volume);
    modify_smooth(x3d, y3d, z3d, nx, ny, k,
                  coef_e_xi, coef_e_et,
                  a_xi, b_xi, c_xi, a_et, b_et, c_et, d_et);
    modify_bdry(a_xi, b_xi, c_xi, a_et, b_et, c_et, d_et,
                nx, ny, epsilon_x, bdry_x_itype, epsilon_y, bdry_y_itype);
    solve_et_block(a_et, b_et, c_et, d_et, g_xi, nx, ny);
    solve_xi_block(a_xi, b_xi, c_xi, g_xi, xyz, nx, ny);
    assign_coords(x3d, y3d, z3d, nx, ny, k, xyz,
                  epsilon_x, bdry_x_itype, epsilon_y, bdry_y_itype);
    fprintf(stdout, "number of layer is %d\n", k);
    fflush(stdout);
  }

  /* arc-length stretching in z */
  if (flag_stretch == 1) {
    float *x3d_temp = (float *)calloc((size_t)nz, sizeof(float));
    float *y3d_temp = (float *)calloc((size_t)nz, sizeof(float));
    float *z3d_temp = (float *)calloc((size_t)nz, sizeof(float));
    float *arc_len  = (float *)calloc((size_t)nz, sizeof(float));
    float *s        = (float *)calloc((size_t)nz, sizeof(float));
    float *u        = (float *)calloc((size_t)nz, sizeof(float));

    /* build arc_len from step array */
    float arc_len_sum = 0;
    for (int kk = 1; kk < nz; kk++) {
      arc_len[kk] = arc_len[kk-1] + step[kk-1];
      arc_len_sum += step[kk-1];
    }
    for (int kk = 0; kk < nz; kk++) {
      arc_len[kk] /= arc_len_sum;
    }

    /* line by line */
    for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {
        /* copy old coords to temp space */
        for (int kk = 0; kk < nz; kk++) {
          size_t ip = (size_t)kk*siz_iz + (size_t)j*siz_iy + (size_t)i;
          x3d_temp[kk] = x3d[ip];
          y3d_temp[kk] = y3d[ip];
          z3d_temp[kk] = z3d[ip];
        }
        /* cal arc length */
        for (int kk = 1; kk < nz; kk++) {
          float xl = x3d_temp[kk] - x3d_temp[kk-1];
          float yl = y3d_temp[kk] - y3d_temp[kk-1];
          float zl = z3d_temp[kk] - z3d_temp[kk-1];
          float dh = sqrtf(xl*xl + yl*yl + zl*zl);
          s[kk] = s[kk-1] + dh;
        }
        /* arc length normalized */
        for (int kk = 0; kk < nz; kk++) {
          u[kk] = s[kk] / s[nz-1];
        }
        for (int kk = 1; kk < nz-1; kk++) {
          float r = arc_len[kk];
          int n = 0;
          for (int m = 0; m < nz-1; m++) {
            if (r >= u[m] && r < u[m+1]) {
              n = m;
              break;
            }
          }
          /* linear interp */
          size_t ip = (size_t)kk*siz_iz + (size_t)j*siz_iy + (size_t)i;
          float xl = x3d_temp[n+1] - x3d_temp[n];
          float yl = y3d_temp[n+1] - y3d_temp[n];
          float zl = z3d_temp[n+1] - z3d_temp[n];
          float ratio = (r - u[n]) / (u[n+1] - u[n]);
          x3d[ip] = x3d_temp[n] + xl*ratio;
          y3d[ip] = y3d_temp[n] + yl*ratio;
          z3d[ip] = z3d_temp[n] + zl*ratio;
        }
      }
    }

    free(x3d_temp);
    free(y3d_temp);
    free(z3d_temp);
    free(arc_len);
    free(s);
    free(u);
  }

  /* flip coordinates if t2b=1 */
  if (t2b == 1) {
    size_t siz_icmp = (size_t)nx * (size_t)ny * (size_t)nz;
    float *tmp_coord_x = (float *)malloc(siz_icmp * sizeof(float));
    float *tmp_coord_y = (float *)malloc(siz_icmp * sizeof(float));
    float *tmp_coord_z = (float *)malloc(siz_icmp * sizeof(float));

    /* copy data */
    for (int kk = 0; kk < nz; kk++) {
      for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
          size_t ip = (size_t)kk*siz_iz + (size_t)j*siz_iy + (size_t)i;
          tmp_coord_x[ip] = x3d[ip];
          tmp_coord_y[ip] = y3d[ip];
          tmp_coord_z[ip] = z3d[ip];
        }
      }
    }
    /* flip coord: nz-1->0, 0->nz-1, i->(nz-1)-i */
    for (int kk = 0; kk < nz; kk++) {
      for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
          size_t ip  = (size_t)kk*siz_iz + (size_t)j*siz_iy + (size_t)i;
          size_t ip1 = (size_t)(nz-1-kk)*siz_iz + (size_t)j*siz_iy + (size_t)i;
          x3d[ip] = tmp_coord_x[ip1];
          y3d[ip] = tmp_coord_y[ip1];
          z3d[ip] = tmp_coord_z[ip1];
        }
      }
    }

    free(tmp_coord_x);
    free(tmp_coord_y);
    free(tmp_coord_z);
  }

  free(coef_e_xi);
  free(coef_e_et);
  free(volume);
  free(a_xi);
  free(b_xi);
  free(c_xi);
  free(g_xi);
  free(a_et);
  free(b_et);
  free(c_et);
  free(d_et);
  free(xyz);
}
