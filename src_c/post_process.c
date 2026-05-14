#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "post_process.h"

/*
 * Post-processing for 3D curvilinear grids.
 *
 * Index convention (row-major, matches NumPy (nz,ny,nx)):
 *   index = k * ny * nx + j * nx + i
 *
 * siz_iy = nx,  siz_iz = nx * ny   (kept as local variables for clarity)
 */

/* ------------------------------------------------------------------ */
/*  Local helpers (from lib_math, inlined to avoid static-link issues)  */
/* ------------------------------------------------------------------ */
static void
cross_product_local(float *A, float *B, float *C)
{
  C[0] = A[1] * B[2] - A[2] * B[1];
  C[1] = A[2] * B[0] - A[0] * B[2];
  C[2] = A[0] * B[1] - A[1] * B[0];
}

static float
dot_product_local(float *A, float *B)
{
  return A[0] * B[0] + A[1] * B[1] + A[2] * B[2];
}

/* Distance from point x0 to plane defined by points x1, x2, x3 */
static float
dist_point2plane_local(float x0[3], float x1[3], float x2[3], float x3[3])
{
  float x12[3], x13[3], p[3];
  x12[0] = x2[0] - x1[0];
  x12[1] = x2[1] - x1[1];
  x12[2] = x2[2] - x1[2];
  x13[0] = x3[0] - x1[0];
  x13[1] = x3[1] - x1[1];
  x13[2] = x3[2] - x1[2];

  cross_product_local(x12, x13, p);
  float d = dot_product_local(p, x1);
  float L = fabsf(dot_product_local(p, x0) - d);
  L = L / sqrtf(dot_product_local(p, p));

  return L;
}

/* ------------------------------------------------------------------ */
/*  sample_interp_c                                                    */
/* ------------------------------------------------------------------ */
void
sample_interp_c(float *x3d, float *y3d, float *z3d,
                float *x3d_new, float *y3d_new, float *z3d_new,
                int nx, int ny, int nz,
                int nx_new, int ny_new, int nz_new)
{
  const size_t siz_iy     = (size_t)nx;
  const size_t siz_iz     = (size_t)(nx * ny);
  const size_t siz_iy_new = (size_t)nx_new;
  const size_t siz_iz_new = (size_t)(nx_new * ny_new);

  float *x3d_temp_y, *y3d_temp_y, *z3d_temp_y;
  float *x3d_temp_x, *y3d_temp_x, *z3d_temp_x;
  float *u_z, *u_y, *u_x;
  size_t iptr, iptr1, iptr2;
  float x_len, y_len, z_len;
  float r, ratio;
  int n;

  /* Allocate temp buffers ------------------------------------------------ */
  u_z = (float *)calloc(nz, sizeof(float));
  u_y = (float *)calloc(ny, sizeof(float));
  u_x = (float *)calloc(nx, sizeof(float));

  x3d_temp_y = (float *)calloc(ny, sizeof(float));
  y3d_temp_y = (float *)calloc(ny, sizeof(float));
  z3d_temp_y = (float *)calloc(ny, sizeof(float));

  x3d_temp_x = (float *)calloc(nx, sizeof(float));
  y3d_temp_x = (float *)calloc(nx, sizeof(float));
  z3d_temp_x = (float *)calloc(nx, sizeof(float));

  /* ---- Step 1: interpolate along zt (k) direction -------------------- */
  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++) {
      /* Normalized coordinate for original z */
      for (int k = 0; k < nz; k++) {
        u_z[k] = (float)k / (nz - 1);
      }
      for (int k1 = 0; k1 < nz_new; k1++) {
        r = (float)k1 / (nz_new - 1);
        /* Find interval */
        n = 0;
        for (int m = 0; m < nz - 1; m++) {
          if (r >= u_z[m] && r < u_z[m + 1]) {
            n = m;
            break;
          }
        }
        /* Linear interpolation */
        iptr  = (size_t)k1 * siz_iz_new + (size_t)j * siz_iy_new + i;
        iptr1 = (size_t)n   * siz_iz     + (size_t)j * siz_iy     + i;
        iptr2 = (size_t)(n + 1) * siz_iz + (size_t)j * siz_iy     + i;
        x_len  = x3d[iptr2] - x3d[iptr1];
        y_len  = y3d[iptr2] - y3d[iptr1];
        z_len  = z3d[iptr2] - z3d[iptr1];
        ratio  = (r - u_z[n]) / (u_z[n + 1] - u_z[n]);
        x3d_new[iptr] = x3d[iptr1] + x_len * ratio;
        y3d_new[iptr] = y3d[iptr1] + y_len * ratio;
        z3d_new[iptr] = z3d[iptr1] + z_len * ratio;
      }
    }
  }

  /* ---- Step 2: interpolate along et (j) direction -------------------- */
  for (int k1 = 0; k1 < nz_new; k1++) {
    for (int i = 0; i < nx; i++) {
      /* Copy z-interpolated coords to temp */
      for (int j = 0; j < ny; j++) {
        iptr = (size_t)k1 * siz_iz_new + (size_t)j * siz_iy_new + i;
        x3d_temp_y[j] = x3d_new[iptr];
        y3d_temp_y[j] = y3d_new[iptr];
        z3d_temp_y[j] = z3d_new[iptr];
      }
      /* Normalized coordinate for original y */
      for (int j = 0; j < ny; j++) {
        u_y[j] = (float)j / (ny - 1);
      }
      for (int j1 = 0; j1 < ny_new; j1++) {
        r = (float)j1 / (ny_new - 1);
        /* Find interval */
        n = 0;
        for (int m = 0; m < ny - 1; m++) {
          if (r >= u_y[m] && r < u_y[m + 1]) {
            n = m;
            break;
          }
        }
        /* Linear interpolation */
        iptr = (size_t)k1 * siz_iz_new + (size_t)j1 * siz_iy_new + i;
        x_len = x3d_temp_y[n + 1] - x3d_temp_y[n];
        y_len = y3d_temp_y[n + 1] - y3d_temp_y[n];
        z_len = z3d_temp_y[n + 1] - z3d_temp_y[n];
        ratio = (r - u_y[n]) / (u_y[n + 1] - u_y[n]);
        x3d_new[iptr] = x3d_temp_y[n] + x_len * ratio;
        y3d_new[iptr] = y3d_temp_y[n] + y_len * ratio;
        z3d_new[iptr] = z3d_temp_y[n] + z_len * ratio;
      }
    }
  }

  /* ---- Step 3: interpolate along xi (i) direction -------------------- */
  for (int k1 = 0; k1 < nz_new; k1++) {
    for (int j1 = 0; j1 < ny_new; j1++) {
      /* Copy y-interpolated coords to temp */
      for (int i = 0; i < nx; i++) {
        iptr = (size_t)k1 * siz_iz_new + (size_t)j1 * siz_iy_new + i;
        x3d_temp_x[i] = x3d_new[iptr];
        y3d_temp_x[i] = y3d_new[iptr];
        z3d_temp_x[i] = z3d_new[iptr];
      }
      /* Normalized coordinate for original x */
      for (int i = 0; i < nx; i++) {
        u_x[i] = (float)i / (nx - 1);
      }
      for (int i1 = 0; i1 < nx_new; i1++) {
        r = (float)i1 / (nx_new - 1);
        /* Find interval */
        n = 0;
        for (int m = 0; m < nx - 1; m++) {
          if (r >= u_x[m] && r < u_x[m + 1]) {
            n = m;
            break;
          }
        }
        /* Linear interpolation */
        iptr = (size_t)k1 * siz_iz_new + (size_t)j1 * siz_iy_new + i1;
        x_len = x3d_temp_x[n + 1] - x3d_temp_x[n];
        y_len = y3d_temp_x[n + 1] - y3d_temp_x[n];
        z_len = z3d_temp_x[n + 1] - z3d_temp_x[n];
        ratio = (r - u_x[n]) / (u_x[n + 1] - u_x[n]);
        x3d_new[iptr] = x3d_temp_x[n] + x_len * ratio;
        y3d_new[iptr] = y3d_temp_x[n] + y_len * ratio;
        z3d_new[iptr] = z3d_temp_x[n] + z_len * ratio;
      }
    }
  }

  /* Free temp buffers */
  free(u_z);
  free(u_y);
  free(u_x);
  free(x3d_temp_y);
  free(y3d_temp_y);
  free(z3d_temp_y);
  free(x3d_temp_x);
  free(y3d_temp_x);
  free(z3d_temp_x);
}

/* ------------------------------------------------------------------ */
/*  cal_min_dist_c                                                      */
/* ------------------------------------------------------------------ */
void
cal_min_dist_c(float *x3d, float *y3d, float *z3d,
               int nx, int ny, int nz,
               int *indx_i, int *indx_j, int *indx_k,
               float *dL_min)
{
  const size_t siz_iy = (size_t)nx;
  const size_t siz_iz = (size_t)(nx * ny);

  float dL_min_global = 1.0e10f;

  for (int k = 1; k < nz - 1; k++) {
    for (int j = 1; j < ny - 1; j++) {
      for (int i = 1; i < nx - 1; i++) {
        size_t iptr = (size_t)i + (size_t)j * siz_iy + (size_t)k * siz_iz;
        float p0[3] = { x3d[iptr], y3d[iptr], z3d[iptr] };

        float dL_min_local = 1.0e10f;

        /* Min distance to 8 planes formed by the 6 face-neighbours */
        for (int kk = -1; kk <= 1; kk += 2) {
          for (int jj = -1; jj <= 1; jj += 2) {
            for (int ii = -1; ii <= 1; ii += 2) {
              /* Three points defining each plane:
               *   p1 = neighbour in xi  direction (offset ii)
               *   p2 = neighbour in eta direction (offset jj)
               *   p3 = neighbour in zt  direction (offset kk)  */
              size_t i1 = (size_t)(i - ii) + (size_t)j * siz_iy + (size_t)k * siz_iz;
              size_t i2 = (size_t)i + (size_t)(j - jj) * siz_iy + (size_t)k * siz_iz;
              size_t i3 = (size_t)i + (size_t)j * siz_iy + (size_t)(k - kk) * siz_iz;

              float p1[3] = { x3d[i1], y3d[i1], z3d[i1] };
              float p2[3] = { x3d[i2], y3d[i2], z3d[i2] };
              float p3[3] = { x3d[i3], y3d[i3], z3d[i3] };

              float L = dist_point2plane_local(p0, p1, p2, p3);
              if (dL_min_local > L) dL_min_local = L;
            }
          }
        }

        if (dL_min_global > dL_min_local) {
          dL_min_global = dL_min_local;
          *dL_min = dL_min_global;
          *indx_i = i;
          *indx_j = j;
          *indx_k = k;
        }
      }
    }
  }
}

/* ------------------------------------------------------------------ */
/*  xi_arc_stretch_c                                                    */
/* ------------------------------------------------------------------ */
void
xi_arc_stretch_c(float *x3d, float *y3d, float *z3d,
                 int nx, int ny, int nz, float *arc_len, int n_arc)
{
  (void)n_arc; /* arc_len size must equal nx; n_arc kept for API clarity */
  const size_t siz_iy = (size_t)nx;
  const size_t siz_iz = (size_t)(nx * ny);

  float *x3d_temp = (float *)calloc(nx, sizeof(float));
  float *y3d_temp = (float *)calloc(nx, sizeof(float));
  float *z3d_temp = (float *)calloc(nx, sizeof(float));
  float *s = (float *)calloc(nx, sizeof(float));
  float *u = (float *)calloc(nx, sizeof(float));

  for (int k = 0; k < nz; k++) {
    for (int j = 0; j < ny; j++) {
      /* Copy coords to temp */
      for (int i = 0; i < nx; i++) {
        size_t iptr = (size_t)k * siz_iz + (size_t)j * siz_iy + i;
        x3d_temp[i] = x3d[iptr];
        y3d_temp[i] = y3d[iptr];
        z3d_temp[i] = z3d[iptr];
      }
      /* Compute cumulative arc length */
      s[0] = 0.0f;
      for (int i = 1; i < nx; i++) {
        float dx = x3d_temp[i] - x3d_temp[i - 1];
        float dy = y3d_temp[i] - y3d_temp[i - 1];
        float dz = z3d_temp[i] - z3d_temp[i - 1];
        s[i] = s[i - 1] + sqrtf(dx * dx + dy * dy + dz * dz);
      }
      /* Normalize arc length to [0,1] */
      for (int i = 0; i < nx; i++) {
        u[i] = s[i] / s[nx - 1];
      }
      /* Redistribute interior points based on arc_len */
      for (int i = 1; i < nx - 1; i++) {
        float r = arc_len[i];
        int n = 0;
        for (int m = 0; m < nx - 1; m++) {
          if (r >= u[m] && r < u[m + 1]) {
            n = m;
            break;
          }
        }
        /* Linear interpolation */
        size_t iptr = (size_t)k * siz_iz + (size_t)j * siz_iy + i;
        float dx = x3d_temp[n + 1] - x3d_temp[n];
        float dy = y3d_temp[n + 1] - y3d_temp[n];
        float dz = z3d_temp[n + 1] - z3d_temp[n];
        float ratio = (r - u[n]) / (u[n + 1] - u[n]);
        x3d[iptr] = x3d_temp[n] + dx * ratio;
        y3d[iptr] = y3d_temp[n] + dy * ratio;
        z3d[iptr] = z3d_temp[n] + dz * ratio;
      }
    }
  }

  free(x3d_temp);
  free(y3d_temp);
  free(z3d_temp);
  free(s);
  free(u);
}

/* ------------------------------------------------------------------ */
/*  et_arc_stretch_c                                                    */
/* ------------------------------------------------------------------ */
void
et_arc_stretch_c(float *x3d, float *y3d, float *z3d,
                 int nx, int ny, int nz, float *arc_len, int n_arc)
{
  (void)n_arc; /* arc_len size must equal ny; n_arc kept for API clarity */
  const size_t siz_iy = (size_t)nx;
  const size_t siz_iz = (size_t)(nx * ny);

  float *x3d_temp = (float *)calloc(ny, sizeof(float));
  float *y3d_temp = (float *)calloc(ny, sizeof(float));
  float *z3d_temp = (float *)calloc(ny, sizeof(float));
  float *s = (float *)calloc(ny, sizeof(float));
  float *u = (float *)calloc(ny, sizeof(float));

  for (int k = 0; k < nz; k++) {
    for (int i = 0; i < nx; i++) {
      /* Copy coords to temp */
      for (int j = 0; j < ny; j++) {
        size_t iptr = (size_t)k * siz_iz + (size_t)j * siz_iy + i;
        x3d_temp[j] = x3d[iptr];
        y3d_temp[j] = y3d[iptr];
        z3d_temp[j] = z3d[iptr];
      }
      /* Compute cumulative arc length */
      s[0] = 0.0f;
      for (int j = 1; j < ny; j++) {
        float dx = x3d_temp[j] - x3d_temp[j - 1];
        float dy = y3d_temp[j] - y3d_temp[j - 1];
        float dz = z3d_temp[j] - z3d_temp[j - 1];
        s[j] = s[j - 1] + sqrtf(dx * dx + dy * dy + dz * dz);
      }
      /* Normalize arc length to [0,1] */
      for (int j = 0; j < ny; j++) {
        u[j] = s[j] / s[ny - 1];
      }
      /* Redistribute interior points based on arc_len */
      for (int j = 1; j < ny - 1; j++) {
        float r = arc_len[j];
        int n = 0;
        for (int m = 0; m < ny - 1; m++) {
          if (r >= u[m] && r < u[m + 1]) {
            n = m;
            break;
          }
        }
        /* Linear interpolation */
        size_t iptr = (size_t)k * siz_iz + (size_t)j * siz_iy + i;
        float dx = x3d_temp[n + 1] - x3d_temp[n];
        float dy = y3d_temp[n + 1] - y3d_temp[n];
        float dz = z3d_temp[n + 1] - z3d_temp[n];
        float ratio = (r - u[n]) / (u[n + 1] - u[n]);
        x3d[iptr] = x3d_temp[n] + dx * ratio;
        y3d[iptr] = y3d_temp[n] + dy * ratio;
        z3d[iptr] = z3d_temp[n] + dz * ratio;
      }
    }
  }

  free(x3d_temp);
  free(y3d_temp);
  free(z3d_temp);
  free(s);
  free(u);
}

/* ------------------------------------------------------------------ */
/*  zt_arc_stretch_c                                                    */
/* ------------------------------------------------------------------ */
void
zt_arc_stretch_c(float *x3d, float *y3d, float *z3d,
                 int nx, int ny, int nz, float *arc_len, int n_arc)
{
  (void)n_arc; /* arc_len size must equal nz; n_arc kept for API clarity */
  const size_t siz_iy = (size_t)nx;
  const size_t siz_iz = (size_t)(nx * ny);

  float *x3d_temp = (float *)calloc(nz, sizeof(float));
  float *y3d_temp = (float *)calloc(nz, sizeof(float));
  float *z3d_temp = (float *)calloc(nz, sizeof(float));
  float *s = (float *)calloc(nz, sizeof(float));
  float *u = (float *)calloc(nz, sizeof(float));

  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++) {
      /* Copy coords to temp */
      for (int k = 0; k < nz; k++) {
        size_t iptr = (size_t)k * siz_iz + (size_t)j * siz_iy + i;
        x3d_temp[k] = x3d[iptr];
        y3d_temp[k] = y3d[iptr];
        z3d_temp[k] = z3d[iptr];
      }
      /* Compute cumulative arc length */
      s[0] = 0.0f;
      for (int k = 1; k < nz; k++) {
        float dx = x3d_temp[k] - x3d_temp[k - 1];
        float dy = y3d_temp[k] - y3d_temp[k - 1];
        float dz = z3d_temp[k] - z3d_temp[k - 1];
        s[k] = s[k - 1] + sqrtf(dx * dx + dy * dy + dz * dz);
      }
      /* Normalize arc length to [0,1] */
      for (int k = 0; k < nz; k++) {
        u[k] = s[k] / s[nz - 1];
      }
      /* Redistribute interior points based on arc_len */
      for (int k = 1; k < nz - 1; k++) {
        float r = arc_len[k];
        int n = 0;
        for (int m = 0; m < nz - 1; m++) {
          if (r >= u[m] && r < u[m + 1]) {
            n = m;
            break;
          }
        }
        /* Linear interpolation */
        size_t iptr = (size_t)k * siz_iz + (size_t)j * siz_iy + i;
        float dx = x3d_temp[n + 1] - x3d_temp[n];
        float dy = y3d_temp[n + 1] - y3d_temp[n];
        float dz = z3d_temp[n + 1] - z3d_temp[n];
        float ratio = (r - u[n]) / (u[n + 1] - u[n]);
        x3d[iptr] = x3d_temp[n] + dx * ratio;
        y3d[iptr] = y3d_temp[n] + dy * ratio;
        z3d[iptr] = z3d_temp[n] + dz * ratio;
      }
    }
  }

  free(x3d_temp);
  free(y3d_temp);
  free(z3d_temp);
  free(s);
  free(u);
}
