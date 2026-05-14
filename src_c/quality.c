#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "quality.h"

/*
 * 3D grid quality check functions.
 *
 * Index convention (row-major, matches NumPy (nz,ny,nx)):
 *   index = k * ny * nx + j * nx + i
 *   siz_iy = nx,  siz_iz = nx * ny
 *
 * The var array is pre-allocated and zeroed by the caller.
 * For orthogonality and jacobian: interior points are k=0..nz-2,
 * j=0..ny-2, i=0..nx-2; boundary values are extended by copying
 * from the nearest interior row/column/layer.
 * For step and smooth: boundary values at the far end of each
 * direction are copied from the adjacent interior value.
 */

#ifndef PI
#define PI 3.14159265358979323846264338327950288
#endif

/* ------------------------------------------------------------------ */
/*  Helper: extend quality var to the far boundaries (i=nx-1,          */
/*          j=ny-1, k=nz-1) by copying from the nearest interior.      */
/* ------------------------------------------------------------------ */
static void
extend_var(float *var, int nx, int ny, int nz)
{
  const size_t siz_iy = (size_t)nx;
  const size_t siz_iz = (size_t)(nx * ny);
  size_t iptr, iptr1;

  /* i = nx-1 : copy from i = nx-2 */
  for (int k = 0; k < nz; k++) {
    for (int j = 0; j < ny; j++) {
      int i = nx - 1;
      iptr  = (size_t)k * siz_iz + (size_t)j * siz_iy + i;
      iptr1 = (size_t)k * siz_iz + (size_t)j * siz_iy + (i - 1);
      var[iptr] = var[iptr1];
    }
  }

  /* j = ny-1 : copy from j = ny-2 */
  for (int k = 0; k < nz; k++) {
    for (int i = 0; i < nx; i++) {
      int j = ny - 1;
      iptr  = (size_t)k * siz_iz + (size_t)j      * siz_iy + i;
      iptr1 = (size_t)k * siz_iz + (size_t)(j - 1) * siz_iy + i;
      var[iptr] = var[iptr1];
    }
  }

  /* k = nz-1 : copy from k = nz-2 */
  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++) {
      int k = nz - 1;
      iptr  = (size_t)k      * siz_iz + (size_t)j * siz_iy + i;
      iptr1 = (size_t)(k - 1) * siz_iz + (size_t)j * siz_iy + i;
      var[iptr] = var[iptr1];
    }
  }
}

/* ------------------------------------------------------------------ */
/*  cal_orth_xiet_c                                                     */
/* ------------------------------------------------------------------ */
void
cal_orth_xiet_c(float *x3d, float *y3d, float *z3d, float *var,
                int nx, int ny, int nz)
{
  const size_t siz_iy = (size_t)nx;
  const size_t siz_iz = (size_t)(nx * ny);

  float trans = (float)(180.0 / PI);
  float dot, len_xi, len_et, cos_angle;
  float x_xi, y_xi, z_xi, x_et, y_et, z_et;

  for (int k = 0; k < nz - 1; k++) {
    for (int j = 0; j < ny - 1; j++) {
      for (int i = 0; i < nx - 1; i++) {
        size_t iptr  = (size_t)k * siz_iz + (size_t)j * siz_iy + i;
        size_t iptr1 = (size_t)k * siz_iz + (size_t)j * siz_iy + (i + 1);

        /* r_xi = r(i+1) - r(i) */
        x_xi = x3d[iptr1] - x3d[iptr];
        y_xi = y3d[iptr1] - y3d[iptr];
        z_xi = z3d[iptr1] - z3d[iptr];

        size_t iptr2 = (size_t)k * siz_iz + (size_t)(j + 1) * siz_iy + i;

        /* r_et = r(j+1) - r(j) */
        x_et = x3d[iptr2] - x3d[iptr];
        y_et = y3d[iptr2] - y3d[iptr];
        z_et = z3d[iptr2] - z3d[iptr];

        /* cos(theta) = (a . b) / (|a| |b|) */
        dot       = x_xi * x_et + y_xi * y_et + z_xi * z_et;
        len_xi    = sqrtf(x_xi * x_xi + y_xi * y_xi + z_xi * z_xi);
        len_et    = sqrtf(x_et * x_et + y_et * y_et + z_et * z_et);
        cos_angle = dot / (len_xi * len_et);

        /* orth = 90 - |angle - 90| */
        var[iptr] = 90.0f - fabsf(acosf(cos_angle) * trans - 90.0f);
      }
    }
  }

  extend_var(var, nx, ny, nz);
}

/* ------------------------------------------------------------------ */
/*  cal_orth_xizt_c                                                     */
/* ------------------------------------------------------------------ */
void
cal_orth_xizt_c(float *x3d, float *y3d, float *z3d, float *var,
                int nx, int ny, int nz)
{
  const size_t siz_iy = (size_t)nx;
  const size_t siz_iz = (size_t)(nx * ny);

  float trans = (float)(180.0 / PI);
  float dot, len_xi, len_zt, cos_angle;
  float x_xi, y_xi, z_xi, x_zt, y_zt, z_zt;

  for (int k = 0; k < nz - 1; k++) {
    for (int j = 0; j < ny - 1; j++) {
      for (int i = 0; i < nx - 1; i++) {
        size_t iptr  = (size_t)k * siz_iz + (size_t)j * siz_iy + i;
        size_t iptr1 = (size_t)k * siz_iz + (size_t)j * siz_iy + (i + 1);

        /* r_xi = r(i+1) - r(i) */
        x_xi = x3d[iptr1] - x3d[iptr];
        y_xi = y3d[iptr1] - y3d[iptr];
        z_xi = z3d[iptr1] - z3d[iptr];

        size_t iptr2 = (size_t)(k + 1) * siz_iz + (size_t)j * siz_iy + i;

        /* r_zt = r(k+1) - r(k) */
        x_zt = x3d[iptr2] - x3d[iptr];
        y_zt = y3d[iptr2] - y3d[iptr];
        z_zt = z3d[iptr2] - z3d[iptr];

        /* cos(theta) = (a . b) / (|a| |b|) */
        dot       = x_xi * x_zt + y_xi * y_zt + z_xi * z_zt;
        len_xi    = sqrtf(x_xi * x_xi + y_xi * y_xi + z_xi * z_xi);
        len_zt    = sqrtf(x_zt * x_zt + y_zt * y_zt + z_zt * z_zt);
        cos_angle = dot / (len_xi * len_zt);

        /* orth = 90 - |angle - 90| */
        var[iptr] = 90.0f - fabsf(acosf(cos_angle) * trans - 90.0f);
      }
    }
  }

  extend_var(var, nx, ny, nz);
}

/* ------------------------------------------------------------------ */
/*  cal_orth_etzt_c                                                     */
/* ------------------------------------------------------------------ */
void
cal_orth_etzt_c(float *x3d, float *y3d, float *z3d, float *var,
                int nx, int ny, int nz)
{
  const size_t siz_iy = (size_t)nx;
  const size_t siz_iz = (size_t)(nx * ny);

  float trans = (float)(180.0 / PI);
  float dot, len_et, len_zt, cos_angle;
  float x_et, y_et, z_et, x_zt, y_zt, z_zt;

  for (int k = 0; k < nz - 1; k++) {
    for (int j = 0; j < ny - 1; j++) {
      for (int i = 0; i < nx - 1; i++) {
        size_t iptr  = (size_t)k * siz_iz + (size_t)j * siz_iy + i;
        size_t iptr1 = (size_t)k * siz_iz + (size_t)(j + 1) * siz_iy + i;

        /* r_et = r(j+1) - r(j) */
        x_et = x3d[iptr1] - x3d[iptr];
        y_et = y3d[iptr1] - y3d[iptr];
        z_et = z3d[iptr1] - z3d[iptr];

        size_t iptr2 = (size_t)(k + 1) * siz_iz + (size_t)j * siz_iy + i;

        /* r_zt = r(k+1) - r(k) */
        x_zt = x3d[iptr2] - x3d[iptr];
        y_zt = y3d[iptr2] - y3d[iptr];
        z_zt = z3d[iptr2] - z3d[iptr];

        /* cos(theta) = (a . b) / (|a| |b|) */
        dot       = x_et * x_zt + y_et * y_zt + z_et * z_zt;
        len_et    = sqrtf(x_et * x_et + y_et * y_et + z_et * z_et);
        len_zt    = sqrtf(x_zt * x_zt + y_zt * y_zt + z_zt * z_zt);
        cos_angle = dot / (len_et * len_zt);

        /* orth = 90 - |angle - 90| */
        var[iptr] = 90.0f - fabsf(acosf(cos_angle) * trans - 90.0f);
      }
    }
  }

  extend_var(var, nx, ny, nz);
}

/* ------------------------------------------------------------------ */
/*  cal_jacobi_c                                                        */
/* ------------------------------------------------------------------ */
void
cal_jacobi_c(float *x3d, float *y3d, float *z3d, float *var,
             int nx, int ny, int nz)
{
  const size_t siz_iy = (size_t)nx;
  const size_t siz_iz = (size_t)(nx * ny);

  float x_xi, y_xi, z_xi;
  float x_et, y_et, z_et;
  float x_zt, y_zt, z_zt;
  float cros_x, cros_y, cros_z;

  /* jacobi (volume) = (r_xi x r_eta) . r_zt */
  for (int k = 0; k < nz - 1; k++) {
    for (int j = 0; j < ny - 1; j++) {
      for (int i = 0; i < nx - 1; i++) {
        size_t iptr  = (size_t)k * siz_iz + (size_t)j * siz_iy + i;
        size_t iptr1 = (size_t)k * siz_iz + (size_t)j * siz_iy + (i + 1);

        /* r_xi */
        x_xi = x3d[iptr1] - x3d[iptr];
        y_xi = y3d[iptr1] - y3d[iptr];
        z_xi = z3d[iptr1] - z3d[iptr];

        size_t iptr2 = (size_t)k * siz_iz + (size_t)(j + 1) * siz_iy + i;

        /* r_et */
        x_et = x3d[iptr2] - x3d[iptr];
        y_et = y3d[iptr2] - y3d[iptr];
        z_et = z3d[iptr2] - z3d[iptr];

        size_t iptr3 = (size_t)(k + 1) * siz_iz + (size_t)j * siz_iy + i;

        /* r_zt */
        x_zt = x3d[iptr3] - x3d[iptr];
        y_zt = y3d[iptr3] - y3d[iptr];
        z_zt = z3d[iptr3] - z3d[iptr];

        /* r_xi x r_et */
        cros_x = y_xi * z_et - z_xi * y_et;
        cros_y = z_xi * x_et - x_xi * z_et;
        cros_z = x_xi * y_et - y_xi * x_et;

        /* scalar triple product */
        var[iptr] = cros_x * x_zt + cros_y * y_zt + cros_z * z_zt;
      }
    }
  }

  extend_var(var, nx, ny, nz);
}

/* ------------------------------------------------------------------ */
/*  cal_step_xi_c                                                       */
/* ------------------------------------------------------------------ */
void
cal_step_xi_c(float *x3d, float *y3d, float *z3d, float *var,
              int nx, int ny, int nz)
{
  const size_t siz_iy = (size_t)nx;
  const size_t siz_iz = (size_t)(nx * ny);

  /* Compute step for i = 0..nx-2 */
  for (int k = 0; k < nz; k++) {
    for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx - 1; i++) {
        size_t iptr  = (size_t)k * siz_iz + (size_t)j * siz_iy + i;
        size_t iptr1 = (size_t)k * siz_iz + (size_t)j * siz_iy + (i + 1);

        float dx = x3d[iptr1] - x3d[iptr];
        float dy = y3d[iptr1] - y3d[iptr];
        float dz = z3d[iptr1] - z3d[iptr];

        var[iptr] = sqrtf(dx * dx + dy * dy + dz * dz);
      }
    }
  }

  /* i = nx-1 : copy from i = nx-2 */
  for (int k = 0; k < nz; k++) {
    for (int j = 0; j < ny; j++) {
      int i = nx - 1;
      size_t iptr  = (size_t)k * siz_iz + (size_t)j * siz_iy + i;
      size_t iptr1 = (size_t)k * siz_iz + (size_t)j * siz_iy + (i - 1);
      var[iptr] = var[iptr1];
    }
  }
}

/* ------------------------------------------------------------------ */
/*  cal_step_et_c                                                       */
/* ------------------------------------------------------------------ */
void
cal_step_et_c(float *x3d, float *y3d, float *z3d, float *var,
              int nx, int ny, int nz)
{
  const size_t siz_iy = (size_t)nx;
  const size_t siz_iz = (size_t)(nx * ny);

  /* Compute step for j = 0..ny-2 */
  for (int k = 0; k < nz; k++) {
    for (int j = 0; j < ny - 1; j++) {
      for (int i = 0; i < nx; i++) {
        size_t iptr  = (size_t)k * siz_iz + (size_t)j * siz_iy + i;
        size_t iptr1 = (size_t)k * siz_iz + (size_t)(j + 1) * siz_iy + i;

        float dx = x3d[iptr1] - x3d[iptr];
        float dy = y3d[iptr1] - y3d[iptr];
        float dz = z3d[iptr1] - z3d[iptr];

        var[iptr] = sqrtf(dx * dx + dy * dy + dz * dz);
      }
    }
  }

  /* j = ny-1 : copy from j = ny-2 */
  for (int k = 0; k < nz; k++) {
    for (int i = 0; i < nx; i++) {
      int j = ny - 1;
      size_t iptr  = (size_t)k * siz_iz + (size_t)j      * siz_iy + i;
      size_t iptr1 = (size_t)k * siz_iz + (size_t)(j - 1) * siz_iy + i;
      var[iptr] = var[iptr1];
    }
  }
}

/* ------------------------------------------------------------------ */
/*  cal_step_zt_c                                                       */
/* ------------------------------------------------------------------ */
void
cal_step_zt_c(float *x3d, float *y3d, float *z3d, float *var,
              int nx, int ny, int nz)
{
  const size_t siz_iy = (size_t)nx;
  const size_t siz_iz = (size_t)(nx * ny);

  /* Compute step for k = 0..nz-2 */
  for (int k = 0; k < nz - 1; k++) {
    for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {
        size_t iptr  = (size_t)k * siz_iz + (size_t)j * siz_iy + i;
        size_t iptr1 = (size_t)(k + 1) * siz_iz + (size_t)j * siz_iy + i;

        float dx = x3d[iptr1] - x3d[iptr];
        float dy = y3d[iptr1] - y3d[iptr];
        float dz = z3d[iptr1] - z3d[iptr];

        var[iptr] = sqrtf(dx * dx + dy * dy + dz * dz);
      }
    }
  }

  /* k = nz-1 : copy from k = nz-2 */
  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++) {
      int k = nz - 1;
      size_t iptr  = (size_t)k      * siz_iz + (size_t)j * siz_iy + i;
      size_t iptr1 = (size_t)(k - 1) * siz_iz + (size_t)j * siz_iy + i;
      var[iptr] = var[iptr1];
    }
  }
}

/* ------------------------------------------------------------------ */
/*  cal_smooth_xi_c                                                     */
/* ------------------------------------------------------------------ */
void
cal_smooth_xi_c(float *x3d, float *y3d, float *z3d, float *var,
                int nx, int ny, int nz)
{
  const size_t siz_iy = (size_t)nx;
  const size_t siz_iz = (size_t)(nx * ny);

  /* smooth_xi for interior i = 1..nx-2 */
  for (int k = 0; k < nz; k++) {
    for (int j = 0; j < ny; j++) {
      for (int i = 1; i < nx - 1; i++) {
        size_t iptr  = (size_t)k * siz_iz + (size_t)j * siz_iy + i;
        size_t iptr1 = (size_t)k * siz_iz + (size_t)j * siz_iy + (i + 1);
        size_t iptr2 = (size_t)k * siz_iz + (size_t)j * siz_iy + (i - 1);

        /* step forward: r(i+1) - r(i) */
        float dx1 = x3d[iptr1] - x3d[iptr];
        float dy1 = y3d[iptr1] - y3d[iptr];
        float dz1 = z3d[iptr1] - z3d[iptr];
        float len1 = sqrtf(dx1 * dx1 + dy1 * dy1 + dz1 * dz1);

        /* step backward: r(i) - r(i-1) */
        float dx2 = x3d[iptr] - x3d[iptr2];
        float dy2 = y3d[iptr] - y3d[iptr2];
        float dz2 = z3d[iptr] - z3d[iptr2];
        float len2 = sqrtf(dx2 * dx2 + dy2 * dy2 + dz2 * dz2);

        float r1 = len1 / len2;
        float r2 = len2 / len1;

        var[iptr] = fmaxf(r1, r2);
      }
    }
  }

  /* i = nx-1 : copy from i = nx-2 */
  for (int k = 0; k < nz; k++) {
    for (int j = 0; j < ny; j++) {
      int i = nx - 1;
      size_t iptr  = (size_t)k * siz_iz + (size_t)j * siz_iy + i;
      size_t iptr1 = (size_t)k * siz_iz + (size_t)j * siz_iy + (i - 1);
      var[iptr] = var[iptr1];
    }
  }

  /* i = 0 : copy from i = 1 */
  for (int k = 0; k < nz; k++) {
    for (int j = 0; j < ny; j++) {
      int i = 0;
      size_t iptr  = (size_t)k * siz_iz + (size_t)j * siz_iy + i;
      size_t iptr1 = (size_t)k * siz_iz + (size_t)j * siz_iy + (i + 1);
      var[iptr] = var[iptr1];
    }
  }
}

/* ------------------------------------------------------------------ */
/*  cal_smooth_et_c                                                     */
/* ------------------------------------------------------------------ */
void
cal_smooth_et_c(float *x3d, float *y3d, float *z3d, float *var,
                int nx, int ny, int nz)
{
  const size_t siz_iy = (size_t)nx;
  const size_t siz_iz = (size_t)(nx * ny);

  /* smooth_et for interior j = 1..ny-2 */
  for (int k = 0; k < nz; k++) {
    for (int j = 1; j < ny - 1; j++) {
      for (int i = 0; i < nx; i++) {
        size_t iptr  = (size_t)k * siz_iz + (size_t)j * siz_iy + i;
        size_t iptr1 = (size_t)k * siz_iz + (size_t)(j + 1) * siz_iy + i;
        size_t iptr2 = (size_t)k * siz_iz + (size_t)(j - 1) * siz_iy + i;

        /* step forward: r(j+1) - r(j) */
        float dx1 = x3d[iptr1] - x3d[iptr];
        float dy1 = y3d[iptr1] - y3d[iptr];
        float dz1 = z3d[iptr1] - z3d[iptr];
        float len1 = sqrtf(dx1 * dx1 + dy1 * dy1 + dz1 * dz1);

        /* step backward: r(j) - r(j-1) */
        float dx2 = x3d[iptr] - x3d[iptr2];
        float dy2 = y3d[iptr] - y3d[iptr2];
        float dz2 = z3d[iptr] - z3d[iptr2];
        float len2 = sqrtf(dx2 * dx2 + dy2 * dy2 + dz2 * dz2);

        float r1 = len1 / len2;
        float r2 = len2 / len1;

        var[iptr] = fmaxf(r1, r2);
      }
    }
  }

  /* j = ny-1 : copy from j = ny-2 */
  for (int k = 0; k < nz; k++) {
    for (int i = 0; i < nx; i++) {
      int j = ny - 1;
      size_t iptr  = (size_t)k * siz_iz + (size_t)j      * siz_iy + i;
      size_t iptr1 = (size_t)k * siz_iz + (size_t)(j - 1) * siz_iy + i;
      var[iptr] = var[iptr1];
    }
  }

  /* j = 0 : copy from j = 1 */
  for (int k = 0; k < nz; k++) {
    for (int i = 0; i < nx; i++) {
      int j = 0;
      size_t iptr  = (size_t)k * siz_iz + (size_t)j * siz_iy + i;
      size_t iptr1 = (size_t)k * siz_iz + (size_t)(j + 1) * siz_iy + i;
      var[iptr] = var[iptr1];
    }
  }
}

/* ------------------------------------------------------------------ */
/*  cal_smooth_zt_c                                                     */
/* ------------------------------------------------------------------ */
void
cal_smooth_zt_c(float *x3d, float *y3d, float *z3d, float *var,
                int nx, int ny, int nz)
{
  const size_t siz_iy = (size_t)nx;
  const size_t siz_iz = (size_t)(nx * ny);

  /* smooth_zt for interior k = 1..nz-2 */
  for (int k = 1; k < nz - 1; k++) {
    for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {
        size_t iptr  = (size_t)k      * siz_iz + (size_t)j * siz_iy + i;
        size_t iptr1 = (size_t)(k + 1) * siz_iz + (size_t)j * siz_iy + i;
        size_t iptr2 = (size_t)(k - 1) * siz_iz + (size_t)j * siz_iy + i;

        /* step forward: r(k+1) - r(k) */
        float dx1 = x3d[iptr1] - x3d[iptr];
        float dy1 = y3d[iptr1] - y3d[iptr];
        float dz1 = z3d[iptr1] - z3d[iptr];
        float len1 = sqrtf(dx1 * dx1 + dy1 * dy1 + dz1 * dz1);

        /* step backward: r(k) - r(k-1) */
        float dx2 = x3d[iptr] - x3d[iptr2];
        float dy2 = y3d[iptr] - y3d[iptr2];
        float dz2 = z3d[iptr] - z3d[iptr2];
        float len2 = sqrtf(dx2 * dx2 + dy2 * dy2 + dz2 * dz2);

        float r1 = len1 / len2;
        float r2 = len2 / len1;

        var[iptr] = fmaxf(r1, r2);
      }
    }
  }

  /* k = nz-1 : copy from k = nz-2 */
  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++) {
      int k = nz - 1;
      size_t iptr  = (size_t)k      * siz_iz + (size_t)j * siz_iy + i;
      size_t iptr1 = (size_t)(k - 1) * siz_iz + (size_t)j * siz_iy + i;
      var[iptr] = var[iptr1];
    }
  }

  /* k = 0 : copy from k = 1 */
  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++) {
      int k = 0;
      size_t iptr  = (size_t)k      * siz_iz + (size_t)j * siz_iy + i;
      size_t iptr1 = (size_t)(k + 1) * siz_iz + (size_t)j * siz_iy + i;
      var[iptr] = var[iptr1];
    }
  }
}

/* ------------------------------------------------------------------ */
/*  cal_ratio_c                                                         */
/* ------------------------------------------------------------------ */
void
cal_ratio_c(float *x3d, float *y3d, float *z3d, float *var,
            int nx, int ny, int nz)
{
  const size_t siz_iy = (size_t)nx;
  const size_t siz_iz = (size_t)(nx * ny);

  float x_xi, y_xi, z_xi, len_xi;
  float x_et, y_et, z_et, len_et;
  float x_zt, y_zt, z_zt, len_zt;
  float r12, r13, r23;

  /* ratio = max of pairwise step-length ratios for each cell */
  for (int k = 0; k < nz - 1; k++) {
    for (int j = 0; j < ny - 1; j++) {
      for (int i = 0; i < nx - 1; i++) {
        size_t iptr  = (size_t)k * siz_iz + (size_t)j * siz_iy + i;
        size_t iptr1 = (size_t)k * siz_iz + (size_t)j * siz_iy + (i + 1);
        size_t iptr2 = (size_t)k * siz_iz + (size_t)(j + 1) * siz_iy + i;
        size_t iptr3 = (size_t)(k + 1) * siz_iz + (size_t)j * siz_iy + i;

        /* r_xi = r(i+1) - r(i) */
        x_xi = x3d[iptr1] - x3d[iptr];
        y_xi = y3d[iptr1] - y3d[iptr];
        z_xi = z3d[iptr1] - z3d[iptr];
        len_xi = sqrtf(x_xi * x_xi + y_xi * y_xi + z_xi * z_xi);

        /* r_et = r(j+1) - r(j) */
        x_et = x3d[iptr2] - x3d[iptr];
        y_et = y3d[iptr2] - y3d[iptr];
        z_et = z3d[iptr2] - z3d[iptr];
        len_et = sqrtf(x_et * x_et + y_et * y_et + z_et * z_et);

        /* r_zt = r(k+1) - r(k) */
        x_zt = x3d[iptr3] - x3d[iptr];
        y_zt = y3d[iptr3] - y3d[iptr];
        z_zt = z3d[iptr3] - z3d[iptr];
        len_zt = sqrtf(x_zt * x_zt + y_zt * y_zt + z_zt * z_zt);

        /* pairwise max ratio */
        if (len_et > 0) r12 = len_xi / len_et; else r12 = 0;
        if (len_xi > 0) { float t = len_et / len_xi; if (t > r12) r12 = t; }
        if (len_zt > 0) r13 = len_xi / len_zt; else r13 = 0;
        if (len_xi > 0) { float t = len_zt / len_xi; if (t > r13) r13 = t; }
        if (len_zt > 0) r23 = len_et / len_zt; else r23 = 0;
        if (len_et > 0) { float t = len_zt / len_et; if (t > r23) r23 = t; }

        var[iptr] = fmaxf(fmaxf(r12, r13), r23);
      }
    }
  }

  extend_var(var, nx, ny, nz);
}
