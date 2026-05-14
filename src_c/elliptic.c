#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>

#include "elliptic.h"

/*
 * 3D Elliptic grid generation kernels.
 * Adapted from the original MPI+C code, with structs replaced by raw
 * parameters and all MPI calls removed (handled by Python instead).
 *
 * Array layout: row-major, index = k*ny*nx + j*nx + i
 * siz_iy = nx, siz_iz = nx*ny  (computed locally)
 * MPI_PROC_NULL is represented by -2 in neighid[]
 */

/* ------------------------------------------------------------------ */
/* 1. SOR iteration (Gauss-Seidel when w=1.0)                         */
/* ------------------------------------------------------------------ */
void update_SOR_c(float *x3d, float *y3d, float *z3d,
                  float *x3d_tmp, float *y3d_tmp, float *z3d_tmp,
                  int nx, int ny, int nz,
                  float *P, float *Q, float *R, float w)
{
  size_t iptr, iptr1, iptr2, iptr3, iptr4, iptr5, iptr6;
  float x_xi, y_xi, z_xi;
  float x_et, y_et, z_et;
  float x_zt, y_zt, z_zt;
  float x_xiet, y_xiet, z_xiet;
  float x_xizt, y_xizt, z_xizt;
  float x_etzt, y_etzt, z_etzt;
  float g11, g22, g33, g12, g13, g23, coef;
  float alpha1, alpha2, alpha3;
  float beta12, beta23, beta13;

  size_t siz_iy = nx;
  size_t siz_iz = nx * ny;

  for (int k = 1; k < nz - 1; k++) {
    for (int j = 1; j < ny - 1; j++) {
      for (int i = 1; i < nx - 1; i++)
      {
        iptr1 = k * siz_iz + j * siz_iy + (i + 1);
        iptr2 = k * siz_iz + j * siz_iy + (i - 1);
        iptr3 = k * siz_iz + (j + 1) * siz_iy + i;
        iptr4 = k * siz_iz + (j - 1) * siz_iy + i;
        iptr5 = (k + 1) * siz_iz + j * siz_iy + i;
        iptr6 = (k - 1) * siz_iz + j * siz_iy + i;

        x_xi = 0.5f * (x3d[iptr1] - x3d_tmp[iptr2]);
        y_xi = 0.5f * (y3d[iptr1] - y3d_tmp[iptr2]);
        z_xi = 0.5f * (z3d[iptr1] - z3d_tmp[iptr2]);

        x_et = 0.5f * (x3d[iptr3] - x3d_tmp[iptr4]);
        y_et = 0.5f * (y3d[iptr3] - y3d_tmp[iptr4]);
        z_et = 0.5f * (z3d[iptr3] - z3d_tmp[iptr4]);

        x_zt = 0.5f * (x3d[iptr5] - x3d_tmp[iptr6]);
        y_zt = 0.5f * (y3d[iptr5] - y3d_tmp[iptr6]);
        z_zt = 0.5f * (z3d[iptr5] - z3d_tmp[iptr6]);

        iptr1 = k * siz_iz + (j - 1) * siz_iy + (i - 1);
        iptr2 = k * siz_iz + (j - 1) * siz_iy + (i + 1);
        iptr3 = k * siz_iz + (j + 1) * siz_iy + (i + 1);
        iptr4 = k * siz_iz + (j + 1) * siz_iy + (i - 1);

        x_xiet = 0.25f * (x3d[iptr3] + x3d_tmp[iptr1] - x3d_tmp[iptr2] - x3d[iptr4]);
        y_xiet = 0.25f * (y3d[iptr3] + y3d_tmp[iptr1] - y3d_tmp[iptr2] - y3d[iptr4]);
        z_xiet = 0.25f * (z3d[iptr3] + z3d_tmp[iptr1] - z3d_tmp[iptr2] - z3d[iptr4]);

        iptr1 = (k - 1) * siz_iz + j * siz_iy + (i - 1);
        iptr2 = (k - 1) * siz_iz + j * siz_iy + (i + 1);
        iptr3 = (k + 1) * siz_iz + j * siz_iy + (i + 1);
        iptr4 = (k + 1) * siz_iz + j * siz_iy + (i - 1);

        x_xizt = 0.25f * (x3d[iptr3] + x3d_tmp[iptr1] - x3d_tmp[iptr2] - x3d[iptr4]);
        y_xizt = 0.25f * (y3d[iptr3] + y3d_tmp[iptr1] - y3d_tmp[iptr2] - y3d[iptr4]);
        z_xizt = 0.25f * (z3d[iptr3] + z3d_tmp[iptr1] - z3d_tmp[iptr2] - z3d[iptr4]);

        iptr1 = (k - 1) * siz_iz + (j - 1) * siz_iy + i;
        iptr2 = (k - 1) * siz_iz + (j + 1) * siz_iy + i;
        iptr3 = (k + 1) * siz_iz + (j + 1) * siz_iy + i;
        iptr4 = (k + 1) * siz_iz + (j - 1) * siz_iy + i;

        x_etzt = 0.25f * (x3d[iptr3] + x3d_tmp[iptr1] - x3d_tmp[iptr2] - x3d[iptr4]);
        y_etzt = 0.25f * (y3d[iptr3] + y3d_tmp[iptr1] - y3d_tmp[iptr2] - y3d[iptr4]);
        z_etzt = 0.25f * (z3d[iptr3] + z3d_tmp[iptr1] - z3d_tmp[iptr2] - z3d[iptr4]);

        g11 = x_xi * x_xi + y_xi * y_xi + z_xi * z_xi;
        g22 = x_et * x_et + y_et * y_et + z_et * z_et;
        g33 = x_zt * x_zt + y_zt * y_zt + z_zt * z_zt;
        g12 = x_xi * x_et + y_xi * y_et + z_xi * z_et;
        g13 = x_xi * x_zt + y_xi * y_zt + z_xi * z_zt;
        g23 = x_et * x_zt + y_et * y_zt + z_et * z_zt;

        alpha1 = g22 * g33 - g23 * g23;
        alpha2 = g11 * g33 - g13 * g13;
        alpha3 = g11 * g22 - g12 * g12;
        beta12 = g13 * g23 - g12 * g33;
        beta23 = g12 * g13 - g11 * g23;
        beta13 = g12 * g23 - g13 * g22;

        coef = 0.5f / (alpha1 + alpha2 + alpha3);

        iptr  = k * siz_iz + j * siz_iy + i;
        iptr1 = k * siz_iz + j * siz_iy + (i + 1);
        iptr2 = k * siz_iz + j * siz_iy + (i - 1);
        iptr3 = k * siz_iz + (j + 1) * siz_iy + i;
        iptr4 = k * siz_iz + (j - 1) * siz_iy + i;
        iptr5 = (k + 1) * siz_iz + j * siz_iy + i;
        iptr6 = (k - 1) * siz_iz + j * siz_iy + i;

        x3d_tmp[iptr] = coef * (alpha1 * (x3d[iptr1] + x3d_tmp[iptr2])
                        + alpha2 * (x3d[iptr3] + x3d_tmp[iptr4])
                        + alpha3 * (x3d[iptr5] + x3d_tmp[iptr6])
                        + 2.0f * (beta12 * x_xiet + beta23 * x_etzt + beta13 * x_xizt)
                        + alpha1 * P[iptr] * x_xi + alpha2 * Q[iptr] * x_et
                        + alpha3 * R[iptr] * x_zt);

        x3d_tmp[iptr] = w * x3d_tmp[iptr] + (1.0f - w) * x3d[iptr];

        y3d_tmp[iptr] = coef * (alpha1 * (y3d[iptr1] + y3d_tmp[iptr2])
                        + alpha2 * (y3d[iptr3] + y3d_tmp[iptr4])
                        + alpha3 * (y3d[iptr5] + y3d_tmp[iptr6])
                        + 2.0f * (beta12 * y_xiet + beta23 * y_etzt + beta13 * y_xizt)
                        + alpha1 * P[iptr] * y_xi + alpha2 * Q[iptr] * y_et
                        + alpha3 * R[iptr] * y_zt);

        y3d_tmp[iptr] = w * y3d_tmp[iptr] + (1.0f - w) * y3d[iptr];

        z3d_tmp[iptr] = coef * (alpha1 * (z3d[iptr1] + z3d_tmp[iptr2])
                        + alpha2 * (z3d[iptr3] + z3d_tmp[iptr4])
                        + alpha3 * (z3d[iptr5] + z3d_tmp[iptr6])
                        + 2.0f * (beta12 * z_xiet + beta23 * z_etzt + beta13 * z_xizt)
                        + alpha1 * P[iptr] * z_xi + alpha2 * Q[iptr] * z_et
                        + alpha3 * R[iptr] * z_zt);

        z3d_tmp[iptr] = w * z3d_tmp[iptr] + (1.0f - w) * z3d[iptr];
      }
    }
  }
}

/* ------------------------------------------------------------------ */
/* 2. Linear transfinite interpolation from 6 boundary surfaces       */
/* ------------------------------------------------------------------ */
void linear_tfi_c(float *x3d, float *y3d, float *z3d,
                  float *x1, float *x2,
                  float *y1, float *y2,
                  float *z1, float *z2,
                  int nx, int ny, int nz,
                  int ni1, int ni2, int nj1, int nj2, int nk1, int nk2,
                  int gni1, int gnj1, int gnk1,
                  int total_nx, int total_ny, int total_nz,
                  int *neighid)
{
  int siz_bx = total_ny * total_nz;
  int siz_by = total_nx * total_nz;
  int siz_bz = total_nx * total_ny;

  size_t siz_iy = nx;
  size_t siz_iz = nx * ny;

  int gni, gnj, gnk;
  float xi, et, zt;
  float a0, a1, b0, b1, c0, c1;
  size_t iptr, iptr1, iptr2, iptr3, iptr4;
  size_t iptr5, iptr6, iptr7, iptr8, iptr9;
  size_t iptr10, iptr11, iptr12;
  float U_x, V_x, W_x, UW_x, UV_x, VW_x, UVW_x;
  float U_y, V_y, W_y, UW_y, UV_y, VW_y, UVW_y;
  float U_z, V_z, W_z, UW_z, UV_z, VW_z, UVW_z;

  /* TFI interpolation for interior points */
  for (int k = nk1; k <= nk2; k++) {
    for (int j = nj1; j <= nj2; j++) {
      for (int i = ni1; i <= ni2; i++)
      {
        gni = gni1 + i;
        gnj = gnj1 + j;
        gnk = gnk1 + k;
        xi = (1.0f * gni) / (total_nx - 1);
        et = (1.0f * gnj) / (total_ny - 1);
        zt = (1.0f * gnk) / (total_nz - 1);
        a0 = 1.0f - xi;
        a1 = xi;
        b0 = 1.0f - et;
        b1 = et;
        c0 = 1.0f - zt;
        c1 = zt;

        /* 6 face points */
        iptr1 = gnk * total_ny + gnj;
        iptr2 = gnk * total_nx + gni;
        iptr3 = gnj * total_nx + gni;
        U_x = a0 * x1[iptr1] + a1 * x2[iptr1];
        V_x = b0 * y1[iptr2] + b1 * y2[iptr2];
        W_x = c0 * z1[iptr3] + c1 * z2[iptr3];

        U_y = a0 * x1[iptr1 + siz_bx] + a1 * x2[iptr1 + siz_bx];
        V_y = b0 * y1[iptr2 + siz_by] + b1 * y2[iptr2 + siz_by];
        W_y = c0 * z1[iptr3 + siz_bz] + c1 * z2[iptr3 + siz_bz];

        U_z = a0 * x1[iptr1 + 2 * siz_bx] + a1 * x2[iptr1 + 2 * siz_bx];
        V_z = b0 * y1[iptr2 + 2 * siz_by] + b1 * y2[iptr2 + 2 * siz_by];
        W_z = c0 * z1[iptr3 + 2 * siz_bz] + c1 * z2[iptr3 + 2 * siz_bz];

        /* 12 edge points: UV from x1/x2 faces */
        iptr1 = gnk * total_ny + 0;
        iptr2 = gnk * total_ny + total_ny - 1;
        iptr3 = gnk * total_ny + 0;
        iptr4 = gnk * total_ny + total_ny - 1;
        UV_x = a0 * b0 * x1[iptr1] + a0 * b1 * x1[iptr2]
             + a1 * b0 * x2[iptr3] + a1 * b1 * x2[iptr4];
        UV_y = a0 * b0 * x1[iptr1 + siz_bx] + a0 * b1 * x1[iptr2 + siz_bx]
             + a1 * b0 * x2[iptr3 + siz_bx] + a1 * b1 * x2[iptr4 + siz_bx];
        UV_z = a0 * b0 * x1[iptr1 + 2 * siz_bx] + a0 * b1 * x1[iptr2 + 2 * siz_bx]
             + a1 * b0 * x2[iptr3 + 2 * siz_bx] + a1 * b1 * x2[iptr4 + 2 * siz_bx];

        /* 12 edge points: UW from z1/z2 faces */
        iptr5 = gnj * total_nx + 0;
        iptr6 = gnj * total_nx + total_nx - 1;
        iptr7 = gnj * total_nx + 0;
        iptr8 = gnj * total_nx + total_nx - 1;
        UW_x = a0 * c0 * z1[iptr5] + a1 * c0 * z1[iptr6]
             + a0 * c1 * z2[iptr7] + a1 * c1 * z2[iptr8];
        UW_y = a0 * c0 * z1[iptr5 + siz_bz] + a1 * c0 * z1[iptr6 + siz_bz]
             + a0 * c1 * z2[iptr7 + siz_bz] + a1 * c1 * z2[iptr8 + siz_bz];
        UW_z = a0 * c0 * z1[iptr5 + 2 * siz_bz] + a1 * c0 * z1[iptr6 + 2 * siz_bz]
             + a0 * c1 * z2[iptr7 + 2 * siz_bz] + a1 * c1 * z2[iptr8 + 2 * siz_bz];

        /* 12 edge points: VW from y1/y2 faces */
        iptr9  = gni;
        iptr10 = (total_nz - 1) * total_nx + gni;
        iptr11 = gni;
        iptr12 = (total_nz - 1) * total_nx + gni;
        VW_x = b0 * c0 * y1[iptr9]  + b0 * c1 * y1[iptr10]
             + b1 * c0 * y2[iptr11] + b1 * c1 * y2[iptr12];
        VW_y = b0 * c0 * y1[iptr9 + siz_by]  + b0 * c1 * y1[iptr10 + siz_by]
             + b1 * c0 * y2[iptr11 + siz_by] + b1 * c1 * y2[iptr12 + siz_by];
        VW_z = b0 * c0 * y1[iptr9 + 2 * siz_by]  + b0 * c1 * y1[iptr10 + 2 * siz_by]
             + b1 * c0 * y2[iptr11 + 2 * siz_by] + b1 * c1 * y2[iptr12 + 2 * siz_by];

        /* 8 corner points */
        iptr1 = 0;
        iptr2 = 0;
        iptr3 = total_ny - 1;
        iptr4 = total_ny - 1;
        iptr5 = (total_nz - 1) * total_ny;
        iptr6 = (total_nz - 1) * total_ny;
        iptr7 = (total_nz - 1) * total_ny + total_ny - 1;
        iptr8 = (total_nz - 1) * total_ny + total_ny - 1;
        UVW_x = a0 * b0 * c0 * x1[iptr1] + a1 * b0 * c0 * x2[iptr2]
              + a1 * b1 * c0 * x2[iptr3] + a0 * b1 * c0 * x1[iptr4]
              + a0 * b0 * c1 * x1[iptr5] + a1 * b0 * c1 * x2[iptr6]
              + a1 * b1 * c1 * x2[iptr7] + a0 * b1 * c1 * x1[iptr8];
        UVW_y = a0 * b0 * c0 * x1[iptr1 + siz_bx] + a1 * b0 * c0 * x2[iptr2 + siz_bx]
              + a1 * b1 * c0 * x2[iptr3 + siz_bx] + a0 * b1 * c0 * x1[iptr4 + siz_bx]
              + a0 * b0 * c1 * x1[iptr5 + siz_bx] + a1 * b0 * c1 * x2[iptr6 + siz_bx]
              + a1 * b1 * c1 * x2[iptr7 + siz_bx] + a0 * b1 * c1 * x1[iptr8 + siz_bx];
        UVW_z = a0 * b0 * c0 * x1[iptr1 + 2 * siz_bx] + a1 * b0 * c0 * x2[iptr2 + 2 * siz_bx]
              + a1 * b1 * c0 * x2[iptr3 + 2 * siz_bx] + a0 * b1 * c0 * x1[iptr4 + 2 * siz_bx]
              + a0 * b0 * c1 * x1[iptr5 + 2 * siz_bx] + a1 * b0 * c1 * x2[iptr6 + 2 * siz_bx]
              + a1 * b1 * c1 * x2[iptr7 + 2 * siz_bx] + a0 * b1 * c1 * x1[iptr8 + 2 * siz_bx];

        iptr = k * siz_iz + j * siz_iy + i;
        x3d[iptr] = U_x + V_x + W_x - UV_x - UW_x - VW_x + UVW_x;
        y3d[iptr] = U_y + V_y + W_y - UV_y - UW_y - VW_y + UVW_y;
        z3d[iptr] = U_z + V_z + W_z - UV_z - UW_z - VW_z + UVW_z;
      }
    }
  }

  /* Assign 6 boundary faces to ghost points (physical boundaries only) */

  /* bdry x1 (i=0) */
  if (neighid[0] == -2)
  {
    for (int k = 0; k < nz; k++)
    {
      for (int j = 0; j < ny; j++)
      {
        iptr = k * siz_iz + j * siz_iy;
        gnk = gnk1 + k;
        gnj = gnj1 + j;
        iptr1 = gnk * total_ny + gnj;

        x3d[iptr] = x1[iptr1];
        y3d[iptr] = x1[iptr1 + siz_bx];
        z3d[iptr] = x1[iptr1 + 2 * siz_bx];
      }
    }
  }

  /* bdry x2 (i=nx-1) */
  if (neighid[1] == -2)
  {
    for (int k = 0; k < nz; k++)
    {
      for (int j = 0; j < ny; j++)
      {
        iptr = k * siz_iz + j * siz_iy + nx - 1;
        gnk = gnk1 + k;
        gnj = gnj1 + j;
        iptr1 = gnk * total_ny + gnj;

        x3d[iptr] = x2[iptr1];
        y3d[iptr] = x2[iptr1 + siz_bx];
        z3d[iptr] = x2[iptr1 + 2 * siz_bx];
      }
    }
  }

  /* bdry y1 (j=0) */
  if (neighid[2] == -2)
  {
    for (int k = 0; k < nz; k++)
    {
      for (int i = 0; i < nx; i++)
      {
        iptr = k * siz_iz + i;
        gnk = gnk1 + k;
        gni = gni1 + i;
        iptr1 = gnk * total_nx + gni;

        x3d[iptr] = y1[iptr1];
        y3d[iptr] = y1[iptr1 + siz_by];
        z3d[iptr] = y1[iptr1 + 2 * siz_by];
      }
    }
  }

  /* bdry y2 (j=ny-1) */
  if (neighid[3] == -2)
  {
    for (int k = 0; k < nz; k++)
    {
      for (int i = 0; i < nx; i++)
      {
        iptr = k * siz_iz + (ny - 1) * siz_iy + i;
        gnk = gnk1 + k;
        gni = gni1 + i;
        iptr1 = gnk * total_nx + gni;

        x3d[iptr] = y2[iptr1];
        y3d[iptr] = y2[iptr1 + siz_by];
        z3d[iptr] = y2[iptr1 + 2 * siz_by];
      }
    }
  }

  /* bdry z1 (k=0) */
  if (neighid[4] == -2)
  {
    for (int j = 0; j < ny; j++)
    {
      for (int i = 0; i < nx; i++)
      {
        iptr = j * siz_iy + i;
        gnj = gnj1 + j;
        gni = gni1 + i;
        iptr1 = gnj * total_nx + gni;

        x3d[iptr] = z1[iptr1];
        y3d[iptr] = z1[iptr1 + siz_bz];
        z3d[iptr] = z1[iptr1 + 2 * siz_bz];
      }
    }
  }

  /* bdry z2 (k=nz-1) */
  if (neighid[5] == -2)
  {
    for (int j = 0; j < ny; j++)
    {
      for (int i = 0; i < nx; i++)
      {
        iptr = (nz - 1) * siz_iz + j * siz_iy + i;
        gnj = gnj1 + j;
        gni = gni1 + i;
        iptr1 = gnj * total_nx + gni;

        x3d[iptr] = z2[iptr1];
        y3d[iptr] = z2[iptr1 + siz_bz];
        z3d[iptr] = z2[iptr1 + 2 * siz_bz];
      }
    }
  }
}

/* ------------------------------------------------------------------ */
/* 3. Ghost point calculation for Dirichlet method                     */
/* ------------------------------------------------------------------ */
void ghost_cal_c(float *x3d, float *y3d, float *z3d,
                 int nx, int ny, int nz,
                 float *p_x, float *p_y, float *p_z,
                 float *g11_x, float *g22_y, float *g33_z,
                 int *neighid)
{
  size_t iptr, iptr1, iptr2, iptr3, iptr4;
  float x_xi, y_xi, z_xi;
  float x_et, y_et, z_et;
  float x_zt, y_zt, z_zt;
  float x_xi0, y_xi0, z_xi0;
  float x_et0, y_et0, z_et0;
  float x_zt0, y_zt0, z_zt0;
  float vn_x, vn_y, vn_z;
  float vn_x0, vn_y0, vn_z0;
  float len_vn, coef;
  float *p_x1, *p_x2;
  float *p_y1, *p_y2;
  float *p_z1, *p_z2;
  float *g11_x1, *g11_x2;
  float *g22_y1, *g22_y2;
  float *g33_z1, *g33_z2;
  size_t siz_iy = nx;
  size_t siz_iz = nx * ny;
  size_t size;

  p_x1 = p_x;
  p_x2 = p_x + ny * nz * 3;
  p_y1 = p_y;
  p_y2 = p_y + nx * nz * 3;
  p_z1 = p_z;
  p_z2 = p_z + nx * ny * 3;
  g11_x1 = g11_x;
  g11_x2 = g11_x + ny * nz;
  g22_y1 = g22_y;
  g22_y2 = g22_y + nx * nz;
  g33_z1 = g33_z;
  g33_z2 = g33_z + nx * ny;

  /* bdry x1: r_et X r_zt */
  if (neighid[0] == -2)
  {
    for (int k = 1; k < nz - 1; k++) {
      for (int j = 1; j < ny - 1; j++)
      {
        iptr1 = k * siz_iz + j * siz_iy + 1;
        iptr2 = k * siz_iz + j * siz_iy + 0;
        x_xi0 = x3d[iptr1] - x3d[iptr2];
        y_xi0 = y3d[iptr1] - y3d[iptr2];
        z_xi0 = z3d[iptr1] - z3d[iptr2];
        iptr1 = k * siz_iz + (j + 1) * siz_iy + 0;
        iptr2 = k * siz_iz + (j - 1) * siz_iy + 0;
        x_et = 0.5f * (x3d[iptr1] - x3d[iptr2]);
        y_et = 0.5f * (y3d[iptr1] - y3d[iptr2]);
        z_et = 0.5f * (z3d[iptr1] - z3d[iptr2]);
        iptr3 = (k + 1) * siz_iz + j * siz_iy + 0;
        iptr4 = (k - 1) * siz_iz + j * siz_iy + 0;
        x_zt = 0.5f * (x3d[iptr3] - x3d[iptr4]);
        y_zt = 0.5f * (y3d[iptr3] - y3d[iptr4]);
        z_zt = 0.5f * (z3d[iptr3] - z3d[iptr4]);
        /* orthogonal vector: cross product */
        vn_x = y_et * z_zt - z_et * y_zt;
        vn_y = z_et * x_zt - x_et * z_zt;
        vn_z = x_et * y_zt - y_et * x_zt;
        len_vn = sqrt(vn_x * vn_x + vn_y * vn_y + vn_z * vn_z);
        /* normalize */
        vn_x0 = vn_x / len_vn;
        vn_y0 = vn_y / len_vn;
        vn_z0 = vn_z / len_vn;
        /* projection from r_xi0 to vn: dot product */
        coef = x_xi0 * vn_x0 + y_xi0 * vn_y0 + z_xi0 * vn_z0;

        x_xi = coef * vn_x0;
        y_xi = coef * vn_y0;
        z_xi = coef * vn_z0;

        iptr = k * ny + j;
        iptr1 = k * siz_iz + j * siz_iy + 0;
        size = ny * nz;
        p_x1[iptr + 0 * size] = x3d[iptr1] - x_xi;
        p_x1[iptr + 1 * size] = y3d[iptr1] - y_xi;
        p_x1[iptr + 2 * size] = z3d[iptr1] - z_xi;
        g11_x1[iptr] = x_xi * x_xi + y_xi * y_xi + z_xi * z_xi;
      }
    }
  }

  /* bdry x2: r_et X r_zt */
  if (neighid[1] == -2)
  {
    for (int k = 1; k < nz - 1; k++) {
      for (int j = 1; j < ny - 1; j++)
      {
        iptr1 = k * siz_iz + j * siz_iy + nx - 1;
        iptr2 = k * siz_iz + j * siz_iy + nx - 2;
        x_xi0 = x3d[iptr1] - x3d[iptr2];
        y_xi0 = y3d[iptr1] - y3d[iptr2];
        z_xi0 = z3d[iptr1] - z3d[iptr2];
        iptr1 = k * siz_iz + (j + 1) * siz_iy + nx - 1;
        iptr2 = k * siz_iz + (j - 1) * siz_iy + nx - 1;
        x_et = 0.5f * (x3d[iptr1] - x3d[iptr2]);
        y_et = 0.5f * (y3d[iptr1] - y3d[iptr2]);
        z_et = 0.5f * (z3d[iptr1] - z3d[iptr2]);
        iptr3 = (k + 1) * siz_iz + j * siz_iy + nx - 1;
        iptr4 = (k - 1) * siz_iz + j * siz_iy + nx - 1;
        x_zt = 0.5f * (x3d[iptr3] - x3d[iptr4]);
        y_zt = 0.5f * (y3d[iptr3] - y3d[iptr4]);
        z_zt = 0.5f * (z3d[iptr3] - z3d[iptr4]);
        /* orthogonal vector: cross product */
        vn_x = y_et * z_zt - z_et * y_zt;
        vn_y = z_et * x_zt - x_et * z_zt;
        vn_z = x_et * y_zt - y_et * x_zt;
        len_vn = sqrt(vn_x * vn_x + vn_y * vn_y + vn_z * vn_z);
        /* normalize */
        vn_x0 = vn_x / len_vn;
        vn_y0 = vn_y / len_vn;
        vn_z0 = vn_z / len_vn;
        /* projection from r_xi0 to vn: dot product */
        coef = x_xi0 * vn_x0 + y_xi0 * vn_y0 + z_xi0 * vn_z0;

        x_xi = coef * vn_x0;
        y_xi = coef * vn_y0;
        z_xi = coef * vn_z0;

        iptr = k * ny + j;
        iptr1 = k * siz_iz + j * siz_iy + nx - 1;
        size = ny * nz;
        p_x2[iptr + 0 * size] = x3d[iptr1] + x_xi;
        p_x2[iptr + 1 * size] = y3d[iptr1] + y_xi;
        p_x2[iptr + 2 * size] = z3d[iptr1] + z_xi;
        g11_x2[iptr] = x_xi * x_xi + y_xi * y_xi + z_xi * z_xi;
      }
    }
  }

  /* bdry y1: r_zt X r_xi */
  if (neighid[2] == -2)
  {
    for (int k = 1; k < nz - 1; k++) {
      for (int i = 1; i < nx - 1; i++)
      {
        iptr1 = k * siz_iz + 1 * siz_iy + i;
        iptr2 = k * siz_iz + 0 * siz_iy + i;
        x_et0 = x3d[iptr1] - x3d[iptr2];
        y_et0 = y3d[iptr1] - y3d[iptr2];
        z_et0 = z3d[iptr1] - z3d[iptr2];
        iptr1 = k * siz_iz + 0 * siz_iy + (i + 1);
        iptr2 = k * siz_iz + 0 * siz_iy + (i - 1);
        x_xi = 0.5f * (x3d[iptr1] - x3d[iptr2]);
        y_xi = 0.5f * (y3d[iptr1] - y3d[iptr2]);
        z_xi = 0.5f * (z3d[iptr1] - z3d[iptr2]);
        iptr3 = (k + 1) * siz_iz + 0 * siz_iy + i;
        iptr4 = (k - 1) * siz_iz + 0 * siz_iy + i;
        x_zt = 0.5f * (x3d[iptr3] - x3d[iptr4]);
        y_zt = 0.5f * (y3d[iptr3] - y3d[iptr4]);
        z_zt = 0.5f * (z3d[iptr3] - z3d[iptr4]);
        /* orthogonal vector: cross product */
        vn_x = y_zt * z_xi - z_zt * y_xi;
        vn_y = z_zt * x_xi - x_zt * z_xi;
        vn_z = x_zt * y_xi - y_zt * x_xi;
        len_vn = sqrt(vn_x * vn_x + vn_y * vn_y + vn_z * vn_z);
        /* normalize */
        vn_x0 = vn_x / len_vn;
        vn_y0 = vn_y / len_vn;
        vn_z0 = vn_z / len_vn;
        /* projection from r_et0 to vn: dot product */
        coef = x_et0 * vn_x0 + y_et0 * vn_y0 + z_et0 * vn_z0;

        x_et = coef * vn_x0;
        y_et = coef * vn_y0;
        z_et = coef * vn_z0;

        iptr = k * nx + i;
        iptr1 = k * siz_iz + 0 * siz_iy + i;
        size = nx * nz;
        p_y1[iptr + 0 * size] = x3d[iptr1] - x_et;
        p_y1[iptr + 1 * size] = y3d[iptr1] - y_et;
        p_y1[iptr + 2 * size] = z3d[iptr1] - z_et;
        g22_y1[iptr] = x_et * x_et + y_et * y_et + z_et * z_et;
      }
    }
  }

  /* bdry y2: r_zt X r_xi */
  if (neighid[3] == -2)
  {
    for (int k = 1; k < nz - 1; k++) {
      for (int i = 1; i < nx - 1; i++)
      {
        iptr1 = k * siz_iz + (ny - 1) * siz_iy + i;
        iptr2 = k * siz_iz + (ny - 2) * siz_iy + i;
        x_et0 = x3d[iptr1] - x3d[iptr2];
        y_et0 = y3d[iptr1] - y3d[iptr2];
        z_et0 = z3d[iptr1] - z3d[iptr2];
        iptr1 = k * siz_iz + (ny - 1) * siz_iy + (i + 1);
        iptr2 = k * siz_iz + (ny - 1) * siz_iy + (i - 1);
        x_xi = 0.5f * (x3d[iptr1] - x3d[iptr2]);
        y_xi = 0.5f * (y3d[iptr1] - y3d[iptr2]);
        z_xi = 0.5f * (z3d[iptr1] - z3d[iptr2]);
        iptr3 = (k + 1) * siz_iz + (ny - 1) * siz_iy + i;
        iptr4 = (k - 1) * siz_iz + (ny - 1) * siz_iy + i;
        x_zt = 0.5f * (x3d[iptr3] - x3d[iptr4]);
        y_zt = 0.5f * (y3d[iptr3] - y3d[iptr4]);
        z_zt = 0.5f * (z3d[iptr3] - z3d[iptr4]);
        /* orthogonal vector: cross product */
        vn_x = y_zt * z_xi - z_zt * y_xi;
        vn_y = z_zt * x_xi - x_zt * z_xi;
        vn_z = x_zt * y_xi - y_zt * x_xi;
        len_vn = sqrt(vn_x * vn_x + vn_y * vn_y + vn_z * vn_z);
        /* normalize */
        vn_x0 = vn_x / len_vn;
        vn_y0 = vn_y / len_vn;
        vn_z0 = vn_z / len_vn;
        /* projection from r_et0 to vn: dot product */
        coef = x_et0 * vn_x0 + y_et0 * vn_y0 + z_et0 * vn_z0;

        x_et = coef * vn_x0;
        y_et = coef * vn_y0;
        z_et = coef * vn_z0;

        iptr = k * nx + i;
        iptr1 = k * siz_iz + (ny - 1) * siz_iy + i;
        size = nx * nz;
        p_y2[iptr + 0 * size] = x3d[iptr1] + x_et;
        p_y2[iptr + 1 * size] = y3d[iptr1] + y_et;
        p_y2[iptr + 2 * size] = z3d[iptr1] + z_et;
        g22_y2[iptr] = x_et * x_et + y_et * y_et + z_et * z_et;
      }
    }
  }

  /* bdry z1: r_xi X r_et */
  if (neighid[4] == -2)
  {
    for (int j = 1; j < ny - 1; j++) {
      for (int i = 1; i < nx - 1; i++)
      {
        iptr1 = 1 * siz_iz + j * siz_iy + i;
        iptr2 = 0 * siz_iz + j * siz_iy + i;
        x_zt0 = x3d[iptr1] - x3d[iptr2];
        y_zt0 = y3d[iptr1] - y3d[iptr2];
        z_zt0 = z3d[iptr1] - z3d[iptr2];
        iptr1 = 0 * siz_iz + j * siz_iy + (i + 1);
        iptr2 = 0 * siz_iz + j * siz_iy + (i - 1);
        x_xi = 0.5f * (x3d[iptr1] - x3d[iptr2]);
        y_xi = 0.5f * (y3d[iptr1] - y3d[iptr2]);
        z_xi = 0.5f * (z3d[iptr1] - z3d[iptr2]);
        iptr3 = 0 * siz_iz + (j + 1) * siz_iy + i;
        iptr4 = 0 * siz_iz + (j - 1) * siz_iy + i;
        x_et = 0.5f * (x3d[iptr3] - x3d[iptr4]);
        y_et = 0.5f * (y3d[iptr3] - y3d[iptr4]);
        z_et = 0.5f * (z3d[iptr3] - z3d[iptr4]);
        /* orthogonal vector: cross product */
        vn_x = y_xi * z_et - z_xi * y_et;
        vn_y = z_xi * x_et - x_xi * z_et;
        vn_z = x_xi * y_et - y_xi * x_et;
        len_vn = sqrt(vn_x * vn_x + vn_y * vn_y + vn_z * vn_z);
        /* normalize */
        vn_x0 = vn_x / len_vn;
        vn_y0 = vn_y / len_vn;
        vn_z0 = vn_z / len_vn;
        /* projection from r_zt0 to vn: dot product */
        coef = x_zt0 * vn_x0 + y_zt0 * vn_y0 + z_zt0 * vn_z0;

        x_zt = coef * vn_x0;
        y_zt = coef * vn_y0;
        z_zt = coef * vn_z0;

        iptr = j * nx + i;
        iptr1 = 0 * siz_iz + j * siz_iy + i;
        size = nx * ny;
        p_z1[iptr + 0 * size] = x3d[iptr1] - x_zt;
        p_z1[iptr + 1 * size] = y3d[iptr1] - y_zt;
        p_z1[iptr + 2 * size] = z3d[iptr1] - z_zt;
        g33_z1[iptr] = x_zt * x_zt + y_zt * y_zt + z_zt * z_zt;
      }
    }
  }

  /* bdry z2: r_xi X r_et */
  if (neighid[5] == -2)
  {
    for (int j = 1; j < ny - 1; j++) {
      for (int i = 1; i < nx - 1; i++)
      {
        iptr1 = (nz - 1) * siz_iz + j * siz_iy + i;
        iptr2 = (nz - 2) * siz_iz + j * siz_iy + i;
        x_zt0 = x3d[iptr1] - x3d[iptr2];
        y_zt0 = y3d[iptr1] - y3d[iptr2];
        z_zt0 = z3d[iptr1] - z3d[iptr2];
        iptr1 = (nz - 1) * siz_iz + j * siz_iy + (i + 1);
        iptr2 = (nz - 1) * siz_iz + j * siz_iy + (i - 1);
        x_xi = 0.5f * (x3d[iptr1] - x3d[iptr2]);
        y_xi = 0.5f * (y3d[iptr1] - y3d[iptr2]);
        z_xi = 0.5f * (z3d[iptr1] - z3d[iptr2]);
        iptr3 = (nz - 1) * siz_iz + (j + 1) * siz_iy + i;
        iptr4 = (nz - 1) * siz_iz + (j - 1) * siz_iy + i;
        x_et = 0.5f * (x3d[iptr3] - x3d[iptr4]);
        y_et = 0.5f * (y3d[iptr3] - y3d[iptr4]);
        z_et = 0.5f * (z3d[iptr3] - z3d[iptr4]);
        /* orthogonal vector: cross product */
        vn_x = y_xi * z_et - z_xi * y_et;
        vn_y = z_xi * x_et - x_xi * z_et;
        vn_z = x_xi * y_et - y_xi * x_et;
        len_vn = sqrt(vn_x * vn_x + vn_y * vn_y + vn_z * vn_z);
        /* normalize */
        vn_x0 = vn_x / len_vn;
        vn_y0 = vn_y / len_vn;
        vn_z0 = vn_z / len_vn;
        /* projection from r_zt0 to vn: dot product */
        coef = x_zt0 * vn_x0 + y_zt0 * vn_y0 + z_zt0 * vn_z0;

        x_zt = coef * vn_x0;
        y_zt = coef * vn_y0;
        z_zt = coef * vn_z0;

        iptr = j * nx + i;
        iptr1 = (nz - 1) * siz_iz + j * siz_iy + i;
        size = nx * ny;
        p_z2[iptr + 0 * size] = x3d[iptr1] + x_zt;
        p_z2[iptr + 1 * size] = y3d[iptr1] + y_zt;
        p_z2[iptr + 2 * size] = z3d[iptr1] + z_zt;
        g33_z2[iptr] = x_zt * x_zt + y_zt * y_zt + z_zt * z_zt;
      }
    }
  }
}

/* ------------------------------------------------------------------ */
/* 4. Compute maximum residual across local domain                     */
/* ------------------------------------------------------------------ */
void compute_residual_c(float *x3d, float *y3d, float *z3d,
                        float *x3d_tmp, float *y3d_tmp, float *z3d_tmp,
                        int nx, int ny, int nz,
                        float *local_max)
{
  size_t siz_iy = nx;
  size_t siz_iz = nx * ny;
  size_t iptr, iptr1, iptr2, iptr3;
  float dif, dif1, dif2, dif3;
  float dif_x, dif_y, dif_z;
  float resi, resj, resk;
  float max_resi = 0.0f;
  float max_resj = 0.0f;
  float max_resk = 0.0f;

  for (int k = 1; k < nz - 1; k++) {
    for (int j = 1; j < ny - 1; j++) {
      for (int i = 1; i < nx - 1; i++)
      {
        iptr  = k * siz_iz + j * siz_iy + i;
        iptr1 = k * siz_iz + j * siz_iy + i + 1;
        iptr2 = k * siz_iz + (j + 1) * siz_iy + i;
        iptr3 = (k + 1) * siz_iz + j * siz_iy + i;

        dif_x = x3d_tmp[iptr] - x3d[iptr];
        dif_y = y3d_tmp[iptr] - y3d[iptr];
        dif_z = z3d_tmp[iptr] - z3d[iptr];
        dif = sqrt(dif_x * dif_x + dif_y * dif_y + dif_z * dif_z);

        dif_x = x3d[iptr1] - x3d[iptr];
        dif_y = y3d[iptr1] - y3d[iptr];
        dif_z = z3d[iptr1] - z3d[iptr];
        dif1 = sqrt(dif_x * dif_x + dif_y * dif_y + dif_z * dif_z);

        dif_x = x3d[iptr2] - x3d[iptr];
        dif_y = y3d[iptr2] - y3d[iptr];
        dif_z = z3d[iptr2] - z3d[iptr];
        dif2 = sqrt(dif_x * dif_x + dif_y * dif_y + dif_z * dif_z);

        dif_x = x3d[iptr3] - x3d[iptr];
        dif_y = y3d[iptr3] - y3d[iptr];
        dif_z = z3d[iptr3] - z3d[iptr];
        dif3 = sqrt(dif_x * dif_x + dif_y * dif_y + dif_z * dif_z);

        resi = dif / dif1;
        resj = dif / dif2;
        resk = dif / dif3;
        max_resi = fmax(max_resi, resi);
        max_resj = fmax(max_resj, resj);
        max_resk = fmax(max_resk, resk);
      }
    }
  }

  local_max[0] = max_resi;
  local_max[1] = max_resj;
  local_max[2] = max_resk;
}

/* ------------------------------------------------------------------ */
/* 5. Interpolate source terms from 6 boundary faces to interior       */
/* ------------------------------------------------------------------ */
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
                           float *coef, int n_coef)
{
  size_t siz_iy = nx;
  size_t siz_iz = nx * ny;
  int gni, gnj, gnk;
  float xi, et, zt;
  float c0, c1, r0, r1;
  size_t iptr, iptr1;

  /* x-direction interpolation: P linear, Q/R exponential */
  for (int k = 1; k < nz - 1; k++) {
    for (int j = 1; j < ny - 1; j++) {
      for (int i = 1; i < nx - 1; i++)
      {
        gni = gni1 + i;
        gnk = gnk1 + k - 1;
        gnj = gnj1 + j - 1;
        xi = (1.0f * gni) / (total_nx - 1);

        c0 = 1.0f - xi;
        c1 = xi;

        r0 = exp(-coef[0] * xi);
        r1 = exp(-coef[1] * (1.0f - xi));

        iptr  = k * siz_iz + j * siz_iy + i;
        iptr1 = gnk * (total_ny - 2) + gnj;
        P[iptr] = c0 * P_x1[iptr1] + c1 * P_x2[iptr1];
        Q[iptr] = r0 * Q_x1[iptr1] + r1 * Q_x2[iptr1];
        R[iptr] = r0 * R_x1[iptr1] + r1 * R_x2[iptr1];
      }
    }
  }

  /* y-direction interpolation: Q linear, P/R exponential */
  for (int k = 1; k < nz - 1; k++) {
    for (int j = 1; j < ny - 1; j++) {
      for (int i = 1; i < nx - 1; i++)
      {
        gnj = gnj1 + j;
        gnk = gnk1 + k - 1;
        gni = gni1 + i - 1;
        et = (1.0f * gnj) / (total_ny - 1);

        c0 = 1.0f - et;
        c1 = et;

        r0 = exp(-coef[2] * et);
        r1 = exp(-coef[3] * (1.0f - et));

        iptr  = k * siz_iz + j * siz_iy + i;
        iptr1 = gnk * (total_nx - 2) + gni;
        P[iptr] = P[iptr] + r0 * P_y1[iptr1] + r1 * P_y2[iptr1];
        Q[iptr] = Q[iptr] + c0 * Q_y1[iptr1] + c1 * Q_y2[iptr1];
        R[iptr] = R[iptr] + r0 * R_y1[iptr1] + r1 * R_y2[iptr1];
      }
    }
  }

  /* z-direction interpolation: R linear, P/Q exponential */
  for (int k = 1; k < nz - 1; k++) {
    for (int j = 1; j < ny - 1; j++) {
      for (int i = 1; i < nx - 1; i++)
      {
        gnk = gnk1 + k;     /* include bdry point */
        gnj = gnj1 + j - 1; /* not include bdry point */
        gni = gni1 + i - 1;
        zt = (1.0f * gnk) / (total_nz - 1);
        c0 = 1.0f - zt;
        c1 = zt;

        r0 = exp(-coef[4] * zt);
        r1 = exp(-coef[5] * (1.0f - zt));

        iptr  = k * siz_iz + j * siz_iy + i;
        iptr1 = gnj * (total_nx - 2) + gni;
        P[iptr] = P[iptr] + r0 * P_z1[iptr1] + r1 * P_z2[iptr1];
        Q[iptr] = Q[iptr] + r0 * Q_z1[iptr1] + r1 * Q_z2[iptr1];
        R[iptr] = R[iptr] + c0 * R_z1[iptr1] + c1 * R_z2[iptr1];
      }
    }
  }
}

/* ------------------------------------------------------------------ */
/* 6. Set source terms for Dirichlet method on all 6 boundaries        */
/* ------------------------------------------------------------------ */
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
                    int total_nx, int total_ny, int total_nz)
{
  size_t siz_iy = nx;
  size_t siz_iz = nx * ny;
  int gni, gnj, gnk;
  size_t size;

  size_t iptr, iptr1, iptr2, iptr3, iptr4;
  float temp_x, temp_y, temp_z;
  float temp1, temp2, temp3;
  float x_xi, y_xi, z_xi;
  float x_et, y_et, z_et;
  float x_zt, y_zt, z_zt;
  float x_xixi, y_xixi, z_xixi;
  float x_etet, y_etet, z_etet;
  float x_ztzt, y_ztzt, z_ztzt;
  float x_xizt, y_xizt, z_xizt;
  float x_xiet, y_xiet, z_xiet;
  float x_etzt, y_etzt, z_etzt;
  float g11, g22, g33, g12, g13, g23;
  float c1, c2, c3, c4;

  float *p_x1, *p_x2;
  float *p_y1, *p_y2;
  float *p_z1, *p_z2;
  float *g11_x1, *g11_x2;
  float *g22_y1, *g22_y2;
  float *g33_z1, *g33_z2;

  p_x1 = p_x;
  p_x2 = p_x + ny * nz * 3;
  p_y1 = p_y;
  p_y2 = p_y + nx * nz * 3;
  p_z1 = p_z;
  p_z2 = p_z + nx * ny * 3;
  g11_x1 = g11_x;
  g11_x2 = g11_x + ny * nz;
  g22_y1 = g22_y;
  g22_y2 = g22_y + nx * nz;
  g33_z1 = g33_z;
  g33_z2 = g33_z + nx * ny;

  /* ---- bdry x1 (xi=0) ---- */
  if (neighid[0] == -2)
  {
    for (int k = 1; k < nz - 1; k++) {
      for (int j = 1; j < ny - 1; j++)
      {
        iptr  = k * siz_iz + j * siz_iy + 0;
        iptr1 = k * siz_iz + (j + 1) * siz_iy + 0;
        iptr2 = k * siz_iz + (j - 1) * siz_iy + 0;
        iptr3 = (k + 1) * siz_iz + j * siz_iy + 0;
        iptr4 = (k - 1) * siz_iz + j * siz_iy + 0;
        x_et = 0.5f * (x3d[iptr1] - x3d[iptr2]);
        y_et = 0.5f * (y3d[iptr1] - y3d[iptr2]);
        z_et = 0.5f * (z3d[iptr1] - z3d[iptr2]);
        x_zt = 0.5f * (x3d[iptr3] - x3d[iptr4]);
        y_zt = 0.5f * (y3d[iptr3] - y3d[iptr4]);
        z_zt = 0.5f * (z3d[iptr3] - z3d[iptr4]);
        x_etet = x3d[iptr1] + x3d[iptr2] - 2.0f * x3d[iptr];
        y_etet = y3d[iptr1] + y3d[iptr2] - 2.0f * y3d[iptr];
        z_etet = z3d[iptr1] + z3d[iptr2] - 2.0f * z3d[iptr];
        x_ztzt = x3d[iptr3] + x3d[iptr4] - 2.0f * x3d[iptr];
        y_ztzt = y3d[iptr3] + y3d[iptr4] - 2.0f * y3d[iptr];
        z_ztzt = z3d[iptr3] + z3d[iptr4] - 2.0f * z3d[iptr];

        iptr1 = (k - 1) * siz_iz + (j - 1) * siz_iy + 0;
        iptr2 = (k - 1) * siz_iz + (j + 1) * siz_iy + 0;
        iptr3 = (k + 1) * siz_iz + (j + 1) * siz_iy + 0;
        iptr4 = (k + 1) * siz_iz + (j - 1) * siz_iy + 0;

        x_etzt = 0.25f * (x3d[iptr1] + x3d[iptr3] - x3d[iptr2] - x3d[iptr4]);
        y_etzt = 0.25f * (y3d[iptr1] + y3d[iptr3] - y3d[iptr2] - y3d[iptr4]);
        z_etzt = 0.25f * (z3d[iptr1] + z3d[iptr3] - z3d[iptr2] - z3d[iptr4]);

        iptr1 = k * siz_iz + j * siz_iy + 1;
        iptr2 = k * ny + j;
        size = ny * nz;
        x_xi = x3d[iptr1] - x3d[iptr];
        y_xi = y3d[iptr1] - y3d[iptr];
        z_xi = z3d[iptr1] - z3d[iptr];
        x_xixi = p_x1[iptr2 + 0 * size] + x3d[iptr1] - 2.0f * x3d[iptr];
        y_xixi = p_x1[iptr2 + 1 * size] + y3d[iptr1] - 2.0f * y3d[iptr];
        z_xixi = p_x1[iptr2 + 2 * size] + z3d[iptr1] - 2.0f * z3d[iptr];

        g22 = x_et * x_et + y_et * y_et + z_et * z_et;
        g33 = x_zt * x_zt + y_zt * y_zt + z_zt * z_zt;
        g23 = x_et * x_zt + y_et * y_zt + z_et * z_zt;

        c1 = 1.0f / (g22 * g33 - g23 * g23);
        c2 = g23 / g33;
        c3 = g23 / g22;
        c4 = 1.0f / g11_x1[iptr2];

        temp1 = -c4 * (x_xi * x_xixi + y_xi * y_xixi + z_xi * z_xixi);
        temp2 = -c4 * ((x_et - c2 * x_zt) * x_xixi + (y_et - c2 * y_zt) * y_xixi
                      + (z_et - c2 * z_zt) * z_xixi);
        temp3 = -c4 * ((x_zt - c3 * x_et) * x_xixi + (y_zt - c3 * y_et) * y_xixi
                      + (z_zt - c3 * z_et) * z_xixi);
        temp_x = g33 * x_etet + g22 * x_ztzt - 2.0f * g23 * x_etzt;
        temp_y = g33 * y_etet + g22 * y_ztzt - 2.0f * g23 * y_etzt;
        temp_z = g33 * z_etet + g22 * z_ztzt - 2.0f * g23 * z_etzt;

        gnj = gnj1 + (j - 1);
        gnk = gnk1 + (k - 1);
        iptr = gnk * (total_ny - 2) + gnj;
        P_x1_loc[iptr] = temp1 - c1 * (x_xi * temp_x + y_xi * temp_y + z_xi * temp_z);
        Q_x1_loc[iptr] = temp2 - c1 * ((x_et - c2 * x_zt) * temp_x
                       + (y_et - c2 * y_zt) * temp_y + (z_et - c2 * z_zt) * temp_z);
        R_x1_loc[iptr] = temp3 - c1 * ((x_zt - c3 * x_et) * temp_x
                       + (y_zt - c3 * y_et) * temp_y + (z_zt - c3 * z_et) * temp_z);
      }
    }
  }

  /* ---- bdry x2 (xi=1) ---- */
  if (neighid[1] == -2)
  {
    for (int k = 1; k < nz - 1; k++) {
      for (int j = 1; j < ny - 1; j++)
      {
        iptr = k * siz_iz + j * siz_iy + nx - 1;
        iptr1 = k * siz_iz + (j + 1) * siz_iy + nx - 1;
        iptr2 = k * siz_iz + (j - 1) * siz_iy + nx - 1;
        iptr3 = (k + 1) * siz_iz + j * siz_iy + nx - 1;
        iptr4 = (k - 1) * siz_iz + j * siz_iy + nx - 1;
        x_et = 0.5f * (x3d[iptr1] - x3d[iptr2]);
        y_et = 0.5f * (y3d[iptr1] - y3d[iptr2]);
        z_et = 0.5f * (z3d[iptr1] - z3d[iptr2]);
        x_zt = 0.5f * (x3d[iptr3] - x3d[iptr4]);
        y_zt = 0.5f * (y3d[iptr3] - y3d[iptr4]);
        z_zt = 0.5f * (z3d[iptr3] - z3d[iptr4]);
        x_etet = x3d[iptr1] + x3d[iptr2] - 2.0f * x3d[iptr];
        y_etet = y3d[iptr1] + y3d[iptr2] - 2.0f * y3d[iptr];
        z_etet = z3d[iptr1] + z3d[iptr2] - 2.0f * z3d[iptr];
        x_ztzt = x3d[iptr3] + x3d[iptr4] - 2.0f * x3d[iptr];
        y_ztzt = y3d[iptr3] + y3d[iptr4] - 2.0f * y3d[iptr];
        z_ztzt = z3d[iptr3] + z3d[iptr4] - 2.0f * z3d[iptr];

        iptr1 = (k - 1) * siz_iz + (j - 1) * siz_iy + nx - 1;
        iptr2 = (k - 1) * siz_iz + (j + 1) * siz_iy + nx - 1;
        iptr3 = (k + 1) * siz_iz + (j + 1) * siz_iy + nx - 1;
        iptr4 = (k + 1) * siz_iz + (j - 1) * siz_iy + nx - 1;

        x_etzt = 0.25f * (x3d[iptr1] + x3d[iptr3] - x3d[iptr2] - x3d[iptr4]);
        y_etzt = 0.25f * (y3d[iptr1] + y3d[iptr3] - y3d[iptr2] - y3d[iptr4]);
        z_etzt = 0.25f * (z3d[iptr1] + z3d[iptr3] - z3d[iptr2] - z3d[iptr4]);

        iptr1 = k * siz_iz + j * siz_iy + nx - 2;
        iptr2 = k * ny + j;
        size = ny * nz;
        x_xi = x3d[iptr] - x3d[iptr1];
        y_xi = y3d[iptr] - y3d[iptr1];
        z_xi = z3d[iptr] - z3d[iptr1];
        x_xixi = p_x2[iptr2 + 0 * size] + x3d[iptr1] - 2.0f * x3d[iptr];
        y_xixi = p_x2[iptr2 + 1 * size] + y3d[iptr1] - 2.0f * y3d[iptr];
        z_xixi = p_x2[iptr2 + 2 * size] + z3d[iptr1] - 2.0f * z3d[iptr];

        g22 = x_et * x_et + y_et * y_et + z_et * z_et;
        g33 = x_zt * x_zt + y_zt * y_zt + z_zt * z_zt;
        g23 = x_et * x_zt + y_et * y_zt + z_et * z_zt;

        c1 = 1.0f / (g22 * g33 - g23 * g23);
        c2 = g23 / g33;
        c3 = g23 / g22;
        c4 = 1.0f / g11_x2[iptr2];

        temp1 = -c4 * (x_xi * x_xixi + y_xi * y_xixi + z_xi * z_xixi);
        temp2 = -c4 * ((x_et - c2 * x_zt) * x_xixi + (y_et - c2 * y_zt) * y_xixi
                      + (z_et - c2 * z_zt) * z_xixi);
        temp3 = -c4 * ((x_zt - c3 * x_et) * x_xixi + (y_zt - c3 * y_et) * y_xixi
                      + (z_zt - c3 * z_et) * z_xixi);
        temp_x = g33 * x_etet + g22 * x_ztzt - 2.0f * g23 * x_etzt;
        temp_y = g33 * y_etet + g22 * y_ztzt - 2.0f * g23 * y_etzt;
        temp_z = g33 * z_etet + g22 * z_ztzt - 2.0f * g23 * z_etzt;

        gnj = gnj1 + (j - 1);
        gnk = gnk1 + (k - 1);
        iptr = gnk * (total_ny - 2) + gnj;
        P_x2_loc[iptr] = temp1 - c1 * (x_xi * temp_x + y_xi * temp_y + z_xi * temp_z);
        Q_x2_loc[iptr] = temp2 - c1 * ((x_et - c2 * x_zt) * temp_x
                       + (y_et - c2 * y_zt) * temp_y + (z_et - c2 * z_zt) * temp_z);
        R_x2_loc[iptr] = temp3 - c1 * ((x_zt - c3 * x_et) * temp_x
                       + (y_zt - c3 * y_et) * temp_y + (z_zt - c3 * z_et) * temp_z);
      }
    }
  }

  /* ---- bdry y1 (et=0) ---- */
  if (neighid[2] == -2)
  {
    for (int k = 1; k < nz - 1; k++) {
      for (int i = 1; i < nx - 1; i++)
      {
        iptr  = k * siz_iz + 0 * siz_iy + i;
        iptr1 = k * siz_iz + 0 * siz_iy + i + 1;
        iptr2 = k * siz_iz + 0 * siz_iy + i - 1;
        iptr3 = (k + 1) * siz_iz + 0 * siz_iy + i;
        iptr4 = (k - 1) * siz_iz + 0 * siz_iy + i;
        x_xi = 0.5f * (x3d[iptr1] - x3d[iptr2]);
        y_xi = 0.5f * (y3d[iptr1] - y3d[iptr2]);
        z_xi = 0.5f * (z3d[iptr1] - z3d[iptr2]);
        x_zt = 0.5f * (x3d[iptr3] - x3d[iptr4]);
        y_zt = 0.5f * (y3d[iptr3] - y3d[iptr4]);
        z_zt = 0.5f * (z3d[iptr3] - z3d[iptr4]);
        x_xixi = x3d[iptr1] + x3d[iptr2] - 2.0f * x3d[iptr];
        y_xixi = y3d[iptr1] + y3d[iptr2] - 2.0f * y3d[iptr];
        z_xixi = z3d[iptr1] + z3d[iptr2] - 2.0f * z3d[iptr];
        x_ztzt = x3d[iptr3] + x3d[iptr4] - 2.0f * x3d[iptr];
        y_ztzt = y3d[iptr3] + y3d[iptr4] - 2.0f * y3d[iptr];
        z_ztzt = z3d[iptr3] + z3d[iptr4] - 2.0f * z3d[iptr];

        iptr1 = (k - 1) * siz_iz + 0 * siz_iy + i - 1;
        iptr2 = (k - 1) * siz_iz + 0 * siz_iy + i + 1;
        iptr3 = (k + 1) * siz_iz + 0 * siz_iy + i + 1;
        iptr4 = (k + 1) * siz_iz + 0 * siz_iy + i - 1;

        x_xizt = 0.25f * (x3d[iptr1] + x3d[iptr3] - x3d[iptr2] - x3d[iptr4]);
        y_xizt = 0.25f * (y3d[iptr1] + y3d[iptr3] - y3d[iptr2] - y3d[iptr4]);
        z_xizt = 0.25f * (z3d[iptr1] + z3d[iptr3] - z3d[iptr2] - z3d[iptr4]);

        iptr1 = k * siz_iz + 1 * siz_iy + i;
        iptr2 = k * nx + i;
        size = nx * nz;
        x_et = x3d[iptr1] - x3d[iptr];
        y_et = y3d[iptr1] - y3d[iptr];
        z_et = z3d[iptr1] - z3d[iptr];
        x_etet = p_y1[iptr2 + 0 * size] + x3d[iptr1] - 2.0f * x3d[iptr];
        y_etet = p_y1[iptr2 + 1 * size] + y3d[iptr1] - 2.0f * y3d[iptr];
        z_etet = p_y1[iptr2 + 2 * size] + z3d[iptr1] - 2.0f * z3d[iptr];

        g11 = x_xi * x_xi + y_xi * y_xi + z_xi * z_xi;
        g33 = x_zt * x_zt + y_zt * y_zt + z_zt * z_zt;
        g13 = x_xi * x_zt + y_xi * y_zt + z_xi * z_zt;

        c1 = 1.0f / (g11 * g33 - g13 * g13);
        c2 = g13 / g33;
        c3 = g13 / g11;
        c4 = 1.0f / g22_y1[iptr2];

        temp1 = -c4 * ((x_xi - c2 * x_zt) * x_etet + (y_xi - c2 * y_zt) * y_etet
                      + (z_xi - c2 * z_zt) * z_etet);
        temp2 = -c4 * (x_et * x_etet + y_et * y_etet + z_et * z_etet);
        temp3 = -c4 * ((x_zt - c3 * x_xi) * x_etet + (y_zt - c3 * y_xi) * y_etet
                      + (z_zt - c3 * z_xi) * z_etet);
        temp_x = g33 * x_xixi + g11 * x_ztzt - 2.0f * g13 * x_xizt;
        temp_y = g33 * y_xixi + g11 * y_ztzt - 2.0f * g13 * y_xizt;
        temp_z = g33 * z_xixi + g11 * z_ztzt - 2.0f * g13 * z_xizt;

        gni = gni1 + (i - 1);
        gnk = gnk1 + (k - 1);
        iptr = gnk * (total_nx - 2) + gni;
        P_y1_loc[iptr] = temp1 - c1 * ((x_xi - c2 * x_zt) * temp_x
                       + (y_xi - c2 * y_zt) * temp_y + (z_xi - c2 * z_zt) * temp_z);
        Q_y1_loc[iptr] = temp2 - c1 * (x_et * temp_x + y_et * temp_y + z_et * temp_z);
        R_y1_loc[iptr] = temp3 - c1 * ((x_zt - c3 * x_xi) * temp_x
                       + (y_zt - c3 * y_xi) * temp_y + (z_zt - c3 * z_xi) * temp_z);
      }
    }
  }

  /* ---- bdry y2 (et=1) ---- */
  if (neighid[3] == -2)
  {
    for (int k = 1; k < nz - 1; k++) {
      for (int i = 1; i < nx - 1; i++)
      {
        iptr  = k * siz_iz + (ny - 1) * siz_iy + i;
        iptr1 = k * siz_iz + (ny - 1) * siz_iy + i + 1;
        iptr2 = k * siz_iz + (ny - 1) * siz_iy + i - 1;
        iptr3 = (k + 1) * siz_iz + (ny - 1) * siz_iy + i;
        iptr4 = (k - 1) * siz_iz + (ny - 1) * siz_iy + i;
        x_xi = 0.5f * (x3d[iptr1] - x3d[iptr2]);
        y_xi = 0.5f * (y3d[iptr1] - y3d[iptr2]);
        z_xi = 0.5f * (z3d[iptr1] - z3d[iptr2]);
        x_zt = 0.5f * (x3d[iptr3] - x3d[iptr4]);
        y_zt = 0.5f * (y3d[iptr3] - y3d[iptr4]);
        z_zt = 0.5f * (z3d[iptr3] - z3d[iptr4]);
        x_xixi = x3d[iptr1] + x3d[iptr2] - 2.0f * x3d[iptr];
        y_xixi = y3d[iptr1] + y3d[iptr2] - 2.0f * y3d[iptr];
        z_xixi = z3d[iptr1] + z3d[iptr2] - 2.0f * z3d[iptr];
        x_ztzt = x3d[iptr3] + x3d[iptr4] - 2.0f * x3d[iptr];
        y_ztzt = y3d[iptr3] + y3d[iptr4] - 2.0f * y3d[iptr];
        z_ztzt = z3d[iptr3] + z3d[iptr4] - 2.0f * z3d[iptr];

        iptr1 = (k - 1) * siz_iz + (ny - 1) * siz_iy + i - 1;
        iptr2 = (k - 1) * siz_iz + (ny - 1) * siz_iy + i + 1;
        iptr3 = (k + 1) * siz_iz + (ny - 1) * siz_iy + i + 1;
        iptr4 = (k + 1) * siz_iz + (ny - 1) * siz_iy + i - 1;

        x_xizt = 0.25f * (x3d[iptr1] + x3d[iptr3] - x3d[iptr2] - x3d[iptr4]);
        y_xizt = 0.25f * (y3d[iptr1] + y3d[iptr3] - y3d[iptr2] - y3d[iptr4]);
        z_xizt = 0.25f * (z3d[iptr1] + z3d[iptr3] - z3d[iptr2] - z3d[iptr4]);

        iptr1 = k * siz_iz + (ny - 2) * siz_iy + i;
        iptr2 = k * nx + i;
        size = nx * nz;
        x_et = x3d[iptr] - x3d[iptr1];
        y_et = y3d[iptr] - y3d[iptr1];
        z_et = z3d[iptr] - z3d[iptr1];
        x_etet = p_y2[iptr2 + 0 * size] + x3d[iptr1] - 2.0f * x3d[iptr];
        y_etet = p_y2[iptr2 + 1 * size] + y3d[iptr1] - 2.0f * y3d[iptr];
        z_etet = p_y2[iptr2 + 2 * size] + z3d[iptr1] - 2.0f * z3d[iptr];

        g11 = x_xi * x_xi + y_xi * y_xi + z_xi * z_xi;
        g33 = x_zt * x_zt + y_zt * y_zt + z_zt * z_zt;
        g13 = x_xi * x_zt + y_xi * y_zt + z_xi * z_zt;

        c1 = 1.0f / (g11 * g33 - g13 * g13);
        c2 = g13 / g33;
        c3 = g13 / g11;
        c4 = 1.0f / g22_y2[iptr2];

        temp1 = -c4 * ((x_xi - c2 * x_zt) * x_etet + (y_xi - c2 * y_zt) * y_etet
                      + (z_xi - c2 * z_zt) * z_etet);
        temp2 = -c4 * (x_et * x_etet + y_et * y_etet + z_et * z_etet);
        temp3 = -c4 * ((x_zt - c3 * x_xi) * x_etet + (y_zt - c3 * y_xi) * y_etet
                      + (z_zt - c3 * z_xi) * z_etet);
        temp_x = g33 * x_xixi + g11 * x_ztzt - 2.0f * g13 * x_xizt;
        temp_y = g33 * y_xixi + g11 * y_ztzt - 2.0f * g13 * y_xizt;
        temp_z = g33 * z_xixi + g11 * z_ztzt - 2.0f * g13 * z_xizt;

        gni = gni1 + (i - 1);
        gnk = gnk1 + (k - 1);
        iptr = gnk * (total_nx - 2) + gni;
        P_y2_loc[iptr] = temp1 - c1 * ((x_xi - c2 * x_zt) * temp_x
                       + (y_xi - c2 * y_zt) * temp_y + (z_xi - c2 * z_zt) * temp_z);
        Q_y2_loc[iptr] = temp2 - c1 * (x_et * temp_x + y_et * temp_y + z_et * temp_z);
        R_y2_loc[iptr] = temp3 - c1 * ((x_zt - c3 * x_xi) * temp_x
                       + (y_zt - c3 * y_xi) * temp_y + (z_zt - c3 * z_xi) * temp_z);
      }
    }
  }

  /* ---- bdry z1 (zt=0) ---- */
  if (neighid[4] == -2)
  {
    for (int j = 1; j < ny - 1; j++) {
      for (int i = 1; i < nx - 1; i++)
      {
        iptr  = 0 * siz_iz + j * siz_iy + i;
        iptr1 = 0 * siz_iz + j * siz_iy + i + 1;
        iptr2 = 0 * siz_iz + j * siz_iy + i - 1;
        iptr3 = 0 * siz_iz + (j + 1) * siz_iy + i;
        iptr4 = 0 * siz_iz + (j - 1) * siz_iy + i;
        x_xi = 0.5f * (x3d[iptr1] - x3d[iptr2]);
        y_xi = 0.5f * (y3d[iptr1] - y3d[iptr2]);
        z_xi = 0.5f * (z3d[iptr1] - z3d[iptr2]);
        x_et = 0.5f * (x3d[iptr3] - x3d[iptr4]);
        y_et = 0.5f * (y3d[iptr3] - y3d[iptr4]);
        z_et = 0.5f * (z3d[iptr3] - z3d[iptr4]);
        x_xixi = x3d[iptr1] + x3d[iptr2] - 2.0f * x3d[iptr];
        y_xixi = y3d[iptr1] + y3d[iptr2] - 2.0f * y3d[iptr];
        z_xixi = z3d[iptr1] + z3d[iptr2] - 2.0f * z3d[iptr];
        x_etet = x3d[iptr3] + x3d[iptr4] - 2.0f * x3d[iptr];
        y_etet = y3d[iptr3] + y3d[iptr4] - 2.0f * y3d[iptr];
        z_etet = z3d[iptr3] + z3d[iptr4] - 2.0f * z3d[iptr];

        iptr1 = 0 * siz_iz + (j - 1) * siz_iy + i - 1;
        iptr2 = 0 * siz_iz + (j - 1) * siz_iy + i + 1;
        iptr3 = 0 * siz_iz + (j + 1) * siz_iy + i + 1;
        iptr4 = 0 * siz_iz + (j + 1) * siz_iy + i - 1;

        x_xiet = 0.25f * (x3d[iptr1] + x3d[iptr3] - x3d[iptr2] - x3d[iptr4]);
        y_xiet = 0.25f * (y3d[iptr1] + y3d[iptr3] - y3d[iptr2] - y3d[iptr4]);
        z_xiet = 0.25f * (z3d[iptr1] + z3d[iptr3] - z3d[iptr2] - z3d[iptr4]);

        iptr1 = 1 * siz_iz + j * siz_iy + i;
        iptr2 = j * nx + i;
        size = nx * ny;
        x_zt = x3d[iptr1] - x3d[iptr];
        y_zt = y3d[iptr1] - y3d[iptr];
        z_zt = z3d[iptr1] - z3d[iptr];
        x_ztzt = p_z1[iptr2 + 0 * size] + x3d[iptr1] - 2.0f * x3d[iptr];
        y_ztzt = p_z1[iptr2 + 1 * size] + y3d[iptr1] - 2.0f * y3d[iptr];
        z_ztzt = p_z1[iptr2 + 2 * size] + z3d[iptr1] - 2.0f * z3d[iptr];

        g11 = x_xi * x_xi + y_xi * y_xi + z_xi * z_xi;
        g22 = x_et * x_et + y_et * y_et + z_et * z_et;
        g12 = x_xi * x_et + y_xi * y_et + z_xi * z_et;

        c1 = 1.0f / (g11 * g22 - g12 * g12);
        c2 = g12 / g22;
        c3 = g12 / g11;
        c4 = 1.0f / g33_z1[iptr2];

        temp1 = -c4 * ((x_xi - c2 * x_et) * x_ztzt + (y_xi - c2 * y_et) * y_ztzt
                      + (z_xi - c2 * z_et) * z_ztzt);
        temp2 = -c4 * ((x_et - c3 * x_xi) * x_ztzt + (y_et - c3 * y_xi) * y_ztzt
                      + (z_et - c3 * z_xi) * z_ztzt);
        temp3 = -c4 * (x_zt * x_ztzt + y_zt * y_ztzt + z_zt * z_ztzt);
        temp_x = g22 * x_xixi + g11 * x_etet - 2.0f * g12 * x_xiet;
        temp_y = g22 * y_xixi + g11 * y_etet - 2.0f * g12 * y_xiet;
        temp_z = g22 * z_xixi + g11 * z_etet - 2.0f * g12 * z_xiet;

        gni = gni1 + (i - 1);
        gnj = gnj1 + (j - 1);
        iptr = gnj * (total_nx - 2) + gni;
        P_z1_loc[iptr] = temp1 - c1 * ((x_xi - c2 * x_et) * temp_x
                       + (y_xi - c2 * y_et) * temp_y + (z_xi - c2 * z_et) * temp_z);
        Q_z1_loc[iptr] = temp2 - c1 * ((x_et - c3 * x_xi) * temp_x
                       + (y_et - c3 * y_xi) * temp_y + (z_et - c3 * z_xi) * temp_z);
        R_z1_loc[iptr] = temp3 - c1 * (x_zt * temp_x + y_zt * temp_y + z_zt * temp_z);
      }
    }
  }

  /* ---- bdry z2 (zt=1) ---- */
  if (neighid[5] == -2)
  {
    for (int j = 1; j < ny - 1; j++) {
      for (int i = 1; i < nx - 1; i++)
      {
        iptr  = (nz - 1) * siz_iz + j * siz_iy + i;
        iptr1 = (nz - 1) * siz_iz + j * siz_iy + i + 1;
        iptr2 = (nz - 1) * siz_iz + j * siz_iy + i - 1;
        iptr3 = (nz - 1) * siz_iz + (j + 1) * siz_iy + i;
        iptr4 = (nz - 1) * siz_iz + (j - 1) * siz_iy + i;
        x_xi = 0.5f * (x3d[iptr1] - x3d[iptr2]);
        y_xi = 0.5f * (y3d[iptr1] - y3d[iptr2]);
        z_xi = 0.5f * (z3d[iptr1] - z3d[iptr2]);
        x_et = 0.5f * (x3d[iptr3] - x3d[iptr4]);
        y_et = 0.5f * (y3d[iptr3] - y3d[iptr4]);
        z_et = 0.5f * (z3d[iptr3] - z3d[iptr4]);
        x_xixi = x3d[iptr1] + x3d[iptr2] - 2.0f * x3d[iptr];
        y_xixi = y3d[iptr1] + y3d[iptr2] - 2.0f * y3d[iptr];
        z_xixi = z3d[iptr1] + z3d[iptr2] - 2.0f * z3d[iptr];
        x_etet = x3d[iptr3] + x3d[iptr4] - 2.0f * x3d[iptr];
        y_etet = y3d[iptr3] + y3d[iptr4] - 2.0f * y3d[iptr];
        z_etet = z3d[iptr3] + z3d[iptr4] - 2.0f * z3d[iptr];

        iptr1 = (nz - 1) * siz_iz + (j - 1) * siz_iy + i - 1;
        iptr2 = (nz - 1) * siz_iz + (j - 1) * siz_iy + i + 1;
        iptr3 = (nz - 1) * siz_iz + (j + 1) * siz_iy + i + 1;
        iptr4 = (nz - 1) * siz_iz + (j + 1) * siz_iy + i - 1;

        x_xiet = 0.25f * (x3d[iptr1] + x3d[iptr3] - x3d[iptr2] - x3d[iptr4]);
        y_xiet = 0.25f * (y3d[iptr1] + y3d[iptr3] - y3d[iptr2] - y3d[iptr4]);
        z_xiet = 0.25f * (z3d[iptr1] + z3d[iptr3] - z3d[iptr2] - z3d[iptr4]);

        iptr1 = (nz - 2) * siz_iz + j * siz_iy + i;
        iptr2 = j * nx + i;
        size = nx * ny;
        x_zt = x3d[iptr] - x3d[iptr1];
        y_zt = y3d[iptr] - y3d[iptr1];
        z_zt = z3d[iptr] - z3d[iptr1];
        x_ztzt = p_z2[iptr2 + 0 * size] + x3d[iptr1] - 2.0f * x3d[iptr];
        y_ztzt = p_z2[iptr2 + 1 * size] + y3d[iptr1] - 2.0f * y3d[iptr];
        z_ztzt = p_z2[iptr2 + 2 * size] + z3d[iptr1] - 2.0f * z3d[iptr];

        g11 = x_xi * x_xi + y_xi * y_xi + z_xi * z_xi;
        g22 = x_et * x_et + y_et * y_et + z_et * z_et;
        g12 = x_xi * x_et + y_xi * y_et + z_xi * z_et;

        c1 = 1.0f / (g11 * g22 - g12 * g12);
        c2 = g12 / g22;
        c3 = g12 / g11;
        c4 = 1.0f / g33_z2[iptr2];

        temp1 = -c4 * ((x_xi - c2 * x_et) * x_ztzt + (y_xi - c2 * y_et) * y_ztzt
                      + (z_xi - c2 * z_et) * z_ztzt);
        temp2 = -c4 * ((x_et - c3 * x_xi) * x_ztzt + (y_et - c3 * y_xi) * y_ztzt
                      + (z_et - c3 * z_xi) * z_ztzt);
        temp3 = -c4 * (x_zt * x_ztzt + y_zt * y_ztzt + z_zt * z_ztzt);
        temp_x = g22 * x_xixi + g11 * x_etet - 2.0f * g12 * x_xiet;
        temp_y = g22 * y_xixi + g11 * y_etet - 2.0f * g12 * y_xiet;
        temp_z = g22 * z_xixi + g11 * z_etet - 2.0f * g12 * z_xiet;

        gni = gni1 + (i - 1);
        gnj = gnj1 + (j - 1);
        iptr = gnj * (total_nx - 2) + gni;
        P_z2_loc[iptr] = temp1 - c1 * ((x_xi - c2 * x_et) * temp_x
                       + (y_xi - c2 * y_et) * temp_y + (z_xi - c2 * z_et) * temp_z);
        Q_z2_loc[iptr] = temp2 - c1 * ((x_et - c3 * x_xi) * temp_x
                       + (y_et - c3 * y_xi) * temp_y + (z_et - c3 * z_xi) * temp_z);
        R_z2_loc[iptr] = temp3 - c1 * (x_zt * temp_x + y_zt * temp_y + z_zt * temp_z);
      }
    }
  }

  /* Copy _loc arrays to non-_loc arrays (Python handles MPI_Allreduce) */
  size_t siz_bx = (size_t)(total_ny - 2) * (total_nz - 2);
  size_t siz_by = (size_t)(total_nx - 2) * (total_nz - 2);
  size_t siz_bz = (size_t)(total_nx - 2) * (total_ny - 2);

  memcpy(P_x1, P_x1_loc, siz_bx * sizeof(float));
  memcpy(Q_x1, Q_x1_loc, siz_bx * sizeof(float));
  memcpy(R_x1, R_x1_loc, siz_bx * sizeof(float));
  memcpy(P_x2, P_x2_loc, siz_bx * sizeof(float));
  memcpy(Q_x2, Q_x2_loc, siz_bx * sizeof(float));
  memcpy(R_x2, R_x2_loc, siz_bx * sizeof(float));

  memcpy(P_y1, P_y1_loc, siz_by * sizeof(float));
  memcpy(Q_y1, Q_y1_loc, siz_by * sizeof(float));
  memcpy(R_y1, R_y1_loc, siz_by * sizeof(float));
  memcpy(P_y2, P_y2_loc, siz_by * sizeof(float));
  memcpy(Q_y2, Q_y2_loc, siz_by * sizeof(float));
  memcpy(R_y2, R_y2_loc, siz_by * sizeof(float));

  memcpy(P_z1, P_z1_loc, siz_bz * sizeof(float));
  memcpy(Q_z1, Q_z1_loc, siz_bz * sizeof(float));
  memcpy(R_z1, R_z1_loc, siz_bz * sizeof(float));
  memcpy(P_z2, P_z2_loc, siz_bz * sizeof(float));
  memcpy(Q_z2, Q_z2_loc, siz_bz * sizeof(float));
  memcpy(R_z2, R_z2_loc, siz_bz * sizeof(float));
}

/* ------------------------------------------------------------------ */
/* 8. Distance calculation for Higenstock method                       */
/*     Compute normal distances from boundary faces                    */
/* ------------------------------------------------------------------ */
void dist_cal_c(float *x3d, float *y3d, float *z3d,
                int nx, int ny, int nz,
                float *dx1, float *dx2,
                float *dy1, float *dy2,
                float *dz1, float *dz2,
                int *neighid)
{
  size_t iptr, iptr1, iptr2;
  size_t siz_iy = nx;
  size_t siz_iz = nx * ny;
  float x_xi, y_xi, z_xi;
  float x_et, y_et, z_et;
  float x_zt, y_zt, z_zt;
  float x_xi0, y_xi0, z_xi0;
  float x_et0, y_et0, z_et0;
  float x_zt0, y_zt0, z_zt0;
  float vn_x, vn_y, vn_z, len_vn;
  float vn_x0, vn_y0, vn_z0;

  /* bdry x1: r_et X r_zt */
  if (neighid[0] == -2) {
    for (int k = 1; k < nz-1; k++) {
      for (int j = 1; j < ny-1; j++) {
        iptr1 = k*siz_iz + j*siz_iy + 1;
        iptr2 = k*siz_iz + j*siz_iy + 0;
        x_xi0 = x3d[iptr1] - x3d[iptr2];
        y_xi0 = y3d[iptr1] - y3d[iptr2];
        z_xi0 = z3d[iptr1] - z3d[iptr2];
        iptr1 = k*siz_iz + (j+1)*siz_iy + 0;
        iptr2 = k*siz_iz + (j-1)*siz_iy + 0;
        x_et = 0.5f*(x3d[iptr1] - x3d[iptr2]);
        y_et = 0.5f*(y3d[iptr1] - y3d[iptr2]);
        z_et = 0.5f*(z3d[iptr1] - z3d[iptr2]);
        iptr1 = (k+1)*siz_iz + j*siz_iy + 0;
        iptr2 = (k-1)*siz_iz + j*siz_iy + 0;
        x_zt = 0.5f*(x3d[iptr1] - x3d[iptr2]);
        y_zt = 0.5f*(y3d[iptr1] - y3d[iptr2]);
        z_zt = 0.5f*(z3d[iptr1] - z3d[iptr2]);
        vn_x = y_et*z_zt - z_et*y_zt;
        vn_y = z_et*x_zt - x_et*z_zt;
        vn_z = x_et*y_zt - y_et*x_zt;
        len_vn = sqrtf(vn_x*vn_x + vn_y*vn_y + vn_z*vn_z);
        vn_x0 = vn_x/len_vn; vn_y0 = vn_y/len_vn; vn_z0 = vn_z/len_vn;
        iptr = k*ny + j;
        dx1[iptr] = x_xi0*vn_x0 + y_xi0*vn_y0 + z_xi0*vn_z0;
      }
    }
  }
  /* bdry x2 */
  if (neighid[1] == -2) {
    for (int k = 1; k < nz-1; k++) {
      for (int j = 1; j < ny-1; j++) {
        iptr1 = k*siz_iz + j*siz_iy + nx-1;
        iptr2 = k*siz_iz + j*siz_iy + nx-2;
        x_xi0 = x3d[iptr1] - x3d[iptr2];
        y_xi0 = y3d[iptr1] - y3d[iptr2];
        z_xi0 = z3d[iptr1] - z3d[iptr2];
        iptr1 = k*siz_iz + (j+1)*siz_iy + nx-1;
        iptr2 = k*siz_iz + (j-1)*siz_iy + nx-1;
        x_et = 0.5f*(x3d[iptr1] - x3d[iptr2]);
        y_et = 0.5f*(y3d[iptr1] - y3d[iptr2]);
        z_et = 0.5f*(z3d[iptr1] - z3d[iptr2]);
        iptr1 = (k+1)*siz_iz + j*siz_iy + nx-1;
        iptr2 = (k-1)*siz_iz + j*siz_iy + nx-1;
        x_zt = 0.5f*(x3d[iptr1] - x3d[iptr2]);
        y_zt = 0.5f*(y3d[iptr1] - y3d[iptr2]);
        z_zt = 0.5f*(z3d[iptr1] - z3d[iptr2]);
        vn_x = y_et*z_zt - z_et*y_zt;
        vn_y = z_et*x_zt - x_et*z_zt;
        vn_z = x_et*y_zt - y_et*x_zt;
        len_vn = sqrtf(vn_x*vn_x + vn_y*vn_y + vn_z*vn_z);
        vn_x0 = vn_x/len_vn; vn_y0 = vn_y/len_vn; vn_z0 = vn_z/len_vn;
        iptr = k*ny + j;
        dx2[iptr] = x_xi0*vn_x0 + y_xi0*vn_y0 + z_xi0*vn_z0;
      }
    }
  }
  /* bdry y1: r_zt X r_xi */
  if (neighid[2] == -2) {
    for (int k = 1; k < nz-1; k++) {
      for (int i = 1; i < nx-1; i++) {
        iptr1 = k*siz_iz + 1*siz_iy + i;
        iptr2 = k*siz_iz + 0*siz_iy + i;
        x_et0 = x3d[iptr1] - x3d[iptr2];
        y_et0 = y3d[iptr1] - y3d[iptr2];
        z_et0 = z3d[iptr1] - z3d[iptr2];
        iptr1 = k*siz_iz + 0*siz_iy + (i+1);
        iptr2 = k*siz_iz + 0*siz_iy + (i-1);
        x_xi = 0.5f*(x3d[iptr1] - x3d[iptr2]);
        y_xi = 0.5f*(y3d[iptr1] - y3d[iptr2]);
        z_xi = 0.5f*(z3d[iptr1] - z3d[iptr2]);
        iptr1 = (k+1)*siz_iz + 0*siz_iy + i;
        iptr2 = (k-1)*siz_iz + 0*siz_iy + i;
        x_zt = 0.5f*(x3d[iptr1] - x3d[iptr2]);
        y_zt = 0.5f*(y3d[iptr1] - y3d[iptr2]);
        z_zt = 0.5f*(z3d[iptr1] - z3d[iptr2]);
        vn_x = y_zt*z_xi - z_zt*y_xi;
        vn_y = z_zt*x_xi - x_zt*z_xi;
        vn_z = x_zt*y_xi - y_zt*x_xi;
        len_vn = sqrtf(vn_x*vn_x + vn_y*vn_y + vn_z*vn_z);
        vn_x0 = vn_x/len_vn; vn_y0 = vn_y/len_vn; vn_z0 = vn_z/len_vn;
        iptr = k*nx + i;
        dy1[iptr] = x_et0*vn_x0 + y_et0*vn_y0 + z_et0*vn_z0;
      }
    }
  }
  /* bdry y2 */
  if (neighid[3] == -2) {
    for (int k = 1; k < nz-1; k++) {
      for (int i = 1; i < nx-1; i++) {
        iptr1 = k*siz_iz + (ny-1)*siz_iy + i;
        iptr2 = k*siz_iz + (ny-2)*siz_iy + i;
        x_et0 = x3d[iptr1] - x3d[iptr2];
        y_et0 = y3d[iptr1] - y3d[iptr2];
        z_et0 = z3d[iptr1] - z3d[iptr2];
        iptr1 = k*siz_iz + (ny-1)*siz_iy + (i+1);
        iptr2 = k*siz_iz + (ny-1)*siz_iy + (i-1);
        x_xi = 0.5f*(x3d[iptr1] - x3d[iptr2]);
        y_xi = 0.5f*(y3d[iptr1] - y3d[iptr2]);
        z_xi = 0.5f*(z3d[iptr1] - z3d[iptr2]);
        iptr1 = (k+1)*siz_iz + (ny-1)*siz_iy + i;
        iptr2 = (k-1)*siz_iz + (ny-1)*siz_iy + i;
        x_zt = 0.5f*(x3d[iptr1] - x3d[iptr2]);
        y_zt = 0.5f*(y3d[iptr1] - y3d[iptr2]);
        z_zt = 0.5f*(z3d[iptr1] - z3d[iptr2]);
        vn_x = y_zt*z_xi - z_zt*y_xi;
        vn_y = z_zt*x_xi - x_zt*z_xi;
        vn_z = x_zt*y_xi - y_zt*x_xi;
        len_vn = sqrtf(vn_x*vn_x + vn_y*vn_y + vn_z*vn_z);
        vn_x0 = vn_x/len_vn; vn_y0 = vn_y/len_vn; vn_z0 = vn_z/len_vn;
        iptr = k*nx + i;
        dy2[iptr] = x_et0*vn_x0 + y_et0*vn_y0 + z_et0*vn_z0;
      }
    }
  }
  /* bdry z1: r_xi X r_et */
  if (neighid[4] == -2) {
    for (int j = 1; j < ny-1; j++) {
      for (int i = 1; i < nx-1; i++) {
        iptr1 = 1*siz_iz + j*siz_iy + i;
        iptr2 = 0*siz_iz + j*siz_iy + i;
        x_zt0 = x3d[iptr1] - x3d[iptr2];
        y_zt0 = y3d[iptr1] - y3d[iptr2];
        z_zt0 = z3d[iptr1] - z3d[iptr2];
        iptr1 = 0*siz_iz + j*siz_iy + (i+1);
        iptr2 = 0*siz_iz + j*siz_iy + (i-1);
        x_xi = 0.5f*(x3d[iptr1] - x3d[iptr2]);
        y_xi = 0.5f*(y3d[iptr1] - y3d[iptr2]);
        z_xi = 0.5f*(z3d[iptr1] - z3d[iptr2]);
        iptr1 = 0*siz_iz + (j+1)*siz_iy + i;
        iptr2 = 0*siz_iz + (j-1)*siz_iy + i;
        x_et = 0.5f*(x3d[iptr1] - x3d[iptr2]);
        y_et = 0.5f*(y3d[iptr1] - y3d[iptr2]);
        z_et = 0.5f*(z3d[iptr1] - z3d[iptr2]);
        vn_x = y_xi*z_et - z_xi*y_et;
        vn_y = z_xi*x_et - x_xi*z_et;
        vn_z = x_xi*y_et - y_xi*x_et;
        len_vn = sqrtf(vn_x*vn_x + vn_y*vn_y + vn_z*vn_z);
        vn_x0 = vn_x/len_vn; vn_y0 = vn_y/len_vn; vn_z0 = vn_z/len_vn;
        iptr = j*nx + i;
        dz1[iptr] = x_zt0*vn_x0 + y_zt0*vn_y0 + z_zt0*vn_z0;
      }
    }
  }
  /* bdry z2 */
  if (neighid[5] == -2) {
    for (int j = 1; j < ny-1; j++) {
      for (int i = 1; i < nx-1; i++) {
        iptr1 = (nz-1)*siz_iz + j*siz_iy + i;
        iptr2 = (nz-2)*siz_iz + j*siz_iy + i;
        x_zt0 = x3d[iptr1] - x3d[iptr2];
        y_zt0 = y3d[iptr1] - y3d[iptr2];
        z_zt0 = z3d[iptr1] - z3d[iptr2];
        iptr1 = (nz-1)*siz_iz + j*siz_iy + (i+1);
        iptr2 = (nz-1)*siz_iz + j*siz_iy + (i-1);
        x_xi = 0.5f*(x3d[iptr1] - x3d[iptr2]);
        y_xi = 0.5f*(y3d[iptr1] - y3d[iptr2]);
        z_xi = 0.5f*(z3d[iptr1] - z3d[iptr2]);
        iptr1 = (nz-1)*siz_iz + (j+1)*siz_iy + i;
        iptr2 = (nz-1)*siz_iz + (j-1)*siz_iy + i;
        x_et = 0.5f*(x3d[iptr1] - x3d[iptr2]);
        y_et = 0.5f*(y3d[iptr1] - y3d[iptr2]);
        z_et = 0.5f*(z3d[iptr1] - z3d[iptr2]);
        vn_x = y_xi*z_et - z_xi*y_et;
        vn_y = z_xi*x_et - x_xi*z_et;
        vn_z = x_xi*y_et - y_xi*x_et;
        len_vn = sqrtf(vn_x*vn_x + vn_y*vn_y + vn_z*vn_z);
        vn_x0 = vn_x/len_vn; vn_y0 = vn_y/len_vn; vn_z0 = vn_z/len_vn;
        iptr = j*nx + i;
        dz2[iptr] = x_zt0*vn_x0 + y_zt0*vn_y0 + z_zt0*vn_z0;
      }
    }
  }
}

/* ------------------------------------------------------------------ */
/* 9. Higenstock source term setting                                   */
/*     Computes orthogonality and distance-based source terms          */
/* ------------------------------------------------------------------ */
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
                     int total_nx, int total_ny, int total_nz)
{
  size_t siz_iy = nx;
  size_t siz_iz = nx * ny;
  float theta0 = 1.5707963267948966f; /* PI/2 */
  float a = 0.2f;
  size_t iptr, iptr1, iptr2, iptr3, iptr4;
  float dot_val, len_xi, len_et, len_zt, dif_dis;
  float cos_theta, theta, dif_theta;
  float x_xi, y_xi, z_xi;
  float x_et, y_et, z_et;
  float x_zt, y_zt, z_zt;
  int gni, gnj, gnk;
  size_t siz_bx = (size_t)(total_ny-2) * (total_nz-2);
  size_t siz_by = (size_t)(total_nx-2) * (total_nz-2);
  size_t siz_bz = (size_t)(total_nx-2) * (total_ny-2);

  /* Do NOT clear _loc arrays — Higenstock accumulates source terms
     across iterations, matching C original set_src_higen behavior */

  /* bdry x1: xi=0 */
  if (neighid[0] == -2) {
    for (int k = 1; k < nz-1; k++) {
      for (int j = 1; j < ny-1; j++) {
        iptr  = k*siz_iz + j*siz_iy + 0;
        iptr1 = k*siz_iz + (j+1)*siz_iy + 0;
        iptr2 = k*siz_iz + (j-1)*siz_iy + 0;
        iptr3 = (k+1)*siz_iz + j*siz_iy + 0;
        iptr4 = (k-1)*siz_iz + j*siz_iy + 0;
        x_et = 0.5f*(x3d[iptr1]-x3d[iptr2]);
        y_et = 0.5f*(y3d[iptr1]-y3d[iptr2]);
        z_et = 0.5f*(z3d[iptr1]-z3d[iptr2]);
        x_zt = 0.5f*(x3d[iptr3]-x3d[iptr4]);
        y_zt = 0.5f*(y3d[iptr3]-y3d[iptr4]);
        z_zt = 0.5f*(z3d[iptr3]-z3d[iptr4]);
        iptr1 = k*siz_iz + j*siz_iy + 1;
        x_xi = x3d[iptr1] - x3d[iptr];
        y_xi = y3d[iptr1] - y3d[iptr];
        z_xi = z3d[iptr1] - z3d[iptr];
        gnj = gnj1 + (j-1);
        gnk = gnk1 + (k-1);
        iptr = gnk*(total_ny-2) + gnj;
        len_xi = sqrtf(x_xi*x_xi + y_xi*y_xi + z_xi*z_xi);
        len_et = sqrtf(x_et*x_et + y_et*y_et + z_et*z_et);
        len_zt = sqrtf(x_zt*x_zt + y_zt*y_zt + z_zt*z_zt);
        dot_val = x_xi*x_et + y_xi*y_et + z_xi*z_et;
        cos_theta = dot_val/(len_xi*len_et);
        theta = acosf(cos_theta);
        dif_theta = (theta0-theta)/theta0;
        Q_x1_loc[iptr] -= a*tanhf(dif_theta);
        dot_val = x_xi*x_zt + y_xi*y_zt + z_xi*z_zt;
        cos_theta = dot_val/(len_xi*len_zt);
        theta = acosf(cos_theta);
        dif_theta = (theta0-theta)/theta0;
        R_x1_loc[iptr] -= a*tanhf(dif_theta);
        iptr1 = k*ny + j;
        dif_dis = (dx1[iptr1]-len_xi)/dx1[iptr1];
        P_x1_loc[iptr] += a*tanhf(dif_dis);
      }
    }
  }
  /* bdry x2: xi=1 */
  if (neighid[1] == -2) {
    for (int k = 1; k < nz-1; k++) {
      for (int j = 1; j < ny-1; j++) {
        iptr  = k*siz_iz + j*siz_iy + nx-1;
        iptr1 = k*siz_iz + (j+1)*siz_iy + nx-1;
        iptr2 = k*siz_iz + (j-1)*siz_iy + nx-1;
        iptr3 = (k+1)*siz_iz + j*siz_iy + nx-1;
        iptr4 = (k-1)*siz_iz + j*siz_iy + nx-1;
        x_et = 0.5f*(x3d[iptr1]-x3d[iptr2]);
        y_et = 0.5f*(y3d[iptr1]-y3d[iptr2]);
        z_et = 0.5f*(z3d[iptr1]-z3d[iptr2]);
        x_zt = 0.5f*(x3d[iptr3]-x3d[iptr4]);
        y_zt = 0.5f*(y3d[iptr3]-y3d[iptr4]);
        z_zt = 0.5f*(z3d[iptr3]-z3d[iptr4]);
        iptr1 = k*siz_iz + j*siz_iy + nx-2;
        x_xi = x3d[iptr] - x3d[iptr1];
        y_xi = y3d[iptr] - y3d[iptr1];
        z_xi = z3d[iptr] - z3d[iptr1];
        gnj = gnj1 + (j-1);
        gnk = gnk1 + (k-1);
        iptr = gnk*(total_ny-2) + gnj;
        len_xi = sqrtf(x_xi*x_xi + y_xi*y_xi + z_xi*z_xi);
        len_et = sqrtf(x_et*x_et + y_et*y_et + z_et*z_et);
        len_zt = sqrtf(x_zt*x_zt + y_zt*y_zt + z_zt*z_zt);
        dot_val = x_xi*x_et + y_xi*y_et + z_xi*z_et;
        cos_theta = dot_val/(len_xi*len_et);
        theta = acosf(cos_theta);
        dif_theta = (theta0-theta)/theta0;
        Q_x2_loc[iptr] += a*tanhf(dif_theta);
        dot_val = x_xi*x_zt + y_xi*y_zt + z_xi*z_zt;
        cos_theta = dot_val/(len_xi*len_zt);
        theta = acosf(cos_theta);
        dif_theta = (theta0-theta)/theta0;
        R_x2_loc[iptr] += a*tanhf(dif_theta);
        iptr1 = k*ny + j;
        dif_dis = (dx2[iptr1]-len_xi)/dx2[iptr1];
        P_x2_loc[iptr] -= a*tanhf(dif_dis);
      }
    }
  }
  /* bdry y1: et=0 */
  if (neighid[2] == -2) {
    for (int k = 1; k < nz-1; k++) {
      for (int i = 1; i < nx-1; i++) {
        iptr  = k*siz_iz + 0*siz_iy + i;
        iptr1 = k*siz_iz + 0*siz_iy + i+1;
        iptr2 = k*siz_iz + 0*siz_iy + i-1;
        iptr3 = (k+1)*siz_iz + 0*siz_iy + i;
        iptr4 = (k-1)*siz_iz + 0*siz_iy + i;
        x_xi = 0.5f*(x3d[iptr1]-x3d[iptr2]);
        y_xi = 0.5f*(y3d[iptr1]-y3d[iptr2]);
        z_xi = 0.5f*(z3d[iptr1]-z3d[iptr2]);
        x_zt = 0.5f*(x3d[iptr3]-x3d[iptr4]);
        y_zt = 0.5f*(y3d[iptr3]-y3d[iptr4]);
        z_zt = 0.5f*(z3d[iptr3]-z3d[iptr4]);
        iptr1 = k*siz_iz + 1*siz_iy + i;
        x_et = x3d[iptr1] - x3d[iptr];
        y_et = y3d[iptr1] - y3d[iptr];
        z_et = z3d[iptr1] - z3d[iptr];
        gni = gni1 + (i-1);
        gnk = gnk1 + (k-1);
        iptr = gnk*(total_nx-2) + gni;
        len_xi = sqrtf(x_xi*x_xi + y_xi*y_xi + z_xi*z_xi);
        len_et = sqrtf(x_et*x_et + y_et*y_et + z_et*z_et);
        len_zt = sqrtf(x_zt*x_zt + y_zt*y_zt + z_zt*z_zt);
        dot_val = x_xi*x_et + y_xi*y_et + z_xi*z_et;
        cos_theta = dot_val/(len_xi*len_et);
        theta = acosf(cos_theta);
        dif_theta = (theta0-theta)/theta0;
        P_y1_loc[iptr] -= a*tanhf(dif_theta);
        dot_val = x_et*x_zt + y_et*y_zt + z_et*z_zt;
        cos_theta = dot_val/(len_et*len_zt);
        theta = acosf(cos_theta);
        dif_theta = (theta0-theta)/theta0;
        R_y1_loc[iptr] -= a*tanhf(dif_theta);
        iptr1 = k*nx + i;
        dif_dis = (dy1[iptr1]-len_et)/dy1[iptr1];
        Q_y1_loc[iptr] += a*tanhf(dif_dis);
      }
    }
  }
  /* bdry y2: et=1 */
  if (neighid[3] == -2) {
    for (int k = 1; k < nz-1; k++) {
      for (int i = 1; i < nx-1; i++) {
        iptr  = k*siz_iz + (ny-1)*siz_iy + i;
        iptr1 = k*siz_iz + (ny-1)*siz_iy + i+1;
        iptr2 = k*siz_iz + (ny-1)*siz_iy + i-1;
        iptr3 = (k+1)*siz_iz + (ny-1)*siz_iy + i;
        iptr4 = (k-1)*siz_iz + (ny-1)*siz_iy + i;
        x_xi = 0.5f*(x3d[iptr1]-x3d[iptr2]);
        y_xi = 0.5f*(y3d[iptr1]-y3d[iptr2]);
        z_xi = 0.5f*(z3d[iptr1]-z3d[iptr2]);
        x_zt = 0.5f*(x3d[iptr3]-x3d[iptr4]);
        y_zt = 0.5f*(y3d[iptr3]-y3d[iptr4]);
        z_zt = 0.5f*(z3d[iptr3]-z3d[iptr4]);
        iptr1 = k*siz_iz + (ny-2)*siz_iy + i;
        x_et = x3d[iptr] - x3d[iptr1];
        y_et = y3d[iptr] - y3d[iptr1];
        z_et = z3d[iptr] - z3d[iptr1];
        gni = gni1 + (i-1);
        gnk = gnk1 + (k-1);
        iptr = gnk*(total_nx-2) + gni;
        len_xi = sqrtf(x_xi*x_xi + y_xi*y_xi + z_xi*z_xi);
        len_et = sqrtf(x_et*x_et + y_et*y_et + z_et*z_et);
        len_zt = sqrtf(x_zt*x_zt + y_zt*y_zt + z_zt*z_zt);
        dot_val = x_xi*x_et + y_xi*y_et + z_xi*z_et;
        cos_theta = dot_val/(len_xi*len_et);
        theta = acosf(cos_theta);
        dif_theta = (theta0-theta)/theta0;
        P_y2_loc[iptr] += a*tanhf(dif_theta);
        dot_val = x_et*x_zt + y_et*y_zt + z_et*z_zt;
        cos_theta = dot_val/(len_et*len_zt);
        theta = acosf(cos_theta);
        dif_theta = (theta0-theta)/theta0;
        R_y2_loc[iptr] += a*tanhf(dif_theta);
        iptr1 = k*nx + i;
        dif_dis = (dy2[iptr1]-len_et)/dy2[iptr1];
        Q_y2_loc[iptr] -= a*tanhf(dif_dis);
      }
    }
  }
  /* bdry z1: zt=0 */
  if (neighid[4] == -2) {
    for (int j = 1; j < ny-1; j++) {
      for (int i = 1; i < nx-1; i++) {
        iptr  = 0*siz_iz + j*siz_iy + i;
        iptr1 = 0*siz_iz + j*siz_iy + i+1;
        iptr2 = 0*siz_iz + j*siz_iy + i-1;
        iptr3 = 0*siz_iz + (j+1)*siz_iy + i;
        iptr4 = 0*siz_iz + (j-1)*siz_iy + i;
        x_xi = 0.5f*(x3d[iptr1]-x3d[iptr2]);
        y_xi = 0.5f*(y3d[iptr1]-y3d[iptr2]);
        z_xi = 0.5f*(z3d[iptr1]-z3d[iptr2]);
        x_et = 0.5f*(x3d[iptr3]-x3d[iptr4]);
        y_et = 0.5f*(y3d[iptr3]-y3d[iptr4]);
        z_et = 0.5f*(z3d[iptr3]-z3d[iptr4]);
        iptr1 = 1*siz_iz + j*siz_iy + i;
        x_zt = x3d[iptr1] - x3d[iptr];
        y_zt = y3d[iptr1] - y3d[iptr];
        z_zt = z3d[iptr1] - z3d[iptr];
        gni = gni1 + (i-1);
        gnj = gnj1 + (j-1);
        iptr = gnj*(total_nx-2) + gni;
        len_xi = sqrtf(x_xi*x_xi + y_xi*y_xi + z_xi*z_xi);
        len_et = sqrtf(x_et*x_et + y_et*y_et + z_et*z_et);
        len_zt = sqrtf(x_zt*x_zt + y_zt*y_zt + z_zt*z_zt);
        dot_val = x_xi*x_zt + y_xi*y_zt + z_xi*z_zt;
        cos_theta = dot_val/(len_xi*len_zt);
        theta = acosf(cos_theta);
        dif_theta = (theta0-theta)/theta0;
        P_z1_loc[iptr] -= a*tanhf(dif_theta);
        dot_val = x_et*x_zt + y_et*y_zt + z_et*z_zt;
        cos_theta = dot_val/(len_et*len_zt);
        theta = acosf(cos_theta);
        dif_theta = (theta0-theta)/theta0;
        Q_z1_loc[iptr] -= a*tanhf(dif_theta);
        iptr1 = j*nx + i;
        dif_dis = (dz1[iptr1]-len_zt)/dz1[iptr1];
        R_z1_loc[iptr] += a*tanhf(dif_dis);
      }
    }
  }
  /* bdry z2: zt=1 */
  if (neighid[5] == -2) {
    for (int j = 1; j < ny-1; j++) {
      for (int i = 1; i < nx-1; i++) {
        iptr  = (nz-1)*siz_iz + j*siz_iy + i;
        iptr1 = (nz-1)*siz_iz + j*siz_iy + i+1;
        iptr2 = (nz-1)*siz_iz + j*siz_iy + i-1;
        iptr3 = (nz-1)*siz_iz + (j+1)*siz_iy + i;
        iptr4 = (nz-1)*siz_iz + (j-1)*siz_iy + i;
        x_xi = 0.5f*(x3d[iptr1]-x3d[iptr2]);
        y_xi = 0.5f*(y3d[iptr1]-y3d[iptr2]);
        z_xi = 0.5f*(z3d[iptr1]-z3d[iptr2]);
        x_et = 0.5f*(x3d[iptr3]-x3d[iptr4]);
        y_et = 0.5f*(y3d[iptr3]-y3d[iptr4]);
        z_et = 0.5f*(z3d[iptr3]-z3d[iptr4]);
        iptr1 = (nz-2)*siz_iz + j*siz_iy + i;
        x_zt = x3d[iptr] - x3d[iptr1];
        y_zt = y3d[iptr] - y3d[iptr1];
        z_zt = z3d[iptr] - z3d[iptr1];
        gni = gni1 + (i-1);
        gnj = gnj1 + (j-1);
        iptr = gnj*(total_nx-2) + gni;
        len_xi = sqrtf(x_xi*x_xi + y_xi*y_xi + z_xi*z_xi);
        len_et = sqrtf(x_et*x_et + y_et*y_et + z_et*z_et);
        len_zt = sqrtf(x_zt*x_zt + y_zt*y_zt + z_zt*z_zt);
        dot_val = x_xi*x_zt + y_xi*y_zt + z_xi*z_zt;
        cos_theta = dot_val/(len_xi*len_zt);
        theta = acosf(cos_theta);
        dif_theta = (theta0-theta)/theta0;
        P_z2_loc[iptr] += a*tanhf(dif_theta);
        dot_val = x_et*x_zt + y_et*y_zt + z_et*z_zt;
        cos_theta = dot_val/(len_et*len_zt);
        theta = acosf(cos_theta);
        dif_theta = (theta0-theta)/theta0;
        Q_z2_loc[iptr] += a*tanhf(dif_theta);
        iptr1 = j*nx + i;
        dif_dis = (dz2[iptr1]-len_zt)/dz2[iptr1];
        R_z2_loc[iptr] -= a*tanhf(dif_dis);
      }
    }
  }

  /* Copy local to global (in serial, these are the same) */
  memcpy(P_x1, P_x1_loc, siz_bx * sizeof(float));
  memcpy(Q_x1, Q_x1_loc, siz_bx * sizeof(float));
  memcpy(R_x1, R_x1_loc, siz_bx * sizeof(float));
  memcpy(P_x2, P_x2_loc, siz_bx * sizeof(float));
  memcpy(Q_x2, Q_x2_loc, siz_bx * sizeof(float));
  memcpy(R_x2, R_x2_loc, siz_bx * sizeof(float));
  memcpy(P_y1, P_y1_loc, siz_by * sizeof(float));
  memcpy(Q_y1, Q_y1_loc, siz_by * sizeof(float));
  memcpy(R_y1, R_y1_loc, siz_by * sizeof(float));
  memcpy(P_y2, P_y2_loc, siz_by * sizeof(float));
  memcpy(Q_y2, Q_y2_loc, siz_by * sizeof(float));
  memcpy(R_y2, R_y2_loc, siz_by * sizeof(float));
  memcpy(P_z1, P_z1_loc, siz_bz * sizeof(float));
  memcpy(Q_z1, Q_z1_loc, siz_bz * sizeof(float));
  memcpy(R_z1, R_z1_loc, siz_bz * sizeof(float));
  memcpy(P_z2, P_z2_loc, siz_bz * sizeof(float));
  memcpy(Q_z2, Q_z2_loc, siz_bz * sizeof(float));
  memcpy(R_z2, R_z2_loc, siz_bz * sizeof(float));
}
