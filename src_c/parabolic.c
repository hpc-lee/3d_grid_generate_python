#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "parabolic.h"

/*
 * Predict points at layers k and k+1 from known layer k-1.
 * Algorithm:
 *   1. Compute unit normal vector at each (i,j) on layer k-1
 *   2. Blend normal-projection and linear-interpolation points
 *      using clustering factor cs = exp(-coef * zt)
 *   3. Apply geometric-symmetry boundary conditions on 4 edges
 */
void
predict_point_c(float *x3d, float *y3d, float *z3d,
                int nx, int ny, int nz,
                int k, int t2b, float coef,
                float *step_len)
{
  size_t siz_iy = nx;
  size_t siz_iz = nx * ny;
  float zt = (1.0f * k) / (nz - 1);
  float cs = expf(-coef * zt);

  for (int j = 1; j < ny - 1; j++) {
    for (int i = 1; i < nx - 1; i++)
    {
      /* central differences on layer k-1 */
      size_t ip1 = (k-1)*siz_iz + j*siz_iy + (i+1);
      size_t ip2 = (k-1)*siz_iz + j*siz_iy + (i-1);
      float x_xi = 0.5f * (x3d[ip1] - x3d[ip2]);
      float y_xi = 0.5f * (y3d[ip1] - y3d[ip2]);
      float z_xi = 0.5f * (z3d[ip1] - z3d[ip2]);

      size_t ip3 = (k-1)*siz_iz + (j+1)*siz_iy + i;
      size_t ip4 = (k-1)*siz_iz + (j-1)*siz_iy + i;
      float x_et = 0.5f * (x3d[ip3] - x3d[ip4]);
      float y_et = 0.5f * (y3d[ip3] - y3d[ip4]);
      float z_et = 0.5f * (z3d[ip3] - z3d[ip4]);

      /* normal vector (cross product of xi and et tangent vectors) */
      float vn_x = y_xi * z_et - z_xi * y_et;
      float vn_y = z_xi * x_et - x_xi * z_et;
      float vn_z = x_xi * y_et - y_xi * x_et;
      if (t2b == 1) { vn_x = -vn_x; vn_y = -vn_y; vn_z = -vn_z; }
      float vn_len = sqrtf(vn_x*vn_x + vn_y*vn_y + vn_z*vn_z);
      vn_x /= vn_len;  vn_y /= vn_len;  vn_z /= vn_len;

      /* vector from (i,j,k-1) to (i,j,nz-1) */
      size_t ip_src = (k-1)*siz_iz + j*siz_iy + i;
      size_t ip_tgt = (nz-1)*siz_iz + j*siz_iy + i;
      float R_x = x3d[ip_tgt] - x3d[ip_src];
      float R_y = y3d[ip_tgt] - y3d[ip_src];
      float R_z = z3d[ip_tgt] - z3d[ip_src];
      float R = sqrtf(R_x*R_x + R_y*R_y + R_z*R_z);

      /* clustering factors */
      float R1 = step_len[nz-1] - step_len[k-1];
      float r1 = step_len[k+1] - step_len[k-1];
      float r2 = step_len[k]   - step_len[k-1];
      float c1 = r1 / R1;
      float c2 = r2 / r1;

      /* normal-projection point */
      float x0 = x3d[ip_src] + vn_x * c1 * R;
      float y0 = y3d[ip_src] + vn_y * c1 * R;
      float z0 = z3d[ip_src] + vn_z * c1 * R;

      /* linear-interpolation point */
      float xs = x3d[ip_src] + c1 * R_x;
      float ys = y3d[ip_src] + c1 * R_y;
      float zs = z3d[ip_src] + c1 * R_z;

      if (k < nz - 2)
      {
        /* predict layer k+1 */
        size_t ip_k1 = (k+1)*siz_iz + j*siz_iy + i;
        x3d[ip_k1] = cs * x0 + (1 - cs) * xs;
        y3d[ip_k1] = cs * y0 + (1 - cs) * ys;
        z3d[ip_k1] = cs * z0 + (1 - cs) * zs;

        /* predict layer k by interpolation between k-1 and k+1 */
        size_t ip_k = k*siz_iz + j*siz_iy + i;
        x3d[ip_k] = x3d[ip_src] + c2 * (x3d[ip_k1] - x3d[ip_src]);
        y3d[ip_k] = y3d[ip_src] + c2 * (y3d[ip_k1] - y3d[ip_src]);
        z3d[ip_k] = z3d[ip_src] + c2 * (z3d[ip_k1] - z3d[ip_src]);
      }
      if (k == nz - 2)
      {
        /* bottom boundary is fixed; only predict layer k */
        size_t ip_k1 = (k+1)*siz_iz + j*siz_iy + i;
        size_t ip_k  = k*siz_iz + j*siz_iy + i;
        x3d[ip_k] = x3d[ip_src] + c2 * (x3d[ip_k1] - x3d[ip_src]);
        y3d[ip_k] = y3d[ip_src] + c2 * (y3d[ip_k1] - y3d[ip_src]);
        z3d[ip_k] = z3d[ip_src] + c2 * (z3d[ip_k1] - z3d[ip_src]);
      }
    }
  }

  /* geometric-symmetry boundaries for layers k and k+1 */
  for (int j = 0; j < ny; j++) {
    /* i = 0: extrapolate from i=1 and i=2 */
    for (int kk = k; kk <= k+1; kk++) {
      size_t p0 = kk*siz_iz + j*siz_iy;
      size_t p1 = p0 + 1;
      size_t p2 = p0 + 2;
      x3d[p0] = 2*x3d[p1] - x3d[p2];
      y3d[p0] = 2*y3d[p1] - y3d[p2];
      z3d[p0] = 2*z3d[p1] - z3d[p2];
      /* i = nx-1 */
      size_t pn = kk*siz_iz + j*siz_iy + (nx-1);
      size_t pn1 = pn - 1;
      size_t pn2 = pn - 2;
      x3d[pn] = 2*x3d[pn1] - x3d[pn2];
      y3d[pn] = 2*y3d[pn1] - y3d[pn2];
      z3d[pn] = 2*z3d[pn1] - z3d[pn2];
    }
  }
  for (int i = 0; i < nx; i++) {
    for (int kk = k; kk <= k+1; kk++) {
      /* j = 0 */
      size_t p0 = kk*siz_iz + i;
      size_t p1 = kk*siz_iz + siz_iy + i;
      size_t p2 = kk*siz_iz + 2*siz_iy + i;
      x3d[p0] = 2*x3d[p1] - x3d[p2];
      y3d[p0] = 2*y3d[p1] - y3d[p2];
      z3d[p0] = 2*z3d[p1] - z3d[p2];
      /* j = ny-1 */
      size_t pn = kk*siz_iz + (ny-1)*siz_iy + i;
      size_t pn1 = kk*siz_iz + (ny-2)*siz_iy + i;
      size_t pn2 = kk*siz_iz + (ny-3)*siz_iy + i;
      x3d[pn] = 2*x3d[pn1] - x3d[pn2];
      y3d[pn] = 2*y3d[pn1] - y3d[pn2];
      z3d[pn] = 2*z3d[pn1] - z3d[pn2];
    }
  }
}

/*
 * Update (correct) points at layer k by solving the parabolic PDE:
 *   g22*g33 * f_{,xxi} + g11*g33 * f_{,eet} + g11*g22 * f_{,zzt}
 *   + 2*(beta12*f_{,xiet} + beta23*f_{,etzt} + beta13*f_{,xizt}) = 0
 * solved via Thomas algorithm along xi-direction for each j-line.
 */
void
update_point_c(float *x3d, float *y3d, float *z3d,
               int nx, int ny, int nz,
               int k)
{
  size_t siz_iy = nx;
  size_t siz_iz = nx * ny;
  int siz_vec = nx - 2;

  /* allocate work arrays: a, b, c, d_x, d_y, d_z, u_x, u_y, u_z (9 arrays) */
  float *var_th = (float *)calloc(siz_vec * 9, sizeof(float));
  float *coord  = (float *)calloc(nx * ny * 3, sizeof(float));
  float *a   = var_th;
  float *b   = var_th + 1*siz_vec;
  float *c   = var_th + 2*siz_vec;
  float *d_x = var_th + 3*siz_vec;
  float *d_y = var_th + 4*siz_vec;
  float *d_z = var_th + 5*siz_vec;
  float *u_x = var_th + 6*siz_vec;
  float *u_y = var_th + 7*siz_vec;
  float *u_z = var_th + 8*siz_vec;
  float *coord_x = coord;
  float *coord_y = coord + nx*ny;
  float *coord_z = coord + 2*nx*ny;

  for (int j = 1; j < ny - 1; j++) {
    for (int i = 1; i < nx - 1; i++)
    {
      /* first derivatives (central differences) */
      size_t ip1 = k*siz_iz + j*siz_iy + (i+1);
      size_t ip2 = k*siz_iz + j*siz_iy + (i-1);
      float x_xi = 0.5f*(x3d[ip1] - x3d[ip2]);
      float y_xi = 0.5f*(y3d[ip1] - y3d[ip2]);
      float z_xi = 0.5f*(z3d[ip1] - z3d[ip2]);

      size_t ip3 = k*siz_iz + (j+1)*siz_iy + i;
      size_t ip4 = k*siz_iz + (j-1)*siz_iy + i;
      float x_et = 0.5f*(x3d[ip3] - x3d[ip4]);
      float y_et = 0.5f*(y3d[ip3] - y3d[ip4]);
      float z_et = 0.5f*(z3d[ip3] - z3d[ip4]);

      size_t ip5 = (k+1)*siz_iz + j*siz_iy + i;
      size_t ip6 = (k-1)*siz_iz + j*siz_iy + i;
      float x_zt = 0.5f*(x3d[ip5] - x3d[ip6]);
      float y_zt = 0.5f*(y3d[ip5] - y3d[ip6]);
      float z_zt = 0.5f*(z3d[ip5] - z3d[ip6]);

      /* mixed derivatives */
      size_t im1jm = k*siz_iz + (j-1)*siz_iy + (i-1);
      size_t ip1jm = k*siz_iz + (j-1)*siz_iy + (i+1);
      size_t ip1jp = k*siz_iz + (j+1)*siz_iy + (i+1);
      size_t im1jp = k*siz_iz + (j+1)*siz_iy + (i-1);
      float x_xiet = 0.25f*(x3d[ip1jp] + x3d[im1jm] - x3d[ip1jm] - x3d[im1jp]);
      float y_xiet = 0.25f*(y3d[ip1jp] + y3d[im1jm] - y3d[ip1jm] - y3d[im1jp]);
      float z_xiet = 0.25f*(z3d[ip1jp] + z3d[im1jm] - z3d[ip1jm] - z3d[im1jp]);

      size_t im1km = (k-1)*siz_iz + j*siz_iy + (i-1);
      size_t ip1km = (k-1)*siz_iz + j*siz_iy + (i+1);
      size_t ip1kp = (k+1)*siz_iz + j*siz_iy + (i+1);
      size_t im1kp = (k+1)*siz_iz + j*siz_iy + (i-1);
      float x_xizt = 0.25f*(x3d[ip1kp] + x3d[im1km] - x3d[ip1km] - x3d[im1kp]);
      float y_xizt = 0.25f*(y3d[ip1kp] + y3d[im1km] - y3d[ip1km] - y3d[im1kp]);
      float z_xizt = 0.25f*(z3d[ip1kp] + z3d[im1km] - z3d[ip1km] - z3d[im1kp]);

      size_t jm1km = (k-1)*siz_iz + (j-1)*siz_iy + i;
      size_t jp1km = (k-1)*siz_iz + (j+1)*siz_iy + i;
      size_t jp1kp = (k+1)*siz_iz + (j+1)*siz_iy + i;
      size_t jm1kp = (k+1)*siz_iz + (j-1)*siz_iy + i;
      float x_etzt = 0.25f*(x3d[jp1kp] + x3d[jm1km] - x3d[jp1km] - x3d[jm1kp]);
      float y_etzt = 0.25f*(y3d[jp1kp] + y3d[jm1km] - y3d[jp1km] - y3d[jm1kp]);
      float z_etzt = 0.25f*(z3d[jp1kp] + z3d[jm1km] - z3d[jp1km] - z3d[jm1kp]);

      /* metric tensor */
      float g11 = x_xi*x_xi + y_xi*y_xi + z_xi*z_xi;
      float g22 = x_et*x_et + y_et*y_et + z_et*z_et;
      float g33 = x_zt*x_zt + y_zt*y_zt + z_zt*z_zt;
      float g12 = x_xi*x_et + y_xi*y_et + z_xi*z_et;
      float g13 = x_xi*x_zt + y_xi*y_zt + z_xi*z_zt;
      float g23 = x_et*x_zt + y_et*y_zt + z_et*z_zt;

      float alpha1 = g22*g33 - g23*g23;
      float alpha2 = g11*g33 - g13*g13;
      float alpha3 = g11*g22 - g12*g12;
      float beta12 = g13*g23 - g12*g33;
      float beta23 = g12*g13 - g11*g23;
      float beta13 = g12*g23 - g13*g22;

      /* tridiagonal system coefficients */
      a[i-1] = alpha1;
      b[i-1] = -2.0f*(alpha1 + alpha2 + alpha3);
      c[i-1] = alpha1;

      /* RHS: includes contributions from j±1 and k±1 neighbours + mixed terms */
      size_t jp = k*siz_iz + (j+1)*siz_iy + i;
      size_t jm = k*siz_iz + (j-1)*siz_iy + i;
      size_t kp = (k+1)*siz_iz + j*siz_iy + i;
      size_t km = (k-1)*siz_iz + j*siz_iy + i;

      d_x[i-1] = -alpha2*(x3d[jp]+x3d[jm]) - alpha3*(x3d[kp]+x3d[km])
                 - 2.0f*(beta12*x_xiet + beta23*x_etzt + beta13*x_xizt);
      d_y[i-1] = -alpha2*(y3d[jp]+y3d[jm]) - alpha3*(y3d[kp]+y3d[km])
                 - 2.0f*(beta12*y_xiet + beta23*y_etzt + beta13*y_xizt);
      d_z[i-1] = -alpha2*(z3d[jp]+z3d[jm]) - alpha3*(z3d[kp]+z3d[km])
                 - 2.0f*(beta12*z_xiet + beta23*z_etzt + beta13*z_xizt);
    }

    /* boundary modifications */
    size_t ip0 = k*siz_iz + j*siz_iy;
    d_x[0] -= a[0]*x3d[ip0];
    d_y[0] -= a[0]*y3d[ip0];
    d_z[0] -= a[0]*z3d[ip0];

    size_t ipn = k*siz_iz + j*siz_iy + (nx-1);
    d_x[nx-3] -= c[nx-3]*x3d[ipn];
    d_y[nx-3] -= c[nx-3]*y3d[ipn];
    d_z[nx-3] -= c[nx-3]*z3d[ipn];

    /* Thomas algorithm (forward elimination) */
    for (int i = 1; i < siz_vec; i++)
    {
      float factor = a[i] / b[i-1];
      b[i]   = b[i]   - factor * c[i-1];
      d_x[i] = d_x[i] - factor * d_x[i-1];
      d_y[i] = d_y[i] - factor * d_y[i-1];
      d_z[i] = d_z[i] - factor * d_z[i-1];
    }
    /* back substitution */
    u_x[siz_vec-1] = d_x[siz_vec-1] / b[siz_vec-1];
    u_y[siz_vec-1] = d_y[siz_vec-1] / b[siz_vec-1];
    u_z[siz_vec-1] = d_z[siz_vec-1] / b[siz_vec-1];
    for (int i = siz_vec-2; i >= 0; i--)
    {
      u_x[i] = (d_x[i] - c[i]*u_x[i+1]) / b[i];
      u_y[i] = (d_y[i] - c[i]*u_y[i+1]) / b[i];
      u_z[i] = (d_z[i] - c[i]*u_z[i+1]) / b[i];
    }

    /* store solved coordinates */
    for (int i = 1; i < nx-1; i++)
    {
      size_t idx = j*nx + i;
      coord_x[idx] = u_x[i-1];
      coord_y[idx] = u_y[i-1];
      coord_z[idx] = u_z[i-1];
    }
  }

  /* write back coordinates */
  for (int j = 1; j < ny-1; j++) {
    for (int i = 1; i < nx-1; i++)
    {
      size_t iptr  = k*siz_iz + j*siz_iy + i;
      size_t idx   = j*nx + i;
      x3d[iptr] = coord_x[idx];
      y3d[iptr] = coord_y[idx];
      z3d[iptr] = coord_z[idx];
    }
  }

  /* geometric-symmetry boundaries */
  for (int j = 0; j < ny; j++) {
    size_t p0 = k*siz_iz + j*siz_iy;
    x3d[p0] = 2*x3d[p0+1] - x3d[p0+2];
    y3d[p0] = 2*y3d[p0+1] - y3d[p0+2];
    z3d[p0] = 2*z3d[p0+1] - z3d[p0+2];

    size_t pn = k*siz_iz + j*siz_iy + (nx-1);
    x3d[pn] = 2*x3d[pn-1] - x3d[pn-2];
    y3d[pn] = 2*y3d[pn-1] - y3d[pn-2];
    z3d[pn] = 2*z3d[pn-1] - z3d[pn-2];
  }
  for (int i = 0; i < nx; i++) {
    size_t p0 = k*siz_iz + i;
    size_t p1 = k*siz_iz + siz_iy + i;
    size_t p2 = k*siz_iz + 2*siz_iy + i;
    x3d[p0] = 2*x3d[p1] - x3d[p2];
    y3d[p0] = 2*y3d[p1] - y3d[p2];
    z3d[p0] = 2*z3d[p1] - z3d[p2];

    size_t pn = k*siz_iz + (ny-1)*siz_iy + i;
    size_t pn1 = k*siz_iz + (ny-2)*siz_iy + i;
    size_t pn2 = k*siz_iz + (ny-3)*siz_iy + i;
    x3d[pn] = 2*x3d[pn1] - x3d[pn2];
    y3d[pn] = 2*y3d[pn1] - y3d[pn2];
    z3d[pn] = 2*z3d[pn1] - z3d[pn2];
  }

  free(var_th);
  free(coord);
}
