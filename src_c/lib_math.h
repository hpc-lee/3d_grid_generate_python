#ifndef LIB_MATH_3D_H
#define LIB_MATH_3D_H

/* 3x3 matrix and 3-vector operations for 3D grid generation. */

int mat_invert3x3(float m[][3]);
int mat_mul3x3(float A[][3], float B[][3], float C[][3]);
int cross_product(float *A, float *B, float *C);
float dot_product(float *A, float *B);
int mat_mul3x1(float A[][3], float *B, float *C);
int mat_add3x3(float A[][3], float B[][3], float C[][3]);
int vec_add3x1(float *A, float *B, float *C);
int vec_sub3x1(float *A, float *B, float *C);
int mat_sub3x3(float A[][3], float B[][3], float C[][3]);
int mat_copy3x3(float A[][3], float B[][3]);
int mat_iden3x3(float A[][3]);
float dist_point2plane(float x0[3], float x1[3], float x2[3], float x3[3]);

#endif /* LIB_MATH_3D_H */
