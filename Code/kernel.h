__global__ void Mat2DMult(float* A, float* B, float* C, int N);
__global__ void Mat2DAdd(float *A, float *B, float *C, int N);
__global__ void Mat2D_float_Add(float *A, float num, float *C, int N);