__global__ void Mat2DMult(double* A, double* B, double* C, int dimX, int dimY);
__global__ void Mat2DAdd(double *A, double *B, double *C, int dimX, int dimY);
__global__ void Mat2D_double_Add(double *A, double num, double *C, int dimX, int dimY);