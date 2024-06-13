#include <cuda.h>
#include <cuda_runtime.h>

#include <math.h>

#include "kernel.h"

void matrixMultiplication(float *A, float *B, float *C, int N)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    Mat2DMult<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);

    cudaDeviceSynchronize();
}

void matrixAddition(float *A, float *B, float *C, int N)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    Mat2DAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);

    cudaDeviceSynchronize();
}

void matrix_float_Addition(float *A, float num, float *C, int N)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    Mat2D_float_Add<<<blocksPerGrid, threadsPerBlock>>>(A, num, C, N);

    cudaDeviceSynchronize();
}