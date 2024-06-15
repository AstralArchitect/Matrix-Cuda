#include <cuda.h>
#include <cuda_runtime.h>

__global__ void Mat2DMult(float* A, float* B, float* C, int dimX, int dimY)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < dimX && column < dimY)
    {
        float tmpSum = 0.0f;
        for (int i = 0; i < dimY; i++)
        {
            tmpSum += A[row * dimX + i] * B[i * dimY + column];
        }
        C[row * dimX + column] = tmpSum;
    }
}

__global__ void Mat2DAdd(float *A, float *B, float *C, int dimX, int dimY)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;

    int pos = row * dimX + column;

    C[pos] = A[pos] + B[pos];
}

__global__ void Mat2D_float_Add(float *A, float num, float *C, int dimX, int dimY)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;

    int pos = row * dimX + column;

    C[pos] = A[pos] + num;
}