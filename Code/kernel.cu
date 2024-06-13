#include <cuda.h>
#include <cuda_runtime.h>

__global__ void Mat2DMult(float* A, float* B, float* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && column < N)
    {
        float tmpSum = 0.0f;
        for (int i = 0; i < N; i++)
        {
            tmpSum += A[row * N + i] * B[i * N + column];
        }
        C[row * N + column] = tmpSum;
    }
}

__global__ void Mat2DAdd(float *A, float *B, float *C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;

    int pos = row * N + column;

    C[pos] = A[pos] + B[pos];
}

__global__ void Mat2D_float_Add(float *A, float num, float *C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;

    int pos = row * N + column;

    C[pos] = A[pos] + num;
}