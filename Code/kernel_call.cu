#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <math.h>

#include "kernel.h"

cublasHandle_t handle;

void initCuBLAS()
{
    // First, create a cuBLAS handle:
    cublasStatus_t cublasStat = cublasCreate_v2(&handle);

    // Set the math mode to allow cuBLAS to use Tensor Cores:
    cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
}

void destroyCuBLAS()
{
    cublasDestroy(handle);
}

void matrixMultiplication(double *A, double *B, double *C, int dimX, int dimY)
{
    /*dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((dimX + threadsPerBlock.x - 1) / threadsPerBlock.x, (dimY + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    Mat2DMult<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, dimX, dimY);
    cudaDeviceSynchronize();
    
    return;*/

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, dimX, dimY, dimY, &alpha,
                 A, CUDA_R_64F, dimX,
                 B, CUDA_R_64F, dimX,
                 &beta, C, CUDA_R_64F, dimX,
                 CUDA_R_64F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void matrixAddition(double *A, double *B, double *C, int dimX, int dimY)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((dimX + threadsPerBlock.x - 1) / threadsPerBlock.x, (dimY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    Mat2DAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, dimX, dimY);

    cudaDeviceSynchronize();
}

void matrix_double_Addition(double *A, double num, double *C, int dimX, int dimY)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((dimX + threadsPerBlock.x - 1) / threadsPerBlock.x, (dimY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    Mat2D_double_Add<<<blocksPerGrid, threadsPerBlock>>>(A, num, C, dimX, dimY);

    cudaDeviceSynchronize();
}