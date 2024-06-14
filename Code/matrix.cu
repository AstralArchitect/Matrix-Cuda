#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "matrix.h"
#include "kernel_call.h"

Matrix::Matrix(unsigned int dimX, unsigned int dimY)
{
    cudaError_t result = cudaMalloc((void**)&matrix_d, dimX * dimY * sizeof(double));
    if (result != cudaSuccess)
    {
        printf("Error: failed to allocate memory on device (%s).\n", cudaGetErrorString(result));
        exit(1);
    }
    size.x = dimX;
    size.y = dimY;
}

Matrix::Matrix(const Matrix& other)
{
    size.x = other.size.x;
    size.y = other.size.y;
    cudaError_t result = cudaMalloc((void**)&matrix_d, size.x * size.y * sizeof(double));
    if (result != cudaSuccess)
    {
        printf("Error: failed to allocate memory on device (%s).\n", cudaGetErrorString(result));
        exit(1);
    }
    result = cudaMemcpy(matrix_d, other.matrix_d, size.x * size.y * sizeof(double), cudaMemcpyDeviceToDevice);
    if (result != cudaSuccess)
    {
        printf("Error: failed to copy device memory (%s).\n", cudaGetErrorString(result));
        exit(1);
    }
}

Matrix::~Matrix()
{
    cudaError_t result = cudaFree(matrix_d);
    if (result != cudaSuccess)
    {
        printf("Error: failed to free device memory (%s).\n", cudaGetErrorString(result));
    }
}

Matrix& Matrix::operator=(const Matrix& other)
{
    if (this != &other)
    {
        if (matrix_d)
        {
            cudaError_t result = cudaFree(matrix_d);
            if (result != cudaSuccess)
            {
                printf("Error: failed to free device memory (%s).\n", cudaGetErrorString(result));
            }
        }

        size = other.size;
        cudaError_t result = cudaMalloc((void**)&matrix_d, size.x * size.y * sizeof(double));
        if (result != cudaSuccess)
        {
            printf("Error: failed to allocate memory on device (%s).\n", cudaGetErrorString(result));
            exit(1);
        }
        result = cudaMemcpy(matrix_d, other.matrix_d, size.x * size.y * sizeof(double), cudaMemcpyDeviceToDevice);
        if (result != cudaSuccess)
        {
            printf("Error: failed to copy device memory (%s).\n", cudaGetErrorString(result));
            exit(1);
        }
    }
    return *this;
}

Matrix Matrix::operator*(Matrix const& matrix)
{
    if (matrix.size.x != size.x && matrix.size.y != size.y)
    {
        printf("Error: can't multiply two matrices of different sizes\n");
        exit(1);
    }
    
    Matrix res(size.x, size.y);
    matrixMultiplication(matrix_d, matrix.matrix_d, res.matrix_d, size.x, size.y);

    return res;
}

Matrix Matrix::operator+(Matrix const& matrix)
{
    if (matrix.size.x != size.x && matrix.size.y != size.y)
    {
        printf("Error: can't add two matrices of different sizes\n");
        exit(1);
    }
    
    Matrix res(size.x, size.y);
    matrixAddition(matrix_d, matrix.matrix_d, res.matrix_d, size.x, size.y);

    return res;
}

Matrix Matrix::operator+(double num)
{
    Matrix res(size.x, size.y);
    matrix_double_Addition(matrix_d, num, res.matrix_d, size.x, size.y);

    return res;
}

double Matrix::get(int x, int y)
{
    double element;
    cudaError_t result = cudaMemcpy(&element, matrix_d + size.x * y + x, sizeof(double), cudaMemcpyDeviceToHost);
    if (result != cudaSuccess)
    {
        printf("Error: failed to copy to host memory (%s)\n", cudaGetErrorString(result));
        exit(1);
    }
    
    return element;
}

void Matrix::set(int x, int y, double value)
{
    cudaError_t result = cudaMemcpy(matrix_d + size.x * y + x, &value, sizeof(double), cudaMemcpyHostToDevice);
    if (result != cudaSuccess)
    {
        printf("Error: failed to copy to device memory (%s)\n", cudaGetErrorString(result));
        exit(1);
    }
}

double *Matrix::getPtr()
{
    return matrix_d;
}