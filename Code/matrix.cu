#include <stdio.h>

#include "matrix.h"
#include "kernel_call.h"

void matrixMultiplication(float *A, float *B, float *C, int N);
void matrixAddition(float *A, float *B, float *C, int N);
void matrix_float_Addition(float *A, float num, float *C, int N);

Matrix::Matrix(int N)
{
    cudaError_t result = cudaMalloc(&matrix_d, N * N * sizeof(float));
    if (result != cudaSuccess)
    {
        printf("Error: failed to allocate memory on device (%s).\n", cudaGetErrorString(result));
        exit(1);
    }
    size = N;
}

Matrix::Matrix(const Matrix& other)
{
    size = other.size;
    cudaError_t result = cudaMalloc(&matrix_d, size * size * sizeof(float));
    if (result != cudaSuccess)
    {
        printf("Error: failed to allocate memory on device (%s).\n", cudaGetErrorString(result));
        exit(1);
    }
    result = cudaMemcpy(matrix_d, other.matrix_d, size * size * sizeof(float), cudaMemcpyDeviceToDevice);
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
        cudaError_t result = cudaMalloc(&matrix_d, size * size * sizeof(float));
        if (result != cudaSuccess)
        {
            printf("Error: failed to allocate memory on device (%s).\n", cudaGetErrorString(result));
            exit(1);
        }
        result = cudaMemcpy(matrix_d, other.matrix_d, size * size * sizeof(float), cudaMemcpyDeviceToDevice);
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
    if (matrix.size != size)
    {
        printf("Error: can't multiply two matrices of different sizes\n");
        exit(1);
    }
    
    Matrix res(size);
    matrixMultiplication(matrix_d, matrix.matrix_d, res.matrix_d, size);

    return res;
}

Matrix Matrix::operator+(Matrix const& matrix)
{
    if (matrix.size != size)
    {
        printf("Error: can't add two matrices of different sizes\n");
        exit(1);
    }
    
    Matrix res(size);
    matrixAddition(matrix_d, matrix.matrix_d, res.matrix_d, size);

    return res;
}

Matrix Matrix::operator+(float num)
{
    Matrix res(size);
    matrix_float_Addition(matrix_d, num, res.matrix_d, size);

    return res;
}

float Matrix::get(int x, int y)
{
    float element;
    cudaError_t result = cudaMemcpy(&element, matrix_d + size * y + x, sizeof(float), cudaMemcpyDeviceToHost);
    if (result != cudaSuccess)
    {
        printf("Error: failed to copy to host memory (%s)\n", cudaGetErrorString(result));
        exit(1);
    }
    
    return element;
}

void Matrix::set(int x, int y, float value)
{
    cudaError_t result = cudaMemcpy(matrix_d + size * y + x, &value, sizeof(float), cudaMemcpyHostToDevice);
    if (result != cudaSuccess)
    {
        printf("Error: failed to copy to device memory (%s)\n", cudaGetErrorString(result));
        exit(1);
    }
}