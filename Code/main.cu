#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
    #include <windows.h>
#endif

#include <time.h>
#include <random>

#include "matrix.h"

#include <cuda.h>
#include <cuda_runtime.h>

void initCuBLAS();
void destroyCuBLAS();

const unsigned int dimX = 4;
const unsigned int dimY = dimX;

int main()
{
    #ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    #endif
    initCuBLAS();

    Matrix matrixA(dimX, dimY, 10.0f);

    std::srand(time(NULL));

    printf("First Matrix:\n");
    for (int i = 0; i < dimX; i++)
    {
        for (int j = 0; j < dimY; j++)
        {
            printf("%f\n", matrixA.get(i, j));
        }
    }

    Matrix matrixB(4, 4, 5.0f);

    matrixA *= matrixA;

    matrixB += matrixA;

    printf("Second Matrix A:\n");
    for (int i = 0; i < dimX; i++)
    {
        for (int j = 0; j < dimY; j++)
        {
            printf("%f\n", matrixA.get(i, j));
        }
    }

    printf("Second Matrix B:\n");
    for (int i = 0; i < dimX; i++)
    {
        for (int j = 0; j < dimY; j++)
        {
            printf("%f\n", matrixB.get(i, j));
        }
    }

    Matrix matrixC(4, 4, 0.0f);

    matrixC = matrixA * matrixB;

    printf("First Matrix C:\n");
    for (int i = 0; i < dimX; i++)
    {
        for (int j = 0; j < dimY; j++)
        {
            printf("%f\n", matrixC.get(i, j));
        }
    }

    destroyCuBLAS();

    return 0;
}
