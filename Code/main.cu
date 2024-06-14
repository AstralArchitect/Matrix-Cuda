#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
    #include <windows.h>
#endif

#include <time.h>

#include "matrix.h"

#include <cuda.h>
#include <cuda_runtime.h>

void initCuBLAS();
void destroyCuBLAS();

const unsigned int dimX = 15000;
const unsigned int dimY = dimX;

int main()
{
    #ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    #endif
    initCuBLAS();

    double *matrix_h = (double*)malloc(dimX * dimY * sizeof(double));
    if (matrix_h == NULL)
    {
        printf("Erreur d'allocation dynamique de mémoire\n");
    }
    
    double *matrix_d = NULL;

    for (int i = 0; i < dimX * dimY; i++)
    {
        matrix_h[i] = 5.0f;
    }

    Matrix myMatrix(dimX, dimY);

    matrix_d = myMatrix.getPtr();

    cudaMemcpy(matrix_d, matrix_h, (unsigned long long)dimX * (unsigned long long)dimY * sizeof(double), cudaMemcpyHostToDevice);

    free(matrix_h);

    printf("Multiplication...\n");

    time_t start, stop;

    start = time(NULL);

    myMatrix = myMatrix * myMatrix;

    stop = time(NULL);

    printf("terminé en %lld secondes\n", stop - start);

    destroyCuBLAS();

    return 0;
}
