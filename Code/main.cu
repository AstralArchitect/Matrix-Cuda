#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

#include "matrix.h"

int main()
{
    Matrix myMatrix(4);

    printf("\033[2J\033[HStart Matrix:\n");

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            myMatrix.set(i, j, 10.0f);
        }
    }
    
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            printf("x %d, y %d: %.2f\n", i, j, myMatrix.get(i, j));
        }
    }

    Sleep(10000);

    myMatrix = myMatrix * myMatrix;

    printf("\033[2J\033[HMatrice multiplie par elle meme\n");

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            printf("x %d, y %d: %.2f\n", i, j, myMatrix.get(i, j));
        }
    }

    Sleep(10000);

    myMatrix = myMatrix + myMatrix;

    printf("\033[2J\033[HMatrice additionne par elle meme\n");

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            printf("x %d, y %d: %.2f\n", i, j, myMatrix.get(i, j));
        }
    }

    Sleep(10000);

    myMatrix = myMatrix + 1.0f;

    printf("\033[2J\033[HMatrice + 1.0\n");

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            printf("x %d, y %d: %.2f\n", i, j, myMatrix.get(i, j));
        }
    }

    myMatrix.set(3, 2, 10.0f);

    Sleep(10000);

    printf("\033[2J\033[HMatrice element x3, y2 = 10.0\n");

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            printf("x %d, y %d: %.2f\n", i, j, myMatrix.get(i, j));
        }
    }

    return 0;
}
