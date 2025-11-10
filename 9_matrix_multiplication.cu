#include <stdio.h>
#include <stdlib.h>

int main()
{
    const int N = 100; // We'll use 100×100 matrices

    // Allocate host memory
    int *h_A = (int *)malloc(N * N * sizeof(int));
    int *h_B = (int *)malloc(N * N * sizeof(int));
    int *h_C = (int *)malloc(N * N * sizeof(int));

    // Initialize A and B with some test values
    for (int i = 0; i < N * N; ++i)
    {
        h_A[i] = i % 100;
        h_B[i] = (i % 100) + 1;
    }

    // Matrix multiplication on host: C = A × B
    for (int row = 0; row < N; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            int sum = 0;
            for (int k = 0; k < N; ++k)
            {
                sum += h_A[row * N + k] * h_B[k * N + col];
            }
            h_C[row * N + col] = sum;
        }
    }

    // Print top-left 5×5 region of result matrix C
    printf("Matrix C = A × B (top-left 5x5):\n");
    for (int i = 0; i < 5; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            printf("%6d ", h_C[i * N + j]);
        }
        printf("\n");
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
