#include <stdio.h>
#include <stdlib.h>

int main()
{
    const int N = 1 << 20;
    size_t size = N * sizeof(int);

    int *h_A = (int *)malloc(size);
    int *h_B = (int *)malloc(size);
    int *h_C = (int *)malloc(size);

    for (int i = 0; i < N; ++i)
    {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    int *d_A, *d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Add on the host instead
    for (int i = 0; i < N; ++i)
    {
        h_C[i] = h_A[i] + h_B[i];
    }

    printf("Sample results:\n");
    for (int i = 0; i < 10; ++i)
    {
        printf("A[%d] + B[%d] = %d + %d = %d\n", i, i, h_A[i], h_B[i], h_C[i]);
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}