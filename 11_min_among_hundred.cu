#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

int main()
{
    const int N = 100;
    int h_input[N];

    // Fill with random values
    srand(42);
    for (int i = 0; i < N; ++i)
    {
        h_input[i] = rand() % 1000 + 1;
    }

    // Allocate device memory just to simulate CUDA memory workflow
    int *d_input;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Perform reduction on host
    int min_val = INT_MAX;
    for (int i = 0; i < N; ++i)
    {
        if (h_input[i] < min_val)
            min_val = h_input[i];
    }

    printf("First 10 values:\n");
    for (int i = 0; i < 10; ++i)
        printf("%d ", h_input[i]);
    printf("\nMinimum value = %d\n", min_val);

    cudaFree(d_input);
    return 0;
}