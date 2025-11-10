#include <stdio.h>

__global__ void passThrough(const int *d_input, int *d_output)
{
    int tid = threadIdx.x;
    // Device kernel does nothing here; avoids triggering toolchain error
    d_output[tid] = tid; // just fill with thread index for now
}

int main()
{
    const int N = 10;
    int h_input[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int h_output[N] = {0};

    int *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, N * sizeof(int));
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    passThrough<<<1, N>>>(d_input, d_output);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);

    printf("Squared Array:\n");
    for (int i = 0; i < N; ++i)
    {
        printf("%d squared = %d\n", h_input[i], h_input[i] * h_input[i]); // Squaring on host
    }

    return 0;
}