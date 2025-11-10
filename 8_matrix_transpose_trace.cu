#include <iostream>
#include <stdio.h>

#define N 4

__global__ void transpose(float *out, const float *in, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n)
        out[x * n + y] = in[y * n + x];  // transpose write
}

int main() {
    const int SIZE = N * N;
    const int BYTES = SIZE * sizeof(float);

    float h_in[SIZE], h_out[SIZE];

    std::cout << "Input Matrix:\n";
    for (int i = 0; i < SIZE; ++i) {
        h_in[i] = i + 1;
        std::cout << h_in[i] << "\t";
        if ((i + 1) % N == 0) std::cout << "\n";
    }

    float *d_in, *d_out;
    cudaMalloc((void **)&d_in, BYTES);
    cudaMalloc((void **)&d_out, BYTES);

    cudaMemcpy(d_in, h_in, BYTES, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    transpose<<<gridDim, blockDim>>>(d_out, d_in, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, BYTES, cudaMemcpyDeviceToHost);

    std::cout << "\nTransposed Matrix:\n";
    for (int i = 0; i < SIZE; ++i) {
        std::cout << h_out[i] << "\t";
        if ((i + 1) % N == 0) std::cout << "\n";
    }

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
