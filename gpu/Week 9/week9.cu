#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define N 4

__global__ void matrixMultiplyKernel(float *A, float *B, float *C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        float value = 0;
        for (int k = 0; k < N; ++k)
            value += A[row * N + k] * B[k * N + col];
        C[row * N + col] = value;
    }
}

int main()
{
    int size = N * N * sizeof(float);

    int *h_A = new int[N * N];
    int *h_B = new int[N * N];
    int *h_C = new int[N * N];

    // Initialize matrices
    for (int i = 0; i < N * N; i++)
        h_A[i] = i + 1;
    for (int i = 0; i < N * N; i++)
        h_B[i] = (i % 4) + 1;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);
    matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cout << "Result Matrix (C = A * B):\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            cout << h_C[i * N + j] << " ";
        cout << endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
