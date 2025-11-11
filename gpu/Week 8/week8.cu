// Write a CUDA program to find the transpose and trace of a matrix

#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define N 4

__global__ void transpose(int *input, int *output, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
        output[col * n + row] = input[row * n + col];
}

__global__ void trace(int *matrix, int *result, int n)
{
    int sum = 0;
    for (int i = 0; i < n; ++i)
        sum += matrix[i * n + i];
    *result = sum;
}

int main()
{
    int *h_matrix, *h_transposed, *h_trace;

    h_matrix = new int[N * N];
    h_transposed = new int[N * N];
    h_trace = new int;

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            h_matrix[i * N + j] = i * N + j + 1; // Initialize matrix with some values

    int *d_matrix, *d_transposed, *d_trace;
    cudaMalloc(&d_matrix, N * N * sizeof(int));
    cudaMalloc(&d_transposed, N * N * sizeof(int));
    cudaMalloc(&d_trace, sizeof(int));
    cudaMemcpy(d_matrix, h_matrix, N * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    transpose<<<numBlocks, threadsPerBlock>>>(d_matrix, d_transposed, N);
    trace<<<1, 1>>>(d_matrix, d_trace, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_transposed, d_transposed, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_trace, d_trace, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "Transposed Matrix:" << endl;
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
            cout << h_transposed[i * N + j] << " ";
        cout << endl;
    }

    cout << "Trace of the Matrix: " << *h_trace << endl;
    cudaFree(d_matrix);
    cudaFree(d_transposed);
    cudaFree(d_trace);
    delete[] h_matrix;
    delete[] h_transposed;
    delete h_trace;

    return 0;
}