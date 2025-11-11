// Write a CUDA program to demonstrate adding 2 large arrays using CUDA kernel.

#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define N 100000000 // 10^8, to demonstrate large array processing

__global__ void add(int *a, int *b, int *c, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
        c[index] = a[index] + b[index];
}

int main()
{
    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;

    int size = N * sizeof(int);

    h_a = new int[N];
    h_b = new int[N];
    h_c = new int[N];

    for (int i = 0; i < N; i++)
    {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i)
        cout << h_c[i] << " "; // Print first 10 results
    cout << endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}