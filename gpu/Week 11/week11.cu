#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define N 100
#define THREADS 128

__global__ void findMinKernel(int *input, int *result, int n)
{
    __shared__ int sharedData[THREADS];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (idx < n)
        sharedData[tid] = input[idx];
    else
        sharedData[tid] = INT_MAX; // pad with large number
    __syncthreads();

    // Parallel reduction for minimum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            sharedData[tid] = min(sharedData[tid], sharedData[tid + stride]);
        __syncthreads();
    }

    // Thread 0 writes block result
    if (tid == 0)
        result[blockIdx.x] = sharedData[0];
}

int main()
{
    int h_input[N];
    for (int i = 0; i < N; ++i)
        h_input[i] = rand() % 1000 + 1; // random numbers between 1–1000

    cout << "Input Values:\n";
    for (int i = 0; i < N; ++i)
    {
        cout << h_input[i] << " ";
        if ((i + 1) % 20 == 0)
            cout << endl;
    }

    int *d_input, *d_result;
    cudaMalloc(&d_input, N * sizeof(int));

    // Max possible blocks
    int blocks = (N + THREADS - 1) / THREADS;
    cudaMalloc(&d_result, blocks * sizeof(int));

    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // First kernel reduces within blocks
    findMinKernel<<<blocks, THREADS>>>(d_input, d_result, N);
    cudaDeviceSynchronize();

    // If multiple blocks, reduce again on CPU (since 100 is small)
    int *h_partial = new int[blocks];
    cudaMemcpy(h_partial, d_result, blocks * sizeof(int), cudaMemcpyDeviceToHost);

    int minVal = h_partial[0];
    for (int i = 1; i < blocks; ++i)
        minVal = min(minVal, h_partial[i]);

    cout << "\n\nMinimum value found = " << minVal << endl;

    cudaFree(d_input);
    cudaFree(d_result);

    return 0;
}
