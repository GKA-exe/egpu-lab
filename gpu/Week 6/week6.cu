// Write a CUDA program to demonstrate squaring an array using CUDA kernel.

#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define N 100000000 // 10^8, to demonstrate large array processing

__global__ void square(int *input, int *output, int n)
{
    // Each thread computes one element, to find the index, we get the current block, then the number of threads per block and finally the thread index within the block
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) // Check if index is less than our total size
        output[index] = input[index] * input[index];
}

int main()
{
    int size = N * sizeof(int);
    int *host_array = new int[N], *host_result = new int[N]; // Creating large arrays on the CPU (RAM) for input and result
    for (int i = 0; i < N; ++i)
        host_array[i] = i + 1; // Assigning values of 1, 2, 3, ..., 10^8 in the array

    int *device_array, *device_result; // Creating the GPU arrays
    cudaMalloc(&device_array, size);   // Allocating memory on the GPU, same like malloc on the CPU
    cudaMalloc(&device_result, size);

    cudaMemcpy(device_array, host_array, size, cudaMemcpyHostToDevice); // Copying data from CPU (Host) to GPU (Device)

    int threads_per_block = 256;                                                    // How many elements each block will process
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;          // Calculating how many blocks are needed to process all elements (N + B - 1) / B is ceiling of N / B, we can use ceil(N / B) instead
    square<<<blocks_per_grid, threads_per_block>>>(device_array, device_result, N); // square<<<number of blocks, number of threads per block>>>
    cudaDeviceSynchronize();                                                        // Wait for the GPU to complete it's operation, else it'll fail. VERY IMPORTANT LINE!

    cudaMemcpy(host_result, device_result, size, cudaMemcpyDeviceToHost); // Copying data back from GPU to CPU
    for (int i = 0; i < 10; ++i)
        cout << "Square of " << host_array[i] << " is " << host_result[i] << endl;

    // Freeing the memory
    cudaFree(device_array);
    cudaFree(device_result);
    delete[] host_array;

    return 0;
}