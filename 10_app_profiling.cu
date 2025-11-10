#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define N 10000000

__global__ void squareKernel(float *input, float *output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        output[idx] = input[idx] * input[idx];
}

int main()
{
    int size = N * sizeof(float);
    float *h_input = new float[N];
    float *h_output = new float[N];

    for (int i = 0; i < N; ++i)
        h_input[i] = i * 0.5f;

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float time_memcpy_h2d, time_kernel, time_memcpy_d2h;

    // --- Measure Host → Device copy ---
    cudaEventRecord(start, 0);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_memcpy_h2d, start, stop);

    // --- Measure Kernel execution ---
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cudaEventRecord(start, 0);
    squareKernel<<<blocks, threads>>>(d_input, d_output, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_kernel, start, stop);

    // --- Measure Device → Host copy ---
    cudaEventRecord(start, 0);
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_memcpy_d2h, start, stop);

    cout << "Square of first 5 elements:\n";
    for (int i = 0; i < 5; ++i)
        cout << h_input[i] << "^2 = " << h_output[i] << endl;

    cout << "\n=== Profiling Results ===\n";
    cout << "Host → Device memcpy: " << time_memcpy_h2d << " ms\n";
    cout << "Kernel execution:     " << time_kernel << " ms\n";
    cout << "Device → Host memcpy: " << time_memcpy_d2h << " ms\n";

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
