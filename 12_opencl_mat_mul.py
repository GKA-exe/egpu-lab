import pyopencl as cl 
import numpy as np 
 
# OpenCL kernel for matrix multiplication 
kernel_code = """ 
__kernel void matrix_multiply(__global int* A, __global int* B, 
__global int* C, const int M, const int N, const int K) { 
    int row = get_global_id(0); 
    int col = get_global_id(1); 
    int sum = 0; 
     
    if (row < M && col < K) { 
        for (int i = 0; i < N; i++) { 
            sum += A[row * N + i] * B[i * K + col]; 
        } 
        C[row * K + col] = sum; 
    } 
} 
""" 

M, N, K = 3, 3, 3

matrix_A = np.random.randint(1, 10, size=(M, N), dtype=np.int32) 
matrix_B = np.random.randint(1, 10, size=(N, K), dtype=np.int32) 
 
matrix_C = np.zeros((M, K), dtype=np.int32) 
 
platform = cl.get_platforms()[0] 
device = platform.get_devices()[0] 
ctx = cl.Context([device]) 
queue = cl.CommandQueue(ctx) 
 
program = cl.Program(ctx, kernel_code).build()  
 
buffer_A = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | 
cl.mem_flags.COPY_HOST_PTR, hostbuf=matrix_A) 
buffer_B = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | 
cl.mem_flags.COPY_HOST_PTR, hostbuf=matrix_B) 
buffer_C = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 
matrix_C.nbytes) 
 
program.matrix_multiply(queue, matrix_A.shape, None, buffer_A, 
buffer_B, buffer_C, np.int32(M), np.int32(N), np.int32(K)) 
 
cl.enqueue_copy(queue, matrix_C, buffer_C).wait() 
 
print("Matrix A:") 
print(matrix_A) 
print("Matrix B:") 
print(matrix_B) 
print("Resulting Matrix C:") 
print(matrix_C) 
