import pyopencl as cl
import numpy as np

# Matrix size
N = 4

# Setup OpenCL
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

kernel_code = """
__kernel void transpose(__global float* input, __global float* output, int n) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    if (row < n && col < n) {
        output[col * n + row] = input[row * n + col];
    }
}

__kernel void trace(__global float* matrix, __global float* result, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += matrix[i * n + i];
    }
    result[0] = sum;
}
"""

program = cl.Program(ctx, kernel_code).build()

h_matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
], dtype=np.float32).flatten()

h_transpose = np.zeros(N*N, dtype=np.float32)
h_trace = np.zeros(1, dtype=np.float32)

d_matrix = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_matrix)
d_transpose = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, h_transpose.nbytes)
d_trace = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, h_trace.nbytes)

program.transpose(queue, (N, N), None, d_matrix, d_transpose, np.int32(N))
program.trace(queue, (1,), None, d_matrix, d_trace, np.int32(N))

cl.enqueue_copy(queue, h_transpose, d_transpose).wait()
cl.enqueue_copy(queue, h_trace, d_trace).wait()

print("Original Matrix:")
print(h_matrix.reshape(N, N))

print("\nTranspose:")
print(h_transpose.reshape(N, N))

print(f"\nTrace: {h_trace[0]}")
