import pyopencl as cl 
import numpy as np 

num_work_items = 1000 
num_samples = 100000 

platform = cl.get_platforms()[0] 
device = platform.get_devices()[0] 
ctx = cl.Context([device]) 
queue = cl.CommandQueue(ctx) 

kernel_code = """ 
__kernel void leibniz_pi(__global float *partial_sums, const int iterations) {
    int gid = get_global_id(0);
    int total_workers = get_global_size(0);
    
    float sum = 0.0f;
    for (int i = gid; i < iterations; i += total_workers) {
        float term = 1.0f / (2.0f * i + 1.0f);
        if (i % 2 == 1) term = -term;
        sum += term;
    }
    
    partial_sums[gid] = sum;
}
""" 

program = cl.Program(ctx, kernel_code).build() 

partial_sums = np.zeros(num_work_items, dtype=np.float32)
buffer_result = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, partial_sums.nbytes) 

program.leibniz_pi(queue, (num_work_items,), None, buffer_result, np.int32(num_samples)) 

cl.enqueue_copy(queue, partial_sums, buffer_result).wait() 

estimated_pi = np.sum(partial_sums) * 4
print(f"Estimated value of π: {estimated_pi}")
