import pyopencl as cl 
import numpy as np 

num_work_items = 1000 
num_samples = 100000 

platform = cl.get_platforms()[0] 
device = platform.get_devices()[0] 
ctx = cl.Context([device]) 
queue = cl.CommandQueue(ctx) 

kernel_code = """ 
__kernel void estimate_pi(__global float* results, const uint samples) {
    int id = get_global_id(0);
    uint seed = id * 999983 + 314159;
    uint inside = 0;
    
    for (uint i = 0; i < samples; i++) {
        seed = seed * 1664525 + 1013904223;
        float x = (float)(seed & 0xFFFF) / 65535.0f;
        
        seed = seed * 1664525 + 1013904223;
        float y = (float)(seed & 0xFFFF) / 65535.0f;
        
        if (x*x + y*y <= 1.0f) inside++;
    }
    
    results[id] = 4.0f * (float)inside / (float)samples;
}
""" 

program = cl.Program(ctx, kernel_code).build() 

result = np.empty(num_work_items, dtype=np.float32) 
buffer_result = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, result.nbytes) 

program.estimate_pi(queue, (num_work_items,), None, buffer_result, np.uint32(num_samples)) 

cl.enqueue_copy(queue, result, buffer_result).wait() 

estimated_pi = np.mean(result) 
print(f"Estimated value of π: {estimated_pi}")
