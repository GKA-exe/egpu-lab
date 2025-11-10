import sys
import os
import subprocess

def execute_cuda_file(cuda_file):
    if not cuda_file.endswith('.cu'):
        print("Error: File must be a CUDA file (.cu extension)")
        return

    if not os.path.exists(cuda_file):
        print(f"Error: File {cuda_file} does not exist")
        return

    # Compile the CUDA file
    output_file = cuda_file[:-3] + ".exe"  # Remove .cu and add .exe
    compile_command = f"nvcc {cuda_file} -o {output_file}"
    
    try:
        subprocess.run(compile_command, check=True)
        print(f"Successfully compiled {cuda_file}")
        
        # Execute the compiled file
        print(f"\nExecuting {output_file}...")
        subprocess.run(output_file, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Error during compilation or execution: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python cuda_execute.py <cuda_file_name>")
        sys.exit(1)

    cuda_file = sys.argv[1]
    execute_cuda_file(cuda_file)