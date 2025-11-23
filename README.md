# How to Run a CUDA File on Google Colab

This guide will walk you through the necessary steps to run a CUDA code file on Google Colab.

## Steps

1. **Open Google Colab*

2. **Set the Runtime to GPU**
   - Click on `Runtime` in the menu bar.
   - Select `Change runtime type`.
   - In the Hardware accelerator dropdown, choose `GPU`.
   - Click `Save`.

3. **Upload Your CUDA File**
   - In the left panel, click on the folder icon to open the file explorer.
   - Click on the upload icon and upload your `.cu` CUDA file.

4. **Install NVIDIA CUDA Compiler (if needed)**
   - Google Colab usually has `nvcc` pre-installed. You can check by running `!nvcc --version`.
   - If missing, you may need to install CUDA toolkit (rare case).

5. **Compile Your CUDA File**
   - Use the `nvcc` compiler command to compile your `.cu` file.
   - Run the following command in a code cell:
     ```
     !nvcc your_cuda_file.cu -o your_executable
     ```

6. **Run the Compiled Executable**
   - Execute the compiled program by running:
     ```
     !./your_executable
     ```

7. **Debug and Iterate**
   - View the output for errors or success messages.
   - Make edits to your `.cu` file, re-upload if needed, and re-run steps 5 and 6.

