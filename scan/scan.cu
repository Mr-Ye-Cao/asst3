#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256


// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// Upsweep kernel with robust bounds checking
__global__ void upsweep_kernel(int* output, int two_d, int two_dplus1, int N) {
    // Calculate global thread index
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Convert to the actual array index (using stride pattern)
    i = i * two_dplus1;
    
    // Only perform operations if indices are valid
    if (i + two_d - 1 < N && i + two_dplus1 - 1 < N) {
        output[i + two_dplus1 - 1] += output[i + two_d - 1];
    }
}

// Downsweep kernel with robust bounds checking
__global__ void downsweep_kernel(int* output, int two_d, int two_dplus1, int N) {
    // Calculate global thread index
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Convert to the actual array index (using stride pattern)
    i = i * two_dplus1;
    
    // Only perform operations if indices are valid
    if (i + two_d - 1 < N && i + two_dplus1 - 1 < N) {
        int t = output[i + two_d - 1];
        output[i + two_d - 1] = output[i + two_dplus1 - 1];
        output[i + two_dplus1 - 1] += t;
    }
}

// Exclusive scan implementation
void exclusive_scan(int* input, int N, int* result)
{
    // Copy input to result if they are different
    if (input != result) {
        cudaMemcpy(result, input, N * sizeof(int), cudaMemcpyDeviceToDevice);
    }
    
    // Upsweep phase
    for (int two_d = 1; two_d < N; two_d *= 2) {
        int two_dplus1 = 2 * two_d;
        
        // Calculate number of threads needed based on stride pattern
        int num_elements = (N + two_dplus1 - 1) / two_dplus1;
        if (num_elements == 0) num_elements = 1; // Ensure at least one thread
        
        // Calculate number of blocks needed
        int num_blocks = (num_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        if (num_blocks == 0) num_blocks = 1; // Ensure at least one block
        
        // Launch kernel
        upsweep_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(result, two_d, two_dplus1, N);
        cudaDeviceSynchronize();
    }
    
    // Set last element to 0
    int zero = 0;
    cudaMemcpy(&result[N-1], &zero, sizeof(int), cudaMemcpyHostToDevice);
    
    // Downsweep phase
    for (int two_d = N/2; two_d >= 1; two_d /= 2) {
        int two_dplus1 = 2 * two_d;
        
        // Calculate number of threads needed based on stride pattern
        int num_elements = (N + two_dplus1 - 1) / two_dplus1;
        if (num_elements == 0) num_elements = 1; // Ensure at least one thread
        
        // Calculate number of blocks needed
        int num_blocks = (num_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        if (num_blocks == 0) num_blocks = 1; // Ensure at least one block
        
        // Launch kernel
        downsweep_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(result, two_d, two_dplus1, N);
        cudaDeviceSynchronize();
    }
    
    // Check for any CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}


//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found


// Kernel to identify repeats
__global__ void identify_repeats_kernel(int* input, int length, int* flags) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < length - 1) {
        // Set flag to 1 if element is repeated, 0 otherwise
        flags[i] = (input[i] == input[i+1]) ? 1 : 0;
    } else if (i == length - 1) {
        // Handle the last element (which can't start a repeat)
        flags[i] = 0;
    }
}

// Kernel to collect indices of repeats
__global__ void collect_repeats_kernel(int* flags, int* scanned_flags, int* output, int length) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < length - 1 && flags[i] == 1) {
        // If this is a repeat, put its index in the corresponding position of output
        output[scanned_flags[i]] = i;
    }
}

int find_repeats(int* device_input, int length, int* device_output) {
    // 1. Create a temporary array to mark positions of repeats
    int* device_flags;
    cudaMalloc((void **)&device_flags, length * sizeof(int));
    
    // 2. Launch a kernel to identify repeats
    int num_blocks = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    identify_repeats_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(device_input, length, device_flags);
    
    // 3. Run exclusive scan on the flags to get positions
    int* device_scan_result;
    cudaMalloc((void **)&device_scan_result, length * sizeof(int));
    exclusive_scan(device_flags, length, device_scan_result);
    
    // 4. Get the total count of repeats (last element of scan + last flag)
    int total_repeats;
    int last_flag;
    cudaMemcpy(&last_flag, &device_flags[length-1], sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&total_repeats, &device_scan_result[length-1], sizeof(int), cudaMemcpyDeviceToHost);
    total_repeats += last_flag;
    
    // 5. Launch another kernel to collect the indices
    collect_repeats_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
        device_flags, device_scan_result, device_output, length);
    
    // 6. Clean up
    cudaFree(device_flags);
    cudaFree(device_scan_result);
    
    return total_repeats;
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
