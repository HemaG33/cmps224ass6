
#include "common.h"

#include "timer.h"

#define BLOCK_DIM 1024

__global__ void reduce_kernel(float* input, float* sum, unsigned int N) {

		unsigned int segment = (numElements - 1) - (2 * blockDim.x * blockIdx.x);
		unsigned int i = segment - threadIdx.x;
		__shared__ float input_s[BLOCK_DIM];

		// Reverse data loading direction
		input_s[threadIdx.x] = input[i] + input[i - BLOCK_DIM];

		__syncthreads();

		for(unsigned int stride = BLOCK_DIM/2; stride > WARP_SIZE; stride /= 2) {
			if(threadIdx.x < stride) {
				// Reverse reduction operation
				input_s[threadIdx.x] += input_s[threadIdx.x - stride];
			}
			__syncthreads();
		}

		if(threadIdx.x < WARP_SIZE) {
			float partialSum = input_s[threadIdx.x] + input_s[threadIdx.x - WARP_SIZE];
			for(unsigned int stride = WARP_SIZE/2; stride > 0; stride /= 2) {
				// Reverse reduction operation
				partialSum += __shfl_down_sync(0xffffffff, partialSum, stride);
			}
		}
}

float reduce_gpu(float* input, unsigned int N) {

    Timer timer;

    // Allocate memory
    startTime(&timer);
    float *input_d;
    cudaMalloc((void**) &input_d, N*sizeof(float));
    float *sum_d;
    cudaMalloc((void**) &sum_d, sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);
    cudaMemcpy(input_d, input, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(sum_d, 0, sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Call kernel
    startTime(&timer);
    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    const unsigned int numElementsPerBlock = 2*numThreadsPerBlock;
    const unsigned int numBlocks = (N + numElementsPerBlock - 1)/numElementsPerBlock;
    reduce_kernel <<< numBlocks, numThreadsPerBlock >>> (input_d, sum_d, N);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Copy data from GPU
    startTime(&timer);
    float sum;
    cudaMemcpy(&sum, sum_d, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Free memory
    startTime(&timer);
    cudaFree(input_d);
    cudaFree(sum_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

    return sum;

}

