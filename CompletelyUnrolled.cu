#include <iostream>
#include <cuda_runtime.h>

template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, int tid)
{
   if (blockSize >= 64)
      sdata[tid] += sdata[tid + 32];
   if (blockSize >= 32)
      sdata[tid] += sdata[tid + 16];
   if (blockSize >= 16)
      sdata[tid] += sdata[tid + 8];
   if (blockSize >= 8)
      sdata[tid] += sdata[tid + 4];
   if (blockSize >= 4)
      sdata[tid] += sdata[tid + 2];
   if (blockSize >= 2)
      sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduceKernel(
    const int *input,
    int *output,
    unsigned int n)
{
   extern __shared__ int sdata[];

   unsigned int tid = threadIdx.x;
   unsigned int i = blockDim.x * (blockSize * 2) + tid;

   int sum = 0;

   // Each thread loads up to two elements
   if (i < n)
      sum = input[i];

   if (i + blockSize < n)
      sum += input[i + blockSize];

   sdata[tid] = sum;
   __syncthreads();

   // Reduce the block to a partial sum
   if (blockSize >= 512)
   {
      if (tid < 256)
         sdata[tid] += sdata[tid + 256];
      __syncthreads();
   }

   if (blockSize >= 256)
   {
      if (tid < 128)
         sdata[tid] += sdata[tid + 128];
      __syncthreads();
   }

   if (blockSize >= 128)
   {
      if (tid < 64)
         sdata[tid] += sdata[tid + 64];
      __syncthreads();
   }

   if (tid < 32)
   {
      warpReduce<blockSize >= 64 ? 32 : blockSize>(
          sdata, tid);
   }
   if (tid == 0)
      output[blockIdx.x] = sdata[0];
}

int main()
{
   const unsigned int n = 1024;
   const unsigned int blockSize = 256;

   int *h_input = new int[n];
   for (unsigned int i = 0; i < n; i++)
   {
      h_input[i] = 1;
   }

   int *d_input, *d_output;

   unsigned int gridSize = (n + 2 * blockSize - 1) / (2 * blockSize);

   cudaMalloc(&d_input, sizeof(int) * n);
   cudaMalloc(&d_output, gridSize * sizeof(int));

   cudaMemcpy(
       d_input,
       h_input,
       n * sizeof(int),
       cudaMemcpyHostToDevice);

   reduceKernel<blockSize><<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_input, d_output, n);

   int *h_output = new int[gridSize];

   cudaMemcpy(h_output, d_output, gridSize * sizeof(int), cudaMemcpyDeviceToHost);

   int results = 0;
   for (unsigned int i = 0; i < gridSize; i++)
   {
      results += h_output[i];
   }

   std::cout << "Sum: " << results << std::endl;

   delete[] h_input;
   delete[] h_output;
   cudaFree(d_input);
   cudaFree(d_output);

   return 0;
}