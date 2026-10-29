#include <cuda_runtime.h>
#include <iostream>

__device__ void warpReduce(volatile int *sdata, int tid)
{
   sdata[tid] += sdata[tid + 32];
   sdata[tid] += sdata[tid + 16];
   sdata[tid] += sdata[tid + 8];
   sdata[tid] += sdata[tid + 4];
   sdata[tid] += sdata[tid + 2];
   sdata[tid] += sdata[tid + 1];
}

__global__ void unroll(int *g_idata, int *g_odata)
{
   extern __shared__ int sdata[];

   unsigned int tid = threadIdx.x;
   unsigned int i = blockDim.x * blockIdx.x + tid;

   sdata[tid] = g_idata[i];

   __syncthreads();

   for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
   {
      if (tid < s)
         sdata[tid] += sdata[tid + s];
      __syncthreads();
   }

   if (tid < 32)
      warpReduce(sdata, tid);
}

int main()
{
   const int N = 10;

   int h_idata[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

   int h_odata[1];

   int *d_idata;
   int *d_odata;

   cudaMalloc(&d_idata, sizeof(int) * N);
   cudaMalloc(&d_odata, sizeof(int));

   cudaMemcpy(d_idata, h_idata, sizeof(int) * N, cudaMemcpyHostToDevice);

   unroll<<<1, N, N * sizeof(int)>>>(d_idata, d_odata);

   cudaMemcpy(h_odata, d_odata, sizeof(int), cudaMemcpyDeviceToHost);

   std::cout << "Sum: " << h_odata[0] << std::endl;
   cudaFree(d_idata);
   cudaFree(d_odata);

   return 0;
}