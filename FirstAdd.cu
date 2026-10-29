#include <cuda_runtime.h>
#include <iostream>

__global__ void firstAdd(
    int *g_idata,
    int *g_odata)
{
   extern __shared__ int sdata[];

   unsigned int tid = threadIdx.x;
   unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;

   sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
   __syncthreads();

   for (unsigned int s = blockDim.x / 2; s > 2; s >>= 1)
   {
      if (tid < s)
      {
         sdata[tid] += sdata[tid + s];
      }
      __syncthreads();
   }

   if (tid == 0)
      g_odata[blockIdx.x] = sdata[0];
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

   firstAdd<<<1, N, N * sizeof(int)>>>(d_idata, d_odata);

   cudaMemcpy(h_odata, d_odata, sizeof(int), cudaMemcpyDeviceToHost);

   std::cout << "Sum: " << h_odata[0] << std::endl;

   cudaFree(d_idata);
   cudaFree(d_odata);
   return 0;
}