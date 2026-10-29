#include <iostream>
#include <cuda_runtime.h>

__global__ void reduce0(
    int *g_idata,
    int *g_odata)
{
   extern __shared__ int sdata[];

   unsigned int tid = threadIdx.x;
   unsigned int i = blockDim.x * blockIdx.x + tid;

   // Load one element into shared memory
   sdata[tid] = g_idata[i];

   __syncthreads();

   for (unsigned int s = 1; s < blockDim.x; s *= 2)
   {
      if (tid % (s * 2) == 0)
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

   // Host Output
   int h_odata[1];

   // Device Pointers
   int *d_idata;
   int *d_odata;

   // Allocate memory on GPU
   cudaMalloc(&d_idata, sizeof(N) * N);
   cudaMalloc(&d_odata, sizeof(int));

   // Copy input from CPU to GPU
   cudaMemcpy(d_idata,h_idata,N*sizeof(int),cudaMemcpyHostToDevice);

   reduce0<<<1,N,N*sizeof(int)>>>(d_idata,d_odata);

   cudaMemcpy(h_odata,d_odata,sizeof(int),cudaMemcpyDeviceToHost);

   // Print result
   std::cout<<"Sum = "<<h_odata[0]<<std::endl;

   // Free GPU memory
   cudaFree(d_idata);
   cudaFree(d_odata);

   return 0;
}