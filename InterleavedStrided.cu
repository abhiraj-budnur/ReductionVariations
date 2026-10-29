#include <cuda_runtime.h>
#include <iostream>

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
      int index = 2 * s * tid;
      if (index < blockDim.x)
      {
         sdata[index] += sdata[index + s];
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

   // Allocate Memory on GPU
   cudaMalloc(&d_idata, sizeof(N) * N);
   cudaMalloc(&d_odata, sizeof(int));

   // Copy host data to device
   cudaMemcpy(d_idata,h_idata,N*sizeof(int),cudaMemcpyHostToDevice);

   reduce0<<<1,N,N*sizeof(int)>>>(d_idata,d_odata);

   cudaMemcpy(h_odata,d_odata,sizeof(int),cudaMemcpyDeviceToHost);

   std::cout<<"Sum = "<<h_odata[0]<<std::endl;

   cudaFree(d_idata);
   cudaFree(d_odata);

   return 0;
}