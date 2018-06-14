#include <iostream>
using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %d %s %s %d\n", code, cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void vecAddKernel(float *A, float *B, float *C, int n) {
  size_t index = blockIdx.x  * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (int i = index; i < n; i += stride)
    C[i] = A[i] + B[i];
}

__global__ void init(int n, float *x, float *y) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
}

void vecAdd(float *h_arrayA, float *h_arrayB, float *h_arrayC, int n) {
 vecAddKernel<<<65535, 1024>>>(h_arrayA, h_arrayB, h_arrayC, n);
 cudaDeviceSynchronize();
}

int main() {
  unsigned long long size = 1 << 28;
  cout << size << endl;

  float *arrayA;
  float *arrayB;
  float *arrayC;

  gpuErrchk(cudaMallocManaged(&arrayA, size  * sizeof(float)));
  gpuErrchk(cudaMallocManaged(&arrayB, size  * sizeof(float)));
  gpuErrchk(cudaMallocManaged(&arrayC, size  * sizeof(float)));

  init<<<65535,1024>>>(size, arrayA, arrayB);

  vecAdd(arrayA, arrayB, arrayC, size);
  cout << arrayC[0] << ' '<<arrayC[size - 1 ] << endl;

  cudaFree(arrayA);
  cudaFree(arrayB);
  cudaFree(arrayC);
}
