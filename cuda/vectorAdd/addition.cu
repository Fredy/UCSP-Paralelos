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
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
    C[i] = A[i] + B[i];
}

void vecAdd(float *h_arrayA, float *h_arrayB, float *h_arrayC, int n) {
  int size = n * sizeof(float);
  float *d_arrayA, *d_arrayB, *d_arrayC;
  gpuErrchk(cudaMalloc(&d_arrayA, size));
  gpuErrchk(cudaMemcpy(d_arrayA, h_arrayA, size, cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc(&d_arrayB, size));
  gpuErrchk(cudaMemcpy(d_arrayB, h_arrayB, size, cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc(&d_arrayC, size));

  vecAddKernel<<<ceil(n/1024.0), 1024>>>(d_arrayA, d_arrayB, d_arrayC, n);

  gpuErrchk(cudaMemcpy(h_arrayC, d_arrayC, size, cudaMemcpyDeviceToHost));

  cudaFree(d_arrayC);
  cudaFree(d_arrayA);
  cudaFree(d_arrayB);
}

int main() {
  unsigned long long size = 1 << 28;
  cout << size << endl;

  float *arrayA = (float *)malloc(size * sizeof(float));
  float *arrayB = (float *)malloc(size * sizeof(float));
  float *arrayC = (float *)malloc(size * sizeof(float));

  for (int i = 0; i < size; i++) {
    arrayA[i] = 1.0f;
    arrayB[i] = 2.0f;
  }

  vecAdd(arrayA, arrayB, arrayC, size);
  cout << arrayC[0] << ' '<<arrayC[size - 1 ] << endl;
}
