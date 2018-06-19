#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %d %s %s %d\n", code, cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__
void matrixAddKernel(float *matA, float *matB, float *matC, int size) {
  size_t indexX = blockIdx.x  * blockDim.x + threadIdx.x;
  size_t strideX = blockDim.x * gridDim.x;

  size_t indexY = blockIdx.y  * blockDim.y + threadIdx.y;
  size_t strideY = blockDim.y * gridDim.y;

  for (size_t i = indexX; i < n; i += strideX)
    for (size_t j = indexY; j < n; j += strideY)
      C[i*size + j] = A[i* size + j] + B[i*size  + j];
}

// 1.B Un thread por elemento
__global__ 
void matrixAddKernel_B(float *matA, float *matB, float *matC, int size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i < size and i < size)
    C[i*size + j] = A[i * size + j] + B[i * size + j];
}

// 1.C Un thread por fila
__global__ 
void matrixAddKernel_C(float *matA, float *matB, float *matC, int size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i < size and i < size)
    C[i*size + j] = A[i * size + j] + B[i * size + j];
}

// 1.D Un thread por columna

// 1.E 

// 1.A
void matrixAddKernel(float *matA, float *matB, float *matC, int size) {
  int size = size * sizeof(float);
  float *d_matA, *d_matB, *d_matC;

  gpuErrchk(cudaMalloc(&d_matA, size * size));
  gpuErrchk(cudaMemcpy(d_matA, matA, size * size, cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc(&d_matB, size * size));
  gpuErrchk(cudaMemcpy(d_matB, matB, size * size, cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc(&d_matC, size * size));

  matrixAddKernel<<<65535, 1024>>>(d_matA, d_matB, d_matC, size);

  gpuErrchk(cudaMemcpy(matC, d_matC, size * size, cudaMemcpyDeviceToHost));

  cudaFree(d_matC);
  cudaFree(d_matA);
  cudaFree(d_matB);
}


