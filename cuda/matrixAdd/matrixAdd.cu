#include <cstdio>
#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %d %s %s %d\n", code, cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}

__global__ void matrixAddKernel(float *matA, float *matB, float *matC,
                                int size) {
  size_t indexX = blockIdx.x * blockDim.x + threadIdx.x;
  size_t strideX = blockDim.x * gridDim.x;

  size_t indexY = blockIdx.y * blockDim.y + threadIdx.y;
  size_t strideY = blockDim.y * gridDim.y;

  for (size_t i = indexX; i < size; i += strideX)
    for (size_t j = indexY; j < size; j += strideY)
      matC[i * size + j] = matA[i * size + j] + matB[i * size + j];
}

// 1.B Un thread por elemento
__global__ void matrixAddKernel_B(float *matA, float *matB, float *matC,
                                  int size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  if (i < size and j < size)
    matC[i * size + j] = matA[i * size + j] + matB[i * size + j];
}

// 1.C Un thread por fila
__global__ void matrixAddKernel_C(float *matA, float *matB, float *matC,
                                  int size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) {
    for (size_t j = 0; j < size; j++)
      matC[i * size + j] = matA[i * size + j] + matB[i * size + j];
  }
}

// 1.D Un thread por columna
__global__ void matrixAddKernel_D(float *matA, float *matB, float *matC,
                                  int size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) {
    for (size_t j = 0; j < size; j++)
      matC[j * size + i] = matA[j * size + i] + matB[j * size + i];
  }
}

// 1.A
void matrixAdd(float *matA, float *matB, float *matC, int size) {
  size_t sizeM = size * size * sizeof(float);
  float *d_matA, *d_matB, *d_matC;

  gpuErrchk(cudaMalloc(&d_matA, sizeM));
  gpuErrchk(cudaMemcpy(d_matA, matA, sizeM, cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc(&d_matB, sizeM));
  gpuErrchk(cudaMemcpy(d_matB, matB, sizeM, cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc(&d_matC, sizeM));

  // Execute the kernel

  // 1.B
  dim3 threads(16, 16);
  dim3 blocks(ceil(size / threads.x), ceil(size / threads.y));
  matrixAddKernel_B<<<blocks, threads>>>(d_matA, d_matB, d_matC, size);

  // 1.C
  threads = dim3(16);
  blocks = dim3(ceil(size / threads.x));
  matrixAddKernel_C<<<blocks, threads>>>(d_matA, d_matB, d_matC, size);

  // 1.D
  threads = dim3(16);
  blocks = dim3(ceil(size / threads.x));
  matrixAddKernel_D<<<blocks, threads>>>(d_matA, d_matB, d_matC, size);

  gpuErrchk(cudaMemcpy(matC, d_matC, sizeM, cudaMemcpyDeviceToHost));

  cudaFree(d_matC);
  cudaFree(d_matA);
  cudaFree(d_matB);
}

int main() {
  size_t size = 16;
  float *matA = new float[size * size];
  float *matB = new float[size * size];
  float *matC = new float[size * size];

  for (size_t i = 0; i < size * size; i++) {
    matA[i] = 1.0;
    matB[i] = 2.0;
  }

  matrixAdd(matA, matB, matC, size);
}
