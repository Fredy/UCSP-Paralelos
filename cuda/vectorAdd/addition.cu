__global__ void vecAddKernel(float *A, float *B, float *C, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
    C[i] = A[i] + B[i];
}

void vecAdd(float *h_arrayA, float *h_arrayB, float *h_arrayC, int n) {
  int size = n * sizeof(float);
  float *d_arrayA, *d_arrayB, *d_arrayC;
  cudaMalloc((void **)&d_arrayA, size);
  cudaMemcpy(d_arrayA, h_arrayA, size, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&d_arrayB, size);
  cudaMemcpy(d_arrayB, h_arrayB, size, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&d_arrayC, size);

  vecAddKernel<<<ceil(n / 256.0), 256>>>(d_arrayA, d_arrayB, d_arrayC, n);
  cudaMemcpy(h_arrayC, d_arrayC, size, cudaMemcpyDeviceToHost);

  cudaFree(d_arrayC);
  cudaFree(d_arrayA);
  cudaFree(d_arrayB);
}

int main() {
  int size = 10;

  float *arrayA = (float *)malloc(size * sizeof(float));
  float *arrayB = (float *)malloc(size * sizeof(float));
  float *arrayC = (float *)malloc(size * sizeof(float));

  for (int i = 0; i < size; i++) {
    arrayA[i] = 1.0f;
    arrayB[i] = 2.0f;
  }
  vecAdd(arrayA, arrayB, arrayC, size);

  for (int i = 0; i < size; i++) {
    printf("%f, ", arrayC[i]);
  }
}
