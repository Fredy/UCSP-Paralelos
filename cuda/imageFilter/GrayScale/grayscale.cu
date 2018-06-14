__global__
void cudaGrayScale(float *R, float *G, float *B, float* gray, int n){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i < n) {
        gray[i] = static_cast<float>((R[i] * 0.21 + G[i] * 0.71 + B[i] * 0.07) / 350.0);
        //gray[i] = static_cast<float>((R[i] + G[i] + B[i]) / (3 * 500.0));
    }
}

void grayscale(float* R, float* G, float* B, float* grayscale, int n){
    int size = n * sizeof(float);
    float *d_R, *d_G, *d_B, *d_gray;
    cudaMalloc((void **) &d_R, size);
    cudaMemcpy(d_R, R, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_G, size);
    cudaMemcpy(d_G, G, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_B, size);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_gray, size);

    cudaGrayScale<<<ceil(n/1024.0), 1024>>>(d_R, d_G, d_B, d_gray, n);
    cudaMemcpy(grayscale, d_gray, size, cudaMemcpyDeviceToHost);

    cudaFree(d_R);
    cudaFree(d_G);
    cudaFree(d_B);
    cudaFree(d_gray);
}
