#include "blurimagecuda.h"
#include <cuda_runtime.h>
#include <cuda.h>

__global__
void blurimageCudaDevice(float * R, float* blurredimage, int w, int h) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int BLUR_SIZE = 20;
	if(col < w && row < h){
		int pixval = 0;
		int pixels = 0;
		for(int blurrow = -BLUR_SIZE; blurrow < BLUR_SIZE; ++blurrow){
			for(int blurcol = -BLUR_SIZE; blurcol < BLUR_SIZE; ++blurcol){
				int currow = row + blurrow;
				int curcol = col + blurcol;
				if(currow > -1 && currow < h && curcol > -1 && curcol < w){
					pixval += R[currow * w + curcol];
					pixels++;
				}
			}
		}
		blurredimage[row * w + col] = ((float)pixval/(float)pixels)/400.0;
	}
}


void blurimage(float * R, float * G, float * B, float* blurredR, float* blurredB, float* blurredG, int w, int h){
	int size = w * h * sizeof(float);
	float *d_R, *d_G, *d_B, *d_blurR, *d_blurG, *d_blurB;
	cudaMalloc((void **) &d_R, size);
	cudaMemcpy(d_R, R, size, cudaMemcpyHostToDevice);
	cudaMalloc((void **) &d_G, size);
	cudaMemcpy(d_G, G, size, cudaMemcpyHostToDevice);
	cudaMalloc((void **) &d_B, size);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
	cudaMalloc((void **) &d_blurR, size);
	cudaMalloc((void **) &d_blurG, size);
	cudaMalloc((void **) &d_blurB, size);
	dim3 dimGrid(ceil((float)(w*h)/32.0), ceil((float)(w*h)/32.0), 1);
	dim3 dimBlock(32,32,1);
	blurimageCudaDevice<<<dimGrid, dimBlock>>>(d_R, d_blurR, w, h);
	cudaMemcpy(blurredR, d_blurR, size, cudaMemcpyDeviceToHost);
	cudaFree(d_R);
	cudaFree(d_blurR);
	blurimageCudaDevice<<<dimGrid, dimBlock>>>(d_G, d_blurG, w, h);
	cudaMemcpy(blurredG, d_blurG, size, cudaMemcpyDeviceToHost);
	cudaFree(d_G);
	cudaFree(d_blurG);
	blurimageCudaDevice<<<dimGrid, dimBlock>>>(d_B, d_blurB, w, h);
	cudaMemcpy(blurredB, d_blurB, size, cudaMemcpyDeviceToHost);
	cudaFree(d_B);
	cudaFree(d_blurB);
}

__global__
void blurimageCudaDevice1(float * R, float* blurredimage, int w, int h) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int BLUR_SIZE = 2;
	if(col < w && row < h){
		int pixval = 0;
		int pixels = 0;
		for(int blurrow = -BLUR_SIZE; blurrow < BLUR_SIZE; ++blurrow){
			for(int blurcol = -BLUR_SIZE; blurcol < BLUR_SIZE; ++blurcol){
				int currow = row + blurrow;
				int curcol = col + blurcol;
				if(currow > -1 && currow < h && curcol > -1 && curcol < w){
					pixval += R[currow * w + curcol];
					pixels++;
				}
			}
		}
		blurredimage[row * w + col] = (unsigned char)(pixval/pixels);
	}
}

void bluruchar(unsigned char* ucharmat, int w, int h, unsigned char* ucharupdated){
	int size = w * h * sizeof(float);
	float *d_R, *d_blurR;
	cudaMalloc((void **) &d_R, size);
	cudaMemcpy(d_R, ucharmat, size, cudaMemcpyHostToDevice);
	cudaMalloc((void **) &d_blurR, size);
	dim3 dimGrid(ceil((float)(w*h)/32.0), ceil((float)(w*h)/32.0), 1);
	dim3 dimBlock(32,32,1);
	blurimageCudaDevice1<<<dimGrid, dimBlock>>>(d_R, d_blurR, w, h);
	cudaMemcpy(ucharupdated, d_blurR, size, cudaMemcpyDeviceToHost);
	cudaFree(d_R);
	cudaFree(d_blurR);
}
