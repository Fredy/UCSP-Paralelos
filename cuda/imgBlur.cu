#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "GpuTimer.h"
#include <time.h>

using namespace cv;
using namespace std;

#define FILTER_SIZE 11
#define BLOCK_SIZE 16

// imgBlurGPU blurs an image on the GPU
__global__ void imgBlurGPU(unsigned char* outImg, unsigned char* inImg, int width, int height) {
    int filterRow, filterCol;
    int cornerRow, cornerCol;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x; int by = blockIdx.y;
    int filterSize = 2*FILTER_SIZE + 1;

    // compute global thread coordinates
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    // make sure thread is within image boundaries
    if ((row < height) && (col < width)) {
        // instantiate accumulator
        int numPixels = 0;
        int cumSum = 0;

        // top-left corner coordinates
        cornerRow = row - FILTER_SIZE;
        cornerCol = col - FILTER_SIZE;

        // accumulate values inside filter
        for (int i = 0; i < filterSize; i++) {
            for (int j = 0; j < filterSize; j++) {
                // filter coordinates
                filterRow = cornerRow + i;
                filterCol = cornerCol + j;

                // accumulate sum
                if ((filterRow >= 0) && (filterRow <= height) && (filterCol >= 0) && (filterCol <= width)) {
                    cumSum += inImg[filterRow*width + filterCol];
                    numPixels++;
                }
            }
        }
        // set the value of output
        outImg[row*width + col] = (unsigned char)(cumSum / numPixels);
    }
}

__global__
void blurKernel(unsingned char* in, unsigned char* out, int w, int h){
    int Col = blockIdx.x * blockDim.x + ThreadIdx.x;
    int Row = blockIdx.y * blockDim.y + ThreadIdx.y;

    if (Col < w && Row < h){
        int pixVal = 0;
        int pixels = 0;
        for(int blurRow= -BLUR_SIZE; blurRow < BLUR_SIZE +1; ++blurRow)
            for(int blurCol= -BLUR_SIZE; blurCol < BLUR_SIZE +1; ++blurCol){
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                if (curRow > -1 && curRow < h && curCol >-1 && curCol < w){
                    pixVal += in[curRow*w + curCol];
                    pixels ++;
                }
            }
    }

    out[ Row*w +Col ] = (unsigned char) (pixelVal/ pixels);
}

int main(int argc, char *argv[]) {
    // make sure filename given
    if (argc == 1) {
        printf("[!] Filename expected.\n");
        return 0;
    }

    // read image
    Mat img;
    img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    if (img.empty()) {
        printf("Cannot read image file %s", argv[1]);
        exit(1);
    }

    // define img params and timers
    int imgWidth = img.cols;
    int imgHeight = img.rows;
    size_t imgSize = sizeof(unsigned char)*imgWidth*imgHeight;
    GpuTimer timer;

    // allocate mem for host output image vectors
    unsigned char* h_outImg = (unsigned char*)malloc(imgSize);
    unsigned char* h_outImg_CPU = (unsigned char*)malloc(imgSize);

    // grab pointer to host input image
    unsigned char* h_inImg = img.data;

    // allocate mem for device input and output
    unsigned char* d_inImg;
    unsigned char* d_outImg;

    cudaMalloc((void**)&d_inImg, imgSize);
    cudaMalloc((void**)&d_outImg, imgSize);

    // copy the input image from the host to the device and record the needed time
    cudaMemcpy(d_inImg, h_inImg, imgSize, cudaMemcpyHostToDevice);

    // execution configuration parameters + kernel launch
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(ceil(imgWidth/16.0), ceil(imgHeight/16.0), 1);

    timer.Start();
    //imgBlurGPU<<<dimGrid, dimBlock>>>(d_outImg, d_inImg, imgWidth, imgHeight);
    blurKernel<<<dimGrid, dimBlock>>>(d_outImg, d_inImg, imgWidth, imgHeight);
    timer.Stop();
    float d_t2 = timer.Elapsed();
    printf("Implemented CUDA code ran in: %f msecs.\n", d_t2);

    // copy output image from device to host
    cudaMemcpy(h_outImg, d_outImg, imgSize, cudaMemcpyDeviceToHost);

    // display images
    Mat img1(imgHeight, imgWidth, CV_8UC1, h_outImg);
    Mat img2(imgHeight, imgWidth, CV_8UC1, h_outImg_CPU);
    namedWindow("Before", WINDOW_NORMAL);
    imshow("Before", img);
    namedWindow("After (GPU)", WINDOW_NORMAL);
    imshow("After (GPU)", img1);
    namedWindow("After (CPU)", WINDOW_NORMAL);
    imshow("After (CPU)", img2);
    waitKey(0);

    // free host and device memory
    img.release(); img1.release(); img2.release();
    free(h_outImg_CPU); free(h_outImg);
    cudaFree(d_outImg); cudaFree(d_inImg);

    return 0;
}
