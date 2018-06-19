#include "blur.cuh"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;

const size_t FILTER_SIZE =  11;
const size_t BLOCK_SIZE =  16;

__global__ void imgBlurKernel(unsigned char *outImg, unsigned char *inImg,
                           int width, int height) {
  int filterRow, filterCol;
  int cornerRow, cornerCol;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int filterSize = 2 * FILTER_SIZE + 1;

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
        if ((filterRow >= 0) && (filterRow <= height) && (filterCol >= 0) &&
            (filterCol <= width)) {
          cumSum += inImg[filterRow * width + filterCol];
          numPixels++;
        }
      }
    }
    // set the value of output
    outImg[row * width + col] = (unsigned char)(cumSum / numPixels);
  }
}

void imageBlur(string imageName) {
  // read image
  Mat img;
  img = imread(imageName, CV_LOAD_IMAGE_GRAYSCALE);
  if (img.empty()) {
    cout << "Cannot read image file " << imageName;
    exit(1);
  }

  // define img params
  int imgWidth = img.cols;
  int imgHeight = img.rows;
  size_t imgSize = sizeof(unsigned char) * imgWidth * imgHeight;

  // allocate mem for host output image vectors
  unsigned char *h_outImg = (unsigned char *)malloc(imgSize);

  // grab pointer to host input image
  unsigned char *h_inImg = img.data;

  // allocate mem for device input and output
  unsigned char *d_inImg;
  unsigned char *d_outImg;
  cudaMalloc(&d_inImg, imgSize);
  cudaMalloc(&d_outImg, imgSize);

  // copy the input image from the host to the device
  cudaMemcpy(d_inImg, h_inImg, imgSize, cudaMemcpyHostToDevice);

  // execution configuration parameters + kernel launch
  dim3 dimBlock(16, 16);
  dim3 dimGrid(ceil(imgWidth / 16.0), ceil(imgHeight / 16.0));
  imgBlurKernel<<<dimGrid, dimBlock>>>(d_outImg, d_inImg, imgWidth, imgHeight);

  // copy output image from device to host
  cudaMemcpy(h_outImg, d_outImg, imgSize, cudaMemcpyDeviceToHost);

  // display images
  Mat imgBlur(imgHeight, imgWidth, CV_8UC1, h_outImg);

  namedWindow("blur", WINDOW_NORMAL);
  imshow("blur",imgBlur);
  waitKey(0);

  imwrite("./blur.jpg", imgBlur);

  // free host and device memory
  img.release();
  imgBlur.release();
  free(h_outImg);
  cudaFree(d_outImg);
  cudaFree(d_inImg);
}
