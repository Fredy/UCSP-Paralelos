#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
using namespace std;

//http://www.coldvision.io/2015/11/17/rgb-grayscale-conversion-cuda-opencv/
void rgbaToGrayscale(string inputFileName, string outputFileName) {
	// pointers to images in CPU's memory (h_) and GPU's memory (d_)
	uchar4        *h_rgbaImage, *d_rgbaImage;
	unsigned char *h_greyImage, *d_greyImage;
 
	//load the image and give us our input and output pointers
	preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, input_file);
 
	GpuTimer timer;
	timer.Start();
	// here is where the conversion actually happens
	rgbaToGreyscaleCuda(h_rgbaImage, d_rgbaImage, d_greyImage, numRows(), numCols());
	timer.Stop();
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
 
	int err = printf("Implemented CUDA code ran in: %f msecs.\n", timer.Elapsed());
 
	if (err < 0) {
		//Couldn't print!
		std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
		exit(1);
	}
 
	size_t numPixels = numRows()*numCols();
	checkCudaErrors(cudaMemcpy(h_greyImage, d_greyImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));
 
	//check results and output the grey image
	postProcess(output_file, h_greyImage);
}

int main() {

}
