#pragma once

#include <string>
using namespace std;

__global__ void rgb2grayKernel(unsigned char *Pout, unsigned char *Pin, int width,
                            int height, int numChannels);

void grayScale(string fileName);
