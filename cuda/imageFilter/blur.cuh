#include <string>
using namespace std;

__global__ void imgBlurKernel(unsigned char *outImg, unsigned char *inImg,
                           int width, int height);

void imageBlur(string imageName);
