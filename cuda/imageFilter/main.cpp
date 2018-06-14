#include <iostream>
#include "GrayScale/GrayScale.h"
#include "Blur/blurImage.h"

using namespace std;



int main(int argc, char** argv) {
    if (argc < 3) {
        cout << argv[0] << ": needs two arguments\n" << "<image_path> <option>\n";
        return 0;
    }

    string image_path(argv[1]), option(argv[2]);
    if (option == "gray") {
        auto gs = GrayScale(image_path);
        gs.conversionToGrayScaleParallelized();
    }
    else{
        auto gs = blurImage(image_path);
        gs.conversion();
    }
    //gs.conversionToGrayScaleParallelized();
    //gs.conversionToGrayScaleSerial();
    return 0;
}