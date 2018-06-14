#pragma once

#include <string>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

class GrayScale {
public:
    explicit GrayScale(string path_to_image);
    void conversionToGrayScaleSerial();
    void conversionToGrayScaleParallelized();
    void export_to_image();
    string get_path();
private:
    Mat matrix_image;
    Mat matrix_filtered;
    string result_image_path;
};

