#pragma once
#include <string>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;


class blurImage {
public:
    explicit blurImage(string path_to_file);
    void conversion();

private:
    Mat matrix_image;
};
