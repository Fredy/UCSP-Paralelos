#include "GrayScale.h"
#include "grayscalecuda.h"

GrayScale::GrayScale(string path_to_image) {
    this->matrix_image = imread(path_to_image, CV_LOAD_IMAGE_COLOR);
    if (!(this->matrix_image).data){
        cout << "Error reading image" << endl;
    }
}

void GrayScale::conversionToGrayScaleSerial() {
    int cols = this->matrix_image.cols;
    int rows = this->matrix_image.rows;
    int* Blue = new int[cols * rows];
    int* Green = new int[cols * rows];
    int* Red = new int[cols * rows];
    float* GrayScaleMatrix = new float[cols * rows];
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            int pos = cols * i + j;
            Blue[pos] = this->matrix_image.at<cv::Vec3b>(i, j)[0];
            Green[pos] = this->matrix_image.at<cv::Vec3b>(i, j)[1];
            Red[pos] = this->matrix_image.at<cv::Vec3b>(i, j)[2];
            GrayScaleMatrix[pos] = (float)(((float)Blue[pos]*0.07 + (float)Green[pos]*0.72 + (float)Red[pos]*0.21)/1000.0);
        }

    }

    Mat gray = Mat(rows, cols, CV_32FC1, GrayScaleMatrix);
}

void GrayScale::export_to_image() {

}

string GrayScale::get_path() {
    return std::string();
}

void GrayScale::conversionToGrayScaleParallelized() {
    int cols = this->matrix_image.cols;
    int rows = this->matrix_image.rows;
    float* Blue = new float[cols * rows];
    float* Green = new float[cols * rows];
    float* Red = new float[cols * rows];
    float* GrayScaleMatrix = new float[cols * rows];
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            int pos = cols * i + j;
            Blue[pos] = (float)this->matrix_image.at<cv::Vec3b>(i, j)[0];
            Green[pos] = (float)this->matrix_image.at<cv::Vec3b>(i, j)[1];
            Red[pos] = (float)this->matrix_image.at<cv::Vec3b>(i, j)[2];
        }
    }
    grayscale(Red, Green, Blue, GrayScaleMatrix, cols * rows);
    Mat gray = Mat(rows, cols, CV_32FC1, GrayScaleMatrix);
    gray.convertTo(gray, CV_8UC3, 255.0);
    imwrite("./grayscale.jpg", gray);
}
