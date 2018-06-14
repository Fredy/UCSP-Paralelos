#include "blurImage.h"
#include "blurimagecuda.h"
#include <vector>
using namespace std;
using namespace cv;

blurImage::blurImage(string path_to_image) {
    this->matrix_image = imread(path_to_image, CV_LOAD_IMAGE_COLOR);
    if (!(this->matrix_image).data){
        cout << "Error reading image" << endl;
    }
}

void blurImage::conversion() {
    int cols = this->matrix_image.cols;
    int rows = this->matrix_image.rows;
    uchar *ucharmat = this->matrix_image.data;
    uchar* ucharupdated = new uchar[cols * rows];
    float* Blue = new float[cols * rows];
    float* Green = new float[cols * rows];
    float* Red = new float[cols * rows];
    float* BlurredR = new float[cols * rows];
    float* BlurredB = new float[cols * rows];
    float* BlurredG= new float[cols * rows];
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            int pos = cols * i + j;
            Blue[pos] = this->matrix_image.at<cv::Vec3b>(i, j)[0];
            Green[pos] = this->matrix_image.at<cv::Vec3b>(i, j)[1];
            Red[pos] = this->matrix_image.at<cv::Vec3b>(i, j)[2];
        }
    }
    blurimage(Red, Green, Blue, BlurredR, BlurredB, BlurredG ,cols ,rows);
    /*bluruchar(ucharmat, cols, rows, ucharupdated);
    Mat finalmat = Mat(rows, cols, CV_16UC3, ucharupdated);
    namedWindow("newwin", WINDOW_AUTOSIZE);
    imshow("newwi", finalmat);
    waitKey(0);*/
    //return;
    Mat gray = Mat(rows, cols, CV_32FC3);
    std::vector<cv::Mat> channels;
    Mat Rm(rows, cols, CV_32FC1, BlurredR);
    Mat Gm(rows, cols, CV_32FC1, BlurredG);
    Mat Bm(rows, cols, CV_32FC1, BlurredB);
    channels.push_back(Bm);
    channels.push_back(Gm);
    channels.push_back(Rm);
    cv::merge(channels, gray);

    imwrite("./blur.jpg", gray);
}
