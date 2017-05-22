#include "stdlib.h"
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

void Set_the_LM(vector<Point2f> input_LM,vector<Point2f> *output_LM);// Cnehra LM + Lab LM and plus eye LM //

void Set_the_LM_2(Mat Reg_LM_pick_data,vector<Point2f> *output_LM);

void push_data(Mat src,Mat &output,int count);// pca ¹w³B²z //

void dlib_LM_set(std::vector<cv::Point2f> dlib_in, std::vector<cv::Point2f>* dlib_out);