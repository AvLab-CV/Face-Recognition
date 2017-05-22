#include "stdlib.h"
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

void find_LM_center(vector<Point2f> input_LM, Point2f* center_out);
void move_img(Mat input_img, Mat* output_img, Point2f src_point, Point2f dst_point);
void move_img_ch1(Mat input_img, Mat* output_img, Point2f src_point, Point2f dst_point);
void move_LM_point(vector<Point2f> input_LM,vector<Point2f> *output_LM, Point2f src_point, Point2f dst_point);
void find_theate(vector<Point2f> src_LM,vector<Point2f> dst_LM, float* theate);
void find_theate_zero(vector<Point2f> src_LM, float* theate);
void find_theate2(vector<Point2f> src_LM,vector<Point2f> dst_LM, float* theate);
void rotate_LM(vector<Point2f> input_LM,vector<Point2f> *output_LM, float theate);
void find_scale(vector<Point2f> src_LM,vector<Point2f> dst_LM, float* scale);
void find_scale2(vector<Point2f> src_LM,vector<Point2f> dst_LM, float* scale);
void scale_LM(vector<Point2f> input_LM,vector<Point2f> *output_LM, float scale);