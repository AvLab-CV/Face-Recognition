#include "Chehra_Linker.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace Chehra_Tracker;

void Chehra_Plot(Mat &img,Mat &pts,Mat &eyes, vector<Point2f>* outLMp);//不畫LM,只回傳LM//
void Chehra_Plot_withDraw(Mat &img,Mat &pts,Mat &eyes, vector<Point2f>* outLMp);//畫LM,只回傳LM//
void Chehra_Plot_camera(Mat &img,Mat &pts,Mat &eyes);//畫LM,不回傳LM//
void Chehra_Plot_camera_no(Mat &img,Mat &pts,Mat &eyes);//不畫LM,不回傳LM//