#include "Chehra_Linker.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace Chehra_Tracker;

void Chehra_Plot(Mat &img,Mat &pts,Mat &eyes, vector<Point2f>* outLMp)
{

	int i,n=49,m=14;
	Point p1; Scalar c;

	c = CV_RGB(0,255,0);

	vector<Point2f> temp;

	for(i = 0; i < 49; i++)
	{    
		p1 = Point(pts.at<float>(i,0),pts.at<float>(i+n,0));
		//cout<<"x : "<<pts.at<float>(i,0)<<"　y : "<<pts.at<float>(i+n,0)<<endl;
		//circle(img,p1,1,c,-1);
		temp.push_back(p1);
	}

	*outLMp=temp;
}

void Chehra_Plot_withDraw(Mat &img,Mat &pts,Mat &eyes, vector<Point2f>* outLMp)
{

	int i,n=49,m=14;
	Point p1; Scalar c;

	c = CV_RGB(0,255,0);

	vector<Point2f> temp;

	for(i = 0; i < 49; i++)
	{    
		p1 = Point(pts.at<float>(i,0),pts.at<float>(i+n,0));
		//cout<<"x : "<<pts.at<float>(i,0)<<"　y : "<<pts.at<float>(i+n,0)<<endl;
		circle(img,p1,1,c,-1);
		temp.push_back(p1);
	}

	*outLMp=temp;
}

void Chehra_Plot_camera(Mat &img,Mat &pts,Mat &eyes)
{

	int i,n=49,m=14;
	Point p1; Scalar c;

	c = CV_RGB(0,255,0);

	for(i = 0; i < 49; i++)
	{    
		p1 = Point(pts.at<float>(i,0),pts.at<float>(i+n,0));
		circle(img,p1,1,c,-1);
	}
	//cout<<"x : "<<pts.at<float>(13,0)<<"　y : "<<pts.at<float>(13+n,0)<<endl;
}

void Chehra_Plot_camera_no(Mat &img,Mat &pts,Mat &eyes) //不畫找到的點
{

	int i,n=49,m=14;
	Point p1; Scalar c;

	c = CV_RGB(0,255,0);

	for(i = 0; i < 49; i++)
	{    
		p1 = Point(pts.at<float>(i,0),pts.at<float>(i+n,0));
		//circle(img,p1,1,c,-1);
	}
	//cout<<"x : "<<pts.at<float>(13,0)<<"　y : "<<pts.at<float>(13+n,0)<<endl;
}