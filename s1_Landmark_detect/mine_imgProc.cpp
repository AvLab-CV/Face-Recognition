#include "stdlib.h"
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

// image process //
void find_LM_center(vector<Point2f> input_LM, Point2f* center_out)
{
	Point2f sum_point(0,0);
	int total_point_num=0;
	for (int i=0;i<input_LM.size();i++)
	{
		sum_point=sum_point+input_LM[i];
		total_point_num++;
	}
	sum_point.x=sum_point.x/total_point_num;
	sum_point.y=sum_point.y/total_point_num;

	*center_out=sum_point;
}
void move_img(Mat input_img, Mat* output_img, Point2f src_point, Point2f dst_point)
{
	// 將測試影像移至Model影像位置 //
	// input_img : 原測試影像
	// output_img : 平移後的測試影像
	// dst_point : 測試影像LM 中心點
	// src_point : Model影像LM 中心點

	float move_x=-(dst_point.x-src_point.x);
	float move_y=-(dst_point.y-src_point.y);
	Mat temp_img=input_img.clone();
	temp_img.setTo(0);

	vector<Mat> input_img_s;
	vector<Mat> temp_img_s;

	split(input_img,input_img_s);
	split(temp_img,temp_img_s);

	for (int i=0;i<input_img.rows;i++)
	{
		for (int j=0;j<input_img.cols;j++)
		{
			if (i+move_y>0 && j+move_x>0)
			{
				if (i+move_y<input_img.rows && j+move_x<input_img.cols)
				{
					temp_img_s[0].at<uchar>(i+move_y,j+move_x)=input_img_s[0].at<uchar>(i,j);
					temp_img_s[1].at<uchar>(i+move_y,j+move_x)=input_img_s[1].at<uchar>(i,j);
					temp_img_s[2].at<uchar>(i+move_y,j+move_x)=input_img_s[2].at<uchar>(i,j);
				}
			}
		}
	}

	merge(temp_img_s,temp_img);
	*output_img=temp_img;
	//imshow("temp_img",temp_img);
	//waitKey(0);
}
void move_img_ch1(Mat input_img, Mat* output_img, Point2f src_point, Point2f dst_point)
{
	// 將測試影像移至Model影像位置 //
	// input_img : 原測試影像
	// output_img : 平移後的測試影像
	// dst_point : 測試影像LM 中心點
	// src_point : Model影像LM 中心點

	float move_x=-(dst_point.x-src_point.x);
	float move_y=-(dst_point.y-src_point.y);
	Mat temp_img=input_img.clone();
	temp_img.setTo(0);

	Mat img_in=input_img.clone();
	Mat img_temp=input_img.clone();img_temp.setTo(0);

	for (int i=0;i<input_img.rows;i++)
	{
		for (int j=0;j<input_img.cols;j++)
		{
			if (i+move_y>0 && j+move_x>0)
			{
				if (i+move_y<input_img.rows && j+move_x<input_img.cols)
				{
					img_temp.at<uchar>(i+move_y,j+move_x)=img_in.at<uchar>(i,j);
				}
			}
		}
	}


	*output_img=img_temp;
	//imshow("temp_img",temp_img);
	//waitKey(0);
}
void move_LM_point(vector<Point2f> input_LM,vector<Point2f> *output_LM, Point2f src_point, Point2f dst_point)
{
	float move_x=-(dst_point.x-src_point.x);
	float move_y=-(dst_point.y-src_point.y);

	vector<Point2f> temp_LM;
	for (int i=0;i<input_LM.size();i++)
	{
		Point2f temp=Point2f(input_LM[i].x+move_x,input_LM[i].y+move_y);
		temp_LM.push_back(temp);
	}
	*output_LM=temp_LM;
}
void find_theate(vector<Point2f> src_LM,vector<Point2f> dst_LM, float* theate)
{
	float src_slope=(src_LM[20-1].y-src_LM[29-1].y)/(src_LM[20-1].x-src_LM[29-1].x);
	float dst_slope=(dst_LM[20-1].y-dst_LM[29-1].y)/(dst_LM[20-1].x-dst_LM[29-1].x);

	//cout<<src_slope<<endl;
	//cout<<dst_slope<<endl;

	double angle_1=(src_slope-dst_slope)/(1+src_slope*dst_slope);
	double angle_2=-(src_slope-dst_slope)/(1+src_slope*dst_slope);

	double angle_arct_1=atan(angle_1);
	double angle_arct_2=atan(angle_2);

	angle_arct_1=angle_arct_1*180/3.141592;
	angle_arct_2=angle_arct_2*180/3.141592;

	//cout<<"angle_arct_1 : "<<angle_arct_1<<endl;
	//cout<<"angle_arct_2 : "<<angle_arct_2<<endl;
	if (src_slope > dst_slope)
	{
		*theate=(angle_arct_1>angle_arct_2)?angle_arct_1:angle_arct_2;
		//cout<<"theate : "<<*theate<<endl;
	} 
	else
	{
		*theate=(angle_arct_1<angle_arct_2)?angle_arct_1:angle_arct_2;
		//cout<<"theate : "<<*theate<<endl;
	}
}
void find_theate_zero(vector<Point2f> src_LM, float* theate)
{
	
	float src_slope=(src_LM[20-1].y-src_LM[29-1].y)/(src_LM[20-1].x-src_LM[29-1].x);
	float dst_slope=0.0;

	//cout<<src_slope<<endl;
	//cout<<dst_slope<<endl;

	double angle_1=(src_slope-dst_slope)/(1+src_slope*dst_slope);
	double angle_2=-(src_slope-dst_slope)/(1+src_slope*dst_slope);

	double angle_arct_1=atan(angle_1);
	double angle_arct_2=atan(angle_2);

	angle_arct_1=angle_arct_1*180/3.141592;
	angle_arct_2=angle_arct_2*180/3.141592;

	//cout<<"angle_arct_1 : "<<angle_arct_1<<endl;
	//cout<<"angle_arct_2 : "<<angle_arct_2<<endl;
	if (src_slope > dst_slope)
	{
		*theate=(angle_arct_1>angle_arct_2)?angle_arct_1:angle_arct_2;
		//cout<<"theate : "<<*theate<<endl;
	} 
	else
	{
		*theate=(angle_arct_1<angle_arct_2)?angle_arct_1:angle_arct_2;
		//cout<<"theate : "<<*theate<<endl;
	}
}
void find_theate2(vector<Point2f> src_LM,vector<Point2f> dst_LM, float* theate)
{
	float src_slope=(src_LM[6-1].y-src_LM[12-1].y)/(src_LM[6-1].x-src_LM[12-1].x);
	float dst_slope=(dst_LM[6-1].y-dst_LM[12-1].y)/(dst_LM[6-1].x-dst_LM[12-1].x);

	//cout<<src_slope<<endl;
	//cout<<dst_slope<<endl;

	double angle_1=(src_slope-dst_slope)/(1+src_slope*dst_slope);
	double angle_2=-(src_slope-dst_slope)/(1+src_slope*dst_slope);

	double angle_arct_1=atan(angle_1);
	double angle_arct_2=atan(angle_2);

	angle_arct_1=angle_arct_1*180/3.141592;
	angle_arct_2=angle_arct_2*180/3.141592;

	//cout<<"angle_arct_1 : "<<angle_arct_1<<endl;
	//cout<<"angle_arct_2 : "<<angle_arct_2<<endl;
	if (src_slope > dst_slope)
	{
		*theate=(angle_arct_1>angle_arct_2)?angle_arct_1:angle_arct_2;
		//cout<<"theate : "<<*theate<<endl;
	} 
	else
	{
		*theate=(angle_arct_1<angle_arct_2)?angle_arct_1:angle_arct_2;
		//cout<<"theate : "<<*theate<<endl;
	}
}
void rotate_LM(vector<Point2f> input_LM,vector<Point2f> *output_LM, float theate)
{
	float cos_a=cos(theate*3.141592/180);
	float sin_a=sin(theate*3.141592/180);

	Point2f center_out;
	find_LM_center(input_LM, &center_out);

	vector<Point2f> temp;
	for (int i=0;i<input_LM.size();i++)
	{
		input_LM[i].x=input_LM[i].x-center_out.x;
		input_LM[i].y=input_LM[i].y-center_out.y;

		float ii=cos_a*input_LM[i].x+sin_a*input_LM[i].y;
		float jj=-sin_a*input_LM[i].x+cos_a*input_LM[i].y;

		ii=ii+center_out.x;
		jj=jj+center_out.y;
		Point2f temp_tt=Point2f(ii,jj);
		temp.push_back(temp_tt);
	}

	*output_LM=temp;
}
void find_scale(vector<Point2f> src_LM,vector<Point2f> dst_LM, float* scale)
{
	//Point2f Leye=Point2f((src_LM[20-1].x+src_LM[23-1].x)/2,(src_LM[20-1].y+src_LM[23-1].y)/2);
	//Point2f Reye=Point2f((src_LM[26-1].x+src_LM[29-1].x)/2,(src_LM[26-1].y+src_LM[29-1].y)/2);
	Point2f Leye=Point2f((src_LM[20-1].x+src_LM[21-1].x+src_LM[22-1].x+src_LM[23-1].x+src_LM[24-1].x+src_LM[25-1].x)/6,(src_LM[20-1].y+src_LM[21-1].y+src_LM[22-1].y+src_LM[23-1].y+src_LM[24-1].y+src_LM[25-1].y)/6);
	Point2f Reye=Point2f((src_LM[26-1].x+src_LM[27-1].x+src_LM[28-1].x+src_LM[29-1].x+src_LM[30-1].x+src_LM[31-1].x)/6,(src_LM[26-1].y+src_LM[27-1].y+src_LM[28-1].y+src_LM[29-1].y+src_LM[30-1].y+src_LM[31-1].y)/6);
	float src_distance=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));

	//Leye=Point2f((dst_LM[20-1].x+dst_LM[23-1].x)/2,(dst_LM[20-1].y+dst_LM[23-1].y)/2);
	//Reye=Point2f((dst_LM[26-1].x+dst_LM[29-1].x)/2,(dst_LM[26-1].y+dst_LM[29-1].y)/2);
	Leye=Point2f((dst_LM[20-1].x+dst_LM[21-1].x+dst_LM[22-1].x+dst_LM[23-1].x+dst_LM[24-1].x+dst_LM[25-1].x)/6,(dst_LM[20-1].y+dst_LM[21-1].y+dst_LM[22-1].y+dst_LM[23-1].y+dst_LM[24-1].y+dst_LM[25-1].y)/6);
	Reye=Point2f((dst_LM[26-1].x+dst_LM[27-1].x+dst_LM[28-1].x+dst_LM[29-1].x+dst_LM[30-1].x+dst_LM[31-1].x)/6,(dst_LM[26-1].y+dst_LM[27-1].y+dst_LM[28-1].y+dst_LM[29-1].y+dst_LM[30-1].y+dst_LM[31-1].y)/6);
	float dst_distance=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));

	//float src_distance=sqrt((src_LM[26-1].x-src_LM[23-1].x)*(src_LM[26-1].x-src_LM[23-1].x)+(src_LM[26-1].y-src_LM[23-1].y)*(src_LM[26-1].y-src_LM[23-1].y));
	//float dst_distance=sqrt((dst_LM[26-1].x-dst_LM[23-1].x)*(dst_LM[26-1].x-dst_LM[23-1].x)+(dst_LM[26-1].y-dst_LM[23-1].y)*(dst_LM[26-1].y-dst_LM[23-1].y));

	*scale=dst_distance/src_distance;
}
void find_scale2(vector<Point2f> src_LM,vector<Point2f> dst_LM, float* scale)
{
	float src_distance=sqrt((src_LM[6-1].x-src_LM[12-1].x)*(src_LM[6-1].x-src_LM[12-1].x)+(src_LM[6-1].y-src_LM[12-1].y)*(src_LM[6-1].y-src_LM[12-1].y));
	float dst_distance=sqrt((dst_LM[6-1].x-dst_LM[12-1].x)*(dst_LM[6-1].x-dst_LM[12-1].x)+(dst_LM[6-1].y-dst_LM[12-1].y)*(dst_LM[6-1].y-dst_LM[12-1].y));

	*scale=dst_distance/src_distance;
}
void scale_LM(vector<Point2f> input_LM,vector<Point2f> *output_LM, float scale)
{
	Point2f center_out;
	find_LM_center(input_LM, &center_out);

	vector<Point2f> temp;
	for (int i=0;i<input_LM.size();i++)
	{
		//input_LM[i].x=input_LM[i].x-center_out.x;
		//input_LM[i].y=input_LM[i].y-center_out.y;

		float ii=input_LM[i].x*scale;
		float jj=input_LM[i].y*scale;

		//ii=ii+center_out.x;
		//jj=jj+center_out.y;
		Point2f temp_tt=Point2f(ii,jj);
		temp.push_back(temp_tt);
	}

	*output_LM=temp;
}