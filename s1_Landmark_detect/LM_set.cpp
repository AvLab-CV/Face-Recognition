#include "stdlib.h"
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;


void Set_the_LM(vector<Point2f> input_LM,vector<Point2f> *output_LM)
{
	vector<Point2f> temp = input_LM;

	temp[54-1].x=temp[53-1].x;
	temp[54-1].y=temp[20-1].y;

	temp[58-1].x=temp[57-1].x;
	temp[58-1].y=temp[29-1].y;

	temp[50-1].x=temp[41-1].x;

	//註冊人左眼下對稱點
	float Mlm = (input_LM[20-1].y-input_LM[23-1].y)/(input_LM[20-1].x-input_LM[23-1].x);
	float Mla = -Mlm;
	float Mlb = 1.03;
	float Mlc = -(input_LM[20-1].y-(Mlm*input_LM[23-1].x));
	for (int i = 1; i <= 5; i++)
	{
		Point2f temp_tt=cv::Point2f((input_LM[i-1].x-(2*Mla)*((Mla*input_LM[i-1].x+Mlb*input_LM[i-1].y+Mlc)/(Mla*Mla+Mlb*Mlb))),(input_LM[i-1].y-(2*Mlb)*((Mla*input_LM[i-1].x+Mlb*input_LM[i-1].y+Mlc)/(Mla*Mla+Mlb*Mlb))));
		temp.push_back(temp_tt);
	}

	//註冊人右眼下對稱點
	float Mrm = (input_LM[26-1].y-input_LM[29-1].y)/(input_LM[26-1].x-input_LM[29-1].x);
	float Mra = -Mrm;
	float Mrb = 1.03;
	float Mrc = -(input_LM[26-1].y-(Mrm*input_LM[29-1].x));
	for (int i = 6; i <= 10; i++)
	{
		Point2f temp_tt=cv::Point2f((input_LM[i-1].x-(2*Mra)*((Mra*input_LM[i-1].x+Mrb*input_LM[i-1].y+Mrc)/(Mra*Mra+Mrb*Mrb))),(input_LM[i-1].y-(2*Mrb)*((Mra*input_LM[i-1].x+Mrb*input_LM[i-1].y+Mrc)/(Mra*Mra+Mrb*Mrb))));
		temp.push_back(temp_tt);
	}	

	*output_LM=temp;
}

void Set_the_LM_2(Mat Reg_LM_pick_data,vector<Point2f> *output_LM)
{
	vector<Point2f> temp;
	int Mask_point_glass_mid[]={17,15,19,11,23,24,21,20,1,3,5,26,31,28,29,10,8,6};//17,15,19,23,20,26,29
	vector<int> number(Mask_point_glass_mid, Mask_point_glass_mid + sizeof(Mask_point_glass_mid)/sizeof(Mask_point_glass_mid[0]));
	int size_number=number.size();
	for (int i = 0; i < size_number; i++)
	{
		int m=cvRound(Reg_LM_pick_data.at<float>(0,number[i]-1));
		int n=cvRound(Reg_LM_pick_data.at<float>(1,number[i]-1));
		temp.push_back(Point2f(m,n));
	}

	*output_LM=temp;
}

// for pca //
void push_data(Mat src,Mat &output,int count)
{
	//將資料38*38轉換成1*1444
	int c=0;
	for (int i=0;i<src.rows;i++)
	{
		for(int j=0;j<src.cols;j++)
		{
			output.at<float>(count,c)=src.at<uchar>(i,j);
			c++;
		}
	}
}

void dlib_LM_set(std::vector<cv::Point2f> dlib_in, std::vector<cv::Point2f>* dlib_out)
{
	std::vector<cv::Point2f> temp_vec;
	cv::Point2f temp;

	// 1 //
	temp=dlib_in[18-1];
	temp_vec.push_back(temp);
	// 2 //
	temp=dlib_in[19-1];
	temp_vec.push_back(temp);
	// 3 //
	temp=dlib_in[20-1];
	temp_vec.push_back(temp);
	// 4 //
	temp=dlib_in[21-1];
	temp_vec.push_back(temp);
	// 5 //
	temp=dlib_in[22-1];
	temp_vec.push_back(temp);
	// 6 //
	temp=dlib_in[23-1];
	temp_vec.push_back(temp);
	// 7 //
	temp=dlib_in[24-1];
	temp_vec.push_back(temp);
	// 8 //
	temp=dlib_in[25-1];
	temp_vec.push_back(temp);
	// 9 //
	temp=dlib_in[26-1];
	temp_vec.push_back(temp);
	// 10 //
	temp=dlib_in[27-1];
	temp_vec.push_back(temp);

	// 11 //
	temp=dlib_in[28-1];
	temp_vec.push_back(temp);
	// 12 //
	temp=dlib_in[29-1];
	temp_vec.push_back(temp);
	// 13 //
	temp=dlib_in[30-1];
	temp_vec.push_back(temp);
	// 14 //
	temp=dlib_in[31-1];
	temp_vec.push_back(temp);
	// 15 //
	temp=dlib_in[32-1];
	temp_vec.push_back(temp);
	// 16 //
	temp=dlib_in[33-1];
	temp_vec.push_back(temp);
	// 17 //
	temp=dlib_in[34-1];
	temp_vec.push_back(temp);
	// 18 //
	temp=dlib_in[35-1];
	temp_vec.push_back(temp);
	// 19 //
	temp=dlib_in[36-1];
	temp_vec.push_back(temp);

	// 20 //
	temp=dlib_in[37-1];
	temp_vec.push_back(temp);
	// 21 //
	temp=dlib_in[38-1];
	temp_vec.push_back(temp);
	// 22 //
	temp=dlib_in[39-1];
	temp_vec.push_back(temp);
	// 23 //
	temp=dlib_in[40-1];
	temp_vec.push_back(temp);
	// 24 //
	temp=dlib_in[41-1];
	temp_vec.push_back(temp);
	// 25 //
	temp=dlib_in[42-1];
	temp_vec.push_back(temp);

	// 26 //
	temp=dlib_in[43-1];
	temp_vec.push_back(temp);
	// 27 //
	temp=dlib_in[44-1];
	temp_vec.push_back(temp);
	// 28 //
	temp=dlib_in[45-1];
	temp_vec.push_back(temp);
	// 29 //
	temp=dlib_in[46-1];
	temp_vec.push_back(temp);
	// 30 //
	temp=dlib_in[47-1];
	temp_vec.push_back(temp);
	// 31 //
	temp=dlib_in[48-1];
	temp_vec.push_back(temp);

	// 32 //
	temp=dlib_in[49-1];
	temp_vec.push_back(temp);
	// 33 //
	temp=dlib_in[50-1];
	temp_vec.push_back(temp);
	// 34 //
	temp=dlib_in[51-1];
	temp_vec.push_back(temp);
	// 35 //
	temp=dlib_in[52-1];
	temp_vec.push_back(temp);
	// 36 //
	temp=dlib_in[53-1];
	temp_vec.push_back(temp);
	// 37 //
	temp=dlib_in[54-1];
	temp_vec.push_back(temp);
	// 38 //
	temp=dlib_in[55-1];
	temp_vec.push_back(temp);
	// 39 //
	temp=dlib_in[56-1];
	temp_vec.push_back(temp);
	// 40 //
	temp=dlib_in[57-1];
	temp_vec.push_back(temp);
	// 41 //
	temp=dlib_in[58-1];
	temp_vec.push_back(temp);
	// 42 //
	temp=dlib_in[59-1];
	temp_vec.push_back(temp);
	// 43 //
	temp=dlib_in[60-1];
	temp_vec.push_back(temp);

	// 44 //
	temp=dlib_in[62-1];
	temp_vec.push_back(temp);
	// 45 //
	temp=dlib_in[63-1];
	temp_vec.push_back(temp);
	// 46 //
	temp=dlib_in[64-1];
	temp_vec.push_back(temp);
	// 47 //
	temp=dlib_in[66-1];
	temp_vec.push_back(temp);
	// 48 //
	temp=dlib_in[67-1];
	temp_vec.push_back(temp);
	// 49 //
	temp=dlib_in[68-1];
	temp_vec.push_back(temp);

	// 50 //
	temp=dlib_in[9-1];
	temp_vec.push_back(temp);
	// 51 //
	temp=dlib_in[7-1];
	temp_vec.push_back(temp);
	// 52 //
	temp=dlib_in[5-1];
	temp_vec.push_back(temp);
	// 53 //
	temp=dlib_in[3-1];
	temp_vec.push_back(temp);
	// 54 //
	temp=dlib_in[1-1];
	temp_vec.push_back(temp);
	// 55 //
	temp=dlib_in[11-1];
	temp_vec.push_back(temp);
	// 56 //
	temp=dlib_in[13-1];
	temp_vec.push_back(temp);
	// 57 //
	temp=dlib_in[15-1];
	temp_vec.push_back(temp);
	// 58 //
	temp=dlib_in[17-1];
	temp_vec.push_back(temp);

	// 59 //
	temp=dlib_in[2-1];
	temp_vec.push_back(temp);
	// 60 //
	temp=dlib_in[4-1];
	temp_vec.push_back(temp);
	// 61 //
	temp=dlib_in[6-1];
	temp_vec.push_back(temp);
	// 62 //
	temp=dlib_in[8-1];
	temp_vec.push_back(temp);
	// 63 //
	temp=dlib_in[10-1];
	temp_vec.push_back(temp);
	// 64 //
	temp=dlib_in[12-1];
	temp_vec.push_back(temp);
	// 65 //
	temp=dlib_in[14-1];
	temp_vec.push_back(temp);
	// 66 //
	temp=dlib_in[16-1];
	temp_vec.push_back(temp);

	// 67 //
	temp.x=dlib_in[28-1].x;
	temp.y=(dlib_in[22-1].y+dlib_in[23-1].y)/2;
	temp_vec.push_back(temp);

	*dlib_out=temp_vec;
}
