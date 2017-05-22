// This is the main function for TT-Normalization write by Ruby.
//
//#include "stdafx.h"
#include "opencv2/opencv.hpp"
#include "opencv2/legacy/compat.hpp"


CvMat Normalize8(CvMat *Mat1,CvMat *Mat4)
{
	// Mat1 input
	// Mat4 output

	// Max & Min
	double max_val; double min_val;
	CvPoint max_location; CvPoint min_location;
	cvMinMaxLoc(Mat1,&min_val,&max_val,&min_location,&max_location);

	double sub_data = (max_val) - (min_val); // だダ场だ计

	CvMat *Mat2 = cvCreateMat(Mat1->rows,Mat1->cols,CV_32FC1);
	cvSubS(Mat1,cvScalar(min_val),Mat2); // だ场だ (Mat2)

	CvMat *Mat3 = cvCreateMat(Mat1->rows,Mat1->cols,CV_32FC1);
	cvZero(Mat3);
	cvAddS(Mat3,cvScalar(sub_data),Mat3);// だダ场だ эΘ痻皚 (Mat3)

	// 埃
	cvDiv(Mat2,Mat3,Mat4,1);

	cvConvertScale(Mat4,Mat4,255,0);

	// 礚兵ン秈
	for (int i = 0 ; i < Mat4->rows ; i++) // row
	{
		for (int j = 0 ; j < Mat4->cols ; j++) // col
		{
			cvmSet(Mat4,i,j,ceil(cvmGet(Mat4,i,j)));
		}
	}
	cvReleaseMat(&Mat2);
	cvReleaseMat(&Mat3);
	return *Mat4;
}

CvMat robust_postprocessor(CvMat *Mat6,CvMat *Mat13)
{
	// Mat6 input
	// Mat13 output
	double alfa = 0.1;
	double tao = 10;
	double m1, m2, m3;

	// Part 1
	CvMat *Mat7 = cvCreateMat((Mat6->rows) * (Mat6->cols),1,CV_32FC1);

	int Row = 0;
	for (int j = 0 ; j < (Mat6->cols) ; j++)
	{
		for (int i = 0 ; i < (Mat6->rows) ; i++)
		{
			cvmSet(Mat7,Row,0,abs(cvmGet(Mat6,i,j)));
			Row = Row + 1;
		}
	}

	m1 = cvMean(Mat7);
	m2 = pow(m1,alfa);
	m3 = pow(m2,(1/alfa));

	CvMat *Mat8 = cvCreateMat(Mat6->rows,Mat6->cols,CV_32FC1);
	cvZero(Mat8);
	cvAddS(Mat8,cvScalar(m3),Mat8); // だダ场だ

	CvMat *Mat9 = cvCreateMat(Mat6->rows,Mat6->cols,CV_32FC1);
	cvDiv(Mat6,Mat8,Mat9,1); // Mat9 材场だ挡狦

	cvReleaseMat(&Mat7);
	cvReleaseMat(&Mat8);

	// Part 2
	//2-1
	CvMat *taoMat = cvCreateMat(1,(Mat6->rows) * (Mat6->cols),CV_32FC1);
	cvZero(taoMat);
	cvAddS(taoMat,cvScalar(tao),taoMat);

	//2-2
	CvMat *Mat9Mat = cvCreateMat(1,(Mat6->rows) * (Mat6->cols),CV_32FC1);
	int Col = 0;
	for (int j = 0 ; j < (Mat6->cols) ; j++)
	{
		for (int i = 0 ; i < (Mat6->rows) ; i++)
		{
			cvmSet(Mat9Mat,0,Col,abs(cvmGet(Mat9,i,j)));
			Col = Col + 1;
		}
	}

	//2-3
	CvMat *Mat10 = cvCreateMat(1,(Mat6->rows) * (Mat6->cols),CV_32FC1);
	cvMin(taoMat,Mat9Mat,Mat10);

	//2-4
	cvPow(Mat10,Mat10,alfa);

	//2-5
	double MeanMat10 = cvMean(Mat10);

	//だダ场だ
	double PowMeanMat10 = pow(MeanMat10,(1/alfa));
	CvMat *Mat11 = cvCreateMat(Mat6->rows,Mat6->cols,CV_32FC1);
	cvZero(Mat11);
	cvAddS(Mat11,cvScalar(PowMeanMat10),Mat11);

	CvMat *Mat12 = cvCreateMat(Mat6->rows,Mat6->cols,CV_32FC1);
	cvDiv(Mat9,Mat11,Mat12,1); // Mat12穝挡狦

	double MatData;
	for (int i = 0 ; i < (Mat13->rows) ; i++)
	{
		for (int j = 0 ; j < (Mat13->cols) ; j++)
		{
			MatData = cvmGet(Mat12,i,j);
			cvmSet(Mat13,i,j,tao*tanh(MatData/tao));
		}
	}
	cvReleaseMat(&Mat9);
	cvReleaseMat(&taoMat);
	cvReleaseMat(&Mat9Mat);
	cvReleaseMat(&Mat10);
	cvReleaseMat(&Mat11);
	cvReleaseMat(&Mat12);
	return *Mat13;
}

CvMat TT(CvMat *ResizeImage, CvSize ImageSize, CvMat *Result, int Type)
{
	// Image -> Matrix
	/*CvMat *ImageMatrix = cvCreateMat(ImageSize.height,ImageSize.width,CV_8UC1);
	cvGetMat(ResizeImage,ImageMatrix);*/

	// %% Gamma correction
	// %% (we could use 255*imadjust(X,[],[0,1],gamma), but would add dependencies
	// %% to the image processing toolbox); we use our implementation

	// Adjust Region [0 1]
	CvMat *Matrix = cvCreateMat(ImageSize.height,ImageSize.width,CV_32FC1);
	cvNormalize(ResizeImage,Matrix,1,0,CV_MINMAX);
	//cvReleaseMat(&ResizeImage);

	// Gamma Correction
	cv::Mat Matrix_temp(Matrix);
	cv::pow(Matrix_temp,0.2,Matrix_temp);
	CvMat temp = Matrix_temp; 
	cvCopy(&temp, Matrix);
	//cvPow(Matrix,Matrix,0.2);

	// Adjust Region [0 255]
	cvNormalize(Matrix,Matrix,255,0,CV_MINMAX);

	/*
	CvMat *GMat = cvCreateMat(ImageSize.height,ImageSize.width,CV_8UC1);
	cvConvertScale(Matrix,GMat,1,0);
	IplImage *GImage = cvCreateImage(ImageSize,IPL_DEPTH_8U,1);
	cvGetImage(GMat,GImage);
	cvNamedWindow("Gamma Correction");
	cvShowImage("Gamma Correction",GImage);
	*/

	//%% Dog filtering
	// Normalize 8
	CvMat *Normalize8Matrix = cvCreateMat(ImageSize.height,ImageSize.width,CV_32FC1);
	Normalize8(Matrix,Normalize8Matrix);
	cvReleaseMat(&Matrix);
	
	// Difference of Gaussian
	double Sigma1 = 0.5; double Sigma2 = 2;
	int GaussianSize1 = 2 * ceil( 3 * Sigma1 ) + 1; int GaussianSize2 = 2 * ceil( 3 * Sigma2 ) + 1;

	CvMat *DoG1 = cvCreateMat(ImageSize.height,ImageSize.width,CV_32FC1);
	CvMat *DoG2 = cvCreateMat(ImageSize.height,ImageSize.width,CV_32FC1);
	cvSmooth(Normalize8Matrix,DoG1,CV_GAUSSIAN,GaussianSize1,GaussianSize1,Sigma1);
	cvSmooth(Normalize8Matrix,DoG2,CV_GAUSSIAN,GaussianSize2,GaussianSize2,Sigma2);

	CvMat *DoG = cvCreateMat(ImageSize.height,ImageSize.width,CV_32FC1);
	cvSub(DoG1,DoG2,DoG);

	cvReleaseMat(&Normalize8Matrix);
	cvReleaseMat(&DoG1);
	cvReleaseMat(&DoG2);

	//%% Postprocessing
	// Robust Postprocessor
	CvMat *RPMat = cvCreateMat(ImageSize.height,ImageSize.width,CV_32FC1);
	robust_postprocessor(DoG,RPMat);
	cvReleaseMat(&DoG);

	// Normalization 8
	//CvMat *ResultOutput = cvCreateMat(ImageSize.height,ImageSize.width,CV_32FC1);
	if ( Type==1 )
	{
		Normalize8(RPMat,Result);
		cvReleaseMat(&RPMat);
	}
	else if (Type==0)
		cvCopy(RPMat,Result);

	return *Result;
}

//double cvMean(CvMat *Mat_m)
//{
//	double MatData;
//
//	for (int i = 0 ; i < (Mat_m->rows) ; i++)
//	{
//		for (int j = 0 ; j < (Mat_m->cols) ; j++)
//		{
//			MatData += cvmGet(Mat_m,i,j);			
//		}
//	}
//
//	MatData = MatData / (Mat_m->rows*Mat_m->cols);
//
//	return MatData;
//}