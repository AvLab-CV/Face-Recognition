#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>

#include "opencv.hpp"
#include <Windows.h> //using for Virtual-Key Codes ex: VK_ESCAPE

#include "file_read_mine.h"
#include "GaborFR.h"
#include "Ill_Nor.h"
#include "SolveHomotopy.h"
#include "SRC_sort.h"

using namespace std;
using namespace cv;

int main()
{
	string name_xml="../../using_data/src_test_data/name_index.xml";
	string data_xml="../../using_data/src_test_data/data_save.xml";

	// load train image //
	GaborFR Gabor;// initial class Gabor
	string train_img_path="../../using_data/src_test_data/train/003/";
	vector<string> train_img_name;
	Load_insideFile_name(train_img_path,&train_img_name);

	Mat Data_Tar_regist;
	for (vector<string>::iterator in=train_img_name.begin(); in!=train_img_name.end(); ++in)
	{
		string train_img_path_in=train_img_path+*in+"/";
		vector<string> train_img_name_in;
		Load_insideFile_name(train_img_path_in,&train_img_name_in);
		for (vector<string>::iterator in_in=train_img_name_in.begin(); in_in!=train_img_name_in.end(); ++in_in)
		{
			string train_img_name_total=train_img_path_in+*in_in;
			cout<<train_img_name_total<<endl;

			Mat train_img=imread(train_img_name_total,0);//¦Ç¶¥Åª¤J
			//-----Illumiantion Normalize image by TT
			CvMat *Mat_TT_in = cvCreateMat(train_img.rows,train_img.cols,CV_8UC1);
			CvMat temp = train_img;
			cvCopy(&temp, Mat_TT_in);
			//Mat_TT_in = &train_img.operator CvMat();
			CvMat *Mat_TT_out = cvCreateMat(train_img.rows,train_img.cols,CV_32FC1);
			//cvShowImage("tt", Mat_TT_in);
			//waitKey(1);

			TT(Mat_TT_in,cvSize(train_img.cols,train_img.rows),Mat_TT_out,0);
			
			//-----Feature Extract by Gabor filter
			Mat m_ori = Mat_TT_out;//CvMat->Mat
			Mat Gabor_out;
			Gabor.Gabor_compute(m_ori,Gabor_out,0);
			
			//-----store Regrister Data
			Data_Tar_regist.push_back(Gabor_out.reshape(0,1));

			//-----Release Memory
			if( Mat_TT_in )
				cvDecRefData(&Mat_TT_in);
			if( Mat_TT_out )
				cvDecRefData(&Mat_TT_out);
		}
	}
	//-----Save the Data into xml
	FileStorage FS_NT;
	FS_NT.open(name_xml, FileStorage::WRITE);
	FS_NT << "Name_Tar" << train_img_name;
	FS_NT.release();
	FileStorage FS_DT;
	FS_DT.open(data_xml, FileStorage::WRITE);
	FS_DT << "Data_Tar" << Data_Tar_regist;
	FS_DT.release();

	//--Sparse Coefficient
	double sparsity = 0.1; //--create sparse X 
	double tol = 0.01;//0.01 //--sparse homotopy tolerance(control)
	int iter = 0;//iteration

	double lambda = 0.000001;//--homotopy of lambda(control)
	int maxIter = 80;//--max iter number
	int n,m;
	int n_b,m_b;

	double normB, normX0, normA;
	normB  = 0;
	normX0 = 0;

	int FR_SRC = 0;
	int Total_TestNum = 0;

	GaborFR Gabor_test;

	//-----Read the Data from xml
	Mat Data_Tar_regist_fotTest;
	vector<string> train_img_name_fotTest;
	FileStorage FS_NT_test;
	FS_NT_test.open(name_xml, FileStorage::READ);
	FS_NT_test["Name_Tar"] >> train_img_name_fotTest;
	FS_NT_test.release();
	FileStorage FS_DT_test;
	FS_DT_test.open(data_xml, FileStorage::READ);
	FS_DT_test["Data_Tar"] >> Data_Tar_regist_fotTest;
	FS_DT_test.release();

	//---------------------Construct the Target matrix of A---------------------//
	m = Data_Tar_regist_fotTest.cols,n = Data_Tar_regist_fotTest.rows;
	double *A  = new double[m*n];

	for(int i = 0 ; i < m; i++)
	{
		normA  = 0;
		for(int j = 0 ; j < n; j++)
		{
			A[j*m + i] = Data_Tar_regist_fotTest.at<float>(j,i); //--put the target data
			normA += A[j*m + i]*A[j*m + i];
		}
		normA = sqrt(normA);
		for(int j = 0 ; j < n; j++)
		{
			A[j*m + i] = (double)(A[j*m + i] / normA);
		}
	}

	// test image load //
	string test_img_path="../../using_data/src_test_data/test/204/";
	vector<string> test_img_name;
	Load_insideFile_name(test_img_path,&test_img_name);
	for (vector<string>::iterator in=test_img_name.begin(); in!=test_img_name.end(); ++in)
	{
		string test_img_path_in=test_img_path+*in+"/";
		vector<string> test_img_name_in;
		Load_insideFile_name(test_img_path_in,&test_img_name_in);
		for (vector<string>::iterator in_in=test_img_name_in.begin(); in_in!=test_img_name_in.end(); ++in_in)
		{
			string test_img_name_total=test_img_path_in+*in_in;
			cout<<test_img_name_total<<endl;

			Mat test_img=imread(test_img_name_total,0);//¦Ç¶¥Åª¤J
			//-----Illumiantion Normalize image by TT
			CvMat *Mat_TT_in = cvCreateMat(test_img.rows,test_img.cols,CV_8UC1);
			CvMat temp = test_img;
			cvCopy(&temp, Mat_TT_in);
			CvMat *Mat_TT_out = cvCreateMat(test_img.rows,test_img.cols,CV_32FC1);

			TT(Mat_TT_in,cvSize(test_img.cols,test_img.rows),Mat_TT_out,0);
			//-----Feature Extract by Gabor filter
			Mat m_ori = Mat_TT_out;//CvMat->Mat
			Mat Gabor_out;
			Gabor_test.Gabor_compute(m_ori,Gabor_out);

			//----B---//
			m_b = Gabor_out.rows;
			double *b  = new double[m_b];
			for(int j = 0 ; j < m_b; j++)
			{
				b[j] = Gabor_out.at<float>(j,0) ;
			}
			for(int j = 0; j < m_b; j++)
			{
				normB += b[j]*b[j];
			}
			normB = sqrt(normB);
			for(int j = 0 ; j < m_b; j++)
			{
				b[j] = (double)(b[j]/normB);
			}

			double *x_predict  = new double[n];

			//----SRC----//
			SolveHomotopy(x_predict, iter, b, A, tol, lambda, maxIter, m, n);

			//string char_str(regist_name);
			string char_str="test";
			FR_SRC += SRC_Sort_ShowResult(x_predict,train_img_name_fotTest,n,char_str);

			//Release Memory
			delete [] b;
			delete [] x_predict;
			delete [] A;
		}
	}


	system("pause");
	return 0;
}

