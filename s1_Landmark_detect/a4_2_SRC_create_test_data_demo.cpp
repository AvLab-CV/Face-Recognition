#include "stdlib.h"

#include "dlib/all/source.cpp"
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/gui_widgets.h>

#include <opencv2/opencv.hpp>
#include <opencv2/legacy/legacy.hpp>  
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <direct.h>
#include <algorithm>
#include <openGL-vs2010/glew.h>
#include <openGL-vs2010/freeglut.h>

#include "LandmarkDetect_v2.h"
#include "file_read_mine.h"
#include "LM_set.h"
#include "mine_imgProc.h"

#include <Windows.h> //using for Virtual-Key Codes ex: VK_ESCAPE

#include "GaborFR.h"
#include "Ill_Nor.h"
#include "SolveHomotopy.h"
#include "SRC_sort.h"

using namespace std;
using namespace cv;

string data_file_title="../../using_data/";//工作路徑
int main(int argc, char* argv[])
{	
	// train data save path //
	string light="06";
	//cout<<"input light : ";
	//cin>>light;
	string angle="L15_14_0";
	//cout<<"input angle : ";
	//cin>>angle;
	// F00_05_1 //
	// R90_24_0 R75_01_0 R60_20_0 R45_19_0 R30_04_1 R15_05_0 //
	// L90_11_0 L75_12_0 L60_09_0 L45_08_0 L30_13_0 L15_14_0 //
	string glass_model_type="no_glass_1";
	// glass_frame_full glass_frame_half glass_frame_none no_glass //
	string recog_data_sPath_title=data_file_title+"Test_Data_xml"+"/"+light+"/";
	_mkdir(recog_data_sPath_title.c_str());
	recog_data_sPath_title=data_file_title+"Test_Data_xml"+"/"+light+"/"+angle+"/";
	_mkdir(recog_data_sPath_title.c_str());
	recog_data_sPath_title=data_file_title+"Test_Data_xml"+"/"+light+"/"+angle+"/"+glass_model_type+"/";
	_mkdir(recog_data_sPath_title.c_str());
	string train_sPath=recog_data_sPath_title+"train"+"/";
	_mkdir(train_sPath.c_str());
	string test_sPath=recog_data_sPath_title+"test"+"/";
	_mkdir(test_sPath.c_str());

	// initial class Gabor //
	GaborFR Gabor;

	// load train image //
	string test_image_title=data_file_title+"Test_Data"+"/"+light+"/"+angle+"/"+glass_model_type+"/"+"test"+"/";
	vector<string> test_img_name;
	Load_insideFile_name(test_image_title,&test_img_name);
	double t = (double)getTickCount();
	for (int i = 0; i < test_img_name.size(); i++)
	{
		string test_img_path_in=test_image_title+test_img_name[i]+"/";
		vector<string> test_img_name_in;
		Load_insideFile_name(test_img_path_in,&test_img_name_in);

		Mat Data_Tar_test;
		vector<string> Name_Tar_test;

		test_sPath=recog_data_sPath_title+"test"+"/"+test_img_name[i]+"/";
		_mkdir(test_sPath.c_str());

		for (int j = 0; j < test_img_name_in.size(); j++)
		{
			string img_name_path=test_img_path_in+test_img_name_in[j]+"/";
			vector<string> img_name;
			Load_insideFile_name(img_name_path,&img_name);
			for (int k = 0; k < img_name.size(); k++)
			{
				string img_load_path=img_name_path+img_name[k];

				Mat train_img=imread(img_load_path,0);//灰階讀入
				//-----Illumiantion Normalize image by TT
				CvMat *Mat_TT_in = cvCreateMat(train_img.rows,train_img.cols,CV_8UC1);
				CvMat temp = train_img;
				cvCopy(&temp, Mat_TT_in);
				//Mat_TT_in = &train_img.operator CvMat();
				CvMat *Mat_TT_out = cvCreateMat(train_img.rows,train_img.cols,CV_32FC1);
				//cvShowImage("tt", Mat_TT_in);
				//waitKey(1);

				TT(Mat_TT_in,cvSize(train_img.cols,train_img.rows),Mat_TT_out,0);
				// show TT result //
				//cvShowImage("Mat_TT_in", Mat_TT_in);
				Mat Mat_TT_out_show = Mat(Mat_TT_out, true); //CvMat copy to Mat  
				cv::normalize(Mat_TT_out_show,Mat_TT_out_show,0,1,cv::NORM_MINMAX,-1);
				//imshow("Mat_TT_out_show",Mat_TT_out_show);waitKey(1);
				//cvShowImage("Mat_TT_out",Mat_TT_out);waitKey(0);

				//-----Feature Extract by Gabor filter
				Mat m_ori = Mat_TT_out_show.clone();//CvMat->Mat
				//Mat m_ori = Mat_TT_out;//CvMat->Mat
				Mat Gabor_out;
				Gabor.Gabor_compute(m_ori,Gabor_out,0);
				//imshow("Gabor_out",Gabor_out.reshape(1,Mat_TT_out_show.cols));waitKey(0);
				//cv::normalize(Gabor_out,Gabor_out,0,1,cv::NORM_MINMAX,-1);
				//FILE* fgfg;
				//fgfg=fopen("QUE.txt","w");
				///*cout<<Gabor_out.rows<<endl;
				//cout<<Gabor_out.cols<<endl;
				//cout<<Gabor_out.type()<<endl;*/
				//for (int ggg=0;ggg<6720;ggg++)
				//{
				//	double hhh=Gabor_out.at<float>(ggg);
				//	fprintf(fgfg,"%f\n",hhh);
				//}
				//fclose(fgfg);
				//dct
				//Mat img_dct = Mat( Mat_TT_out_show.rows, Mat_TT_out_show.cols, CV_32FC1);
				//Mat_TT_out_show.convertTo(img_dct, CV_32FC1);

				//resize(img_dct, img_dct, Size(cvRound(img_dct.cols/4),cvRound(img_dct.rows/4)));

				//Mat img_dct_freq;
				//dct(img_dct, img_dct_freq);
				//imshow("img_dct",img_dct);waitKey(1);
				//imshow("img_dct_freq",img_dct_freq);waitKey(1);

				//int start=cvRound(img_dct_freq.rows/8);
				//int use_start=start*1;

				//img_dct_freq.rowRange(use_start,img_dct_freq.rows)=0;
				//img_dct_freq.colRange(use_start,img_dct_freq.cols)=0;

				//cout<<img_name[k].substr(0,3)<<endl;
				//Mat comb_m((Gabor_out.cols*Gabor_out.rows+img_dct_freq.cols*img_dct_freq.rows),1,CV_32FC1);

				//for (int i = 0; i < Gabor_out.rows; i++)
				//{
				//	comb_m.at<float>(i,0)=Gabor_out.at<float>(i,0);
				//}
				//Mat temp_i=img_dct_freq.reshape(0,img_dct_freq.rows*img_dct_freq.cols);
				//for (int i = Gabor_out.rows; i < Gabor_out.rows+temp_i.rows; i++)
				//{
				//	comb_m.at<float>(i,0)=temp_i.at<float>(i-Gabor_out.rows,0);
				//}

				//-----store Regrister Data
				Data_Tar_test.push_back(Gabor_out);
				//Data_Tar_test.push_back(img_dct_freq.reshape(0,img_dct_freq.rows*img_dct_freq.cols));
				//Data_Tar_test.push_back(Mat_TT_out_show.reshape(0,Mat_TT_out_show.rows*Mat_TT_out_show.cols));
				//Data_Tar_test.push_back(comb_m);
				Name_Tar_test.push_back(img_name[k].substr(0,3));

				//-----Release Memory
				if( Mat_TT_in )
					cvDecRefData(&Mat_TT_in);
				if( Mat_TT_out )
					cvDecRefData(&Mat_TT_out);

			}
		}

		string name_xml=recog_data_sPath_title+"test"+"/"+test_img_name[i]+"/"+"name.xml";
		string data_xml=recog_data_sPath_title+"test"+"/"+test_img_name[i]+"/"+"data.xml";
		//-----Save the Data into xml
		FileStorage FS_NT;
		FS_NT.open(name_xml, FileStorage::WRITE);
		FS_NT << "Name_Tar" << Name_Tar_test;
		FS_NT.release();
		FileStorage FS_DT;
		FS_DT.open(data_xml, FileStorage::WRITE);
		FS_DT << "Data_Tar" << Data_Tar_test;
		FS_DT.release();
	}
	t = ((double)getTickCount() - t)/getTickFrequency();
	cout<<"total : "<< t <<" sec."<<endl;
	cout << '\a';
	//system("pause");
	return 0;
}