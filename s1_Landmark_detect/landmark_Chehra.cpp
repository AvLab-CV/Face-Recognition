// test chehra landmark model //
#include "stdlib.h"
#include "Chehra_Linker.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <dirent.h>

#include "LandmarkDetect_v2.h"
#include "Chehra_draw.h"
#include "file_read_mine.h"

using namespace std;
using namespace cv;
using namespace Chehra_Tracker;

// Chehra landmark //
#define use_Chehra 1
// Lab face landmark
#define use_Lab_LM_model 1

#define use_camera 1

// Load file name //
//void Load_insideFile_name(string input_path, vector<string>* output_path);//讀file內所有檔名//

int main()
{
	// set Chehra LM //
#if use_Chehra
	//set other tracking parameters

	//Failure Checker Interval <integer>
	int fcheck_interval=5; //Failure checker to be executed after every 5 frames

	//Failure Checker Score Treshold [-1,1]
	float fcheck_score_treshold=-0.25;	//threshold for detecting tracking failure

	//Number of Consecutive Frames Failed To Activate Redetection
	int fcheck_fail_treshold=2;	//reinitialize after failure on 2 consecutive frames

	//models
	char regFile[256],fdFile[256];
	strcpy(regFile,"../using_data/Chehra-model/Chehra_t1.0.model");
	strcpy(fdFile,"../using_data/Chehra-model/haarcascade_frontalface_alt_tree.xml");

	//loading opencv's face detector model
	CascadeClassifier face_cascade;
	if(!face_cascade.load(fdFile))
	{
		cout<<"--(!)Error Loading Face Detector Model\n";
		return -1;
	}

	//load chehra model
	std::auto_ptr<Chehra_Tracker::Chehra_Linker> pModel(Chehra_Tracker::CreateChehraLinker(regFile));
	Chehra_Tracker::Chehra_Linker &ChehraObj = *pModel.get();
#endif // use_Chehra

	// set Lab face LM model //
#if use_Lab_LM_model
	int posemap[18],temp=90;
	string Lab_LM_title_path = "../using_data/";
	string face_LM_model_path=Lab_LM_title_path+"LM_Model"+"/"+"MPIE_glasses_v2_final.mat";

	LandmarkDetector detector; //宣告model格式
	detector.Load(face_LM_model_path.c_str()); //讀取model
	//printf("model : %s\n",face_LM_model_path_FRGC);

	// set up the threshold //
	detector.model_->thresh_=-0.8;

	// define the mapping from view-specific mixture id to viewpoint //
	if (detector.model_->components_.size()==13)
		for (int i=0;i<13;i++)
		{
			posemap[i]=temp;
			temp-=15;
		}
	else if (detector.model_->components_.size()==18)
		for (int i=0;i<18;i++)
		{
			if (i>5 && i<11){
				posemap[i]=0;}
			else{
				posemap[i]=temp;
				temp-=15;}
		}
	else if (detector.model_->components_.size()==1)
		posemap[0]=0;
	else
		printf("Can not recognize this model");

#endif

#if use_camera
	//camera set//
	VideoCapture src;// open the default camera
	printf("Connect to camera...\n");
	src.open(0);
	if(!src.isOpened())// check if we succeeded
	{
		printf("not found\n");
		system("pause");
		return -1;
	}
	src.set(CV_CAP_PROP_FRAME_HEIGHT,480);
	src.set(CV_CAP_PROP_FRAME_WIDTH,640);

	Mat camera_img_ori,camera_img_ori_g,camera_img_ori_t;
	int inputKey;
	while(1)
	{
		if( !src.grab() )
			break;
		src.retrieve(camera_img_ori);
		cvtColor(camera_img_ori,camera_img_ori_g,CV_BGR2GRAY);
		vector<Point2f> LM_each;
		if (ChehraObj.TrackFrame(camera_img_ori_g,fcheck_interval,fcheck_score_treshold,fcheck_fail_treshold,face_cascade) == 0)
		{
			//Chehra_Plot(camera_img_ori,ChehraObj._bestFaceShape,ChehraObj._bestEyeShape,&LM_each);
			Chehra_Plot_camera_no(camera_img_ori,ChehraObj._bestFaceShape,ChehraObj._bestEyeShape);

			// LM detect progess //
			vector<Data_bs>bs;vector<Data_bs> top;int point;
			float scale_LM_img=0.4;
			resize(camera_img_ori,camera_img_ori_t, Size(cvRound(scale_LM_img*camera_img_ori.cols),cvRound(scale_LM_img*camera_img_ori.rows)), scale_LM_img, scale_LM_img ,INTER_LINEAR );
			char *modelname_Char = new char[face_LM_model_path.length() + 1];
			strcpy(modelname_Char, face_LM_model_path.c_str());
			detector.detect(bs,camera_img_ori_t ,detector.model_ ,modelname_Char);
			delete [] modelname_Char;
			detector.clipboxes(camera_img_ori_t.rows,camera_img_ori_t.cols,bs);
			detector.nmsface(bs,0.3,top);
			if (top.empty() == 1) 
			{
				//printf("Face not found!\n");
			}
			else
			{
				point = detector.model_->components_[top[0].c].defid_.size();
				for (int n=0; n<=point-1; n++) //for (int n=point-1; n>=0; n--) //26
				{
					LM_each.push_back(cvPoint( cvRound(((top[0].xy[n][0]+top[0].xy[n][2])/2)/scale_LM_img) ,cvRound(((top[0].xy[n][1]+top[0].xy[n][3])/2)/scale_LM_img)));
				}
			}
			bs.clear();
			top.clear();
			vector<Data_bs>().swap(bs);
			vector<Data_bs>().swap(top);

			for (vector<Point2f>::iterator LM_pt=LM_each.begin();LM_pt!=LM_each.end();++LM_pt)
			{
				circle(camera_img_ori,*LM_pt,1,CV_RGB(0,255,0),-1);
			}
			vector<Point2f>().swap(LM_each);

		}
		else
			ChehraObj.Reinitialize();

		imshow("camera_img_ori",camera_img_ori);
		inputKey=waitKey(5);
		if(inputKey == VK_ESCAPE) {break;} 
	}
#endif // use_camera

	// read Reg img //
	//int total_Reg_image=0;
	//string Reg_img_title_path = "../using_data/";
	//string Reg_img_file_name="Reg-image/MPIE";
	//string Reg_img_image_path = Reg_img_title_path+Reg_img_file_name+"/";
	//vector<string> Reg_img_name; //all the file name
	//Load_insideFile_name(Reg_img_image_path, &Reg_img_name);

	//Mat Reg_img,Reg_img_g,Reg_img_t;
	//vector<Point2f> LM_glass;
	//for (vector<string>::iterator Reg_name=Reg_img_name.begin();Reg_name!=Reg_img_name.end();++Reg_name)
	//{
	//	string Reg_image_load_path=Reg_img_image_path+*Reg_name;
	//	Reg_img=imread(Reg_image_load_path,1);
	//	// LM detect progess //
	//	vector<Data_bs>bs;vector<Data_bs> top;int point;
	//	float scale_LM_img_1=0.4;
	//	resize(Reg_img,Reg_img_t, Size(cvRound(scale_LM_img_1*Reg_img.cols),cvRound(scale_LM_img_1*Reg_img.rows)), scale_LM_img_1, scale_LM_img_1 ,INTER_LINEAR );
	//	char *modelname_Char = new char[face_LM_model_path.length() + 1];
	//	strcpy(modelname_Char, face_LM_model_path.c_str());
	//	detector.detect(bs,Reg_img_t ,detector.model_ ,modelname_Char);
	//	delete [] modelname_Char;
	//	detector.clipboxes(Reg_img_t.rows,Reg_img_t.cols,bs);
	//	detector.nmsface(bs,0.3,top);
	//	if (top.empty() == 1) 
	//	{
	//		printf("Face not found!\n");
	//	}
	//	else
	//	{
	//		point = detector.model_->components_[top[0].c].defid_.size();
	//		for (int n=0; n<=point-1; n++) //for (int n=point-1; n>=0; n--)
	//		{
	//			LM_glass.push_back(cvPoint( cvRound(((top[0].xy[n][0]+top[0].xy[n][2])/2)/scale_LM_img_1) ,cvRound(((top[0].xy[n][1]+top[0].xy[n][3])/2)/scale_LM_img_1)));
	//		}
	//	}
	//	bs.clear();
	//	top.clear();
	//	vector<Data_bs>().swap(bs);
	//	vector<Data_bs>().swap(top);

	//	for (vector<Point2f>::iterator LM_pt=LM_glass.begin();LM_pt!=LM_glass.end();++LM_pt)
	//	{
	//		circle(Reg_img,*LM_pt,1,CV_RGB(0,255,0),-1);
	//	}
	//	imshow("temp_img",Reg_img);
	//	waitKey(0);
	//	vector<Point2f>().swap(LM_glass);
	//}

	

}

//////////////////////////////////////////
///*          sub function            *///
//////////////////////////////////////////

//void Load_insideFile_name(string input_path, vector<string>* output_path)
//{
//	vector<string> temp_name_save;
//	int testSampleCount = 0; //測試人數統計
//	DIR *DP;
//	struct dirent *DirpathP;
//	DP = opendir(input_path.c_str());
//	while (DirpathP = readdir(DP))
//	{
//		//如果不加下面那一行  讀取出來的檔案會有點點
//		if( strcmp(DirpathP->d_name, ".") != 0 && strcmp(DirpathP->d_name, "..") != 0 )
//		{
//			testSampleCount=testSampleCount+1;
//			//cout<<DirpathP->d_name<<endl;
//			temp_name_save.push_back(DirpathP->d_name);
//			//system("pause");
//		}
//	}
//	//cout<<"共"<<testSampleCount<<"人進行測試"<<endl;
//
//	*output_path=temp_name_save;
//}
