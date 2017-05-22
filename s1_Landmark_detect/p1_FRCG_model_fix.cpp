//  //

#include "stdlib.h"
#include "Chehra_Linker.h"
#include <opencv2/opencv.hpp>
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
#include "Chehra_draw.h"
#include "file_read_mine.h"
#include "LM_set.h"
#include "mine_imgProc.h"

#include <eigen/SparseLU>
#include <eigen/SparseQR>
#include <eigen/SparseCore>
#include <eigen/IterativeLinearSolvers>

using namespace std;
using namespace cv;
using namespace Chehra_Tracker;
using namespace Eigen;

// // Chehra landmark //
#define use_Chehra 1
// Lab face landmark //
#define use_Lab_LM_model 1
// Lab glass landmark //
#define use_Lab_glass_LM_model 1

// Load FRGC Model //
#define read_FRGC_model 0
// Fix FRGC LM & Save //
#define fix_model_lm 0

// fix the landmark // // mouse function // 
vector<Point2f> point_LM;
vector<Point2f> point_LM_temp;
Mat img_temp;// img_temp 办跑计 //
int point_count=1;
void onMouse(int event,int x,int y,int flags,void* param);

int main(void)
{
	string data_file_title="../../using_data/";//隔畖

	// set Chehra LM //
	// 砞竚 Chehra LM ㄏノ把计 //
#if use_Chehra
	//set other tracking parameters

	//Failure Checker Interval <integer>
	int fcheck_interval=5; //Failure checker to be executed after every 5 frames

	//Failure Checker Score Treshold [-1,1]
	float fcheck_score_treshold=-0.65;	//threshold for detecting tracking failure

	//Number of Consecutive Frames Failed To Activate Redetection
	int fcheck_fail_treshold=2;	//reinitialize after failure on 2 consecutive frames

	//models
	char regFile[256],fdFile[256];
	strcpy(regFile,"../../using_data/Chehra-model/Chehra_t1.0.model");
	strcpy(fdFile,"../../using_data/Chehra-model/haarcascade_frontalface_alt_tree.xml");

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
	// 砞竚 Lab face LM ㄏノ把计 //
#if use_Lab_LM_model
	int posemap[18],temp=90;
	string face_LM_model_path=data_file_title+"LM_Model"+"/"+"multipie_35_20_CN3_Sbin3_final.mat";

	LandmarkDetector detector; //modelΑ
	detector.Load(face_LM_model_path.c_str()); //弄model
	//printf("model : %s\n",face_LM_model_path_FRGC);

	// set up the threshold //
	detector.model_->thresh_=-0.95;

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

	// set Lab glass landmark //
	// 砞竚 Lab glass LM ㄏノ把计 //
#if use_Lab_glass_LM_model
	int posemap_glass[18],temp_glass=90;
	string face_glass_LM_model_path=data_file_title+"LM_Model"+"/"+"MPIE_glasses_v2_final.mat";

	LandmarkDetector detector_glass; //modelΑ
	detector_glass.Load(face_glass_LM_model_path.c_str()); //弄model
	//printf("model : %s\n",face_LM_model_path_FRGC);

	// set up the threshold //
	detector_glass.model_->thresh_=-0.95;

	// define the mapping from view-specific mixture id to viewpoint //
	if (detector_glass.model_->components_.size()==13)
		for (int i=0;i<13;i++)
		{
			posemap_glass[i]=temp_glass;
			temp_glass-=15;
		}
	else if (detector_glass.model_->components_.size()==18)
		for (int i=0;i<18;i++)
		{
			if (i>5 && i<11){
				posemap_glass[i]=0;}
			else{
				posemap_glass[i]=temp_glass;
				temp_glass-=15;}
		}
	else if (detector_glass.model_->components_.size()==1)
		posemap_glass[0]=0;
	else
		printf("Can not recognize this model");

#endif

	// read FRGC model //
	// read_FRGC_model=1 : 穝弄 FRGC model 禲 Chehra の Lab face LM //
	// read_FRGC_model=0 : 弄禲Ч FRGC Chehra の Lab face LM //
	string FRGC_model_file_name="FRGC-model-AF";
	string FRGC_model_image_path = data_file_title+FRGC_model_file_name+"/";
	vector<string> FRGC_image_name; //all the file name
	vector<vector<Point2f>> model_LM_all; // all model img landmark
	// -----Save the Data into xml //
	string FRGC_model_name=data_file_title+"xml"+"/"+"FRGC_data"+"/"+"FRGC_Name_Data_AF.xml";
	string FRGC_model_lmpt=data_file_title+"xml"+"/"+"FRGC_data"+"/"+"FRGC_LMPT_Data_AF_fix.xml";
#if read_FRGC_model
	Load_insideFile_name(FRGC_model_image_path, &FRGC_image_name);
	Mat model_img,model_img_g,model_img_t;
	for (vector<string>::iterator model_name=FRGC_image_name.begin();model_name!=FRGC_image_name.end();++model_name)
	{
		string model_image_load_path=FRGC_model_image_path+*model_name+"/"+*model_name+"-ori.ppm";
		//cout<<model_image_load_path<<endl;
		model_img=imread(model_image_load_path,1);
		cvtColor(model_img,model_img_g,CV_BGR2GRAY);
		ChehraObj.Reinitialize();
		vector<Point2f> model_LM_each;
		if (ChehraObj.TrackFrame(model_img_g,fcheck_interval,fcheck_score_treshold,fcheck_fail_treshold,face_cascade) == 0)
		{
			Chehra_Plot(model_img,ChehraObj._bestFaceShape,ChehraObj._bestEyeShape,&model_LM_each);

			// LM detect progess //
#if use_Lab_LM_model
			vector<Data_bs>bs;vector<Data_bs> top;int point;
			float scale_LM_img=0.45;
			resize(model_img,model_img_t, Size(cvRound(scale_LM_img*model_img.cols),cvRound(scale_LM_img*model_img.rows)), scale_LM_img, scale_LM_img ,INTER_LINEAR );
			char *modelname_Char = new char[face_LM_model_path.length() + 1];
			strcpy(modelname_Char, face_LM_model_path.c_str());
			detector.detect(bs,model_img_t ,detector.model_ ,modelname_Char);
			delete [] modelname_Char;
			detector.clipboxes(model_img_t.rows,model_img_t.cols,bs);
			detector.nmsface(bs,0.3,top);
			if (top.empty() == 1) 
			{
				printf("Face not found!\n");
			}
			else
			{
				point = detector.model_->components_[top[0].c].defid_.size();
				for (int n=26; n<=point-1; n++) //for (int n=point-1; n>=0; n--)
				{
					model_LM_each.push_back(cvPoint( cvRound(((top[0].xy[n][0]+top[0].xy[n][2])/2)/scale_LM_img) ,cvRound(((top[0].xy[n][1]+top[0].xy[n][3])/2)/scale_LM_img)));
				}
				//Set_the_LM(model_LM_each,&model_LM_each);
			}
			bs.clear();
			top.clear();
			vector<Data_bs>().swap(bs);
			vector<Data_bs>().swap(top);
#endif
			model_LM_all.push_back(model_LM_each);
		}
		else
		{
			// LM detect progess //
#if use_Lab_LM_model
			vector<Data_bs>bs;vector<Data_bs> top;int point;
			float scale_LM_img=0.45;
			resize(model_img,model_img_t, Size(cvRound(scale_LM_img*model_img.cols),cvRound(scale_LM_img*model_img.rows)), scale_LM_img, scale_LM_img ,INTER_LINEAR );
			char *modelname_Char = new char[face_LM_model_path.length() + 1];
			strcpy(modelname_Char, face_LM_model_path.c_str());
			detector.detect(bs,model_img_t ,detector.model_ ,modelname_Char);
			delete [] modelname_Char;
			detector.clipboxes(model_img_t.rows,model_img_t.cols,bs);
			detector.nmsface(bs,0.3,top);
			if (top.empty() == 1) 
			{
				printf("Face not found!\n");
			}
			else
			{
				point = detector.model_->components_[top[0].c].defid_.size();
				for (int n=26; n<=point-1; n++) //for (int n=point-1; n>=0; n--)
				{
					model_LM_each.push_back(cvPoint( cvRound(((top[0].xy[n][0]+top[0].xy[n][2])/2)/scale_LM_img) ,cvRound(((top[0].xy[n][1]+top[0].xy[n][3])/2)/scale_LM_img)));
				}
				//Set_the_LM(model_LM_each,&model_LM_each);
			}
			bs.clear();
			top.clear();
			vector<Data_bs>().swap(bs);
			vector<Data_bs>().swap(top);
#endif
			model_LM_all.push_back(model_LM_each);
		}
	}

	// -----Save the Data into xml //
	FileStorage FS_NT;
	FS_NT.open(FRGC_model_name, FileStorage::WRITE);
	FS_NT << "FRGC_Name_Tar" << FRGC_image_name;
	FS_NT.release();
	FileStorage FS_LDT;
	FS_LDT.open(FRGC_model_lmpt, FileStorage::WRITE);
	for (int i=0;i<FRGC_image_name.size();i++)
	{
		stringstream int2str;
		string num;
		int2str << i;
		int2str >> num;
		string label="FRGC_LMPT_Data_Tar_"+num;
		FS_LDT << label << model_LM_all[i];
	}
	FS_LDT.release();
#else
	// -----Read the Data from xml //
	FileStorage FS_NT_R;
	FS_NT_R.open(FRGC_model_name, FileStorage::READ);
	FS_NT_R["FRGC_Name_Tar"] >> FRGC_image_name;
	FS_NT_R.release();
	FileStorage FS_LDT_R;
	FS_LDT_R.open(FRGC_model_lmpt, FileStorage::READ);
	for (int i=0;i<FRGC_image_name.size();i++)
	{
		vector<Point2f> temp;
		stringstream int2str;
		string num;
		int2str << i;
		int2str >> num;
		string label="FRGC_LMPT_Data_Tar_"+num;
		FS_LDT_R[label] >> temp;
		model_LM_all.push_back(temp);
	}
	FS_LDT_R.release();
#endif

	// fix the model Landmark //
	// fix_model_lm=1 : タFRGC禲ЧLandmark翴, 纗//
	// fix_model_lm=0 : ぃタFRGC禲ЧLandmark翴//
	// -----Save the fix Data into xml //
	string FRGC_model_lmpt_fix=data_file_title+"xml"+"/"+"FRGC_data"+"/"+"FRGC_LMPT_Data_AF_fix.xml";
#if fix_model_lm
	Mat model_ori_img;
	Mat model_ori_img_d;
	int inputKey;
	vector<vector<Point2f>> model_LM_all_fix; // all model img landmark
	for (int i=0; i<FRGC_image_name.size();i++)
	{
		string img_load_path=FRGC_model_image_path+FRGC_image_name[i]+"/"+FRGC_image_name[i]+"-ori.ppm";
		//cout<<img_load_path<<endl;
		model_ori_img=imread(img_load_path);
		model_ori_img.copyTo(model_ori_img_d);
		imshow("model_ori_img",model_ori_img);waitKey(1);

		for (vector<Point2f>::iterator LM_pt=model_LM_all[i].begin();LM_pt!=model_LM_all[i].end();++LM_pt)
		{
			circle(model_ori_img_d,*LM_pt,1,CV_RGB(0,255,0),-1);
		}
		imshow("model_ori_img_d",model_ori_img_d);inputKey=waitKey(0);

		vector<Point2f> img_LM_each;
		if (char(inputKey) == 'z')
		{
			point_LM.clear();
			point_LM_temp.clear();
			point_count=1;
			model_ori_img.copyTo(img_temp);
			point_LM_temp=model_LM_all[i];

			while(point_count<=58)
			{
				//cout<<point_count<<endl;
				imshow("draw the point",img_temp);
				waitKey(1);
				setMouseCallback("draw the point", onMouse, NULL );
			}
			img_LM_each=point_LM;
		}
		else
		{
			img_LM_each=model_LM_all[i];
		}
		model_LM_all_fix.push_back(img_LM_each);
	}


	FileStorage FS_LDT_fix;
	FS_LDT_fix.open(FRGC_model_lmpt_fix, FileStorage::WRITE);
	for (int i=0;i<FRGC_image_name.size();i++)
	{
		stringstream int2str;
		string num;
		int2str << i;
		int2str >> num;
		string label="FRGC_LMPT_Data_Tar_"+num;
		FS_LDT_fix << label << model_LM_all_fix[i];
	}
	FS_LDT_fix.release();
#else
	Mat model_ori_img;
	Mat model_ori_img_d;
	int inputKey;
	vector<vector<Point2f>> model_LM_all_fix; // all model img landmark
	for (int i=0; i<FRGC_image_name.size();i++)
	{
		string img_load_path=FRGC_model_image_path+FRGC_image_name[i]+"/"+FRGC_image_name[i]+"-ori.ppm";
		//cout<<img_load_path<<endl;
		model_ori_img=imread(img_load_path);
		model_ori_img.copyTo(model_ori_img_d);
		//imshow("model_ori_img",model_ori_img);waitKey(1);

		int point_number=1;
		for (vector<Point2f>::iterator LM_pt=model_LM_all[i].begin();LM_pt!=model_LM_all[i].end();++LM_pt)
		{
			circle(model_ori_img_d,*LM_pt,1,CV_RGB(0,255,0),-1);
			Point2f temp=*LM_pt;
			string point_number_str;
			stringstream int2string;
			int2string<<point_number;
			int2string>>point_number_str;
			putText(model_ori_img_d,point_number_str,cv::Point2f(temp.x-5,temp.y-5),FONT_HERSHEY_PLAIN , 1, CV_RGB(0,0,255), 1);
			int2string.str("");
			int2string.clear();
			point_number=point_number+1;
		}
		imshow("model_ori_img_d",model_ori_img_d);inputKey=waitKey(0);
	}
#endif

	
	
	system("pause");
	return 0;
}

//////////////////////////////////////////
///*          sub function            *///
//////////////////////////////////////////

// fix the landmark // // mouse function // 
void onMouse(int event,int x,int y,int flag,void* param)
{

	if(event==CV_EVENT_LBUTTONDOWN)
	{
		cout<<point_count<<endl;
		point_LM.push_back(Point2f(x,y));
		circle(img_temp,Point2f(x,y),1,CV_RGB(0,255,255),-1);
		imshow("draw the point",img_temp);
		waitKey(1);
		point_count++;
	}
	if(event==CV_EVENT_RBUTTONDOWN)
	{
		cout<<point_count<<endl;
		point_LM.push_back(point_LM_temp[point_count-1]);
		circle(img_temp,point_LM_temp[point_count-1],1,CV_RGB(0,255,255),-1);
		imshow("draw the point",img_temp);
		waitKey(1);
		point_count++;
	}

}
