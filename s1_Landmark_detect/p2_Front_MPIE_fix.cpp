// model :  volumne spline interpolation //
// 2015/06/29 Eric //

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

// Load MPIE Image //
#define read_MPIE_img 0
// Fix MPIE LM & Save //
#define fix_MPIE_lm 0

// Load MPIE glass Image //
#define read_MPIE_img_glass 0 
// Fix MPIE glass LM & Save //
#define fix_MPIE_lm_glass 0
// Fix MPIE only glass LM & Save //
#define fix_MPIE_lm_only_glass 0

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

	// read MPIE image //
	// read_MPIE_img=1 : 穝弄 MPIE no glass 禲 Chehra の Lab face LM //
	// read_MPIE_img=0 : 弄禲Ч MPIE no glass の Lab face LM //
	string MPIE_file_name="Reg-image/MPIE no glass";
	string MPIE_file_image_path = data_file_title+MPIE_file_name+"/";
	vector<string> MPIE_image_name; //MPIE all the file name
	vector<vector<Point2f>> MPIE_LM_all; //MPIE all img landmark
	// -----Save the Data into xml //
	string MPIE_name=data_file_title+"xml"+"/"+"MPIE_no_glass_data"+"/"+"MPIE_no_glass_LM_Name_Data.xml";
	string MPIE_lmpt=data_file_title+"xml"+"/"+"MPIE_no_glass_data"+"/"+"MPIE_no_glass_LM_all_Data_fix.xml";
#if read_MPIE_img
	Load_insideFile_name(MPIE_file_image_path, &MPIE_image_name); 
	Mat model_img,model_img_g,model_img_t;
	for (vector<string>::iterator model_name=MPIE_image_name.begin();model_name!=MPIE_image_name.end();++model_name)
	{
		string model_image_load_path=MPIE_file_image_path+*model_name; //э硂
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
			MPIE_LM_all.push_back(model_LM_each);
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
			MPIE_LM_all.push_back(model_LM_each);
		}
	}

	// -----Save the Data into xml //
	FileStorage FS_NT;
	FS_NT.open(MPIE_name, FileStorage::WRITE);
	FS_NT << "MPIE_Name_Tar" << MPIE_image_name;
	FS_NT.release();
	FileStorage FS_LDT;
	FS_LDT.open(MPIE_lmpt, FileStorage::WRITE);
	for (int i=0;i<MPIE_image_name.size();i++)
	{
		stringstream int2str;
		string num;
		int2str << i;
		int2str >> num;
		string label="MPIE_LMPT_Data_Tar_"+num;
		FS_LDT << label << MPIE_LM_all[i];
	}
	FS_LDT.release();
#else
	// -----Read the Data from xml //
	FileStorage FS_MPIE_NT_R;
	FS_MPIE_NT_R.open(MPIE_name, FileStorage::READ);
	FS_MPIE_NT_R["MPIE_Name_Tar"] >> MPIE_image_name;
	FS_MPIE_NT_R.release();
	FileStorage FS_MPIE_LDT_R;
	FS_MPIE_LDT_R.open(MPIE_lmpt, FileStorage::READ);
	for (int i=0;i<MPIE_image_name.size();i++)
	{
		vector<Point2f> temp;
		stringstream int2str;
		string num;
		int2str << i;
		int2str >> num;
		string label="MPIE_LMPT_Data_Tar_"+num;
		FS_MPIE_LDT_R[label] >> temp;
		MPIE_LM_all.push_back(temp);
	}
	FS_MPIE_LDT_R.release();
#endif

	// fix the MPIE Landmark //
	// fix_MPIE_lm=1 : タ MPIE no glass 禲ЧLandmark翴, 纗//
	// fix_MPIE_lm=0 : ぃタ MPIE no glass 禲ЧLandmark翴//
	// -----Save the fix Data into xml //
	string MPIE_lmpt_fix=data_file_title+"xml"+"/"+"MPIE_no_glass_data"+"/"+"MPIE_no_glass_LM_all_Data_fix.xml";
#if fix_MPIE_lm
	Mat model_ori_img_mpie;
	Mat model_ori_img_d_mpie;
	int inputKey_mpie;
	vector<vector<Point2f>> model_LM_all_fix; // all model img landmark
	for (int i=0; i<MPIE_image_name.size();i++)
	{
		string img_load_path=MPIE_file_image_path+MPIE_image_name[i];
		//cout<<img_load_path<<endl;
		model_ori_img_mpie=imread(img_load_path);
		model_ori_img_mpie.copyTo(model_ori_img_d_mpie);
		imshow("model_ori_img",model_ori_img_mpie);waitKey(1);

		for (vector<Point2f>::iterator LM_pt=MPIE_LM_all[i].begin();LM_pt!=MPIE_LM_all[i].end();++LM_pt)
		{
			circle(model_ori_img_d_mpie,*LM_pt,1,CV_RGB(0,255,0),-1);
		}
		imshow("model_ori_img_d",model_ori_img_d_mpie);inputKey_mpie=waitKey(0);

		vector<Point2f> img_LM_each;
		if (char(inputKey_mpie) == 'z')
		{
			point_LM.clear();
			point_LM_temp.clear();
			point_count=1;
			model_ori_img_mpie.copyTo(img_temp);
			point_LM_temp=MPIE_LM_all[i];

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
			img_LM_each=MPIE_LM_all[i];
		}
		model_LM_all_fix.push_back(img_LM_each);
	}


	FileStorage FS_MPIE_fix;
	FS_MPIE_fix.open(MPIE_lmpt_fix, FileStorage::WRITE);
	for (int i=0;i<MPIE_image_name.size();i++)
	{
		stringstream int2str;
		string num;
		int2str << i;
		int2str >> num;
		string label="MPIE_LMPT_Data_Tar_"+num;
		FS_MPIE_fix << label << model_LM_all_fix[i];
	}
	FS_MPIE_fix.release();
#else
	Mat model_ori_img_mpie;
	Mat model_ori_img_d_mpie;
	int inputKey_mpie;
	for (int i=0; i<MPIE_image_name.size();i++)
	{
		string img_load_path=MPIE_file_image_path+MPIE_image_name[i];
		//cout<<img_load_path<<endl;
		model_ori_img_mpie=imread(img_load_path);
		model_ori_img_mpie.copyTo(model_ori_img_d_mpie);
		//imshow("model_ori_img",model_ori_img_mpie);waitKey(1);

		for (vector<Point2f>::iterator LM_pt=MPIE_LM_all[i].begin();LM_pt!=MPIE_LM_all[i].end();++LM_pt)
		{
			circle(model_ori_img_d_mpie,*LM_pt,1,CV_RGB(0,255,0),-1);
		}
		//imshow("model_ori_img_d",model_ori_img_d_mpie);inputKey_mpie=waitKey(1);
	}
#endif

	// read MPIE glass image //
	// read_MPIE_img_glass=1 : 穝弄 MPIE glass 禲 Chehra の Lab face LM //
	// read_MPIE_img_glass=0 : 弄禲Ч MPIE glass の Lab face LM //
	string MPIE_file_name_glass="Reg-image/MPIE glass";
	string MPIE_file_image_path_glass = data_file_title+MPIE_file_name_glass+"/";
	vector<string> MPIE_image_name_glass; //MPIE all the file name
	vector<vector<Point2f>> MPIE_LM_all_glass; //MPIE all img landmark
	vector<vector<Point2f>> MPIE_LM_all_only_glass; //MPIE all glass img landmark
	// -----Save the Data into xml //
	string MPIE_name_glass=data_file_title+"xml"+"/"+"MPIE_glass_data"+"/"+"MPIE_glass_LM_Name_Data.xml";
	string MPIE_lmpt_glass=data_file_title+"xml"+"/"+"MPIE_glass_data"+"/"+"MPIE_glass_LM_all_Data_fix.xml";
	string MPIE_lmpt_only_glass=data_file_title+"xml"+"/"+"MPIE_glass_data"+"/"+"MPIE_only_glass_LM_all_Data_fix.xml";
#if read_MPIE_img_glass
	Load_insideFile_name(MPIE_file_image_path_glass, &MPIE_image_name_glass); 
	for (vector<string>::iterator model_name=MPIE_image_name_glass.begin();model_name!=MPIE_image_name_glass.end();++model_name)
	{
		Mat model_img,model_img_g,model_img_t;
		string model_image_load_path=MPIE_file_image_path_glass+*model_name; //э硂
		//cout<<model_image_load_path<<endl;
		model_img=imread(model_image_load_path,1);
		cvtColor(model_img,model_img_g,CV_BGR2GRAY);
		ChehraObj.Reinitialize();
		vector<Point2f> model_LM_each;
		vector<Point2f> model_LM_glass;
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
			MPIE_LM_all_glass.push_back(model_LM_each);

			// glass LM detect progess //
#if use_Lab_glass_LM_model
			vector<Data_bs>bs_glass;vector<Data_bs> top_glass;int point_glass;
			float scale_LM_img_glass=0.45;
			resize(model_img,model_img_t, Size(cvRound(scale_LM_img_glass*model_img.cols),cvRound(scale_LM_img_glass*model_img.rows)), scale_LM_img_glass, scale_LM_img_glass ,INTER_LINEAR );
			char *modelname_Char_glass = new char[face_glass_LM_model_path.length() + 1];
			strcpy(modelname_Char_glass, face_glass_LM_model_path.c_str());
			detector_glass.detect(bs_glass,model_img_t ,detector_glass.model_ ,modelname_Char_glass);
			delete [] modelname_Char_glass;
			detector_glass.clipboxes(model_img_t.rows,model_img_t.cols,bs_glass);
			detector_glass.nmsface(bs_glass,0.3,top_glass);
			if (top_glass.empty() == 1) 
			{
				printf("Face not found!\n");
			}
			else
			{
				point_glass = detector_glass.model_->components_[top_glass[0].c].defid_.size();
				for (int n=0; n<=point-1; n++) //for (int n=point-1; n>=0; n--)
				{
					model_LM_glass.push_back(cvPoint( cvRound(((top_glass[0].xy[n][0]+top_glass[0].xy[n][2])/2)/scale_LM_img_glass) ,cvRound(((top_glass[0].xy[n][1]+top_glass[0].xy[n][3])/2)/scale_LM_img_glass)));
				}
				//Set_the_LM(model_LM_each,&model_LM_each);
			}
			bs_glass.clear();
			top_glass.clear();
			vector<Data_bs>().swap(bs_glass);
			vector<Data_bs>().swap(top_glass);

			//for (int i=0;i<model_LM_glass.size();i++)
			//{
			//	circle(model_img,model_LM_glass[i],1,CV_RGB(0,255,0),-1);
			//	cout<<model_LM_glass[i]<<endl;
			//}
			//imshow("model_img",model_img);waitKey(0);
#endif
			MPIE_LM_all_only_glass.push_back(model_LM_glass);

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
			MPIE_LM_all_glass.push_back(model_LM_each);

			// glass LM detect progess //
#if use_Lab_glass_LM_model
			vector<Data_bs>bs_glass;vector<Data_bs> top_glass;int point_glass;
			float scale_LM_img_glass=0.45;
			resize(model_img,model_img_t, Size(cvRound(scale_LM_img_glass*model_img.cols),cvRound(scale_LM_img_glass*model_img.rows)), scale_LM_img_glass, scale_LM_img_glass ,INTER_LINEAR );
			char *modelname_Char_glass = new char[face_glass_LM_model_path.length() + 1];
			strcpy(modelname_Char_glass, face_glass_LM_model_path.c_str());
			detector_glass.detect(bs_glass,model_img_t ,detector_glass.model_ ,modelname_Char_glass);
			delete [] modelname_Char_glass;
			detector_glass.clipboxes(model_img_t.rows,model_img_t.cols,bs_glass);
			detector_glass.nmsface(bs_glass,0.3,top_glass);
			if (top_glass.empty() == 1) 
			{
				printf("Face not found!\n");
			}
			else
			{
				point_glass = detector_glass.model_->components_[top_glass[0].c].defid_.size();
				for (int n=0; n<=point-1; n++) //for (int n=point-1; n>=0; n--)
				{
					model_LM_glass.push_back(cvPoint( cvRound(((top_glass[0].xy[n][0]+top_glass[0].xy[n][2])/2)/scale_LM_img_glass) ,cvRound(((top_glass[0].xy[n][1]+top_glass[0].xy[n][3])/2)/scale_LM_img_glass)));
				}
				//Set_the_LM(model_LM_each,&model_LM_each);
			}
			bs_glass.clear();
			top_glass.clear();
			vector<Data_bs>().swap(bs_glass);
			vector<Data_bs>().swap(top_glass);

			//for (int i=0;i<model_LM_glass.size();i++)
			//{
			//	circle(model_img,model_LM_glass[i],1,CV_RGB(0,255,0),-1);
			//	cout<<model_LM_glass[i]<<endl;
			//}
			//imshow("model_img",model_img);waitKey(0);
#endif
			MPIE_LM_all_only_glass.push_back(model_LM_glass);
		}
	}

	// -----Save the Data into xml //
	FileStorage FS_NT_glass;
	FS_NT_glass.open(MPIE_name_glass, FileStorage::WRITE);
	FS_NT_glass << "MPIE_Name_Tar" << MPIE_image_name_glass;
	FS_NT_glass.release();
	FileStorage FS_LDT_glass;
	FS_LDT_glass.open(MPIE_lmpt_glass, FileStorage::WRITE);
	for (int i=0;i<MPIE_image_name_glass.size();i++)
	{
		stringstream int2str;
		string num;
		int2str << i;
		int2str >> num;
		string label="MPIE_LMPT_Data_Tar_"+num;
		FS_LDT_glass << label << MPIE_LM_all_glass[i];
	}
	FS_LDT_glass.release();
	FileStorage FS_LDT_only_glass;
	FS_LDT_only_glass.open(MPIE_lmpt_only_glass, FileStorage::WRITE);
	for (int i=0;i<MPIE_image_name_glass.size();i++)
	{
		stringstream int2str;
		string num;
		int2str << i;
		int2str >> num;
		string label="MPIE_LMPT_Data_Tar_"+num;
		FS_LDT_only_glass << label << MPIE_LM_all_only_glass[i];
	}
	FS_LDT_only_glass.release();
#else
	// -----Read the Data from xml //
	FileStorage FS_MPIE_NT_R_glass;
	FS_MPIE_NT_R_glass.open(MPIE_name_glass, FileStorage::READ);
	FS_MPIE_NT_R_glass["MPIE_Name_Tar"] >> MPIE_image_name_glass;
	FS_MPIE_NT_R_glass.release();
	FileStorage FS_MPIE_LDT_R_glass;
	FS_MPIE_LDT_R_glass.open(MPIE_lmpt_glass, FileStorage::READ);
	for (int i=0;i<MPIE_image_name_glass.size();i++)
	{
		vector<Point2f> temp;
		stringstream int2str;
		string num;
		int2str << i;
		int2str >> num;
		string label="MPIE_LMPT_Data_Tar_"+num;
		FS_MPIE_LDT_R_glass[label] >> temp;
		MPIE_LM_all_glass.push_back(temp);
	}
	FS_MPIE_LDT_R_glass.release();
	FileStorage FS_MPIE_LDT_R_only_glass;
	FS_MPIE_LDT_R_only_glass.open(MPIE_lmpt_only_glass, FileStorage::READ);
	for (int i=0;i<MPIE_image_name_glass.size();i++)
	{
		vector<Point2f> temp;
		stringstream int2str;
		string num;
		int2str << i;
		int2str >> num;
		string label="MPIE_LMPT_Data_Tar_"+num;
		FS_MPIE_LDT_R_only_glass[label] >> temp;
		MPIE_LM_all_only_glass.push_back(temp);
	}
	FS_MPIE_LDT_R_only_glass.release();
#endif

	// fix the MPIE glass Landmark //
	// fix_MPIE_lm_glass=1 : タ MPIE glass 禲ЧLandmark翴, 纗//
	// fix_MPIE_lm_glass=0 : ぃタ MPIE glass 禲ЧLandmark翴//
	// -----Save the fix Data into xml //
	string MPIE_lmpt_fix_glass=data_file_title+"xml"+"/"+"MPIE_glass_data"+"/"+"MPIE_glass_LM_all_Data_fix.xml";
#if fix_MPIE_lm_glass
	Mat model_ori_img_mpie_glass;
	Mat model_ori_img_d_mpie_glass;
	int inputKey_mpie_glass;
	vector<vector<Point2f>> model_LM_all_fix_glass; // all model img landmark
	for (int i=0; i<MPIE_image_name_glass.size();i++)
	{
		string img_load_path=MPIE_file_image_path_glass+MPIE_image_name_glass[i];
		//cout<<img_load_path<<endl;
		model_ori_img_mpie_glass=imread(img_load_path);
		model_ori_img_mpie_glass.copyTo(model_ori_img_d_mpie_glass);
		imshow("model_ori_img",model_ori_img_mpie_glass);waitKey(1);

		for (vector<Point2f>::iterator LM_pt=MPIE_LM_all_glass[i].begin();LM_pt!=MPIE_LM_all_glass[i].end();++LM_pt)
		{
			circle(model_ori_img_d_mpie_glass,*LM_pt,1,CV_RGB(0,255,0),-1);
		}
		imshow("model_ori_img_d",model_ori_img_d_mpie_glass);inputKey_mpie_glass=waitKey(0);

		vector<Point2f> img_LM_each;
		if (char(inputKey_mpie_glass) == 'z')
		{
			point_LM.clear();
			point_LM_temp.clear();
			point_count=1;
			model_ori_img_mpie_glass.copyTo(img_temp);
			point_LM_temp=MPIE_LM_all_glass[i];

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
			img_LM_each=MPIE_LM_all_glass[i];
		}
		model_LM_all_fix_glass.push_back(img_LM_each);
	}


	FileStorage FS_MPIE_fix_glass;
	FS_MPIE_fix_glass.open(MPIE_lmpt_fix_glass, FileStorage::WRITE);
	for (int i=0;i<model_LM_all_fix_glass.size();i++)
	{
		stringstream int2str;
		string num;
		int2str << i;
		int2str >> num;
		string label="MPIE_LMPT_Data_Tar_"+num;
		FS_MPIE_fix_glass << label << model_LM_all_fix_glass[i];
	}
	FS_MPIE_fix_glass.release();
#else
	Mat model_ori_img_mpie_glass;
	Mat model_ori_img_d_mpie_glass;
	int inputKey_mpie_glass;
	for (int i=0; i<MPIE_image_name_glass.size();i++)
	{
		string img_load_path=MPIE_file_image_path_glass+MPIE_image_name_glass[i];
		//cout<<img_load_path<<endl;
		model_ori_img_mpie_glass=imread(img_load_path);
		model_ori_img_mpie_glass.copyTo(model_ori_img_d_mpie_glass);
		//imshow("model_ori_img",model_ori_img_mpie_glass);waitKey(1);

		for (vector<Point2f>::iterator LM_pt=MPIE_LM_all_glass[i].begin();LM_pt!=MPIE_LM_all_glass[i].end();++LM_pt)
		{
			circle(model_ori_img_d_mpie_glass,*LM_pt,1,CV_RGB(0,255,0),-1);
		}
		//imshow("model_ori_img_d",model_ori_img_d_mpie_glass);inputKey_mpie=waitKey(0);
	}
#endif

	// fix the MPIE only glass Landmark //
	// fix_MPIE_lm_only_glass=1 : タ MPIE only glass 禲ЧLandmark翴, 纗//
	// fix_MPIE_lm_only_glass=0 : ぃタ MPIE only glass 禲ЧLandmark翴//
	// -----Save the fix Data into xml //
	string MPIE_lmpt_fix_only_glass=data_file_title+"xml"+"/"+"MPIE_glass_data"+"/"+"MPIE_only_glass_LM_all_Data_fix.xml";
#if fix_MPIE_lm_only_glass
	Mat model_ori_img_mpie_only_glass;
	Mat model_ori_img_d_mpie_only_glass;
	int inputKey_mpie_only_glass;
	vector<vector<Point2f>> model_LM_all_fix_only_glass; // all model img landmark
	for (int i=0; i<MPIE_image_name_glass.size();i++)
	{
		string img_load_path=MPIE_file_image_path_glass+MPIE_image_name_glass[i];
		//cout<<img_load_path<<endl;
		model_ori_img_mpie_only_glass=imread(img_load_path);
		model_ori_img_mpie_only_glass.copyTo(model_ori_img_d_mpie_only_glass);
		imshow("model_ori_img",model_ori_img_mpie_only_glass);waitKey(1);

		for (vector<Point2f>::iterator LM_pt=MPIE_LM_all_only_glass[i].begin();LM_pt!=MPIE_LM_all_only_glass[i].end();++LM_pt)
		{
			circle(model_ori_img_d_mpie_only_glass,*LM_pt,1,CV_RGB(0,255,0),-1);
		}
		imshow("model_ori_img_d",model_ori_img_d_mpie_only_glass);inputKey_mpie_only_glass=waitKey(0);

		vector<Point2f> img_LM_each;
		if (char(inputKey_mpie_only_glass) == 'z')
		{
			point_LM.clear();
			point_LM_temp.clear();
			point_count=1;
			model_ori_img_mpie_only_glass.copyTo(img_temp);
			point_LM_temp=MPIE_LM_all_only_glass[i];

			while(point_count<=23)
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
			img_LM_each=MPIE_LM_all_only_glass[i];
		}
		model_LM_all_fix_only_glass.push_back(img_LM_each);
	}


	FileStorage FS_MPIE_fix_only_glass;
	FS_MPIE_fix_only_glass.open(MPIE_lmpt_fix_only_glass, FileStorage::WRITE);
	for (int i=0;i<model_LM_all_fix_only_glass.size();i++)
	{
		stringstream int2str;
		string num;
		int2str << i;
		int2str >> num;
		string label="MPIE_LMPT_Data_Tar_"+num;
		FS_MPIE_fix_only_glass << label << model_LM_all_fix_only_glass[i];
	}
	FS_MPIE_fix_only_glass.release();
#else
	Mat model_ori_img_mpie_only_glass;
	Mat model_ori_img_d_mpie_only_glass;
	int inputKey_mpie_only_glass;
	for (int i=0; i<MPIE_image_name_glass.size();i++)
	{
		string img_load_path=MPIE_file_image_path_glass+MPIE_image_name_glass[i];
		//cout<<img_load_path<<endl;
		model_ori_img_mpie_only_glass=imread(img_load_path);
		model_ori_img_mpie_only_glass.copyTo(model_ori_img_d_mpie_only_glass);
		//imshow("model_ori_img",model_ori_img_mpie_only_glass);waitKey(1);

		for (vector<Point2f>::iterator LM_pt=MPIE_LM_all_only_glass[i].begin();LM_pt!=MPIE_LM_all_only_glass[i].end();++LM_pt)
		{
			circle(model_ori_img_d_mpie_only_glass,*LM_pt,1,CV_RGB(0,255,0),-1);
		}
		//imshow("model_ori_img_d2",model_ori_img_d_mpie_only_glass);inputKey_mpie=waitKey(0);
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
