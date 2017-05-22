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

using namespace std;
using namespace cv;

// glass faces //
int nose[8][3]={{1,3,2},
				{3,4,2},
				{3,5,4},
				{5,6,4},
				{1,2,43},
				{2,44,43},
				{43,44,45},
				{44,46,45}};
int L_U[14][3]={{3,7,5},
				{7,8,5},
				{7,9,8},
				{9,10,8},
				{9,11,10},
				{11,12,10},
				{11,13,12},
				{13,14,12},
				{13,15,14},
				{15,16,14},
				{15,17,18},
				{15,18,16},
				{17,19,18},
				{18,19,20}};
int L_Leg[14][3]={{17,22,19},
				{22,24,19},
				{22,21,24},
				{21,23,24},
				{25,26,27},
				{26,28,27},
				{22,26,25},
				{22,25,21},
				{22,26,28},
				{22,28,24},
				{21,25,27},
				{21,27,23},
				{24,28,27},
				{24,27,23}};
int L_B[16][3]={{19,20,29},
				{20,29,30},
				{29,31,30},
				{30,31,32},
				{32,31,33},
				{32,33,34},
				{34,33,35},
				{34,35,36},
				{35,37,36},
				{36,37,38},
				{38,37,39},
				{38,39,40},
				{40,39,41},
				{40,41,42},
				{42,41,4},
				{42,4,6}};
int R_U[14][3]={{43,48,47},
{43,45,48},
{47,48,50},
{47,50,49},
{49,50,52},
{49,52,51},
{51,52,54},
{51,54,53},
{53,54,56},
{53,56,55},
{55,56,58},
{55,58,57},
{58,60,57},
{57,60,59}};
int R_Leg[14][3]={{57,59,64},
{57,64,62},
{62,64,63},
{62,63,61},
{66,65,67},
{66,67,68},
{62,61,65},
{62,65,66},
{61,65,67},
{61,67,63},
{62,66,68},
{62,68,64},
{64,63,67},
{64,67,68}};
int R_B[16][3]={{59,60,70},
{59,70,69},
{69,70,72},
{69,72,71},
{71,72,74},
{71,74,73},
{73,74,76},
{73,76,75},
{75,76,78},
{75,78,77},
{78,80,77},
{77,80,79},
{79,80,82},
{79,82,81},
{81,82,46},
{81,46,44}};
//////////////////////////////////////////
///*          Sub function            *///
//////////////////////////////////////////

// model choose //
void model_choose_LMset_single(vector<Point2f> LM_in, vector<Point2f>* LM_out);
void model_choose_LMset_all(vector<vector<Point2f>> LM_in, vector<vector<Point2f>>* LM_out);
void draw_point(vector<vector<Point2f>> LM_in);
void model_choose(vector<vector<Point2f>> model_LM_all,vector<Point2f> MPIE_LM_each,vector<int>* choose_num, int* min_num);
void angle_calculate(vector<Point2f> LM_in, vector<float> *angle_out);

// depth warp //
void set_persp_aff(std::vector<cv::Point2f> ref_pt, std::vector<std::vector<cv::Point2f>>* out_pt, string name, int* affine_num);
void aff_warp(std::vector<cv::Point2f> src_pnt, std::vector<cv::Point2f> dst_pnt,cv::Mat img_in,cv::Mat *img_out,cv::Mat *mask_out);

// opengl //
void matrix_set(Mat* view,Mat* project);
void model_gravityPoint(Mat mask, Mat model_x, Mat model_y, Mat model_z, Mat *move_matrix);
void R_matrix(float Pitch, float Yaw, float Roll, Mat *r_matrix_out_x, Mat *r_matrix_out_y, Mat *r_matrix_out_z);
void mesh(Mat mask, Mat image, Mat model_x, Mat model_y, Mat model_z, vector<Point3f>* location, vector<Point3i>* faces, vector<Point3f>* img_color , vector<vector<uchar>>* img_color_u);

// model save //
void write_model(string savePath, Mat msak, Mat img, Mat model_x, Mat model_y, Mat model_z);
void write_model_LM(string savePath, Mat LM);
void write_ply_mesh(string savePath, vector<Point3f> location, vector<Point3i> faces, vector<vector<uchar>> img_color_u);
void write_ply_glass(string savePath, vector<Point3f> location, vector<Point3i> faces);

// mask //
void Create_face_mask(vector<Point2f> input_point, Mat &out_mask);
void Create_glass_mask(vector<Point2f> input_point, Mat &out_mask);

// glass model load //
void Load_Model(string path, vector<Point3f>* location, vector<Point3i>* faces);

// glass model create //
void create_glass_model(vector<Point2f> glass_2d_LM, Mat image, Mat model_z, vector<Point3f>* location, vector<Point3i>* faces, vector<Point3f>* img_color , vector<vector<uchar>>* img_color_u);

string data_file_title="../../using_data/";//工作路徑
float glass_color_R=0.0;
float glass_color_G=0.0;
float glass_color_B=0.0;
int main(int argc, char* argv[])
{
	//system("title MPIE glass");

	// save or load ethnicity data //
	string eth_Path=data_file_title+"xml"+"/"+"eth"+"/";

	// save file name //
	string all_id="all";
	//string glass_type_id="glass_frame_full_2"; //set glass type

	// load MPIE DATA //
	string angle="F00_05_1";
	string light="06";
	string glass_type="glass_frame_full"; //set glass type
	// glass_frame_full glass_frame_half glass_frame_none no_glass //
	string MPIE_load_path=data_file_title+"MPIE_classification"+"/"+angle+"/"+light+"/"+all_id+"/"; // title of FRGC file
	vector< string > MPIE_name; // use to save the FRGC model ID
	Load_insideFile_name(MPIE_load_path, &MPIE_name);

	// MPIE LM load path //
	string loadPath_MPIE=data_file_title+"xml"+"/"+"MPIE_data_fix"+"/"+angle+"/";

	// load FRGC DATA //
	string FRGC_CM_Path=data_file_title+"FRGC-model-CM"+"/";
	vector< string > FRGC_CM_name; // use to save the FRGC model ID
	Load_insideFile_name(FRGC_CM_Path, &FRGC_CM_name);

	string FRGC_CF_Path=data_file_title+"FRGC-model-CF"+"/";
	vector< string > FRGC_CF_name; // use to save the FRGC model ID
	Load_insideFile_name(FRGC_CF_Path, &FRGC_CF_name);

	string FRGC_AM_Path=data_file_title+"FRGC-model-AM"+"/";
	vector< string > FRGC_AM_name; // use to save the FRGC model ID
	Load_insideFile_name(FRGC_AM_Path, &FRGC_AM_name);

	string FRGC_AF_Path=data_file_title+"FRGC-model-AF"+"/";
	vector< string > FRGC_AF_name; // use to save the FRGC model ID
	Load_insideFile_name(FRGC_AF_Path, &FRGC_AF_name);

	// FRGC LM load path //
	string loadPath=data_file_title+"xml"+"/"+"FRGC_data_fix"+"/";

	vector<int> eth_no;
	// read ethnicity number //
	string loadPath_2=eth_Path+glass_type;
	string eth_num_load=loadPath_2+"/"+"ethnicity_num"+".xml";
	FileStorage FS_eth_LDT;
	FS_eth_LDT.open(eth_num_load, FileStorage::READ);
	FS_eth_LDT[ "LMPT_Data" ] >> eth_no;
	FS_eth_LDT.release();

	// Load GT glass LM pts //
	vector<vector<Point2f>> image_glass_LM; //MPIE all img landmark
	string load_lmpt=data_file_title+"xml"+"/"+"GT_glass_LM_v4"+"/"+angle+"/"+"LM_all_Data.xml";
	//string load_lmpt=data_file_title+"xml"+"/"+"GT_glass_LM_v5"+"/"+angle+"/"+"LM_all_Data.xml";
	//string load_lmpt=data_file_title+"xml"+"/"+"GT_glass_LM_v6"+"/"+angle+"/"+"LM_all_Data.xml";
	FileStorage FS_MPIE_LDT_R;
	FS_MPIE_LDT_R.open(load_lmpt, FileStorage::READ);
	for (int i=0;i<MPIE_name.size();i++)
	{
		vector<Point2f> temp;
		stringstream int2str;
		string num;
		int2str << i;
		int2str >> num;
		string label="LMPT_Data_Tar_"+num;
		FS_MPIE_LDT_R[label] >> temp;
		image_glass_LM.push_back(temp);
	}
	FS_MPIE_LDT_R.release();

	for (int id=0; id<MPIE_name.size(); id++)//for (int id=0; id<MPIE_name.size(); id++)
	{
		// title FRGC image load Path //
		string MPIE_image_load_path=MPIE_load_path+MPIE_name[id];
		cout<<MPIE_image_load_path<<endl;

		// opencv load image //
		Mat image_MPIE=imread(MPIE_image_load_path,1);
		cv::imshow("image_MPIE",image_MPIE);waitKey(1);
		Mat image_MPIE_after=image_MPIE.clone();image_MPIE_after.setTo(0);
		Mat image_MPIE_draw=image_MPIE.clone();
		Mat image_MPIE_remove=image_MPIE.clone();

		// glass color //
		float R_L=(float)image_MPIE.at<Vec3b>(image_glass_LM[id][8-1].y,image_glass_LM[id][8-1].x)[2];
		float G_L=(float)image_MPIE.at<Vec3b>(image_glass_LM[id][8-1].y,image_glass_LM[id][8-1].x)[1];
		float B_L=(float)image_MPIE.at<Vec3b>(image_glass_LM[id][8-1].y,image_glass_LM[id][8-1].x)[0];

		float R_R=(float)image_MPIE.at<Vec3b>(image_glass_LM[id][23-1].y,image_glass_LM[id][23-1].x)[2];
		float G_R=(float)image_MPIE.at<Vec3b>(image_glass_LM[id][23-1].y,image_glass_LM[id][23-1].x)[1];
		float B_R=(float)image_MPIE.at<Vec3b>(image_glass_LM[id][23-1].y,image_glass_LM[id][23-1].x)[0];
		//imshow("image glass",image_MPIE_draw);waitKey(0);

		glass_color_B=(B_L+B_R)/2;
		glass_color_G=(G_L+G_R)/2;
		glass_color_R=(R_L+R_R)/2;

		cout<<glass_color_R<<" "<<glass_color_G<<" "<<glass_color_B<<endl;

		// remove glass //
		cv::Mat_<uchar> glass_mask(480,640);
		glass_mask.setTo(0);
		Create_glass_mask(image_glass_LM[id],glass_mask);
		//imshow("glass_mask",glass_mask);waitKey(1);
		dilate(glass_mask,glass_mask,Mat(),Point(-1,-1),1);//擴張
		inpaint(image_MPIE, glass_mask, image_MPIE_remove, 3, CV_INPAINT_TELEA);
		//imshow("image_MPIE_remove_glass",image_MPIE_remove);waitKey(0);
		image_MPIE.setTo(0);
		image_MPIE=image_MPIE_remove.clone();

		// -----Read the Data into xml //
		vector<Point2f> image_LM;
		string MPIE_lmpt=loadPath_MPIE+MPIE_name[id].substr(0,3)+"_LM.xml";
		FileStorage FS_MPIE_LDT;
		FS_MPIE_LDT.open(MPIE_lmpt, FileStorage::READ);
		FS_MPIE_LDT[ "LMPT_Data" ] >> image_LM;
		FS_MPIE_LDT.release();

		// show Landmark //
		//for (int j=0; j<image_glass_LM[id].size(); j++)
		//{
		//	circle(image_MPIE_draw,image_glass_LM[id][j],3,CV_RGB(255,0,0),-1);
		//}
		//cv::imshow("image_draw_mySet",image_MPIE_draw);waitKey(0);
		//for (int j=0; j<image_LM.size(); j++)
		//{
		//	circle(image_MPIE_draw,image_LM[j],3,CV_RGB(0,255,0),-1);
		//}
		//cv::imshow("image_draw_mySet",image_MPIE_draw);waitKey(0);

		// put in //
		int eth_ch=eth_no[id];
		// default ethnicity model choose //
		//int eth_ch=1;
		//cout<<"please choose one ethnicity : "<<endl
		//	<<"1 : Caucasian Male"<<endl
		//	<<"2 : Caucasian Female"<<endl
		//	<<"3 : Asian Male"<<endl
		//	<<"4 : Asian Female"<<endl;
		//cin>>eth_ch;
		//eth_no.push_back(eth_ch);

		string FRGC_load_path;
		vector< string > FRGC_name;
		if (eth_ch==1)
		{
			FRGC_load_path=FRGC_CM_Path;
			FRGC_name=FRGC_CM_name;
		}
		else if (eth_ch==2)
		{
			FRGC_load_path=FRGC_CF_Path;
			FRGC_name=FRGC_CF_name;
		}
		else if (eth_ch==3)
		{
			FRGC_load_path=FRGC_AM_Path;
			FRGC_name=FRGC_AM_name;
		}
		else if (eth_ch==4)
		{
			FRGC_load_path=FRGC_AF_Path;
			FRGC_name=FRGC_AF_name;
		}
		else
		{
			cout<<"None of above have been chose, use Caucasian Male as default model."<<endl;
			FRGC_load_path=FRGC_CM_Path;
			FRGC_name=FRGC_CM_name;
		}

		// load model landmark //
		vector< vector<Point2f> > model_LM_all;
		for (int i=0; i<FRGC_name.size(); i++)
		{
			// -----Read the Data into xml //
			vector<Point2f> model_LM;
			string FRGC_model_lmpt=loadPath+FRGC_name[i]+"_LM.xml";
			FileStorage FS_LDT;
			FS_LDT.open(FRGC_model_lmpt, FileStorage::READ);
			FS_LDT[ "LMPT_Data" ] >> model_LM;
			FS_LDT.release();	

			model_LM_all.push_back(model_LM);
		}

		// model choose //
		// step1. MPIE LM norm //
		vector<Point2f> image_LM_mc;
		model_choose_LMset_single(image_LM, &image_LM_mc);
		vector< vector<Point2f> > model_LM_all_mc;
		model_choose_LMset_all(model_LM_all, &model_LM_all_mc);

		vector<int> model_choose_num; //
		int min_num=0; //五官最像
		model_choose(model_LM_all_mc,image_LM_mc, &model_choose_num, &min_num);
		min_num=0;
		//cout<<min_num<<endl;

		// normalized //
		// step.1 basic on eye Landmark to rotate the face at 0 degree // 
		float angle_to_zero; // angle between input and 0
		find_theate_zero(image_LM, &angle_to_zero);
		Point2f center_point; // find the center point of landmark
		find_LM_center(image_LM, &center_point); 
		Mat Rotate_Matrix=getRotationMatrix2D(center_point, angle_to_zero, 1.0); // calculate the rotate matrix [2x3]
		vector<Point2f> img_LM_after; // save the rotated landmark
		vector<Point2f> img_glass_LM_after; // save the rotated landmark
		rotate_LM(image_LM, &img_LM_after, angle_to_zero); 
		rotate_LM(image_glass_LM[id], &img_glass_LM_after, angle_to_zero); 
		cv::warpAffine(image_MPIE, image_MPIE_after, Rotate_Matrix, image_MPIE_after.size()); // rotate the ori. image 
		image_MPIE_draw.setTo(0);
		image_MPIE_draw=image_MPIE_after.clone();
		//for (int j=0; j<img_LM_after.size(); j++)
		//{
		//	circle(image_MPIE_draw,img_LM_after[j],1,CV_RGB(0,255,0),-1);
		//}
		//imshow("image after rotate",image_MPIE_draw);waitKey(1);
		//for (int j=0; j<img_glass_LM_after.size(); j++)
		//{
		//	circle(image_MPIE_draw,img_glass_LM_after[j],1,CV_RGB(255,255,0),-1);
		//}
		//imshow("image after rotate",image_MPIE_draw);waitKey(0);

		// step.2 normalized the length between two eyes;
		Point2f Leye=Point2f((img_LM_after[20-1].x+img_LM_after[23-1].x)/2,(img_LM_after[20-1].y+img_LM_after[23-1].y)/2);
		Point2f Reye=Point2f((img_LM_after[26-1].x+img_LM_after[29-1].x)/2,(img_LM_after[26-1].y+img_LM_after[29-1].y)/2);
		Point2f eye_center=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);
		Point2f dst_pts=Point2f(320,160);
		//imshow("image_after",image_MPIE_after);waitKey(1);
		move_img(image_MPIE_after, &image_MPIE_after, dst_pts, eye_center); // move the image to eyes center
		move_LM_point(img_LM_after, &img_LM_after, dst_pts, eye_center); // 移動測試影像的LM點
		move_LM_point(img_glass_LM_after, &img_glass_LM_after, dst_pts, eye_center); // 移動測試影像的LM點
		image_MPIE_draw.setTo(0);
		image_MPIE_draw=image_MPIE_after.clone();
		//for (int j=0; j<img_LM_after.size(); j++)
		//{
		//	circle(image_MPIE_draw,img_LM_after[j],1,CV_RGB(0,255,0),-1);
		//}
		//imshow("image after move",image_MPIE_draw);waitKey(1);
		//for (int j=0; j<img_glass_LM_after.size(); j++)
		//{
		//	circle(image_MPIE_draw,img_glass_LM_after[j],1,CV_RGB(255,255,0),-1);
		//}
		//imshow("image after rotate",image_MPIE_draw);waitKey(0);

		float eye_length=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
		float norm_length=70.0;
		float scale_IMG=norm_length/eye_length;
		cv::resize(image_MPIE_after,image_MPIE_after,Size(image_MPIE_after.cols*scale_IMG,image_MPIE_after.rows*scale_IMG)); //normlized image
		scale_LM(img_LM_after,&img_LM_after, scale_IMG); //縮放移動後的LM
		scale_LM(img_glass_LM_after,&img_glass_LM_after, scale_IMG); //縮放移動後的LM
		image_MPIE_draw.setTo(0);
		image_MPIE_draw=image_MPIE_after.clone();
		//for (int j=0; j<img_LM_after.size(); j++)
		//{
		//	circle(image_MPIE_draw,img_LM_after[j],1,CV_RGB(0,255,0),-1);
		//}
		//imshow("image after norm",image_MPIE_draw);waitKey(1);
		//for (int j=0; j<img_glass_LM_after.size(); j++)
		//{
		//	circle(image_MPIE_draw,img_glass_LM_after[j],1,CV_RGB(255,255,0),-1);
		//}
		//imshow("image after rotate",image_MPIE_draw);waitKey(0);

		// save ROI //
		Leye=Point2f((img_LM_after[20-1].x+img_LM_after[23-1].x)/2,(img_LM_after[20-1].y+img_LM_after[23-1].y)/2);
		Reye=Point2f((img_LM_after[26-1].x+img_LM_after[29-1].x)/2,(img_LM_after[26-1].y+img_LM_after[29-1].y)/2);
		eye_length=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
		//cout<<eye_length<<endl;
		eye_center=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);
		dst_pts=Point2f(320,160);
		move_img(image_MPIE_after, &image_MPIE_after, dst_pts, eye_center); // move the image to eyes center
		move_LM_point(img_LM_after, &img_LM_after, dst_pts, eye_center); // 移動測試影像的LM點
		move_LM_point(img_glass_LM_after, &img_glass_LM_after, dst_pts, eye_center); // 移動測試影像的LM點
		image_MPIE_draw.setTo(0);
		image_MPIE_draw=image_MPIE_after.clone();
		for (int j=0; j<img_LM_after.size(); j++)
		{
			circle(image_MPIE_draw,img_LM_after[j],1,CV_RGB(0,255,0),-1);
		}
		imshow("image after norm reMove",image_MPIE_draw);waitKey(1);
		//for (int j=0; j<img_glass_LM_after.size(); j++)
		//{
		//	cout<<j+1<<endl;
		//	circle(image_MPIE_draw,img_glass_LM_after[j],1,CV_RGB(255,255,0),-1);
		//	imshow("image after rotate",image_MPIE_draw);waitKey(0);
		//}
		//imshow("image after rotate",image_MPIE_draw);waitKey(0);
		//eye_length=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
		//Mat image_crop=image_MPIE_after(Rect(eye_center.x-100,eye_center.y-65,200,200)).clone();
		//dst_pts=Point2f(100,65);
		//move_LM_point(img_LM_after, &img_LM_after, dst_pts, eye_center); // 移動測試影像的LM點
		//image_MPIE_draw.setTo(0);
		//image_MPIE_draw=image_crop.clone();
		//for (int j=0; j<img_LM_after.size(); j++)
		//{
		//	circle(image_MPIE_draw,img_LM_after[j],1,CV_RGB(0,255,0),-1);
		//}
		//imshow("image norm crop",image_MPIE_draw);waitKey(0);

		// Depth Warp //
		// projectMatrix //
		Mat ViewMatrix;
		Mat ProjMatrix;
		matrix_set(&ViewMatrix,&ProjMatrix);
		// load model x y z //
		string model_savePath=data_file_title+"FRGC_model_xml"+"/";
		Mat model_x,model_y,model_z;
		// -----Read the Data into xml //
		string FRGC_model_save=model_savePath+FRGC_name[min_num]+"_x.xml";
		FileStorage FS_LDT_M;
		FS_LDT_M.open(FRGC_model_save, FileStorage::READ);
		FS_LDT_M[ "M_Data" ] >> model_x;
		FS_LDT_M.release();	

		FRGC_model_save=model_savePath+FRGC_name[min_num]+"_y.xml";
		FS_LDT_M.open(FRGC_model_save, FileStorage::READ);
		FS_LDT_M[ "M_Data" ] >> model_y;
		FS_LDT_M.release();	

		FRGC_model_save=model_savePath+FRGC_name[min_num]+"_z.xml";
		FS_LDT_M.open(FRGC_model_save, FileStorage::READ);
		FS_LDT_M[ "M_Data" ] >> model_z;
		FS_LDT_M.release();	

		cv::Mat_<uchar> affine_Z_mask(480,640);
		affine_Z_mask.setTo(0);
		Create_face_mask(model_LM_all[min_num],affine_Z_mask);
		//imshow("affine_Z_mask",affine_Z_mask);waitKey(1);
		for (int i=0;i<affine_Z_mask.rows;i++)
		{
			for (int j=0;j<affine_Z_mask.cols;j++)
			{
				if (model_z.at<float>(i,j) == 0)
				{
					affine_Z_mask.at<uchar>(i,j)=0;
				}
			}
		}
		imshow("affine_Z_mask_after",affine_Z_mask);waitKey(1);
		int count_warp=0;
		count_warp=cv::countNonZero(affine_Z_mask);
		Mat model_data_ori(4,count_warp,CV_32FC1);model_data_ori.setTo(0);
		int aa=0;
		for (int i = 0; i < affine_Z_mask.rows; i++)
		{
			for (int j = 0; j < affine_Z_mask.cols; j++)
			{
				if (affine_Z_mask.at<uchar>(i,j)!=0)
				{
					model_data_ori.at<float>(0,aa)=model_x.at<float>(i,j);
					model_data_ori.at<float>(1,aa)=model_y.at<float>(i,j);
					model_data_ori.at<float>(2,aa)=model_z.at<float>(i,j);
					model_data_ori.at<float>(3,aa)=1.0;
					aa=aa+1;
				}
			}
		}


		int model_data_LM_num[]={11,12,13,14,20,23,26,29,32,35,38,41,50,51,52,53,54,55,56,57,58,62,61,60,59,63,64,65,66};
		vector<int> number(model_data_LM_num, model_data_LM_num + sizeof(model_data_LM_num)/sizeof(model_data_LM_num[0]));
		Mat model_data_LM_ori(4,number.size(),CV_32FC1);
		for (int i=0; i<number.size(); i++)
		{
			model_data_LM_ori.at<float>(0,i)=model_x.at<float>(model_LM_all[min_num][number[i]-1].y,model_LM_all[min_num][number[i]-1].x);
			model_data_LM_ori.at<float>(1,i)=model_y.at<float>(model_LM_all[min_num][number[i]-1].y,model_LM_all[min_num][number[i]-1].x);
			model_data_LM_ori.at<float>(2,i)=model_z.at<float>(model_LM_all[min_num][number[i]-1].y,model_LM_all[min_num][number[i]-1].x);
			model_data_LM_ori.at<float>(3,i)=1.0;
		}
		//cout<<model_data_LM_ori<<endl;

		// gravity point //
		Mat move_matrix;
		model_gravityPoint(affine_Z_mask, model_x, model_y, model_z, &move_matrix);

		find_theate_zero(model_LM_all[min_num], &angle_to_zero);
		rotate_LM(model_LM_all[min_num], &model_LM_all[min_num], angle_to_zero);

		// rotate matrix //
		Mat rotate_matrix_x,rotate_matrix_y,rotate_matrix_z;
		R_matrix(0.0, 0.0, angle_to_zero, &rotate_matrix_x, &rotate_matrix_y, &rotate_matrix_z);

		Mat model_data_aft=move_matrix.inv()*rotate_matrix_z*rotate_matrix_y*rotate_matrix_x*move_matrix*model_data_ori;
		Mat model_data_aft_D=ViewMatrix*ProjMatrix*move_matrix.inv()*rotate_matrix_z*rotate_matrix_y*rotate_matrix_x*move_matrix*model_data_ori;
		Mat model_data_LM_aft=move_matrix.inv()*rotate_matrix_z*rotate_matrix_y*rotate_matrix_x*move_matrix*model_data_LM_ori;
		Mat model_data_LM_aft_D=ViewMatrix*ProjMatrix*move_matrix.inv()*rotate_matrix_z*rotate_matrix_y*rotate_matrix_x*move_matrix*model_data_LM_ori;

		Mat model_x_in=model_x.clone(); model_x_in.setTo(0);
		Mat model_y_in=model_y.clone(); model_y_in.setTo(0);
		Mat model_z_in=model_z.clone(); model_z_in.setTo(0);
		Mat mask=affine_Z_mask.clone(); mask.setTo(0);
		for(int i=0; i<model_data_aft_D.cols ;i++)
		{
			//if (point_lm_3d.at<float>(2,i)==0){continue;}
			if (model_data_aft_D.at<float>(1,i)<0 || model_data_aft_D.at<float>(1,i)>480){continue;}
			if (model_data_aft_D.at<float>(0,i)<0 || model_data_aft_D.at<float>(0,i)>640){continue;}
			model_x_in.at<float>(model_data_aft_D.at<float>(1,i),model_data_aft_D.at<float>(0,i))=model_data_aft.at<float>(0,i);
			model_y_in.at<float>(model_data_aft_D.at<float>(1,i),model_data_aft_D.at<float>(0,i))=model_data_aft.at<float>(1,i);
			model_z_in.at<float>(model_data_aft_D.at<float>(1,i),model_data_aft_D.at<float>(0,i))=model_data_aft.at<float>(2,i);
			mask.at<uchar>(model_data_aft_D.at<float>(1,i),model_data_aft_D.at<float>(0,i))=255.0;
		}
		//cvtColor(mask,mask,CV_GRAY2BGR);
		//for(int i=0; i<model_data_LM_aft_D.cols ;i++)
		//{
		//	circle(mask,Point2f(model_data_LM_aft_D.at<float>(0,i),model_data_LM_aft_D.at<float>(1,i)),2,CV_RGB(0,255,0),-1);
		//}
		//for(int i=0; i<model_LM_all[min_num].size() ;i++)
		//{
		//	circle(mask,model_LM_all[min_num][i],2,CV_RGB(255,0,0),-1);
		//}
		//imshow("mask",mask);waitKey(0);

		//cout<<model_data_LM_aft<<endl;

		Point2f Leye_FRGC_L=Point2f(model_data_LM_aft.at<float>(0,5-1),model_data_LM_aft.at<float>(1,5-1));
		Point2f Leye_FRGC_R=Point2f(model_data_LM_aft.at<float>(0,6-1),model_data_LM_aft.at<float>(1,6-1));
		Point2f Reye_FRGC_L=Point2f(model_data_LM_aft.at<float>(0,7-1),model_data_LM_aft.at<float>(1,7-1));
		Point2f Reye_FRGC_R=Point2f(model_data_LM_aft.at<float>(0,8-1),model_data_LM_aft.at<float>(1,8-1));
		Point2f Leye_FRGC=Point2f((Leye_FRGC_L.x+Leye_FRGC_R.x)/2,(Leye_FRGC_L.y+Leye_FRGC_R.y)/2);
		Point2f Reye_FRGC=Point2f((Reye_FRGC_L.x+Reye_FRGC_R.x)/2,(Reye_FRGC_L.y+Reye_FRGC_R.y)/2);
		//cout<<Leye_FRGC<<endl;
		//cout<<Reye_FRGC<<endl;
		float eye_length_FRGC=sqrt((Leye_FRGC.x-Reye_FRGC.x)*(Leye_FRGC.x-Reye_FRGC.x)+(Leye_FRGC.y-Reye_FRGC.y)*(Leye_FRGC.y-Reye_FRGC.y));
		//cout<<eye_length_FRGC<<endl;


		rotate_LM(model_LM_all[min_num], &model_LM_all[min_num], -angle_to_zero);
		// warp image //
		std::vector<cv::Point2f> src_pt=model_LM_all[min_num];
		std::vector<cv::Point2f> dst_pt=img_LM_after;
		int aff_num=0;
		std::vector<std::vector<cv::Point2f>> aff_src_pt;
		std::vector<std::vector<cv::Point2f>> aff_dst_pt;
		string title=data_file_title+"old_rec_data/";
		string warp_txt_path=title+"Warp-txt/persp_v6.txt";
		set_persp_aff(src_pt,&aff_src_pt,warp_txt_path,&aff_num);
		set_persp_aff(dst_pt,&aff_dst_pt,warp_txt_path,&aff_num);
		cv::Mat_<uchar> affine_Z_mask_t(480,640);
		affine_Z_mask_t.setTo(0);
		cv::Mat_<uchar> temp_mask(480,640);
		temp_mask.setTo((0));
		cv::Mat_<float> zWarp_temp(480,640);zWarp_temp.setTo(0);
		cv::Mat_<float> zWarp_all(480,640);zWarp_all.setTo(0);
		//Mat zWarp_temp=model_z.clone();zWarp_temp.setTo(0);
		//Mat zWarp_all=model_z.clone();zWarp_all.setTo(0);
		for(int i=0; i<aff_num; i++)
		{
			aff_warp(aff_src_pt[i], aff_dst_pt[i], model_z, &zWarp_temp,&temp_mask);
			for (int k=0;k<temp_mask.rows;k++)
			{
				for (int l=0;l<temp_mask.rows;l++)
				{
					if (temp_mask.at<uchar>(k,l)!=0)
					{
						int step=1;
						int diff=30;
						if (zWarp_temp.at<float>(k,l)==0)
						{
							temp_mask.at<uchar>(k,l)=0;
						}
						else
						{
							if(zWarp_temp.at<float>(k,l) > -1500)
							{
								temp_mask.at<uchar>(k,l)=0;
							}
							if(abs(zWarp_temp.at<float>(k,l)-zWarp_temp.at<float>(k+step,l)) > diff)
							{
								temp_mask.at<uchar>(k,l)=0;
							}
							if(abs(zWarp_temp.at<float>(k,l)-zWarp_temp.at<float>(k-step,l)) > diff)
							{
								temp_mask.at<uchar>(k,l)=0;
							}
							if(abs(zWarp_temp.at<float>(k,l)-zWarp_temp.at<float>(k,l+step)) > diff)
							{
								temp_mask.at<uchar>(k,l)=0;
							}
							if(abs(zWarp_temp.at<float>(k,l)-zWarp_temp.at<float>(k,l-step)) > diff)
							{
								temp_mask.at<uchar>(k,l)=0;
							}
						}
					}
				}
			}
			zWarp_temp.copyTo(zWarp_all,temp_mask);
			zWarp_temp.setTo(0);
			affine_Z_mask_t=affine_Z_mask_t+temp_mask;
			//imshow("temp_mask",temp_mask);waitKey(1);
			//imshow("affine_Z_mask_t",affine_Z_mask_t);waitKey(0);
			temp_mask.setTo((0));
		}
		cv::Mat_<float> model_z_out(480,640);model_z_out.setTo(0);
		//Mat model_z_out=model_z.clone();model_z_out.setTo(0);
		zWarp_all.copyTo(model_z_out);

		//imshow("affine_Z_mask_t",affine_Z_mask_t);waitKey(1);

		cv::Mat_<uchar> affine_Z_mask_i(480,640);
		affine_Z_mask_i.setTo(0);
		Create_face_mask(img_LM_after,affine_Z_mask_i);
		imshow("affine_Z_mask_i",affine_Z_mask_i);waitKey(1);
		// model_x model_y
		Mat model_x_out=model_z.clone();model_x_out.setTo(0);
		Mat model_y_out=model_z.clone();model_y_out.setTo(0);
		for (int i=0; i<affine_Z_mask_i.rows; i++)
		{
			for (int j=0; j<affine_Z_mask_i.cols; j++)
			{
				if (affine_Z_mask_i.at<uchar>(i,j)!=0)
				{
					Mat temp(4,1,CV_32FC1);
					Mat ProjMatrix_t=ProjMatrix.t();
					temp.at<float>(0,0)=j;
					temp.at<float>(1,0)=i;
					temp.at<float>(2,0)=1.0;
					temp.at<float>(3,0)=1.0;
					temp=ProjMatrix.inv()*ViewMatrix.inv()*temp;
					model_x_out.at<float>(i,j)=temp.at<float>(0,0);
					model_y_out.at<float>(i,j)=temp.at<float>(1,0);
				}
			}
		}
		//imshow("model_x",model_x);
		//imshow("model_y",model_y);
		//imshow("model_z",model_z);waitKey(0);
		Point2f Leye_MPIE_L=Point2f(model_x_out.at<float>(img_LM_after[20-1].y,img_LM_after[20-1].x),model_y_out.at<float>(img_LM_after[20-1].y,img_LM_after[20-1].x));
		Point2f Leye_MPIE_R=Point2f(model_x_out.at<float>(img_LM_after[23-1].y,img_LM_after[23-1].x),model_y_out.at<float>(img_LM_after[23-1].y,img_LM_after[23-1].x));
		Point2f Reye_MPIE_L=Point2f(model_x_out.at<float>(img_LM_after[26-1].y,img_LM_after[26-1].x),model_y_out.at<float>(img_LM_after[26-1].y,img_LM_after[26-1].x));
		Point2f Reye_MPIE_R=Point2f(model_x_out.at<float>(img_LM_after[29-1].y,img_LM_after[29-1].x),model_y_out.at<float>(img_LM_after[29-1].y,img_LM_after[29-1].x));
		//cout<<Leye_MPIE_L<<endl;
		//cout<<Leye_MPIE_R<<endl;
		//cout<<Reye_MPIE_L<<endl;
		//cout<<Reye_MPIE_R<<endl;
		Point2f Leye_MPIE=Point2f((Leye_MPIE_L.x+Leye_MPIE_R.x)/2,(Leye_MPIE_L.y+Leye_MPIE_R.y)/2);
		Point2f Reye_MPIE=Point2f((Reye_MPIE_L.x+Reye_MPIE_R.x)/2,(Reye_MPIE_L.y+Reye_MPIE_R.y)/2);
		float eye_length_MPIE=sqrt((Leye_MPIE.x-Reye_MPIE.x)*(Leye_MPIE.x-Reye_MPIE.x)+(Leye_MPIE.y-Reye_MPIE.y)*(Leye_MPIE.y-Reye_MPIE.y));
		//cout<<eye_length_MPIE<<endl;
		float depth_scale=eye_length_MPIE/eye_length_FRGC;
		//cout<<depth_scale<<endl;
		for (int i=0;i<model_z_out.rows;i++)
		{
			for (int j=0;j<model_z_out.cols;j++)
			{
				if (model_z_out.at<float>(i,j)!=0)
				{
					model_z_out.at<float>(i,j)=model_z_out.at<float>(i,j)*depth_scale;
				}
			}
		}
		Leye_MPIE=Point2f((img_LM_after[20-1].x+img_LM_after[23-1].x)/2,(img_LM_after[20-1].y+img_LM_after[23-1].y)/2);
		Reye_MPIE=Point2f((img_LM_after[26-1].x+img_LM_after[29-1].x)/2,(img_LM_after[26-1].y+img_LM_after[29-1].y)/2);
		Point2f eye_center_MPIE=Point2f((Leye_MPIE.x+Reye_MPIE.x)/2,(Leye_MPIE.y+Reye_MPIE.y)/2);
		float depth_z=model_z_out.at<float>(eye_center_MPIE.y,eye_center_MPIE.x);
		//cout<<depth_z<<endl;
		float dst_depth=-1900.0;
		for (int i=0;i<model_z_out.rows;i++)
		{
			for (int j=0;j<model_z_out.cols;j++)
			{
				if (model_z_out.at<float>(i,j) != 0)
				{
					model_z_out.at<float>(i,j)=model_z_out.at<float>(i,j)+(dst_depth-depth_z);
				}
			}
		}

		//Mat model_LM_3d(4,number.size(),CV_32FC1);model_LM_3d.setTo(0);
		//for (int i=0; i<number.size(); i++)
		//{
		//	model_LM_3d.at<float>(0,i)=model_x_out.at<float>(img_LM_after[number[i]-1].y,img_LM_after[number[i]-1].x);
		//	model_LM_3d.at<float>(1,i)=model_y_out.at<float>(img_LM_after[number[i]-1].y,img_LM_after[number[i]-1].x);
		//	model_LM_3d.at<float>(2,i)=model_data_LM_aft.at<float>(2,i)*depth_scale+(dst_depth-depth_z);
		//	model_LM_3d.at<float>(3,i)=1.0;
		//}

		//imshow("affine_Z_mask_t",affine_Z_mask_t);waitKey(0);
		for (int i = 0; i < affine_Z_mask_t.rows; i++)
		{
			for (int j = 0; j < affine_Z_mask_t.cols; j++)
			{
				if (affine_Z_mask_t.at<uchar>(i,j)!=0)
				{
					if (model_z_out.at<float>(i,j) < -1930)
					{
						affine_Z_mask_t.at<uchar>(i,j)=0;
					}
				}
			}
		}
		//imshow("affine_Z_mask_t",affine_Z_mask_t);waitKey(0);

		Mat model_LM_3d(4,number.size(),CV_32FC1);model_LM_3d.setTo(0);
		for (int i=0; i<number.size(); i++)
		{
			//model_LM_3d.at<float>(0,i)=model_x_out.at<float>(img_LM_after[number[i]-1].y,img_LM_after[number[i]-1].x);
			//model_LM_3d.at<float>(1,i)=model_y_out.at<float>(img_LM_after[number[i]-1].y,img_LM_after[number[i]-1].x);
			//model_LM_3d.at<float>(2,i)=model_data_LM_aft.at<float>(2,i)*depth_scale+(dst_depth-depth_z);
			//model_LM_3d.at<float>(3,i)=1.0;

			if (number[i]==20 || number[i]==26 || number[i]==51 || number[i]==52 || number[i]==53 || number[i]==54 || number[i]==62 || number[i]==61 || number[i]==60 || number[i]==59)
			{
				for (int j = 0; j < 50; j++)
				{
					float comp_1=affine_Z_mask_t.at<uchar>(img_LM_after[number[i]-1].y,img_LM_after[number[i]-1].x+j);
					if (comp_1 != 0)
					{
						img_LM_after[number[i]-1].x=img_LM_after[number[i]-1].x+j;
						model_LM_3d.at<float>(0,i)=model_x_out.at<float>(img_LM_after[number[i]-1].y,img_LM_after[number[i]-1].x);
						model_LM_3d.at<float>(1,i)=model_y_out.at<float>(img_LM_after[number[i]-1].y,img_LM_after[number[i]-1].x);
						model_LM_3d.at<float>(2,i)=model_z_out.at<float>(img_LM_after[number[i]-1].y,img_LM_after[number[i]-1].x);
						model_LM_3d.at<float>(3,i)=1.0;
						break;
					}
				}
			}
			else if (number[i]==23 || number[i]==29 || number[i]==55 || number[i]==56 || number[i]==57 || number[i]==58 || number[i]==63 || number[i]==64 || number[i]==65 || number[i]==66)
			{
				for (int j = 0; j < 50; j++)
				{
					float comp_1=affine_Z_mask_t.at<uchar>(img_LM_after[number[i]-1].y,img_LM_after[number[i]-1].x-j);
					if (comp_1 != 0)
					{
						img_LM_after[number[i]-1].x=img_LM_after[number[i]-1].x-j;
						model_LM_3d.at<float>(0,i)=model_x_out.at<float>(img_LM_after[number[i]-1].y,img_LM_after[number[i]-1].x);
						model_LM_3d.at<float>(1,i)=model_y_out.at<float>(img_LM_after[number[i]-1].y,img_LM_after[number[i]-1].x);
						model_LM_3d.at<float>(2,i)=model_z_out.at<float>(img_LM_after[number[i]-1].y,img_LM_after[number[i]-1].x);
						model_LM_3d.at<float>(3,i)=1.0;
						break;
					}
				}
			}
			else
			{
				model_LM_3d.at<float>(0,i)=model_x_out.at<float>(img_LM_after[number[i]-1].y,img_LM_after[number[i]-1].x);
				model_LM_3d.at<float>(1,i)=model_y_out.at<float>(img_LM_after[number[i]-1].y,img_LM_after[number[i]-1].x);
				model_LM_3d.at<float>(2,i)=model_z_out.at<float>(img_LM_after[number[i]-1].y,img_LM_after[number[i]-1].x);
				model_LM_3d.at<float>(3,i)=1.0;
			}
		}

		// glass model create //
		vector<Point3f> location_glass;
		vector<Point3i> faces_glass;
		vector<Point3f> img_color_glass;
		vector<vector<uchar>> img_color_u_glass;
		create_glass_model(img_glass_LM_after, image_MPIE_after, model_z_out, &location_glass, &faces_glass, &img_color_glass, &img_color_u_glass);

		// face mesh //
		vector<Point3f> location;
		vector<Point3i> faces;
		vector<Point3f> img_color;
		vector<vector<uchar>> img_color_u;
		mesh(affine_Z_mask_t, image_MPIE_after, model_x_out, model_y_out, model_z_out, &location, &faces, &img_color, &img_color_u);


		// model save //
		string Regmodel_savePath_light=data_file_title+"Reg_model"+"/"+light;
		_mkdir(Regmodel_savePath_light.c_str());
		string Regmodel_savePath_all=Regmodel_savePath_light+"/"+all_id;
		_mkdir(Regmodel_savePath_all.c_str());
		//string Regmodel_savePath_type=Regmodel_savePath_light+"/"+glass_type_id;
		//_mkdir(Regmodel_savePath_type.c_str());
		string Regmodel_savePath_all_id=Regmodel_savePath_all+"/"+MPIE_name[id].substr(0,3)+"/";
		_mkdir(Regmodel_savePath_all_id.c_str());
		//string Regmodel_savePath_type_id=Regmodel_savePath_type+"/"+MPIE_name[id].substr(0,3)+"/";
		//_mkdir(Regmodel_savePath_type_id.c_str());
		string REG_MODEL, REG_MODEL_LM, REG_MODEL_M, REG_MODEL_G;
		// all //
		REG_MODEL=Regmodel_savePath_all_id+MPIE_name[id].substr(0,3)+".ply";
		REG_MODEL_LM=Regmodel_savePath_all_id+MPIE_name[id].substr(0,3)+"_LM.ply";
		REG_MODEL_M=Regmodel_savePath_all_id+MPIE_name[id].substr(0,3)+"_M.ply";
		REG_MODEL_G=Regmodel_savePath_all_id+MPIE_name[id].substr(0,3)+"_G.ply";
		write_model(REG_MODEL, affine_Z_mask_t, image_MPIE_after, model_x_out, model_y_out, model_z_out);
		write_model_LM(REG_MODEL_LM, model_LM_3d);
		write_ply_mesh(REG_MODEL_M, location,  faces,  img_color_u);
		write_ply_mesh(REG_MODEL_G, location_glass,  faces_glass,  img_color_u_glass);
		//write_ply_glass(REG_MODEL_G, location_aft, faces_glass);
		// type //
		//REG_MODEL=Regmodel_savePath_type_id+MPIE_name[id].substr(0,3)+".ply";
		//REG_MODEL_LM=Regmodel_savePath_type_id+MPIE_name[id].substr(0,3)+"_LM.ply";
		//REG_MODEL_M=Regmodel_savePath_type_id+MPIE_name[id].substr(0,3)+"_M.ply";
		//REG_MODEL_G=Regmodel_savePath_type_id+MPIE_name[id].substr(0,3)+"_G.ply";
		write_model(REG_MODEL, affine_Z_mask_t, image_MPIE_after, model_x_out, model_y_out, model_z_out);
		write_model_LM(REG_MODEL_LM, model_LM_3d);
		write_ply_mesh(REG_MODEL_M, location,  faces,  img_color_u);
		write_ply_mesh(REG_MODEL_G, location_glass,  faces_glass,  img_color_u_glass);
		//write_ply_glass(REG_MODEL_G, location_aft, faces_glass);

		//system("pause");

		// xml save //
		string MODEL_point, MODEL_face, MODEL_color, MODEL_LM, MODEL_glass_p, MODEL_glass_f, MODEL_glass_c;
		string MODEL_x, MODEL_y, MODEL_z, MODEL_m, MODEL_c;
		FileStorage FS_LDT;

		// all //
		MODEL_point=Regmodel_savePath_all_id+MPIE_name[id].substr(0,3)+"_v.xml";
		FS_LDT.open(MODEL_point, FileStorage::WRITE);
		FS_LDT << "V_Data" << location;
		FS_LDT.release();

		MODEL_face=Regmodel_savePath_all_id+MPIE_name[id].substr(0,3)+"_f.xml";
		FS_LDT.open(MODEL_face, FileStorage::WRITE);
		FS_LDT << "F_Data" << faces;
		FS_LDT.release();

		MODEL_color=Regmodel_savePath_all_id+MPIE_name[id].substr(0,3)+"_c.xml";
		FS_LDT.open(MODEL_color, FileStorage::WRITE);
		FS_LDT << "C_Data" << img_color;
		FS_LDT.release();

		MODEL_LM=Regmodel_savePath_all_id+MPIE_name[id].substr(0,3)+"_l.xml";
		FS_LDT.open(MODEL_LM, FileStorage::WRITE);
		FS_LDT << "L_Data" << model_LM_3d;
		FS_LDT.release();

		//MODEL_x=Regmodel_savePath_all_id+MPIE_name[id].substr(0,3)+"_x.xml";
		//FS_LDT.open(MODEL_x, FileStorage::WRITE);
		//FS_LDT << "Data" << model_x_out;
		//FS_LDT.release();

		//MODEL_y=Regmodel_savePath_all_id+MPIE_name[id].substr(0,3)+"_y.xml";
		//FS_LDT.open(MODEL_y, FileStorage::WRITE);
		//FS_LDT << "Data" << model_y_out;
		//FS_LDT.release();

		//MODEL_z=Regmodel_savePath_all_id+MPIE_name[id].substr(0,3)+"_z.xml";
		//FS_LDT.open(MODEL_z, FileStorage::WRITE);
		//FS_LDT << "Data" << model_z_out;
		//FS_LDT.release();

		//MODEL_m=Regmodel_savePath_all_id+MPIE_name[id].substr(0,3)+"_mask.xml";
		//FS_LDT.open(MODEL_m, FileStorage::WRITE);
		//FS_LDT << "Data" << affine_Z_mask_t;
		//FS_LDT.release();

		//MODEL_c=Regmodel_savePath_all_id+MPIE_name[id].substr(0,3)+"_color.xml";
		//FS_LDT.open(MODEL_c, FileStorage::WRITE);
		//FS_LDT << "Data" << image_MPIE_after;
		//FS_LDT.release();

		MODEL_glass_p=Regmodel_savePath_all_id+MPIE_name[id].substr(0,3)+"_vg.xml";
		FS_LDT.open(MODEL_glass_p, FileStorage::WRITE);
		FS_LDT << "V_Data" << location_glass;
		FS_LDT.release();

		MODEL_glass_f=Regmodel_savePath_all_id+MPIE_name[id].substr(0,3)+"_fg.xml";
		FS_LDT.open(MODEL_glass_f, FileStorage::WRITE);
		FS_LDT << "F_Data" << faces_glass;
		FS_LDT.release();

		MODEL_glass_c=Regmodel_savePath_all_id+MPIE_name[id].substr(0,3)+"_cg.xml";
		FS_LDT.open(MODEL_glass_c, FileStorage::WRITE);
		FS_LDT << "C_Data" << img_color_glass;
		FS_LDT.release();

		MODEL_c=Regmodel_savePath_all_id+MPIE_name[id].substr(0,3)+"_eth.xml";
		FS_LDT.open(MODEL_c, FileStorage::WRITE);
		FS_LDT << "Data" << eth_ch;
		FS_LDT.release();

		// type //
		/*MODEL_point=Regmodel_savePath_type_id+MPIE_name[id].substr(0,3)+"_v.xml";
		FS_LDT.open(MODEL_point, FileStorage::WRITE);
		FS_LDT << "V_Data" << location;
		FS_LDT.release();

		MODEL_face=Regmodel_savePath_type_id+MPIE_name[id].substr(0,3)+"_f.xml";
		FS_LDT.open(MODEL_face, FileStorage::WRITE);
		FS_LDT << "F_Data" << faces;
		FS_LDT.release();

		MODEL_color=Regmodel_savePath_type_id+MPIE_name[id].substr(0,3)+"_c.xml";
		FS_LDT.open(MODEL_color, FileStorage::WRITE);
		FS_LDT << "C_Data" << img_color;
		FS_LDT.release();

		MODEL_LM=Regmodel_savePath_type_id+MPIE_name[id].substr(0,3)+"_l.xml";
		FS_LDT.open(MODEL_LM, FileStorage::WRITE);
		FS_LDT << "L_Data" << model_LM_3d;
		FS_LDT.release();*/

		//MODEL_x=Regmodel_savePath_type_id+MPIE_name[id].substr(0,3)+"_x.xml";
		//FS_LDT.open(MODEL_x, FileStorage::WRITE);
		//FS_LDT << "Data" << model_x_out;
		//FS_LDT.release();

		//MODEL_y=Regmodel_savePath_type_id+MPIE_name[id].substr(0,3)+"_y.xml";
		//FS_LDT.open(MODEL_y, FileStorage::WRITE);
		//FS_LDT << "Data" << model_y_out;
		//FS_LDT.release();

		//MODEL_z=Regmodel_savePath_type_id+MPIE_name[id].substr(0,3)+"_z.xml";
		//FS_LDT.open(MODEL_z, FileStorage::WRITE);
		//FS_LDT << "Data" << model_z_out;
		//FS_LDT.release();

		//MODEL_m=Regmodel_savePath_type_id+MPIE_name[id].substr(0,3)+"_mask.xml";
		//FS_LDT.open(MODEL_m, FileStorage::WRITE);
		//FS_LDT << "Data" << affine_Z_mask_t;
		//FS_LDT.release();

		//MODEL_c=Regmodel_savePath_type_id+MPIE_name[id].substr(0,3)+"_color.xml";
		//FS_LDT.open(MODEL_c, FileStorage::WRITE);
		//FS_LDT << "Data" << image_MPIE_after;
		//FS_LDT.release();

		/*MODEL_glass_p=Regmodel_savePath_type_id+MPIE_name[id].substr(0,3)+"_vg.xml";
		FS_LDT.open(MODEL_glass_p, FileStorage::WRITE);
		FS_LDT << "V_Data" << location_glass;
		FS_LDT.release();

		MODEL_glass_f=Regmodel_savePath_type_id+MPIE_name[id].substr(0,3)+"_fg.xml";
		FS_LDT.open(MODEL_glass_f, FileStorage::WRITE);
		FS_LDT << "F_Data" << faces_glass;
		FS_LDT.release();

		MODEL_glass_c=Regmodel_savePath_type_id+MPIE_name[id].substr(0,3)+"_cg.xml";
		FS_LDT.open(MODEL_glass_c, FileStorage::WRITE);
		FS_LDT << "C_Data" << img_color_glass;
		FS_LDT.release();

		MODEL_c=Regmodel_savePath_type_id+MPIE_name[id].substr(0,3)+"_eth.xml";
		FS_LDT.open(MODEL_c, FileStorage::WRITE);
		FS_LDT << "Data" << eth_ch;
		FS_LDT.release();*/
	}

	// save ethnicity number //
	string savePath_2=eth_Path+glass_type;
	_mkdir(savePath_2.c_str());
	string eth_num_save=savePath_2+"/"+"ethnicity_num"+".xml";
	FileStorage FS_LDT;
	FS_LDT.open(eth_num_save, FileStorage::WRITE);
	FS_LDT << "LMPT_Data" << eth_no;
	FS_LDT.release();

	cout << '\a';
	system("pause");
	return 0;
}


//////////////////////////////////////////
///*          Sub function            *///
//////////////////////////////////////////

// model choose //
void model_choose_LMset_single(vector<Point2f> LM_in, vector<Point2f>* LM_out)
{
	vector<Point2f> model_LM_all=LM_in;
	vector<Point2f> model_LM_all_after=LM_in;

	//轉至0度//
	float Rotate_angle;
	find_theate_zero(model_LM_all, &Rotate_angle);
	Point2f center_point;
	find_LM_center(model_LM_all, &center_point); // 尋找移動後測試影像的LM點中心點
	Mat Rotate_Matrix=getRotationMatrix2D(center_point, Rotate_angle, 1.0); // 計算旋轉矩陣[2X3]
	rotate_LM(model_LM_all,&model_LM_all_after, Rotate_angle); //旋轉移動後的LM

	//兩眼間距縮至120//
	Point2f Leye=Point2f((model_LM_all_after[20-1].x+model_LM_all_after[23-1].x)/2,(model_LM_all_after[20-1].y+model_LM_all_after[23-1].y)/2);
	Point2f Reye=Point2f((model_LM_all_after[26-1].x+model_LM_all_after[29-1].x)/2,(model_LM_all_after[26-1].y+model_LM_all_after[29-1].y)/2);
	float default_scale=120.0;
	float scale=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
	float scale_img=default_scale/scale;
	scale_LM(model_LM_all_after,&model_LM_all_after, scale_img); //縮放移動後的LM

	//移至中心原點//
	find_LM_center(model_LM_all_after, &center_point); // 尋找移動後測試影像的LM點中心點
	for(int k=0;k<model_LM_all_after.size();k++)
	{
		model_LM_all_after[k].x=model_LM_all_after[k].x-center_point.x;
		model_LM_all_after[k].y=model_LM_all_after[k].y-center_point.y;
	}

	*LM_out=model_LM_all_after;

}
void model_choose_LMset_all(vector<vector<Point2f>> LM_in, vector<vector<Point2f>>* LM_out)
{
	vector<vector<Point2f>> model_LM_all=LM_in;
	vector<vector<Point2f>> model_LM_all_after=LM_in;

	for (int i=0;i<model_LM_all.size();i++)
	{
		//轉至0度//
		float Rotate_angle;
		find_theate_zero(model_LM_all[i], &Rotate_angle);
		Point2f center_point;
		find_LM_center(model_LM_all[i], &center_point); // 尋找移動後測試影像的LM點中心點
		Mat Rotate_Matrix=getRotationMatrix2D(center_point, Rotate_angle, 1.0); // 計算旋轉矩陣[2X3]
		rotate_LM(model_LM_all[i],&model_LM_all_after[i], Rotate_angle); //旋轉移動後的LM

		//兩眼間距縮至120//
		Point2f Leye=Point2f((model_LM_all_after[i][20-1].x+model_LM_all_after[i][23-1].x)/2,(model_LM_all_after[i][20-1].y+model_LM_all_after[i][23-1].y)/2);
		Point2f Reye=Point2f((model_LM_all_after[i][26-1].x+model_LM_all_after[i][29-1].x)/2,(model_LM_all_after[i][26-1].y+model_LM_all_after[i][29-1].y)/2);
		float default_scale=120.0;
		float scale=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
		float scale_img=default_scale/scale;
		scale_LM(model_LM_all_after[i],&model_LM_all_after[i], scale_img); //縮放移動後的LM

		//移至中心原點//
		find_LM_center(model_LM_all_after[i], &center_point); // 尋找移動後測試影像的LM點中心點
		for(int k=0;k<model_LM_all_after[i].size();k++)
		{
			model_LM_all_after[i][k].x=model_LM_all_after[i][k].x-center_point.x;
			model_LM_all_after[i][k].y=model_LM_all_after[i][k].y-center_point.y;
		}
	}

	*LM_out=model_LM_all_after;
}
void draw_point(vector<vector<Point2f>> LM_in)
{
	const static Scalar colors[] =  { CV_RGB(0,0,255),
		CV_RGB(0,128,255),
		CV_RGB(0,255,255),
		CV_RGB(0,255,0),
		CV_RGB(255,128,0),
		CV_RGB(255,255,0),
		CV_RGB(255,0,0),
		CV_RGB(255,0,255),
		CV_RGB(255,0,128),
		CV_RGB(128,0,255),
		CV_RGB(255,255,255),
		CV_RGB(128,255,0),
		CV_RGB(0,255,128)} ;
	Mat img(600,600,CV_32FC3);
	img.setTo(0);
	int point_number=1;
	//int j=0;
	for (int i=0;i<LM_in.size();i++)
	{
		Scalar color = colors[i%8];
		//j++;
		cout<<i<<endl;
		string point_number_str;
		stringstream int2string;
		int2string<<point_number;
		int2string>>point_number_str;
		putText(img,point_number_str,cv::Point2f(100.0+15*point_number,100.0),FONT_HERSHEY_PLAIN , 1, color, 1);
		for(int k=0;k<LM_in[i].size();k++)
		{
			circle(img,Point2f(LM_in[i][k].x+300,LM_in[i][k].y+300),1,color,-1);
		}
		imshow("point",img);waitKey(0);
		int2string.str("");
		int2string.clear();
		point_number=point_number+1;
	}
}
void model_choose(vector<vector<Point2f>> model_LM_all,vector<Point2f> MPIE_LM_each, vector<int>* choose_num, int* min_num)
{
	vector<float> s_dis;
	for (int i=0;i<model_LM_all.size();i++)
	{
		float s_error=0.0;
		for (int k=0; k<model_LM_all[i].size(); k++)
		{
			float x=MPIE_LM_each[k].x-model_LM_all[i][k].x;
			float y=MPIE_LM_each[k].y-model_LM_all[i][k].y;
			float dis=sqrt(x*x+y*y);
			s_error=s_error+dis;
		}
		s_error=s_error/model_LM_all[i].size();
		s_dis.push_back(s_error);
	}

	vector<float> s_dis_compare=s_dis;
	std::sort(s_dis.begin(),s_dis.end(),less<float>());

	vector<int> num;
	for (int i = 0; i < s_dis.size(); i++)
	{
		for (int k=0; k<s_dis_compare.size(); k++)
		{
			if (s_dis[i]==s_dis_compare[k])
			{
				num.push_back(k);
			}
		}
	}

	*choose_num=num;



	vector<float> angLe_2d;
	angle_calculate(MPIE_LM_each, &angLe_2d);


	vector<float> angle_error;
	for (int i = 0; i < 10; i++) //前10位的model
	{
		vector<float> angLe;
		angle_calculate(model_LM_all[num[i]], &angLe);

		float angle_err=0.0;
		for (int n=0;n<angLe.size();n++)
		{
			angle_err=angle_err+abs(angLe_2d[n]-angLe[n]);
		}
		angle_err=angle_err/angLe.size();
		angle_error.push_back(angle_err);
	}

	vector<float> angle_error_compare=angle_error;
	std::sort(angle_error.begin(),angle_error.end(),less<float>());

	vector<int> angle_num;
	for (int i = 0; i < angle_error.size(); i++)
	{
		for (int k=0; k<angle_error_compare.size(); k++)
		{
			if (angle_error[i]==angle_error_compare[k])
			{
				angle_num.push_back(num[k]);
			}
		}
	}

	*min_num=angle_num[0];

}
void angle_calculate(vector<Point2f> LM_in, vector<float> *angle_out)
{
	vector<float> angLe;

	// 左眉 1~5
	int point_Leyebow[]={1,2,3,4,5};
	vector<int> number_Leyebow(point_Leyebow, point_Leyebow + sizeof(point_Leyebow)/sizeof(point_Leyebow[0]));
	vector<Point2f> v;
	for (int k=0; k<number_Leyebow.size()-1; k++)
	{
		Point2f v_temp=Point2f(LM_in[number_Leyebow[k+1]-1].x-LM_in[number_Leyebow[k]-1].x,LM_in[number_Leyebow[k+1]-1].y-LM_in[number_Leyebow[k]-1].y);
		v.push_back(v_temp);
	}
	for (int m=0;m<v.size()-1;m++)
	{
		float dot_v=v[m].x*v[m+1].x+v[m].y*v[m+1].y;
		float v1_v=sqrt(v[m].x*v[m].x+v[m].y*v[m].y);
		float v2_v=sqrt(v[m+1].x*v[m+1].x+v[m+1].y*v[m+1].y);
		float dd=acos(dot_v/(v1_v*v2_v));
		float theat;
		if (_isnan(double(dd))){theat=180;}
		else{theat=180-(dd*180/3.141592);}
		angLe.push_back(theat);
	}

	//右眉 6~10
	int point_Reyebow[]={6,7,8,9,10};
	vector<int> number_Reyebow(point_Reyebow, point_Reyebow + sizeof(point_Reyebow)/sizeof(point_Reyebow[0]));
	v.clear();
	for (int k=0; k<number_Reyebow.size()-1; k++)
	{
		Point2f v_temp=Point2f(LM_in[number_Reyebow[k+1]-1].x-LM_in[number_Reyebow[k]-1].x,LM_in[number_Reyebow[k+1]-1].y-LM_in[number_Reyebow[k]-1].y);
		v.push_back(v_temp);
	}
	for (int m=0;m<v.size()-1;m++)
	{
		float dot_v=v[m].x*v[m+1].x+v[m].y*v[m+1].y;
		float v1_v=sqrt(v[m].x*v[m].x+v[m].y*v[m].y);
		float v2_v=sqrt(v[m+1].x*v[m+1].x+v[m+1].y*v[m+1].y);
		float dd=acos(dot_v/(v1_v*v2_v));
		float theat;
		if (_isnan(double(dd))){theat=180;}
		else{theat=180-(dd*180/3.141592);}
		angLe.push_back(theat);
	}

	//左眼 20~25
	int point_Leye[]={20,21,22,23,24,25,20};
	vector<int> number_Leye(point_Leye, point_Leye + sizeof(point_Leye)/sizeof(point_Leye[0]));
	v.clear();
	for (int k=0; k<number_Leye.size()-1; k++)
	{
		Point2f v_temp=Point2f(LM_in[number_Leye[k+1]-1].x-LM_in[number_Leye[k]-1].x,LM_in[number_Leye[k+1]-1].y-LM_in[number_Leye[k]-1].y);
		v.push_back(v_temp);
	}
	for (int m=0;m<v.size()-1;m++)
	{
		float dot_v=v[m].x*v[m+1].x+v[m].y*v[m+1].y;
		float v1_v=sqrt(v[m].x*v[m].x+v[m].y*v[m].y);
		float v2_v=sqrt(v[m+1].x*v[m+1].x+v[m+1].y*v[m+1].y);
		float dd=acos(dot_v/(v1_v*v2_v));
		float theat;
		if (_isnan(double(dd))){theat=180;}
		else{theat=180-(dd*180/3.141592);}
		angLe.push_back(theat);
	}
	for (int h=0;h<1;h++)
	{
		float dot_v=v[v.size()-1].x*v[0].x+v[v.size()-1].y*v[0].y;
		float v1_v=sqrt(v[v.size()-1].x*v[v.size()-1].x+v[v.size()-1].y*v[v.size()-1].y);
		float v2_v=sqrt(v[0].x*v[0].x+v[0].y*v[0].y);
		float dd=acos(dot_v/(v1_v*v2_v));
		float theat;
		if (_isnan(double(dd))){theat=180;}
		else{theat=180-(dd*180/3.141592);}
		angLe.push_back(theat);
	}

	//右眼 26~31
	int point_Reye[]={26,27,28,29,30,31,26};
	vector<int> number_Reye(point_Reye, point_Reye + sizeof(point_Reye)/sizeof(point_Reye[0]));
	v.clear();
	for (int k=0; k<number_Reye.size()-1; k++)
	{
		Point2f v_temp=Point2f(LM_in[number_Reye[k+1]-1].x-LM_in[number_Reye[k]-1].x,LM_in[number_Reye[k+1]-1].y-LM_in[number_Reye[k]-1].y);
		v.push_back(v_temp);
	}
	for (int m=0;m<v.size()-1;m++)
	{
		float dot_v=v[m].x*v[m+1].x+v[m].y*v[m+1].y;
		float v1_v=sqrt(v[m].x*v[m].x+v[m].y*v[m].y);
		float v2_v=sqrt(v[m+1].x*v[m+1].x+v[m+1].y*v[m+1].y);
		float dd=acos(dot_v/(v1_v*v2_v));
		float theat;
		if (_isnan(double(dd))){theat=180;}
		else{theat=180-(dd*180/3.141592);}
		angLe.push_back(theat);
	}
	for (int h=0;h<1;h++)
	{
		float dot_v=v[v.size()-1].x*v[0].x+v[v.size()-1].y*v[0].y;
		float v1_v=sqrt(v[v.size()-1].x*v[v.size()-1].x+v[v.size()-1].y*v[v.size()-1].y);
		float v2_v=sqrt(v[0].x*v[0].x+v[0].y*v[0].y);
		float dd=acos(dot_v/(v1_v*v2_v));
		float theat;
		if (_isnan(double(dd))){theat=180;}
		else{theat=180-(dd*180/3.141592);}
		angLe.push_back(theat);
	}

	//鼻子 15~19
	int point_Nose[]={15,16,17,18,19};
	vector<int> number_Nose(point_Nose, point_Nose + sizeof(point_Nose)/sizeof(point_Nose[0]));
	v.clear();
	for (int k=0; k<number_Nose.size()-1; k++)
	{
		Point2f v_temp=Point2f(LM_in[number_Nose[k+1]-1].x-LM_in[number_Nose[k]-1].x,LM_in[number_Nose[k+1]-1].y-LM_in[number_Nose[k]-1].y);
		v.push_back(v_temp);
	}
	for (int m=0;m<v.size()-1;m++)
	{
		float dot_v=v[m].x*v[m+1].x+v[m].y*v[m+1].y;
		float v1_v=sqrt(v[m].x*v[m].x+v[m].y*v[m].y);
		float v2_v=sqrt(v[m+1].x*v[m+1].x+v[m+1].y*v[m+1].y);
		float dd=acos(dot_v/(v1_v*v2_v));
		float theat;
		if (_isnan(double(dd))){theat=180;}
		else{theat=180-(dd*180/3.141592);}
		angLe.push_back(theat);
	}

	//嘴巴 32~43
	int point_Mouth[]={32,33,34,35,36,37,38,39,40,41,42,43,32};
	vector<int> number_Mouth(point_Mouth, point_Mouth + sizeof(point_Mouth)/sizeof(point_Mouth[0]));
	v.clear();
	for (int k=0; k<number_Mouth.size()-1; k++)
	{
		Point2f v_temp=Point2f(LM_in[number_Mouth[k+1]-1].x-LM_in[number_Mouth[k]-1].x,LM_in[number_Mouth[k+1]-1].y-LM_in[number_Mouth[k]-1].y);
		v.push_back(v_temp);
	}
	for (int m=0;m<v.size()-1;m++)
	{
		float dot_v=v[m].x*v[m+1].x+v[m].y*v[m+1].y;
		float v1_v=sqrt(v[m].x*v[m].x+v[m].y*v[m].y);
		float v2_v=sqrt(v[m+1].x*v[m+1].x+v[m+1].y*v[m+1].y);
		float dd=acos(dot_v/(v1_v*v2_v));
		float theat;
		if (_isnan(double(dd))){theat=180;}
		else{theat=180-(dd*180/3.141592);}
		angLe.push_back(theat);
	}
	for (int h=0;h<1;h++)
	{
		float dot_v=v[v.size()-1].x*v[0].x+v[v.size()-1].y*v[0].y;
		float v1_v=sqrt(v[v.size()-1].x*v[v.size()-1].x+v[v.size()-1].y*v[v.size()-1].y);
		float v2_v=sqrt(v[0].x*v[0].x+v[0].y*v[0].y);
		float dd=acos(dot_v/(v1_v*v2_v));
		float theat;
		if (_isnan(double(dd))){theat=180;}
		else{theat=180-(dd*180/3.141592);}
		angLe.push_back(theat);
	}

	//臉輪廓 50~58
	int point_F_contour[]={54,59,53,60,52,61,51,62,50,63,55,64,56,65,57,66,58};
	//int point_F_contour[]={54,53,52,51,50,55,56,57,58};
	vector<int> number_F_contour(point_F_contour, point_F_contour + sizeof(point_F_contour)/sizeof(point_F_contour[0]));
	v.clear();
	for (int k=0; k<number_F_contour.size()-1; k++)
	{
		Point2f v_temp=Point2f(LM_in[number_F_contour[k+1]-1].x-LM_in[number_F_contour[k]-1].x,LM_in[number_F_contour[k+1]-1].y-LM_in[number_F_contour[k]-1].y);
		v.push_back(v_temp);
	}
	for (int m=0;m<v.size()-1;m++)
	{
		float dot_v=v[m].x*v[m+1].x+v[m].y*v[m+1].y;
		float v1_v=sqrt(v[m].x*v[m].x+v[m].y*v[m].y);
		float v2_v=sqrt(v[m+1].x*v[m+1].x+v[m+1].y*v[m+1].y);
		float dd=acos(dot_v/(v1_v*v2_v));
		float theat;
		if (_isnan(double(dd))){theat=180;}
		else{theat=180-(dd*180/3.141592);}
		angLe.push_back(theat);
	}

	*angle_out=angLe;
}

// depth warp //
void set_persp_aff(std::vector<cv::Point2f> ref_pt, std::vector<std::vector<cv::Point2f>>* out_pt, string name, int* affine_num)
{
	FILE* f=fopen(name.c_str(),"rt");
	int a=0,b=0,c=0;
	int aff_num=0;
	fscanf(f,"%d",&aff_num); //區塊數目
	std::vector<std::vector<cv::Point2f>> temp_out;
	for(int j=0;j<aff_num;j++)
	{
		std::vector<Point2f> temp;
		fscanf(f,"%i %i %i",&a,&b,&c);
		temp.push_back(ref_pt[a-1]);
		temp.push_back(ref_pt[b-1]);
		temp.push_back(ref_pt[c-1]);
		temp_out.push_back(temp);
	}
	fclose(f);

	*out_pt=temp_out;
	*affine_num=aff_num;
}
void aff_warp(std::vector<cv::Point2f> src_pnt, std::vector<cv::Point2f> dst_pnt,cv::Mat img_in,cv::Mat *img_out,cv::Mat *mask_out)
{
	cv::Mat_<float> map_matrix(3,3); 
	cv::Mat TMask(img_in.rows, img_in.cols, CV_8UC1);  TMask.setTo((0));
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::Mat tempWarp;
	map_matrix = cv::getAffineTransform(src_pnt, dst_pnt); //getAffineTransform 從ref model 的區域到 每個人的區域
	cv::warpAffine(img_in, tempWarp, map_matrix, img_in.size());
	cv::line(TMask, dst_pnt[0], dst_pnt[1], cv::Scalar(255), 1, CV_AA, 0);
	cv::line(TMask, dst_pnt[1], dst_pnt[2], cv::Scalar(255), 1, CV_AA, 0);
	cv::line(TMask, dst_pnt[2], dst_pnt[0], cv::Scalar(255), 1, CV_AA, 0);
	cv::findContours(TMask, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); //找輪廓
	cv::drawContours(TMask, contours, -1, cv::Scalar(255), -1, CV_AA, hierarchy, 0); //畫輪廓

	//if (img_in.type()==5)
	//{
	//	for (int i=0;i<img_in.rows;i++)
	//	{
	//		for (int j=0;j<img_in.rows;j++)
	//		{
	//			if (img_in.at<float>(i,j)==0)
	//			{
	//				TMask.at<uchar>(i,j)=0;
	//			}
	//		}
	//	}
	//}


	tempWarp.copyTo(*img_out, TMask);
	*mask_out=TMask;
}

// opengl //
void matrix_set(Mat* view,Mat* project)
{
	float mapping_width=640.0;
	float mapping_height=480.0;
	Mat view_matrix(4,4,CV_32FC1);
	view_matrix.at<float>(0,0)=mapping_width/2;
	view_matrix.at<float>(0,1)=0.0;
	view_matrix.at<float>(0,2)=0.0;
	view_matrix.at<float>(0,3)=mapping_width/2;
	view_matrix.at<float>(1,0)=0.0;
	view_matrix.at<float>(1,1)=-mapping_height/2;
	view_matrix.at<float>(1,2)=0.0;
	view_matrix.at<float>(1,3)=mapping_height/2;
	view_matrix.at<float>(2,0)=0.0;
	view_matrix.at<float>(2,1)=0.0;
	view_matrix.at<float>(2,2)=-0.5;
	view_matrix.at<float>(2,3)=0.5;
	view_matrix.at<float>(3,0)=0.0;
	view_matrix.at<float>(3,1)=0.0;
	view_matrix.at<float>(3,2)=0.0;
	view_matrix.at<float>(3,3)=1.0;
	*view=view_matrix;

	float Xmin=-185;
	float Xmax=185;
	float F_four=-(Xmin+Xmax)/(Xmax-Xmin);
	float Ymin=-135;
	float Ymax=135;
	float F_eight=-(Ymin+Ymax)/(Ymax-Ymin);
	float Zmax=-5000.0;
	float Zmin=5000.0;
	float F_ten=2/(Zmax-Zmin);
	float F_fourTeen=-(Zmax+Zmin)/(Zmax-Zmin);
	Mat project_matrix(4,4,CV_32FC1);

	//project_matrix.at<float>(0,0)=0.0054;  
	//project_matrix.at<float>(0,1)=-1.9066*pow(10.0,-6.0);  
	//project_matrix.at<float>(0,2)=5.9645*pow(10,-7.0);  
	//project_matrix.at<float>(0,3)=0.0014;
	//project_matrix.at<float>(1,0)=7.9908*pow(10.0,-6.0);
	//project_matrix.at<float>(1,1)=0.0074;
	//project_matrix.at<float>(1,2)=-2.0505*pow(10,-4.0);
	//project_matrix.at<float>(1,3)=-0.3964;
	//project_matrix.at<float>(2,0)=0.0;
	//project_matrix.at<float>(2,1)=0.0;
	//project_matrix.at<float>(2,2)=F_ten;
	//project_matrix.at<float>(2,3)=F_fourTeen;
	//project_matrix.at<float>(3,0)=0.0;
	//project_matrix.at<float>(3,1)=0.0;
	//project_matrix.at<float>(3,2)=0.0;
	//project_matrix.at<float>(3,3)=1.0;

	project_matrix.at<float>(0,0)=2/(Xmax-Xmin);  
	project_matrix.at<float>(0,1)=0.0;  
	project_matrix.at<float>(0,2)=0.0;  
	project_matrix.at<float>(0,3)=F_four;
	project_matrix.at<float>(1,0)=0.0;
	project_matrix.at<float>(1,1)=2/(Ymax-Ymin);
	project_matrix.at<float>(1,2)=0.0;
	project_matrix.at<float>(1,3)=F_eight;
	project_matrix.at<float>(2,0)=0.0;
	project_matrix.at<float>(2,1)=0.0;
	project_matrix.at<float>(2,2)=F_ten;
	project_matrix.at<float>(2,3)=F_fourTeen;
	project_matrix.at<float>(3,0)=0.0;
	project_matrix.at<float>(3,1)=0.0;
	project_matrix.at<float>(3,2)=0.0;
	project_matrix.at<float>(3,3)=1.0;

	*project=project_matrix;
}
void model_gravityPoint(Mat mask, Mat model_x, Mat model_y, Mat model_z, Mat *move_matrix)
{
	//找model中心
	float mx=0;
	float my=0;
	float mz=0;
	for (int i=0;i<mask.rows;i++)
	{
		for (int j=0;j<mask.cols;j++)
		{
			if (mask.at<uchar>(i,j)!=0)
			{
				mx=mx+model_x.at<float>(i,j);
				my=my+model_y.at<float>(i,j);
				mz=mz+model_z.at<float>(i,j);
			}
		}
	}
	int vertex_num=0;
	vertex_num=cv::countNonZero(mask);
	mx=mx/vertex_num;
	my=my/vertex_num;
	mz=mz/vertex_num;

	Mat move_to_ori=Mat::eye(4,4,CV_32F); //model的平移矩陣
	move_to_ori.at<float>(0,3)=-mx;
	move_to_ori.at<float>(1,3)=-my;
	move_to_ori.at<float>(2,3)=-mz;

	*move_matrix=move_to_ori;
}
void R_matrix(float Pitch, float Yaw, float Roll, Mat *r_matrix_out_x, Mat *r_matrix_out_y, Mat *r_matrix_out_z)
{
	Mat rotate_matrix_x(4,4,CV_32FC1);
	float cos_a=cos(Pitch*3.141592/180);
	float sin_a=sin(Pitch*3.141592/180);
	rotate_matrix_x.at<float>(0,0)=1.0;
	rotate_matrix_x.at<float>(0,1)=0.0;
	rotate_matrix_x.at<float>(0,2)=0.0;
	rotate_matrix_x.at<float>(0,3)=0.0;

	rotate_matrix_x.at<float>(1,0)=0.0;
	rotate_matrix_x.at<float>(1,1)=cos_a;
	rotate_matrix_x.at<float>(1,2)=-sin_a;
	rotate_matrix_x.at<float>(1,3)=0.0;

	rotate_matrix_x.at<float>(2,0)=0.0;
	rotate_matrix_x.at<float>(2,1)=sin_a;
	rotate_matrix_x.at<float>(2,2)=cos_a;
	rotate_matrix_x.at<float>(2,3)=0.0;

	rotate_matrix_x.at<float>(3,0)=0.0;
	rotate_matrix_x.at<float>(3,1)=0.0;
	rotate_matrix_x.at<float>(3,2)=0.0;
	rotate_matrix_x.at<float>(3,3)=1.0;	

	Mat rotate_matrix_y(4,4,CV_32FC1);
	cos_a=cos(Yaw*3.141592/180);
	sin_a=sin(Yaw*3.141592/180);
	rotate_matrix_y.at<float>(0,0)=cos_a;
	rotate_matrix_y.at<float>(0,1)=0.0;
	rotate_matrix_y.at<float>(0,2)=sin_a;
	rotate_matrix_y.at<float>(0,3)=0.0;

	rotate_matrix_y.at<float>(1,0)=0.0;
	rotate_matrix_y.at<float>(1,1)=1.0;
	rotate_matrix_y.at<float>(1,2)=0.0;
	rotate_matrix_y.at<float>(1,3)=0.0;

	rotate_matrix_y.at<float>(2,0)=-sin_a;
	rotate_matrix_y.at<float>(2,1)=0.0;
	rotate_matrix_y.at<float>(2,2)=cos_a;
	rotate_matrix_y.at<float>(2,3)=0.0;

	rotate_matrix_y.at<float>(3,0)=0.0;
	rotate_matrix_y.at<float>(3,1)=0.0;
	rotate_matrix_y.at<float>(3,2)=0.0;
	rotate_matrix_y.at<float>(3,3)=1.0;	

	Mat rotate_matrix_z(4,4,CV_32FC1);
	cos_a=cos(Roll*3.141592/180);
	sin_a=sin(Roll*3.141592/180);
	rotate_matrix_z.at<float>(0,0)=cos_a;
	rotate_matrix_z.at<float>(0,1)=-sin_a;
	rotate_matrix_z.at<float>(0,2)=0.0;
	rotate_matrix_z.at<float>(0,3)=0.0;

	rotate_matrix_z.at<float>(1,0)=sin_a;
	rotate_matrix_z.at<float>(1,1)=cos_a;
	rotate_matrix_z.at<float>(1,2)=0.0;
	rotate_matrix_z.at<float>(1,3)=0.0;

	rotate_matrix_z.at<float>(2,0)=0.0;
	rotate_matrix_z.at<float>(2,1)=0.0;
	rotate_matrix_z.at<float>(2,2)=1.0;
	rotate_matrix_z.at<float>(2,3)=0.0;

	rotate_matrix_z.at<float>(3,0)=0.0;
	rotate_matrix_z.at<float>(3,1)=0.0;
	rotate_matrix_z.at<float>(3,2)=0.0;
	rotate_matrix_z.at<float>(3,3)=1.0;	

	*r_matrix_out_x=rotate_matrix_x;
	*r_matrix_out_y=rotate_matrix_y;
	*r_matrix_out_z=rotate_matrix_z;
}
void mesh(Mat mask, Mat image, Mat model_x, Mat model_y, Mat model_z, vector<Point3f>* location, vector<Point3i>* faces, vector<Point3f>* img_color, vector<vector<uchar>>* img_color_u)
{
	vector<Point3f> location_temp;
	vector<Point3f> color_temp;
	vector<vector<uchar>> img_color_u_temp;
	for (int i=0; i<mask.rows; i++)
	{
		for (int j=0; j<mask.cols; j++)
		{
			if (mask.at<uchar>(i,j)!=0)
			{
				vector<uchar> img_color_each_temp;
				if(model_z.at<float>(i,j)==0){continue;}
				location_temp.push_back(Point3f(model_x.at<float>(i,j),model_y.at<float>(i,j),model_z.at<float>(i,j)));
				color_temp.push_back(Point3f(image.at<Vec3b>(i,j)[2]/255.0,image.at<Vec3b>(i,j)[1]/255.0,image.at<Vec3b>(i,j)[0]/255.0));
				img_color_each_temp.push_back(image.at<Vec3b>(i,j)[2]);
				img_color_each_temp.push_back(image.at<Vec3b>(i,j)[1]);
				img_color_each_temp.push_back(image.at<Vec3b>(i,j)[0]);
				img_color_u_temp.push_back(img_color_each_temp);
			}
		}
	}
	*location=location_temp;
	*img_color=color_temp;
	*img_color_u=img_color_u_temp;

	IplImage *dest;  
	int i,j,total,count;  
	CvMemStorage* storage;  //创建存储器
	CvSubdiv2D* subdiv;  
	vector<CvPoint> points;  
	CvSeqReader reader;  //利用CvSeqReader遍历  
	CvPoint buf[3]; 
	int verticesIdx[3];

	vector<cv::Point3i> face_index;
	vector<cv::Point3i> face_index_re;
	vector<cv::Point2f> points_recode;

	std::set<vector<int>> coll1;

	CvRect rect = { 0, 0, 640, 480 };  //矩形是图像的大小 
	storage = cvCreateMemStorage(0);  
	subdiv = cvCreateSubdivDelaunay2D(rect,storage);  
	count = 0;  

	for (int i=0; i<mask.rows; i++)
	{
		for (int j=0; j<mask.cols; j++)
		{
			if (mask.at<uchar>(i,j) != 0)
			{
				if(model_z.at<float>(i,j)==0){continue;}
				points.push_back(cvPoint(j,i));
				points_recode.push_back(cv::Point2f(j,i));
			}
		}
	}

	//iterate through points inserting them in the subdivision  
	for(int i=0;i<points.size();i++){  
		float x = points.at(i).x;  
		float y = points.at(i).y;  
		CvPoint2D32f floatingPoint = cvPoint2D32f(x, y);  
		CvSubdiv2DPoint *pt = cvSubdivDelaunay2DInsert( subdiv, floatingPoint );//利用插入法进行剖分
		pt->id = i;//为每一个顶点分配一个id
	}  

	//draw image and iterating through all the triangles  
	dest = cvCreateImage(cvSize(640,480),IPL_DEPTH_8U,3);

	total = subdiv->edges->total;  //边的总数  
	int elem_size = subdiv->edges->elem_size;  //边的大小  
	cvStartReadSeq((CvSeq*)(subdiv->edges), &reader, 0);  

	CvNextEdgeType triangleDirections[2] = {CV_NEXT_AROUND_RIGHT,CV_NEXT_AROUND_LEFT};
	for(int tdi = 0;tdi<2;tdi++){  
		CvNextEdgeType triangleDirection = triangleDirections[tdi];  
		for(i = 0; i < total; i++){  
			CvQuadEdge2D* edge = (CvQuadEdge2D*)(reader.ptr);  
			if(CV_IS_SET_ELEM(edge)){  
				CvSubdiv2DEdge t = (CvSubdiv2DEdge)edge;  
				int shouldPaint=1;  
				for(j=0;j<3;j++){  
					CvSubdiv2DPoint* pt = cvSubdiv2DEdgeOrg(t); 
					if(!pt) break;  
					buf[j] = cvPoint(cvRound(pt->pt.x), cvRound(pt->pt.y));  
					verticesIdx[j] = pt->id;
					t = cvSubdiv2DGetEdge(t, triangleDirection);  
					if((pt->pt.x<0)||(pt->pt.x>dest->width))  
						shouldPaint=0;  
					if((pt->pt.y<0)||(pt->pt.y>dest->height))  
						shouldPaint=0;  
				}  
				if(shouldPaint){  
					cv::Point3i temp = cv::Point3i(verticesIdx[2],verticesIdx[1],verticesIdx[0]); 
					count++;  

					vector<int> vec1;
					vec1.push_back(verticesIdx[2]);
					vec1.push_back(verticesIdx[1]);
					vec1.push_back(verticesIdx[0]);
					sort(vec1.begin(), vec1.end());
					//coll1.insert(vec1);
					std::pair<set<vector<int>>::iterator,bool> status = coll1.insert(vec1);
					if (status.second)
					{
						face_index_re.push_back(temp);
					}
					vec1.clear();
				}  
			}  
			CV_NEXT_SEQ_ELEM(elem_size, reader);  
		}     
	}  
	*faces=face_index_re;
}

// model save //
void write_model(string savePath, Mat msak, Mat img, Mat model_x, Mat model_y, Mat model_z)
{
	int count_warp=0;
	count_warp=cv::countNonZero(msak);

	FILE *fs;
	fs=fopen(savePath.c_str(),"wt");//fs=fopen(save_name,"wt");
	fprintf(fs,"ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n",count_warp);	
	for(int m=0; m<msak.rows; m++)
	{
		for(int n=0; n<msak.cols; n++)
		{
			if (msak.at<uchar>(m,n)!=0)
			{
				fprintf(fs,"%f %f %f %d %d %d\n",model_x.at<float>(m,n), model_y.at<float>(m,n), model_z.at<float>(m,n), img.at<Vec3b>(m,n)[2], img.at<Vec3b>(m,n)[1], img.at<Vec3b>(m,n)[0]);
				//fprintf(fs,"%f %f %f %d %d %d\n",model_x.at<float>(m,n), model_y.at<float>(m,n), model_z.at<float>(m,n), (uchar)150.0, (uchar)150.0, (uchar)150.0);
			}
		}
	}
	fclose(fs);
}
void write_model_LM(string savePath, Mat LM)
{
	FILE *fs;
	fs=fopen(savePath.c_str(),"wt");//fs=fopen(save_name,"wt");
	fprintf(fs,"ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n",LM.cols);	
	for(int n=0; n<LM.cols; n++)
	{
		fprintf(fs,"%f %f %f %d %d %d\n",LM.at<float>(0,n), LM.at<float>(1,n), LM.at<float>(2,n), (uchar)255.0, (uchar)0.0, (uchar)0.0);
	}
	fclose(fs);
}
void write_ply_mesh(string savePath, vector<Point3f> location, vector<Point3i> faces, vector<vector<uchar>> img_color_u)
{
	FILE *fs;
	fs=fopen(savePath.c_str(),"wt");
	fprintf(fs,"ply\n");
	fprintf(fs,"format ascii 1.0\n");
	fprintf(fs,"comment VCGLIB generated\n");
	fprintf(fs,"element vertex %d\n",location.size());
	fprintf(fs,"property float x\n");
	fprintf(fs,"property float y\n");
	fprintf(fs,"property float z\n");
	fprintf(fs,"property uchar red\n");
	fprintf(fs,"property uchar green\n");
	fprintf(fs,"property uchar blue\n");
	fprintf(fs,"property uchar alpha\n");
	fprintf(fs,"element face %d\n",faces.size());
	fprintf(fs,"property list uchar int vertex_indices\n");
	fprintf(fs,"end_header\n");

	for (int i = 0; i < location.size(); i++)
	{
		fprintf(fs,"%f %f %f %d %d %d %d\n",location[i].x, location[i].y, location[i].z, img_color_u[i][0], img_color_u[i][1], img_color_u[i][2], (uchar)255.0);
	}

	for (int k=0; k<faces.size(); k++)
	{
		fprintf(fs,"3 %d %d %d\n",faces[k].x,faces[k].y,faces[k].z);
	}
	fclose(fs);
}
void write_ply_glass(string savePath, vector<Point3f> location, vector<Point3i> faces)
{
	FILE *fs;
	fs=fopen(savePath.c_str(),"wt");
	fprintf(fs,"ply\n");
	fprintf(fs,"format ascii 1.0\n");
	fprintf(fs,"comment VCGLIB generated\n");
	fprintf(fs,"element vertex %d\n",location.size());
	fprintf(fs,"property float x\n");
	fprintf(fs,"property float y\n");
	fprintf(fs,"property float z\n");
	fprintf(fs,"property uchar red\n");
	fprintf(fs,"property uchar green\n");
	fprintf(fs,"property uchar blue\n");
	fprintf(fs,"property uchar alpha\n");
	fprintf(fs,"element face %d\n",faces.size());
	fprintf(fs,"property list uchar int vertex_indices\n");
	fprintf(fs,"end_header\n");

	for (int i = 0; i < location.size(); i++)
	{
		fprintf(fs,"%f %f %f %d %d %d %d\n",location[i].x, location[i].y, location[i].z, (uchar)0.0, (uchar)0.0, (uchar)0.0, (uchar)255.0);
	}

	for (int k=0; k<faces.size(); k++)
	{
		fprintf(fs,"3 %d %d %d\n",faces[k].x,faces[k].y,faces[k].z);
	}
	fclose(fs);
}

// mask //
void Create_face_mask(vector<Point2f> input_point, Mat &out_mask)
{
	int Mask_point_glass_mid[]={1,2,3,4,5,67,6,7,8,9,10,58,66,57,65,56,64,55,63,50,62,51,61,52,60,53.59,54};
	vector<int> number(Mask_point_glass_mid, Mask_point_glass_mid + sizeof(Mask_point_glass_mid)/sizeof(Mask_point_glass_mid[0]));

	cv::Mat TMask(out_mask.rows, out_mask.cols, CV_8UC1);
	TMask.setTo((0));

	int size_number=number.size();
	for (int i = 0; i < size_number; i++)
	{
		if (i == size_number-1)
			cv::line(TMask, input_point[number[i]-1], input_point[number[0]-1], cv::Scalar(255), 1, CV_AA, 0);
		else
			cv::line(TMask, input_point[number[i]-1], input_point[number[i+1]-1], cv::Scalar(255), 1, CV_AA, 0);
	}

	//cv::line(TMask, input_point[number[5]-1], Point2f(input_point[number[5]-1].x-20,input_point[number[5]-1].y), cv::Scalar(255), 1, CV_AA, 0);
	//cv::line(TMask, input_point[number[15]-1], Point2f(input_point[number[15]-1].x+20,input_point[number[15]-1].y), cv::Scalar(255), 1, CV_AA, 0);

	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(TMask, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); //找輪廓
	cv::drawContours(TMask, contours, -1, cv::Scalar(255), -1, CV_AA, hierarchy, 0); //畫輪廓

	out_mask=out_mask+TMask;
}
void Create_glass_mask(vector<Point2f> input_point, Mat &out_mask)
{
	int Mask_point_glass_left[]={2,3,4,5,6,7,8,10,11,12,13,14,15,16};
	vector<int> number_left(Mask_point_glass_left, Mask_point_glass_left + sizeof(Mask_point_glass_left)/sizeof(Mask_point_glass_left[0]));

	cv::Mat TMask(out_mask.rows, out_mask.cols, CV_8UC1);
	TMask.setTo((0));

	int size_number=number_left.size();
	for (int i = 0; i < size_number; i++)
	{
		if (i == size_number-1)
			cv::line(TMask, input_point[number_left[i]-1], input_point[number_left[0]-1], cv::Scalar(255), 1, CV_AA, 0);
		else
			cv::line(TMask, input_point[number_left[i]-1], input_point[number_left[i+1]-1], cv::Scalar(255), 1, CV_AA, 0);
	}

	cv::line(TMask, input_point[1-1], input_point[2-1], cv::Scalar(255), 1, CV_AA, 0);
	cv::line(TMask, input_point[8-1], input_point[9-1], cv::Scalar(255), 1, CV_AA, 0);
	cv::line(TMask, input_point[1-1], input_point[17-1], cv::Scalar(255), 1, CV_AA, 0);
	cv::line(TMask, input_point[23-1], input_point[24-1], cv::Scalar(255), 1, CV_AA, 0);

	int Mask_point_glass_right[]={17,18,19,20,21,22,23,25,26,27,28,29,30,31};
	vector<int> number_right(Mask_point_glass_right, Mask_point_glass_right + sizeof(Mask_point_glass_right)/sizeof(Mask_point_glass_right[0]));

	for (int i = 0; i < number_right.size(); i++)
	{
		if (i == number_right.size()-1)
			cv::line(TMask, input_point[number_right[i]-1], input_point[number_right[0]-1], cv::Scalar(255), 1, CV_AA, 0);
		else
			cv::line(TMask, input_point[number_right[i]-1], input_point[number_right[i+1]-1], cv::Scalar(255), 1, CV_AA, 0);
	}

	//cv::line(TMask, input_point[24-1], input_point[32-1], cv::Scalar(255), 1, CV_AA, 0);

	out_mask=out_mask+TMask;
}

// glass model load //
void Load_Model(string path, vector<Point3f>* location, vector<Point3i>* faces)
{
	vector<Point3f> location_temp; 
	vector<Point3i> faces_temp;

	string input_str; //輸入字串	

	char str_v[100];
	char *delim = " "; //判斷字串 遇到時分割
	char * pch;
	vector<string> vectorString;
	int vertex_num=0;
	int faces_num=0;

	ifstream fin;
	fin.open(path);
	//cout<<name_model<<endl;
	if(!fin) { 
		cout << "無法讀取檔案\n"; 
		// 其它處理 
	}
	else
	{
		cout << "讀取檔案成功\n"; 
	}

	// ply 檔讀取
	getline(fin,input_str); //ply
	getline(fin,input_str); //format ascii 1.0
	getline(fin,input_str); //comment VCGLIB generated
	getline(fin,input_str); //element vertex INPUT_NUMBER
	strcpy(str_v, input_str.c_str()); //string to char
	pch = strtok(str_v,delim);
	vectorString.clear();
	while (pch != NULL)
	{
		vectorString.push_back(pch);
		//printf ("%s\n",pch);
		pch = strtok (NULL, delim);
	}
	for(int i=2;i<3;i++)
	{
		stringstream string2float; //string to float
		string2float << vectorString[i];
		string2float >> vertex_num;
		//std::cout <<"vertex_num = "<< vertex_num<< std::endl;
		string2float.str(""); //再次使用前須請空內容
		string2float.clear(); //再次使用前須請空內容
	}
	getline(fin,input_str); //property float x
	getline(fin,input_str); //property float y
	getline(fin,input_str); //property float z
	getline(fin,input_str); //property uchar red
	getline(fin,input_str); //property uchar green
	getline(fin,input_str); //property uchar blue
	getline(fin,input_str); //property uchar alpha
	getline(fin,input_str); //element face 0
	strcpy(str_v, input_str.c_str()); //string to char
	pch = strtok(str_v,delim);
	vectorString.clear();
	while (pch != NULL)
	{
		vectorString.push_back(pch);
		//printf ("%s\n",pch);
		pch = strtok (NULL, delim);
	}
	for(int i=2;i<3;i++)
	{
		stringstream string2float; //string to float
		string2float << vectorString[i];
		string2float >> faces_num;
		//std::cout <<"faces_num = "<< faces_num<< std::endl;
		string2float.str(""); //再次使用前須請空內容
		string2float.clear(); //再次使用前須請空內容
	}
	getline(fin,input_str); //property list uchar int vertex_indices
	getline(fin,input_str); //end_header

	for (int j = 0; j < vertex_num; j++)
	{
		getline(fin,input_str); //得到整行字串
		char str_v[100];
		strcpy(str_v, input_str.c_str()); //string to char
		char *delim = " "; //判斷字串 遇到時分割
		char * pch;
		//printf ("Splitting string \"%s\" into tokens:\n",str_v);
		pch = strtok(str_v,delim);
		vector<string> vectorString;
		while (pch != NULL)
		{
			vectorString.push_back(pch);
			//printf ("%s\n",pch);
			pch = strtok (NULL, delim);
		}
		vector<float> v_temp;
		for(int i=0;i<3;i++) //point location x y z
		{
			float point_of_v;
			stringstream string2float; //string to float
			string2float << vectorString[i];
			string2float >> point_of_v;
			v_temp.push_back(point_of_v);
			//point.at<float>(index_vertex_num,i)=point_of_v;
			//cout<<point[index_vertex_num][i]<<endl;
			string2float.str(""); //再次使用前須請空內容
			string2float.clear(); //再次使用前須請空內容
		}
		location_temp.push_back(Point3f(v_temp[0],v_temp[1],v_temp[2]));
	}

	for (int j = 0; j < faces_num; j++)
	{
		getline(fin,input_str); //得到整行字串
		char str_v[100];
		strcpy(str_v, input_str.c_str()); //string to char
		char *delim = " "; //判斷字串 遇到時分割
		char * pch;
		//printf ("Splitting string \"%s\" into tokens:\n",str_v);
		pch = strtok(str_v,delim);
		vector<string> vectorString;
		while (pch != NULL)
		{
			vectorString.push_back(pch);
			//printf ("%s\n",pch);
			pch = strtok (NULL, delim);
		}
		vector<int> f_temp;
		for(int i=1;i<4;i++) //point location x y z
		{
			float point_of_v;
			stringstream string2float; //string to float
			string2float << vectorString[i];
			string2float >> point_of_v;
			f_temp.push_back(point_of_v);
			//point.at<float>(index_vertex_num,i)=point_of_v;
			//cout<<point[index_vertex_num][i]<<endl;
			string2float.str(""); //再次使用前須請空內容
			string2float.clear(); //再次使用前須請空內容
		}
		faces_temp.push_back(Point3f(f_temp[0],f_temp[1],f_temp[2]));
	}

	//cout<<vertex_num<<endl;
	//cout<<faces_num<<endl;
	*location=location_temp;
	*faces=faces_temp;
}

// glass model create //
void create_glass_model(vector<Point2f> glass_2d_LM, Mat image, Mat model_z, vector<Point3f>* location, vector<Point3i>* faces, vector<Point3f>* img_color , vector<vector<uchar>>* img_color_u)
{
	Mat ViewMatrix;
	Mat ProjMatrix;
	matrix_set(&ViewMatrix,&ProjMatrix);

	vector<Point2f> glass_3d_LM;
	vector<Point3f> img_color_temp_2d;
	vector<vector<uchar>> img_color_u_temp_2d;
	for (int i = 0; i < glass_2d_LM.size(); i++)
	{
		Mat temp(4,1,CV_32FC1);
		Mat ProjMatrix_t=ProjMatrix.t();
		temp.at<float>(0,0)=glass_2d_LM[i].x;
		temp.at<float>(1,0)=glass_2d_LM[i].y;
		temp.at<float>(2,0)=1.0;
		temp.at<float>(3,0)=1.0;
		temp=ProjMatrix.inv()*ViewMatrix.inv()*temp;
		glass_3d_LM.push_back(Point2f(temp.at<float>(0,0),temp.at<float>(1,0)));
		//img_color_temp_2d.push_back(Point3f(image.at<Vec3b>(glass_2d_LM[i].y,glass_2d_LM[i].x)[2]/255.0,image.at<Vec3b>(glass_2d_LM[i].y,glass_2d_LM[i].x)[1]/255.0,image.at<Vec3b>(glass_2d_LM[i].y,glass_2d_LM[i].x)[0]/255.0));
		img_color_temp_2d.push_back(Point3f(glass_color_R/255.0,glass_color_G/255.0,glass_color_B/255.0));
		vector<uchar>img_u_in;
		//img_u_in.push_back(image.at<Vec3b>(glass_2d_LM[i].y,glass_2d_LM[i].x)[2]);
		//img_u_in.push_back(image.at<Vec3b>(glass_2d_LM[i].y,glass_2d_LM[i].x)[1]);
		//img_u_in.push_back(image.at<Vec3b>(glass_2d_LM[i].y,glass_2d_LM[i].x)[0]);
		img_u_in.push_back((uchar)glass_color_R);
		img_u_in.push_back((uchar)glass_color_G);
		img_u_in.push_back((uchar)glass_color_B);
		img_color_u_temp_2d.push_back(img_u_in);
	}

	float depth=model_z.at<float>(glass_2d_LM[0].y,glass_2d_LM[0].x)+2;

	vector<Point3f> location_temp;
	vector<Point3f> img_color_temp;
	vector<vector<uchar>> img_color_u_temp;
	float dis=0.4;
	int index=0;
	
	index=1;
	// 1 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 2 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	
	index=2;
	// 3 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 4 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 5 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 6 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	
	index=3;
	// 7 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 8 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);

	index=4;
	// 9 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 10 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);

	index=5;
	// 11 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 12 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);

	index=6;
	// 13 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 14 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);

	index=7;
	// 15 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 16 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);

	index=8;
	// 17 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 18 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 19 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 20 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);

	index=9;
	// 21 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 22 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 23 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 24 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);

	index=9;
	// 25 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y+dis,depth-50));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 26 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y+dis,depth-50));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 27 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y-dis,depth-50));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 28 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y-dis,depth-50));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	
	index=10;
	// 29 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 30 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);

	index=11;
	// 31 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 32 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);

	index=12;
	// 33 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 34 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);

	index=13;
	// 35 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 36 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);

	index=14;
	// 37 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 38 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);

	index=15;
	// 39 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 40 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);

	index=16;
	// 41 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 42 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);

	index=17;
	// 43 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 44 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 45 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 46 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);

	index=18;
	// 47 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 48 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);

	index=19;
	// 49 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 50 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);

	index=20;
	// 51 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 52 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);

	index=21;
	// 53 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 54 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);

	index=22;
	// 55 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 56 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);

	index=23;
	// 57 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 58 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 59 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 60 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);

	index=24;
	// 61 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 62 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 63 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 64 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);

	index=24;
	// 65 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y+dis,depth-50));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 66 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y+dis,depth-50));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 67 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y-dis,depth-50));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 68 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y-dis,depth-50));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);

	index=25;
	// 69 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 70 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);

	index=26;
	// 71 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 72 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);

	index=27;
	// 73 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 74 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);

	index=28;
	// 75 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 76 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);

	index=29;
	// 77 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 78 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);

	index=30;
	// 79 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 80 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);

	index=31;
	// 81 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x-dis,glass_3d_LM[index-1].y-dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);
	// 82 //
	location_temp.push_back(Point3f(glass_3d_LM[index-1].x+dis,glass_3d_LM[index-1].y+dis,depth));
	img_color_temp.push_back(img_color_temp_2d[index-1]);
	img_color_u_temp.push_back(img_color_u_temp_2d[index-1]);

	vector<Point3i> faces_temp;
	for (int i = 0; i < 8; i++)
	{
		faces_temp.push_back(Point3i(nose[i][0]-1,nose[i][1]-1,nose[i][2]-1));
	}
	for (int i = 0; i < 14; i++)
	{
		faces_temp.push_back(Point3i(L_U[i][0]-1,L_U[i][1]-1,L_U[i][2]-1));
	}
	for (int i = 0; i < 14; i++)
	{
		faces_temp.push_back(Point3i(L_Leg[i][0]-1,L_Leg[i][1]-1,L_Leg[i][2]-1));
	}
	for (int i = 0; i < 16; i++)
	{
		faces_temp.push_back(Point3i(L_B[i][0]-1,L_B[i][1]-1,L_B[i][2]-1));
	}
	for (int i = 0; i < 14; i++)
	{
		faces_temp.push_back(Point3i(R_U[i][0]-1,R_U[i][1]-1,R_U[i][2]-1));
	}
	for (int i = 0; i < 14; i++)
	{
		faces_temp.push_back(Point3i(R_Leg[i][0]-1,R_Leg[i][1]-1,R_Leg[i][2]-1));
	}
	for (int i = 0; i < 16; i++)
	{
		faces_temp.push_back(Point3i(R_B[i][0]-1,R_B[i][1]-1,R_B[i][2]-1));
	}


	*location=location_temp; 
	*faces=faces_temp; 
	*img_color=img_color_temp; 
	*img_color_u=img_color_u_temp;
}