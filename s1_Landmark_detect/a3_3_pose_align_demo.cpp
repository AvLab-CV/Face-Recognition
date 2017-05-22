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

// Lab face landmark //
#define use_Lab_LM_model 0
// Dlib landmark //
#define use_dlib 1
// error type //
#define err_type 0

Mat test_c_u;
Mat reg_c_u;
Mat test_c_b;
Mat reg_c_b;
double error_comp=0.0;
Mat img_test;

//////////////////////////////////////////
//        openGL sub function           //
//////////////////////////////////////////
int CurrentWidth = 640,	CurrentHeight = 480,	WindowHandle = 0;
GLuint myIndex;
GLubyte myLists[250];
Mat project_matrix_t(4,4,CV_32FC1);
Mat move_to_ori_T=Mat::eye(4,4,CV_32F);
Mat move_to_ori_T_inv=Mat::eye(4,4,CV_32F);
int Registration_num_count=1; //openGL繪圖編號, 非肯定時勿修正
float Scale_d=1.0;
float Pitch_d=0.0;
float Yaw_d=0.0;
float Roll_d=0.0;
void Initialize(int argc, char* argv[]);
void ResizeFunction(int, int);
Mat final_pic(480, 640, CV_8UC3);
Mat final_pic_flip(480, 640, CV_8UC3);
void RenderFunction(void);
vector<Point3f> location_d;
vector<Point3i> faces_d;
vector<Point3f> img_color_d;
void Draw_face_Model(vector<Point3f> location, vector<Point3i> faces, vector<Point3f> img_color);
void Draw_glass_Model(vector<Point3f> location, vector<Point3i> faces, vector<Point3f> img_color);
Mat model_x_d;
Mat model_y_d;
Mat model_z_d;
Mat model_mask_d;
Mat model_color_d;
void Draw_face_Model_point(Mat model_x,Mat model_y,Mat model_z,Mat model_mask,Mat model_color);
void matrix_set(Mat* view,Mat* project);
void R_matrix(float Pitch, float Yaw, float Roll, Mat *r_matrix_out_x, Mat *r_matrix_out_y, Mat *r_matrix_out_z);

//////////////////////////////////////////
///*          Sub function            *///
//////////////////////////////////////////

// load model //
void load_REG_model(vector<string>* model_name_out, vector<Mat>* move_matrix_vector_out, vector<Mat>* point_lm_vector_out, vector<int>* eth_no_model);

// rotate image by angle //
void rotate_img_angle(float angle, vector<Point2f> LM_in, Mat image_in, vector<Point2f>* LM_out, Mat* image_out);

// align image //
void align_test_img(float p_angle, Mat img_in, vector<Point2f> LM_in, Mat move_m, Mat model_LM_in, Mat* img_out, vector<Point2f>* LM_out, Point3f angla_m);
void align_test_img_auto(float p_angle, Mat img_in, vector<Point2f> LM_in, Mat move_m, Mat model_LM_in, Mat* img_out, vector<Point2f>* LM_out, Point3f angla_m);
void error_cal_c(float p_angle, Mat img_in, vector<Point2f> LM_in, vector<Point2f> model_in, vector<Point2f>* model_out, double* error_num);
void error_cal(float p_angle, Mat img_in, vector<Point2f> LM_in, vector<Point2f> model_in, vector<Point2f>* model_out, double* error_num);
void find_angle(float src_slope, float dst_slope, float* theate);
void get_error(vector<Point2f> target_LM, vector<Point2f> Reg_LM, double* err_num);

// crop image //
void crop_image_test(float p_angle, Mat image_in, vector<Point2f> LM_in, Mat* image_out);
void crop_image_reg(float p_angle, Mat image_in, Mat testImg, vector<Point2f> LM_in, Mat* image_out);

// scale issue //
void match_scale(Mat testImg, Mat model_Image);
void fix_scale(float ori_scale, Mat oriImg, vector<Point2f> oriLM, float angle, Mat templte,  Mat* Img_out, vector<Point2f>* LM_out);

void show(Mat img, vector<Point2f> LM)
{
	Mat img_show=img.clone();
	for (int i = 0; i < LM.size(); i++)
	{
		circle(img_show,LM[i],1,CV_RGB(255,255,0),-1);
	}
	imshow("show",img_show);waitKey(0);
}
void show_m(Mat img, vector<Point2f> LM, vector<Point2f> LM_model)
{
	Mat img_show=img.clone();
	for (int i = 0; i < LM.size(); i++)
	{
		circle(img_show,LM[i],3,CV_RGB(255,255,0),-1);
		circle(img_show,LM_model[i],3,CV_RGB(255,0,0),-1);
		//imshow("show",img_show);waitKey(0);
	}
	//cout << '\a';
	//cvDestroyWindow("show");
	cvNamedWindow("show",0);
	cvMoveWindow("show",0+640+640,0);
	cvResizeWindow("show",640,480);
	//imshow("show",img_show);waitKey(1);	
	imshow("show",img_show);waitKey(1);	
}
void show_m_2(Mat img, vector<Point2f> LM, vector<Point2f> LM_model)
{
	Mat img_show=img.clone();
	for (int i = 0; i < LM.size(); i++)
	{
		circle(img_show,LM[i],3,CV_RGB(255,255,0),-1);
		//circle(img_show,LM_model[i],3,CV_RGB(255,0,0),-1);
		//imshow("show",img_show);waitKey(0);
	}
	//cout << '\a';
	imshow("show2",img_show);waitKey(0);	
}

string data_file_title="../../using_data/";//工作路徑
int main(int argc, char* argv[])
{	
	// set Lab face LM model //
	// 設置 Lab face LM 使用時的參數 //
#if use_Lab_LM_model
	int posemap[18],temp=90;
	string face_LM_model_path=data_file_title+"LM_Model"+"/"+"multipie_35_20_CN3_Sbin3_final.mat";

	LandmarkDetector detector; //宣告model格式
	detector.Load(face_LM_model_path.c_str()); //讀取model
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

	// set Dlib LM//
#if use_dlib
	// Load face detection and pose estimation models.
	dlib::frontal_face_detector detector_dlib = dlib::get_frontal_face_detector(); //內建人臉偵測modelㄝ, 目測 +-45
	dlib::shape_predictor pose_model;
	string dlib_model_path=data_file_title+"dlib-model/shape_predictor_68_face_landmarks.dat";
	//deserialize("../../using_data/dlib-model/shape_predictor_68_face_landmarks.dat") >> pose_model;
	dlib::deserialize(dlib_model_path.c_str()) >> pose_model;
#endif

	// OpenGL 投影矩陣設置 //
	Initialize(argc, argv);// 初始化 glut //

	// model save file name //
	string glass_model_type_save_name="no_glass_1";

	// set the path to load the landmark result detected by dlib // 
	string loadPath=data_file_title+"xml"+"/"+"MPIE_data_fix"+"/";

	// load model //
	vector<string> model_name;
	vector<Mat> move_matrix_vector; //所有讀取model的移動矩陣//
	vector<Mat> point_lm_vector;
	vector<int> eth_no_model;
	load_REG_model(&model_name, &move_matrix_vector, &point_lm_vector, &eth_no_model);
	//cout<<eth_no_model.size()<<endl;

	// load MPIE DATA //
	float t_angle=-15.0;
	//cout<<"input angle number : ";
	//cin>>t_angle;
	string angle="L15_14_0"; // F00_05_1 //
	//cout<<"input angle : ";
	//cin>>angle;
	// R90_24_0 R75_01_0 R60_20_0 R45_19_0 R30_04_1 R15_05_0 //
	// L90_11_0 L75_12_0 L60_09_0 L45_08_0 L30_13_0 L15_14_0 //
	string light="06";
	string glass_type="all"; //set glass type
	// glass_frame_full glass_frame_half glass_frame_none no_glass //
	string MPIE_load_path=data_file_title+"MPIE_classification"+"/"+angle+"/"+light+"/"+glass_type+"/"; // title of FRGC file
	vector< string > MPIE_name; // use to save the FRGC model ID
	Load_insideFile_name(MPIE_load_path, &MPIE_name);

	// load angle path //
	string anglePath=data_file_title+"Reg_model_angle_v2"+"/"+angle+"/";

	vector<int> eth_no;
	// save or load ethnicity data //
	string eth_Path=data_file_title+"xml"+"/"+"eth"+"/";
	// read ethnicity number //
	string loadPath_2=eth_Path+glass_type;
	string eth_num_load=loadPath_2+"/"+"ethnicity_num"+".xml";
	FileStorage FS_eth_LDT;
	FS_eth_LDT.open(eth_num_load, FileStorage::READ);
	FS_eth_LDT[ "LMPT_Data" ] >> eth_no;
	FS_eth_LDT.release();
	//cout<<eth_no.size()<<endl;

	// data save //
	string test_data_title=data_file_title+"Test_Data"+"/"+light+"/";
	_mkdir(test_data_title.c_str());
	test_data_title=data_file_title+"Test_Data"+"/"+light+"/"+angle+"/";
	_mkdir(test_data_title.c_str());
	test_data_title=data_file_title+"Test_Data"+"/"+light+"/"+angle+"/"+glass_model_type_save_name+"/";
	_mkdir(test_data_title.c_str());
	test_data_title=data_file_title+"Test_Data"+"/"+light+"/"+angle+"/"+glass_model_type_save_name+"/"+"test"+"/";
	_mkdir(test_data_title.c_str());
	string train_data_title=data_file_title+"Test_Data"+"/"+light+"/"+angle+"/"+glass_model_type_save_name+"/"+"train"+"/";
	_mkdir(train_data_title.c_str());

	string test_data_title_r=data_file_title+"Test_Data"+"/"+light+"/"+angle+"/"+glass_model_type_save_name+"_rank"+"/";
	_mkdir(test_data_title_r.c_str());
	test_data_title_r=data_file_title+"Test_Data"+"/"+light+"/"+angle+"/"+glass_model_type_save_name+"_rank"+"/"+"test"+"/";;
	_mkdir(test_data_title_r.c_str());
	string train_data_title_r=data_file_title+"Test_Data"+"/"+light+"/"+angle+"/"+glass_model_type_save_name+"_rank"+"/"+"train"+"/";
	_mkdir(train_data_title_r.c_str());

	// RT Matrix //
	int idd;
	cout<<"編號 : ";
	cin>>idd;
	int model_in_index[]={1};
	vector<int> model_in_number_index(model_in_index, model_in_index + sizeof(model_in_index)/sizeof(model_in_index[0]));
	for (int xx=0;xx<model_in_number_index.size();xx++) //for (int id=0;id<MPIE_name.size();id++)  for (int xx=0;xx<model_in_number_index.size();xx++)
	{
		//int id=model_in_number_index[xx]-1;
		int id=idd-2;

		// data save test IMG //
		string test_data_path=test_data_title+MPIE_name[id].substr(0,3)+"/";
		_mkdir(test_data_path.c_str());
		string train_data_path=train_data_title+MPIE_name[id].substr(0,3)+"/";
		_mkdir(train_data_path.c_str());
		string test_data_path_x=test_data_title_r+MPIE_name[id].substr(0,3)+"/";
		_mkdir(test_data_path_x.c_str());
		string train_data_path_x=train_data_title_r+MPIE_name[id].substr(0,3)+"/";
		_mkdir(train_data_path_x.c_str());
		
		// title FRGC image load Path //
		string MPIE_image_load_path=MPIE_load_path+MPIE_name[id];
		cout<<MPIE_image_load_path<<endl;

		// opencv load image //
		Mat image_MPIE=imread(MPIE_image_load_path,1);
		cv::imshow("image_MPIE",image_MPIE);waitKey(1);

		vector<Point2f> test_LM;
		float test_angle;
		if (test_LM.size()==0)
		{
			test_angle=t_angle;
		}
		// -----Read the Data into xml //
		test_LM.clear();
		string FRGC_model_lmpt=loadPath+angle+"/"+MPIE_name[id].substr(0,3)+"_LM.xml";
		FileStorage FS_LDT;
		FS_LDT.open(FRGC_model_lmpt, FileStorage::READ);
		FS_LDT[ "LMPT_Data" ] >> test_LM;
		FS_LDT.release();	
		//cout<<test_LM.size()<<endl;
		//cout<<test_angle<<endl;

		Mat show_LM=image_MPIE.clone();
		//for (int i = 0; i < test_LM.size(); i++)
		//{
		//	circle(show_LM,test_LM[i],1,CV_RGB(255,255,0),-1);
		//}
		//imshow("show_LM",show_LM);waitKey(1);
		Mat test_crop;
		rotate_img_angle(test_angle, test_LM, image_MPIE, &test_LM, &test_crop);
		
		imshow("test_crop",test_crop);waitKey(1);

		string test_data_path_2=test_data_title+MPIE_name[id].substr(0,3)+"/"+MPIE_name[id].substr(0,3)+"/";
		_mkdir(test_data_path_2.c_str());
		test_data_path_2=test_data_title+MPIE_name[id].substr(0,3)+"/"+MPIE_name[id].substr(0,3)+"/"+MPIE_name[id].substr(0,3)+".png";
		imwrite(test_data_path_2,test_crop);
		//test_data_path_2=test_data_title+MPIE_name[id].substr(0,3)+"/"+MPIE_name[id].substr(0,3)+"/"+MPIE_name[id].substr(0,3)+"_c.png";
		//imwrite(test_data_path_2,test_c_u); // component
		//test_data_path_2=test_data_title+MPIE_name[id].substr(0,3)+"/"+MPIE_name[id].substr(0,3)+"/"+MPIE_name[id].substr(0,3)+"_b.png";
		//imwrite(test_data_path_2,test_c_b); // component
		//test_crop.release();
		string test_data_path_2_r=test_data_title_r+MPIE_name[id].substr(0,3)+"/"+MPIE_name[id].substr(0,3)+"/";
		_mkdir(test_data_path_2_r.c_str());
		test_data_path_2_r=test_data_title_r+MPIE_name[id].substr(0,3)+"/"+MPIE_name[id].substr(0,3)+"/"+MPIE_name[id].substr(0,3)+".png";
		imwrite(test_data_path_2_r,test_crop);
		//test_data_path_2_r=test_data_title_r+MPIE_name[id].substr(0,3)+"/"+MPIE_name[id].substr(0,3)+"/"+MPIE_name[id].substr(0,3)+"_c.png";
		//imwrite(test_data_path_2_r,test_c_u); // component
		//test_data_path_2_r=test_data_title_r+MPIE_name[id].substr(0,3)+"/"+MPIE_name[id].substr(0,3)+"/"+MPIE_name[id].substr(0,3)+"_b.png";
		//imwrite(test_data_path_2_r,test_c_b); // component
		//test_crop.release();

		//continue;

		// -----Read the Data into xml //
		Point3f angle_read;
		string angle_lmpt=anglePath+"/"+MPIE_name[id].substr(0,3)+"_angle.xml";
		FS_LDT.open(angle_lmpt, FileStorage::READ);
		FS_LDT[ "ANGLE_Data" ] >> angle_read;
		FS_LDT.release();	

		//cout<<angle_read<<endl;
		vector<string> reg_name;
		vector<double> error_c;
		vector<Mat> reg_c;
		for (int j=0;j<move_matrix_vector.size();j++)
		{
			//cout<<j<<endl;
			Registration_num_count=j+1;
			//cout<<eth_no[id]<<endl;
			//cout<<eth_no_model[j]<<endl;

			//if (eth_no[id] != eth_no_model[j]){continue;}

			Mat image_Reg_out;
			vector<Point2f> Reg_LM_out;
			Mat reg_crop;
			//angle_read.x=5;
			//angle_read.y=-65;
			//angle_read.x=5;
			//align_test_img(test_angle, test_crop, test_LM, move_matrix_vector[j], point_lm_vector[j], &reg_crop, &Reg_LM_out, angle_read);
			align_test_img_auto(test_angle, test_crop, test_LM, move_matrix_vector[j], point_lm_vector[j], &reg_crop, &Reg_LM_out, angle_read);
			
			imshow("reg_crop",reg_crop);waitKey(1);

			string train_data_path2=train_data_title+MPIE_name[id].substr(0,3)+"/"+model_name[j].substr(0,3)+"/";
			_mkdir(train_data_path2.c_str());
			train_data_path2=train_data_title+MPIE_name[id].substr(0,3)+"/"+model_name[j].substr(0,3)+"/"+model_name[j].substr(0,3)+".png";
			imwrite(train_data_path2,reg_crop);
			//train_data_path2=train_data_title+MPIE_name[id].substr(0,3)+"/"+model_name[j].substr(0,3)+"/"+model_name[j].substr(0,3)+"_c.png";
			//imwrite(train_data_path2,reg_c_u);// component
			//train_data_path2=train_data_title+MPIE_name[id].substr(0,3)+"/"+model_name[j].substr(0,3)+"/"+model_name[j].substr(0,3)+"_b.png";
			//imwrite(train_data_path2,reg_c_b);// component

			//string train_data_path2_r=train_data_title_r+MPIE_name[id].substr(0,3)+"/"+model_name[j].substr(0,3)+"/";
			//_mkdir(train_data_path2_r.c_str());
			//train_data_path2_r=train_data_title_r+MPIE_name[id].substr(0,3)+"/"+model_name[j].substr(0,3)+"/"+model_name[j].substr(0,3)+".png";
			//imwrite(train_data_path2_r,reg_c_u);// component

			reg_name.push_back(model_name[j].substr(0,3));
			error_c.push_back(error_comp);
			reg_c.push_back(reg_crop);

			reg_crop.release();
		}

		vector<double> s_dis_compare=error_c;
		std::sort(error_c.begin(),error_c.end(),less<double>());
		//save rank
		vector<int> num;
		for (int i = 0; i < error_c.size(); i++)
		{
			for (int k=0; k<s_dis_compare.size(); k++)
			{
				if (error_c[i]==s_dis_compare[k])
				{
					num.push_back(k);
				}
			}
		}

		for (int i = 0; i < error_c.size(); i++)
		{
			string point_number_str;
			stringstream int2string;
			int2string<<i+1;
			int2string>>point_number_str;

			string point_number_str2;
			stringstream int2string2;
			int2string2<<s_dis_compare[num[i]];
			int2string2>>point_number_str2;
			
			cout<<"error : "<<s_dis_compare[num[i]]<<endl;
			cout<<"name : "<<reg_name[num[i]]<<endl;
			Mat reg_crop=reg_c[num[i]].clone();
			string train_data_path2=train_data_title_r+MPIE_name[id].substr(0,3)+"/"+point_number_str+"_"+reg_name[num[i]]+"_"+point_number_str2+"/";
			_mkdir(train_data_path2.c_str());
			train_data_path2=train_data_title_r+MPIE_name[id].substr(0,3)+"/"+point_number_str+"_"+reg_name[num[i]]+"_"+point_number_str2+"/"+reg_name[num[i]]+".png";
			imwrite(train_data_path2,reg_crop);
			imshow("reg_crop",reg_crop);waitKey(1);
			//imwrite(train_data_path2,reg_c_u);// component

			int2string2.str("");
			int2string2.clear();
			
			int2string.str("");
			int2string.clear();

			if (!strcmp(reg_name[num[i]].c_str(),MPIE_name[id].substr(0,3).c_str()))
			{
				break;
			}
		}
	}


	cout << '\a';
	system("pause");
	return 0;
}

//////////////////////////////////////////
//        openGL sub function           //
//////////////////////////////////////////
void Initialize(int argc, char* argv[])
{
	glutInit(&argc, argv);

	//以下使用 Context 功能
	/*	glutInitContextVersion(4, 0);
		glutInitContextFlags(GLUT_FORWARD_COMPATIBLE);
		glutInitContextProfile(GLUT_CORE_PROFILE);
		glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,GLUT_ACTION_GLUTMAINLOOP_RETURNS);*/

	//設定 glut 畫布尺寸 與color / depth模式
	glutInitWindowPosition(100, 100); // 設定視窗位置
	glutInitWindowSize(CurrentWidth, CurrentHeight);
	glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA);
	
	//根據已設定好的 glut (如尺寸,color,depth) 向window要求建立一個視窗，接著若失敗則退出程式
	WindowHandle = glutCreateWindow("MPIE Data Example");
	if(WindowHandle < 1) {	fprintf(stderr,"ERROR: Could not create a new rendering window.\n");exit(EXIT_FAILURE);	}
	
	glutReshapeFunc(ResizeFunction); //設定視窗 大小若改變，則跳到"ResizeFunction"這個函數處理應變事項
	glutDisplayFunc(RenderFunction);  //設定視窗 如果要畫圖 則執行"RenderFunction"
	//glutIdleFunc(IdleFunction);		  //閒置時...請系統執行"IdleFunction"

	GLenum GlewInitResult = glewInit();
	if (GlewInitResult != GLEW_OK ) {	fprintf(stderr,"ERROR: %s\n",glewGetErrorString(GlewInitResult)	);	exit(EXIT_FAILURE);	}

	//背景顏色黑
	//glClearColor(0.65f, 0.65f, 0.65f, 1.0f);
	glClearColor(0.0, 0.0, 0.0, 0.0f);

	glEnable(GL_DEPTH_TEST);
	glFlush();
	//exit(EXIT_SUCCESS);

}
void ResizeFunction(int Width, int Height)
{
	CurrentWidth = Width;
	CurrentHeight = Height;
	glViewport(0, 0, CurrentWidth, CurrentHeight);
}
void RenderFunction(void)
{
	// model 顯示 //
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glViewport(0, 0, CurrentWidth, CurrentHeight);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	//glOrtho(-float(CurrentWidth)/2.0,float(CurrentWidth)/2.0,-float(CurrentHeight)/2.0,float(CurrentHeight)/2.0,-CurrentHeight*10.0,CurrentHeight*10.0);
	////gluLookAt(0,0,1000,0,0,0,0,1,0);
	////glOrtho(-float(CurrentWidth)/camera_scale,float(CurrentWidth)/camera_scale,-float(CurrentHeight)/camera_scale,float(CurrentHeight)/camera_scale,-CurrentHeight*5.0,CurrentHeight*5.0);
	////gluLookAt(0,0,500,0,0,0,0,1,0);

	glLoadMatrixf((float*)project_matrix_t.data);

	////double F[16];
	////double Zmax=-5000.0;
	////double Zmin=5000.0;
	////double F_ten=2/(Zmax-Zmin);
	////double F_fourTeen=-(Zmax+Zmin)/(Zmax-Zmin);
	////F[0] = 0.0054                ; F[4] =  -1.9066*pow(10.0,-6.0)   ;  F[8]  =  5.9645*pow(10,-7.0)			; F[12] = 0.0014			;        
	////F[1] = 7.9908*pow(10.0,-6.0) ; F[5] =   0.0074                  ;  F[9]  = -2.0505*pow(10,-4.0)			; F[13] = -0.3964			;
	////F[2] = 0.                    ; F[6] =   0.0                     ;  F[10] = F_ten                        ; F[14] = F_fourTeen              ;  
	////F[3] = 0.                    ; F[7] =   0.0                     ;  F[11] = 0.0                          ; F[15] = 1.0               ;

	////glLoadMatrixd(F);

	glPushMatrix();
	glMultMatrixf((float*)move_to_ori_T_inv.data);
	glRotatef(Roll_d,0,0,1);
	glRotatef(Yaw_d,0,1,0);
	glRotatef(Pitch_d,1,0,0);
	//glScalef(Scale_d,Scale_d,Scale_d);
	glMultMatrixf((float*)move_to_ori_T.data);
	glCallLists(1,GL_UNSIGNED_BYTE,&myLists[Registration_num_count-1]);
	//Draw_Model(mask_d, model_x_d, model_y_d, model_z_d, Reg_img_d);
	glPopMatrix();

	//Mat final_pic(480, 640, CV_8UC3);
	//Mat final_pic_flip(480, 640, CV_8UC3);
	glReadPixels(0,0,CurrentWidth,CurrentHeight,GL_BGR,GL_UNSIGNED_BYTE,final_pic.data);
	flip(final_pic,final_pic_flip,0);
	//imshow("final_pic_flip",final_pic_flip);
	//string num;
	//stringstream string2float; //string to float
	//string2float << Registration_num_count;
	//string2float >> num;
	//string path="../../using_data/openGL_test/"+num+".png";
	//imwrite(path,final_pic_flip);
	//string2float.str(""); //再次使用前須請空內容
	//string2float.clear(); //再次使用前須請空內容
	//waitKey(1);

	//cout<<Registration_num_count<<endl;

	glFlush();
	glutSwapBuffers();
}
void Draw_face_Model(vector<Point3f> location, vector<Point3i> faces, vector<Point3f> img_color)
{
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

	//glEnable(GL_BLEND);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glBegin(GL_TRIANGLES);
	for (int i=0; i<faces.size(); i++)
	{
		glColor4f(img_color[faces[i].x].x, img_color[faces[i].x].y, img_color[faces[i].x].z, 1.0);
		glVertex3f(location[faces[i].x].x, location[faces[i].x].y, location[faces[i].x].z);

		glColor4f(img_color[faces[i].y].x, img_color[faces[i].y].y, img_color[faces[i].y].z, 1.0);
		glVertex3f(location[faces[i].y].x, location[faces[i].y].y, location[faces[i].y].z);

		glColor4f(img_color[faces[i].z].x, img_color[faces[i].z].y, img_color[faces[i].z].z, 1.0);
		glVertex3f(location[faces[i].z].x, location[faces[i].z].y, location[faces[i].z].z);
	}
	glEnd();
}
void Draw_face_Model_point(Mat model_x,Mat model_y,Mat model_z,Mat model_mask,Mat model_color)
{
	glPointSize(3);
	glEnable(GL_DEPTH_TEST);
	glBegin(GL_POINTS);
	for (int i=0;i<model_mask.rows;i++)
	{
		for (int j=0;j<model_mask.cols;j++)
		{
			if (model_mask.at<uchar>(i,j)!=0)
			{				
				if(model_z.at<float>(i,j)==0){continue;}
				glColor4f(model_color.at<Vec3b>(i,j)[2]/255.0,model_color.at<Vec3b>(i,j)[1]/255.0,model_color.at<Vec3b>(i,j)[0]/255.0,0.3);
				glVertex3f(model_x.at<float>(i,j),model_y.at<float>(i,j),model_z.at<float>(i,j));
			}
		}
	}
	glEnd();
}
void Draw_glass_Model(vector<Point3f> location, vector<Point3i> faces, vector<Point3f> img_color)
{
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

	//glEnable(GL_BLEND);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glBegin(GL_TRIANGLES);
	for (int i=0; i<faces.size(); i++)
	{
		glColor4f(img_color[faces[i].x].x, img_color[faces[i].x].y, img_color[faces[i].x].z, 1.0);
		glVertex3f(location[faces[i].x].x, location[faces[i].x].y, location[faces[i].x].z);

		glColor4f(img_color[faces[i].y].x, img_color[faces[i].y].y, img_color[faces[i].y].z, 1.0);
		glVertex3f(location[faces[i].y].x, location[faces[i].y].y, location[faces[i].y].z);

		glColor4f(img_color[faces[i].z].x, img_color[faces[i].z].y, img_color[faces[i].z].z, 1.0);
		glVertex3f(location[faces[i].z].x, location[faces[i].z].y, location[faces[i].z].z);
	}
	glEnd();

	glLineWidth(1.0);
	glBegin(GL_LINES);
	for (int i=0; i<faces.size(); i++)
	{
		glColor4f(img_color[faces[i].x].x, img_color[faces[i].x].y, img_color[faces[i].x].z, 1.0);
		glVertex3f(location[faces[i].x].x, location[faces[i].x].y, location[faces[i].x].z);
		glVertex3f(location[faces[i].y].x, location[faces[i].y].y, location[faces[i].y].z);

		glColor4f(img_color[faces[i].y].x, img_color[faces[i].y].y, img_color[faces[i].y].z, 1.0);
		glVertex3f(location[faces[i].y].x, location[faces[i].y].y, location[faces[i].y].z);
		glVertex3f(location[faces[i].z].x, location[faces[i].z].y, location[faces[i].z].z);

		glColor4f(img_color[faces[i].z].x, img_color[faces[i].z].y, img_color[faces[i].z].z, 1.0);
		glVertex3f(location[faces[i].z].x, location[faces[i].z].y, location[faces[i].z].z);
		glVertex3f(location[faces[i].x].x, location[faces[i].x].y, location[faces[i].x].z);
	}
	glEnd();

}
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


//////////////////////////////////////////
///*          Sub function            *///
//////////////////////////////////////////

// load model //
void load_REG_model(vector<string>* model_name_out, vector<Mat>* move_matrix_vector_out, vector<Mat>* point_lm_vector_out, vector<int>* eth_no_model)
{
	// load model //
	string glass_model_type="all";
	string save_model_title=data_file_title+"Reg_model"+"/"+"06"+"/"+glass_model_type+"/";
	vector<string> model_name;
	Load_insideFile_name(save_model_title, &model_name);
	vector<Mat> move_matrix_vector; //所有讀取model的移動矩陣//
	vector<Mat> point_lm_vector;
	int index=1;
	//cout<<model_name.size()<<endl;
	vector<int> eth_no_model_temp;
	for(int i=0;i<model_name.size();i++) //for(int i=0;i<model_name.size();i++)
	{
		// model path //
		vector<Point3f> location;
		vector<Point3i> faces;
		vector<Point3f> img_color;
		int eth_no_temp;

		string MODEL_point,MODEL_face,MODEL_color,MODEL_eth;
		FileStorage FS_LDT;
		MODEL_point=save_model_title+model_name[i]+"/"+model_name[i]+"_v.xml";
		FS_LDT.open(MODEL_point, FileStorage::READ);
		FS_LDT[ "V_Data" ] >> location;
		FS_LDT.release();
		//cout<<location.size()<<endl;

		MODEL_face=save_model_title+model_name[i]+"/"+model_name[i]+"_f.xml";
		FS_LDT.open(MODEL_face, FileStorage::READ);
		FS_LDT[ "F_Data" ] >> faces;
		FS_LDT.release();
		//cout<<faces.size()<<endl;

		MODEL_color=save_model_title+model_name[i]+"/"+model_name[i]+"_c.xml";
		FS_LDT.open(MODEL_color, FileStorage::READ);
		FS_LDT[ "C_Data" ] >> img_color;
		FS_LDT.release();
		//cout<<img_color.size()<<endl;

		MODEL_eth=save_model_title+model_name[i]+"/"+model_name[i]+"_eth.xml";
		FS_LDT.open(MODEL_eth, FileStorage::READ);
		FS_LDT[ "Data" ] >> eth_no_temp;
		FS_LDT.release();
		eth_no_model_temp.push_back(eth_no_temp);

		// model_path_point //
		/*Mat model_x;
		Mat model_y;
		Mat model_z;
		Mat model_mask;
		Mat model_color;
		MODEL_point=save_model_title+model_name[i]+"/"+model_name[i]+"_x.xml";
		FS_LDT.open(MODEL_point, FileStorage::READ);
		FS_LDT[ "Data" ] >> model_x;
		FS_LDT.release();
		model_x_d=model_x;

		MODEL_point=save_model_title+model_name[i]+"/"+model_name[i]+"_y.xml";
		FS_LDT.open(MODEL_point, FileStorage::READ);
		FS_LDT[ "Data" ] >> model_y;
		FS_LDT.release();
		model_y_d=model_y;

		MODEL_point=save_model_title+model_name[i]+"/"+model_name[i]+"_z.xml";
		FS_LDT.open(MODEL_point, FileStorage::READ);
		FS_LDT[ "Data" ] >> model_z;
		FS_LDT.release();
		model_z_d=model_z;

		MODEL_face=save_model_title+model_name[i]+"/"+model_name[i]+"_mask.xml";
		FS_LDT.open(MODEL_face, FileStorage::READ);
		FS_LDT[ "Data" ] >> model_mask;
		FS_LDT.release();
		model_mask_d=model_mask;

		MODEL_color=save_model_title+model_name[i]+"/"+model_name[i]+"_color.xml";
		FS_LDT.open(MODEL_color, FileStorage::READ);
		FS_LDT[ "Data" ] >> model_color;
		FS_LDT.release();
		model_color_d=model_color;*/

		float mx=0;
		float my=0;
		float mz=0;
		for (int i = 0; i < location.size(); i++)
		{
			mx=mx+location[i].x;
			my=my+location[i].y;
			mz=mz+location[i].z;
		}
		mx=mx/location.size();
		my=my/location.size();
		mz=mz/location.size();

		Mat move_matrix=Mat::eye(4,4,CV_32F);
		move_matrix.at<float>(0,3)=-mx;
		move_matrix.at<float>(1,3)=-my;
		move_matrix.at<float>(2,3)=-mz;

		move_matrix_vector.push_back(move_matrix);

		location_d=location;
		faces_d=faces;
		img_color_d=img_color;

		// landmark //
		string MODEL_LM;
		Mat model_LM_3d;
		MODEL_LM=save_model_title+model_name[i]+"/"+model_name[i]+"_l.xml";
		FS_LDT.open(MODEL_LM, FileStorage::READ);
		FS_LDT[ "L_Data" ] >> model_LM_3d;
		FS_LDT.release();

		//Mat model_LM_3d_alt(4,8,CV_32FC1);
		//for (int i=0;i<8;i++)
		//{
		//	model_LM_3d_alt.at<float>(0,i)=model_LM_3d.at<float>(0,i+4);
		//	model_LM_3d_alt.at<float>(1,i)=model_LM_3d.at<float>(1,i+4);
		//	model_LM_3d_alt.at<float>(2,i)=model_LM_3d.at<float>(2,i+4);
		//	model_LM_3d_alt.at<float>(3,i)=1.0;
		//}
		Mat model_LM_3d_alt(4,29,CV_32FC1);
		for (int i=0;i<29;i++)
		{
			model_LM_3d_alt.at<float>(0,i)=model_LM_3d.at<float>(0,i);
			model_LM_3d_alt.at<float>(1,i)=model_LM_3d.at<float>(1,i);
			model_LM_3d_alt.at<float>(2,i)=model_LM_3d.at<float>(2,i);
			model_LM_3d_alt.at<float>(3,i)=1.0;
		}
		//cout<<model_LM_3d<<endl;
		//cout<<model_LM_3d_alt<<endl;

		point_lm_vector.push_back(model_LM_3d_alt);

		// glass model load //
		string MODEL_glass_p,MODEL_glass_f, MODEL_glass_c;
		vector<Point3f> location_glass; 
		vector<Point3i> faces_glass;
		vector<Point3f> img_color_glass;
		MODEL_glass_p=save_model_title+model_name[i]+"/"+model_name[i]+"_vg.xml";
		FS_LDT.open(MODEL_glass_p, FileStorage::READ);
		FS_LDT[ "V_Data" ] >> location_glass;
		FS_LDT.release();

		MODEL_glass_f=save_model_title+model_name[i]+"/"+model_name[i]+"_fg.xml";
		FS_LDT.open(MODEL_glass_f, FileStorage::READ);
		FS_LDT[ "F_Data" ] >> faces_glass;
		FS_LDT.release();

		MODEL_glass_c=save_model_title+model_name[i]+"/"+model_name[i]+"_cg.xml";
		FS_LDT.open(MODEL_glass_c, FileStorage::READ);
		FS_LDT[ "C_Data" ] >> img_color_glass;
		FS_LDT.release();

		myIndex =  glGenLists(1);
		//cout<<"myIndex : "<<myIndex<<endl;
		glNewList(myIndex, GL_COMPILE); // compile the first one 
		glPushMatrix();
		if(location_glass.size()!=0)
		{
			Draw_glass_Model(location_glass, faces_glass, img_color_glass);
		}
		//Draw_face_Model_point(model_x_d,model_y_d,model_z_d,model_mask_d,model_color_d);
		Draw_face_Model(location_d, faces_d, img_color_d);
		glPopMatrix();
		glEndList(); 
		myLists[index-1] = myIndex;
		index=index+1;
	}

	cout<<" Register Model Pre-load done "<<endl;

	*model_name_out=model_name;
	*move_matrix_vector_out=move_matrix_vector;
	*point_lm_vector_out=point_lm_vector;
	*eth_no_model=eth_no_model_temp;
}

// rotate image by angle //
void rotate_img_angle(float angle, vector<Point2f> LM_in, Mat image_in, vector<Point2f>* LM_out, Mat* image_out)
{
	Mat image_crop=image_in.clone();image_crop.setTo(0);
	Mat image_in_temp;
	vector<Point2f> img_LM_after; // save the rotated landmark
	Mat image_in_temp_2;
	vector<Point2f> img_LM_after_2; // save the rotated landmark
	Mat image_show=image_in.clone();
	float bound=20.0;

	Mat test_crop;

	if (angle <= 45.0 && angle >= -45.0)
	{
		float angle_to_zero; // angle between input and 0
		find_theate_zero(LM_in, &angle_to_zero);
		Point2f center_point; // find the center point of landmark
		find_LM_center(LM_in, &center_point); 
		Mat Rotate_Matrix=getRotationMatrix2D(center_point, angle_to_zero, 1.0); // calculate the rotate matrix [2x3]
		rotate_LM(LM_in, &img_LM_after, angle_to_zero); 
		cv::warpAffine(image_in, image_in_temp, Rotate_Matrix, image_in.size()); // rotate the ori. image 
		//show(image_in_temp, img_LM_after);
		if (angle == 0 )
		{
			Point2f Leye=Point2f((img_LM_after[20-1].x+img_LM_after[23-1].x)/2,(img_LM_after[20-1].y+img_LM_after[23-1].y)/2);
			Point2f Reye=Point2f((img_LM_after[26-1].x+img_LM_after[29-1].x)/2,(img_LM_after[26-1].y+img_LM_after[29-1].y)/2);
			float eye_length=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
			float norm_length=70.0;
			float scale_IMG=norm_length/eye_length;
			cv::resize(image_in_temp,image_in_temp_2,Size(image_in_temp.cols*scale_IMG,image_in_temp.rows*scale_IMG)); //normlized image
			scale_LM(img_LM_after,&img_LM_after_2, scale_IMG); //縮放移動後的LM
			Leye=Point2f((img_LM_after_2[20-1].x+img_LM_after_2[23-1].x)/2,(img_LM_after_2[20-1].y+img_LM_after_2[23-1].y)/2);
			Reye=Point2f((img_LM_after_2[26-1].x+img_LM_after_2[29-1].x)/2,(img_LM_after_2[26-1].y+img_LM_after_2[29-1].y)/2);
			Point2f src_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);
			Point2f dst_pts=Point2f(320,160);

			float min_x=image_in_temp_2.cols;
			float min_y=image_in_temp_2.rows;
			float max_x=0;
			float max_y=0;
			for (int i = 0; i < img_LM_after_2.size(); i++)
			{
				min_x=(min_x<img_LM_after_2[i].x)?min_x:img_LM_after_2[i].x;
				min_y=(min_y<img_LM_after_2[i].y)?min_y:img_LM_after_2[i].y;
				max_x=(max_x>img_LM_after_2[i].x)?max_x:img_LM_after_2[i].x;
				max_y=(max_y>img_LM_after_2[i].y)?max_y:img_LM_after_2[i].y;
			}
			min_x=min_x-bound;
			min_y=min_y-bound;
			max_x=max_x+bound;
			max_y=max_y+bound;

			move_LM_point(img_LM_after_2, &img_LM_after_2, dst_pts, src_pts); // 移動測試影像的LM點
			image_in_temp_2(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(image_crop(Rect(dst_pts.x-(src_pts.x-min_x),dst_pts.y-(src_pts.y-min_y),cvRound(max_x-min_x),cvRound(max_y-min_y))));
		}
		if (angle == 15)
		{
			Point2f Leye=Point2f((img_LM_after[20-1].x+img_LM_after[23-1].x)/2,(img_LM_after[20-1].y+img_LM_after[23-1].y)/2);
			Point2f Reye=Point2f((img_LM_after[26-1].x+img_LM_after[29-1].x)/2,(img_LM_after[26-1].y+img_LM_after[29-1].y)/2);
			float eye_length=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
			float norm_length=68.0;
			float scale_IMG=norm_length/eye_length;
			cv::resize(image_in_temp,image_in_temp_2,Size(image_in_temp.cols*scale_IMG,image_in_temp.rows*scale_IMG)); //normlized image
			scale_LM(img_LM_after,&img_LM_after_2, scale_IMG); //縮放移動後的LM
			Leye=Point2f((img_LM_after_2[20-1].x+img_LM_after_2[23-1].x)/2,(img_LM_after_2[20-1].y+img_LM_after_2[23-1].y)/2);
			Reye=Point2f((img_LM_after_2[26-1].x+img_LM_after_2[29-1].x)/2,(img_LM_after_2[26-1].y+img_LM_after_2[29-1].y)/2);
			Point2f src_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);
			Point2f dst_pts=Point2f(320,160);

			float min_x=image_in_temp_2.cols;
			float min_y=image_in_temp_2.rows;
			float max_x=0;
			float max_y=0;
			for (int i = 0; i < img_LM_after_2.size(); i++)
			{
				min_x=(min_x<img_LM_after_2[i].x)?min_x:img_LM_after_2[i].x;
				min_y=(min_y<img_LM_after_2[i].y)?min_y:img_LM_after_2[i].y;
				max_x=(max_x>img_LM_after_2[i].x)?max_x:img_LM_after_2[i].x;
				max_y=(max_y>img_LM_after_2[i].y)?max_y:img_LM_after_2[i].y;
			}
			min_x=min_x-bound;
			min_y=min_y-bound;
			max_x=max_x+bound;
			max_y=max_y+bound;

			move_LM_point(img_LM_after_2, &img_LM_after_2, dst_pts, src_pts); // 移動測試影像的LM點
			image_in_temp_2(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(image_crop(Rect(dst_pts.x-(src_pts.x-min_x),dst_pts.y-(src_pts.y-min_y),cvRound(max_x-min_x),cvRound(max_y-min_y))));
		}
		if (angle == 30 )
		{
			Point2f Leye=Point2f((img_LM_after[20-1].x+img_LM_after[23-1].x)/2,(img_LM_after[20-1].y+img_LM_after[23-1].y)/2);
			Point2f Reye=Point2f((img_LM_after[26-1].x+img_LM_after[29-1].x)/2,(img_LM_after[26-1].y+img_LM_after[29-1].y)/2);
			float eye_length=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
			float norm_length=62.0;
			float scale_IMG=norm_length/eye_length;
			cv::resize(image_in_temp,image_in_temp_2,Size(image_in_temp.cols*scale_IMG,image_in_temp.rows*scale_IMG)); //normlized image
			scale_LM(img_LM_after,&img_LM_after_2, scale_IMG); //縮放移動後的LM
			Leye=Point2f((img_LM_after_2[20-1].x+img_LM_after_2[23-1].x)/2,(img_LM_after_2[20-1].y+img_LM_after_2[23-1].y)/2);
			Reye=Point2f((img_LM_after_2[26-1].x+img_LM_after_2[29-1].x)/2,(img_LM_after_2[26-1].y+img_LM_after_2[29-1].y)/2);
			Point2f src_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);
			Point2f dst_pts=Point2f(320,160);

			float min_x=image_in_temp_2.cols;
			float min_y=image_in_temp_2.rows;
			float max_x=0;
			float max_y=0;
			for (int i = 0; i < img_LM_after_2.size(); i++)
			{
				min_x=(min_x<img_LM_after_2[i].x)?min_x:img_LM_after_2[i].x;
				min_y=(min_y<img_LM_after_2[i].y)?min_y:img_LM_after_2[i].y;
				max_x=(max_x>img_LM_after_2[i].x)?max_x:img_LM_after_2[i].x;
				max_y=(max_y>img_LM_after_2[i].y)?max_y:img_LM_after_2[i].y;
			}
			min_x=min_x-bound;
			min_y=min_y-bound;
			max_x=max_x+bound;
			max_y=max_y+bound;

			move_LM_point(img_LM_after_2, &img_LM_after_2, dst_pts, src_pts); // 移動測試影像的LM點
			image_in_temp_2(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(image_crop(Rect(dst_pts.x-(src_pts.x-min_x),dst_pts.y-(src_pts.y-min_y),cvRound(max_x-min_x),cvRound(max_y-min_y))));
		}
		if (angle == 45)
		{
			Point2f Leye=img_LM_after[20-1];
			Point2f Reye=img_LM_after[15-1];
			float eye_length=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
			float norm_length=65.0;
			float scale_IMG=norm_length/eye_length;
			cv::resize(image_in_temp,image_in_temp_2,Size(image_in_temp.cols*scale_IMG,image_in_temp.rows*scale_IMG)); //normlized image
			scale_LM(img_LM_after,&img_LM_after_2, scale_IMG); //縮放移動後的LM
			Point2f src_pts=img_LM_after_2[23-1];
			Point2f dst_pts=Point2f(320,160);

			float min_x=image_in_temp_2.cols;
			float min_y=image_in_temp_2.rows;
			float max_x=0;
			float max_y=0;
			for (int i = 0; i < img_LM_after_2.size(); i++)
			{
				min_x=(min_x<img_LM_after_2[i].x)?min_x:img_LM_after_2[i].x;
				min_y=(min_y<img_LM_after_2[i].y)?min_y:img_LM_after_2[i].y;
				max_x=(max_x>img_LM_after_2[i].x)?max_x:img_LM_after_2[i].x;
				max_y=(max_y>img_LM_after_2[i].y)?max_y:img_LM_after_2[i].y;
			}
			min_x=min_x-bound;
			min_y=min_y-bound;
			max_x=max_x+bound;
			max_y=max_y+bound;

			move_LM_point(img_LM_after_2, &img_LM_after_2, dst_pts, src_pts); // 移動測試影像的LM點
			image_in_temp_2(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(image_crop(Rect(dst_pts.x-(src_pts.x-min_x),dst_pts.y-(src_pts.y-min_y),cvRound(max_x-min_x),cvRound(max_y-min_y))));
		}
		if (angle == -15)
		{
			Point2f Leye=Point2f((img_LM_after[20-1].x+img_LM_after[23-1].x)/2,(img_LM_after[20-1].y+img_LM_after[23-1].y)/2);
			Point2f Reye=Point2f((img_LM_after[26-1].x+img_LM_after[29-1].x)/2,(img_LM_after[26-1].y+img_LM_after[29-1].y)/2);
			float eye_length=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
			float norm_length=66.0;
			float scale_IMG=norm_length/eye_length;
			cv::resize(image_in_temp,image_in_temp_2,Size(image_in_temp.cols*scale_IMG,image_in_temp.rows*scale_IMG)); //normlized image
			scale_LM(img_LM_after,&img_LM_after_2, scale_IMG); //縮放移動後的LM

			Leye=Point2f((img_LM_after_2[20-1].x+img_LM_after_2[23-1].x)/2,(img_LM_after_2[20-1].y+img_LM_after_2[23-1].y)/2);
			Reye=Point2f((img_LM_after_2[26-1].x+img_LM_after_2[29-1].x)/2,(img_LM_after_2[26-1].y+img_LM_after_2[29-1].y)/2);
			Point2f src_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);
			Point2f dst_pts=Point2f(320,160);

			float min_x=image_in_temp_2.cols;
			float min_y=image_in_temp_2.rows;
			float max_x=0;
			float max_y=0;
			for (int i = 0; i < img_LM_after_2.size(); i++)
			{
				min_x=(min_x<img_LM_after_2[i].x)?min_x:img_LM_after_2[i].x;
				min_y=(min_y<img_LM_after_2[i].y)?min_y:img_LM_after_2[i].y;
				max_x=(max_x>img_LM_after_2[i].x)?max_x:img_LM_after_2[i].x;
				max_y=(max_y>img_LM_after_2[i].y)?max_y:img_LM_after_2[i].y;
			}
			min_x=min_x-bound;
			min_y=min_y-bound;
			max_x=max_x+bound;
			max_y=max_y+bound;

			move_LM_point(img_LM_after_2, &img_LM_after_2, dst_pts, src_pts); // 移動測試影像的LM點
			image_in_temp_2(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(image_crop(Rect(dst_pts.x-(src_pts.x-min_x),dst_pts.y-(src_pts.y-min_y),cvRound(max_x-min_x),cvRound(max_y-min_y))));
		}
		if (angle == -30 )
		{
			Point2f Leye=Point2f((img_LM_after[20-1].x+img_LM_after[23-1].x)/2,(img_LM_after[20-1].y+img_LM_after[23-1].y)/2);
			Point2f Reye=Point2f((img_LM_after[26-1].x+img_LM_after[29-1].x)/2,(img_LM_after[26-1].y+img_LM_after[29-1].y)/2);
			float eye_length=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
			float norm_length=63.0;
			float scale_IMG=norm_length/eye_length;
			cv::resize(image_in_temp,image_in_temp_2,Size(image_in_temp.cols*scale_IMG,image_in_temp.rows*scale_IMG)); //normlized image
			scale_LM(img_LM_after,&img_LM_after_2, scale_IMG); //縮放移動後的LM

			Leye=Point2f((img_LM_after_2[20-1].x+img_LM_after_2[23-1].x)/2,(img_LM_after_2[20-1].y+img_LM_after_2[23-1].y)/2);
			Reye=Point2f((img_LM_after_2[26-1].x+img_LM_after_2[29-1].x)/2,(img_LM_after_2[26-1].y+img_LM_after_2[29-1].y)/2);
			Point2f src_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);
			Point2f dst_pts=Point2f(320,160);

			float min_x=image_in_temp_2.cols;
			float min_y=image_in_temp_2.rows;
			float max_x=0;
			float max_y=0;
			for (int i = 0; i < img_LM_after_2.size(); i++)
			{
				min_x=(min_x<img_LM_after_2[i].x)?min_x:img_LM_after_2[i].x;
				min_y=(min_y<img_LM_after_2[i].y)?min_y:img_LM_after_2[i].y;
				max_x=(max_x>img_LM_after_2[i].x)?max_x:img_LM_after_2[i].x;
				max_y=(max_y>img_LM_after_2[i].y)?max_y:img_LM_after_2[i].y;
			}
			min_x=min_x-bound;
			min_y=min_y-bound;
			max_x=max_x+bound;
			max_y=max_y+bound;

			move_LM_point(img_LM_after_2, &img_LM_after_2, dst_pts, src_pts); // 移動測試影像的LM點
			image_in_temp_2(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(image_crop(Rect(dst_pts.x-(src_pts.x-min_x),dst_pts.y-(src_pts.y-min_y),cvRound(max_x-min_x),cvRound(max_y-min_y))));
		}
		if (angle == -45  )
		{
			Point2f Leye=img_LM_after[29-1];
			Point2f Reye=img_LM_after[19-1];
			float eye_length=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
			float norm_length=68.0;
			float scale_IMG=norm_length/eye_length;
			cv::resize(image_in_temp,image_in_temp_2,Size(image_in_temp.cols*scale_IMG,image_in_temp.rows*scale_IMG)); //normlized image
			scale_LM(img_LM_after,&img_LM_after_2, scale_IMG); //縮放移動後的LM
			//show(image_in_temp_2, img_LM_after_2);

			Point2f src_pts=img_LM_after_2[26-1];
			Point2f dst_pts=Point2f(320,160);

			float min_x=image_in_temp_2.cols;
			float min_y=image_in_temp_2.rows;
			float max_x=0;
			float max_y=0;
			for (int i = 0; i < img_LM_after_2.size(); i++)
			{
				min_x=(min_x<img_LM_after_2[i].x)?min_x:img_LM_after_2[i].x;
				min_y=(min_y<img_LM_after_2[i].y)?min_y:img_LM_after_2[i].y;
				max_x=(max_x>img_LM_after_2[i].x)?max_x:img_LM_after_2[i].x;
				max_y=(max_y>img_LM_after_2[i].y)?max_y:img_LM_after_2[i].y;
			}
			min_x=min_x-bound;
			min_y=min_y-bound;
			max_x=max_x+bound;
			max_y=max_y+bound;

			move_LM_point(img_LM_after_2, &img_LM_after_2, dst_pts, src_pts); // 移動測試影像的LM點
			image_in_temp_2(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(image_crop(Rect(dst_pts.x-(src_pts.x-min_x),dst_pts.y-(src_pts.y-min_y),cvRound(max_x-min_x),cvRound(max_y-min_y))));
		}
	}
	else if (angle > 45.0)
	{
		float angle_to_zero=0.0; // angle between input and 0
		Point2f center_point; // find the center point of landmark
		find_LM_center(LM_in, &center_point); 
		Mat Rotate_Matrix=getRotationMatrix2D(center_point, angle_to_zero, 1.0); // calculate the rotate matrix [2x3]
		rotate_LM(LM_in, &img_LM_after, angle_to_zero); 
		cv::warpAffine(image_in, image_in_temp, Rotate_Matrix, image_in.size()); // rotate the ori. image 
		if (angle == 60 )
		{
			Point2f Leye=img_LM_after[6-1];
			Point2f Reye=img_LM_after[11-1];
			float eye_length=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
			float norm_length=85;
			float scale_IMG=norm_length/eye_length;
			cv::resize(image_in_temp,image_in_temp_2,Size(image_in_temp.cols*scale_IMG,image_in_temp.rows*scale_IMG)); //normlized image
			scale_LM(img_LM_after,&img_LM_after_2, scale_IMG); //縮放移動後的LM
			Point2f src_pts=Point2f((img_LM_after_2[5-1].x+img_LM_after_2[7-1].x)/2,(img_LM_after_2[5-1].y+img_LM_after_2[7-1].y)/2);
			Point2f dst_pts=Point2f(320,160);

			float min_x=image_in_temp_2.cols;
			float min_y=image_in_temp_2.rows;
			float max_x=0;
			float max_y=0;
			for (int i = 0; i < img_LM_after_2.size(); i++)
			{
				min_x=(min_x<img_LM_after_2[i].x)?min_x:img_LM_after_2[i].x;
				min_y=(min_y<img_LM_after_2[i].y)?min_y:img_LM_after_2[i].y;
				max_x=(max_x>img_LM_after_2[i].x)?max_x:img_LM_after_2[i].x;
				max_y=(max_y>img_LM_after_2[i].y)?max_y:img_LM_after_2[i].y;
			}
			min_x=min_x-bound;
			min_y=min_y-bound;
			max_x=max_x+bound;
			max_y=max_y+bound;

			move_LM_point(img_LM_after_2, &img_LM_after_2, dst_pts, src_pts); // 移動測試影像的LM點
			image_in_temp_2(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(image_crop(Rect(dst_pts.x-(src_pts.x-min_x),dst_pts.y-(src_pts.y-min_y),cvRound(max_x-min_x),cvRound(max_y-min_y))));
		}
		if (angle == 75  )
		{
			Point2f Leye=img_LM_after[6-1];
			Point2f Reye=img_LM_after[11-1];
			float eye_length=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
			float norm_length=86;
			float scale_IMG=norm_length/eye_length;
			cv::resize(image_in_temp,image_in_temp_2,Size(image_in_temp.cols*scale_IMG,image_in_temp.rows*scale_IMG)); //normlized image
			scale_LM(img_LM_after,&img_LM_after_2, scale_IMG); //縮放移動後的LM
			Point2f src_pts=Point2f((img_LM_after_2[5-1].x+img_LM_after_2[7-1].x)/2,(img_LM_after_2[5-1].y+img_LM_after_2[7-1].y)/2);
			Point2f dst_pts=Point2f(320,160);

			float min_x=image_in_temp_2.cols;
			float min_y=image_in_temp_2.rows;
			float max_x=0;
			float max_y=0;
			for (int i = 0; i < img_LM_after_2.size(); i++)
			{
				min_x=(min_x<img_LM_after_2[i].x)?min_x:img_LM_after_2[i].x;
				min_y=(min_y<img_LM_after_2[i].y)?min_y:img_LM_after_2[i].y;
				max_x=(max_x>img_LM_after_2[i].x)?max_x:img_LM_after_2[i].x;
				max_y=(max_y>img_LM_after_2[i].y)?max_y:img_LM_after_2[i].y;
			}
			min_x=min_x-bound;
			min_y=min_y-bound;
			max_x=max_x+bound;
			max_y=max_y+bound;

			move_LM_point(img_LM_after_2, &img_LM_after_2, dst_pts, src_pts); // 移動測試影像的LM點
			image_in_temp_2(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(image_crop(Rect(dst_pts.x-(src_pts.x-min_x),dst_pts.y-(src_pts.y-min_y),cvRound(max_x-min_x),cvRound(max_y-min_y))));
		}
		if (angle == 90  )
		{
			Point2f Leye=img_LM_after[6-1];
			Point2f Reye=img_LM_after[11-1];
			float eye_length=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
			float norm_length=83;
			float scale_IMG=norm_length/eye_length;
			cv::resize(image_in_temp,image_in_temp_2,Size(image_in_temp.cols*scale_IMG,image_in_temp.rows*scale_IMG)); //normlized image
			scale_LM(img_LM_after,&img_LM_after_2, scale_IMG); //縮放移動後的LM
			Point2f src_pts=Point2f((img_LM_after_2[5-1].x+img_LM_after_2[7-1].x)/2,(img_LM_after_2[5-1].y+img_LM_after_2[7-1].y)/2);
			Point2f dst_pts=Point2f(320,160);

			float min_x=image_in_temp_2.cols;
			float min_y=image_in_temp_2.rows;
			float max_x=0;
			float max_y=0;
			for (int i = 0; i < img_LM_after_2.size(); i++)
			{
				min_x=(min_x<img_LM_after_2[i].x)?min_x:img_LM_after_2[i].x;
				min_y=(min_y<img_LM_after_2[i].y)?min_y:img_LM_after_2[i].y;
				max_x=(max_x>img_LM_after_2[i].x)?max_x:img_LM_after_2[i].x;
				max_y=(max_y>img_LM_after_2[i].y)?max_y:img_LM_after_2[i].y;
			}
			min_x=min_x-bound;
			min_y=min_y-bound;
			max_x=max_x+bound;
			max_y=max_y+bound;

			move_LM_point(img_LM_after_2, &img_LM_after_2, dst_pts, src_pts); // 移動測試影像的LM點
			image_in_temp_2(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(image_crop(Rect(dst_pts.x-(src_pts.x-min_x),dst_pts.y-(src_pts.y-min_y),cvRound(max_x-min_x),cvRound(max_y-min_y))));
		}
	}
	else if (angle < -45.0)
	{
		float angle_to_zero=0.0; // angle between input and 0
		//find_theate_zero_big_angle(LM_in, &angle_to_zero);
		Point2f center_point; // find the center point of landmark
		find_LM_center(LM_in, &center_point); 
		Mat Rotate_Matrix=getRotationMatrix2D(center_point, angle_to_zero, 1.0); // calculate the rotate matrix [2x3]
		rotate_LM(LM_in, &img_LM_after, angle_to_zero); 
		cv::warpAffine(image_in, image_in_temp, Rotate_Matrix, image_in.size()); // rotate the ori. image 
		if (angle == -60 )
		{
			Point2f Leye=img_LM_after[6-1];
			Point2f Reye=img_LM_after[11-1];
			float eye_length=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
			float norm_length=76;
			float scale_IMG=norm_length/eye_length;
			cv::resize(image_in_temp,image_in_temp_2,Size(image_in_temp.cols*scale_IMG,image_in_temp.rows*scale_IMG)); //normlized image
			scale_LM(img_LM_after,&img_LM_after_2, scale_IMG); //縮放移動後的LM
			Point2f src_pts=Point2f((img_LM_after_2[5-1].x+img_LM_after_2[7-1].x)/2,(img_LM_after_2[5-1].y+img_LM_after_2[7-1].y)/2);
			Point2f dst_pts=Point2f(320,160);

			float min_x=image_in_temp_2.cols;
			float min_y=image_in_temp_2.rows;
			float max_x=0;
			float max_y=0;
			for (int i = 0; i < img_LM_after_2.size(); i++)
			{
				min_x=(min_x<img_LM_after_2[i].x)?min_x:img_LM_after_2[i].x;
				min_y=(min_y<img_LM_after_2[i].y)?min_y:img_LM_after_2[i].y;
				max_x=(max_x>img_LM_after_2[i].x)?max_x:img_LM_after_2[i].x;
				max_y=(max_y>img_LM_after_2[i].y)?max_y:img_LM_after_2[i].y;
			}
			min_x=min_x-bound;
			min_y=min_y-bound;
			max_x=max_x+bound;
			max_y=max_y+bound;

			move_LM_point(img_LM_after_2, &img_LM_after_2, dst_pts, src_pts); // 移動測試影像的LM點
			image_in_temp_2(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(image_crop(Rect(dst_pts.x-(src_pts.x-min_x),dst_pts.y-(src_pts.y-min_y),cvRound(max_x-min_x),cvRound(max_y-min_y))));
		}
		if (angle == -75  )
		{
			Point2f Leye=img_LM_after[6-1];
			Point2f Reye=img_LM_after[11-1];
			float eye_length=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
			float norm_length=83;
			float scale_IMG=norm_length/eye_length;
			cv::resize(image_in_temp,image_in_temp_2,Size(image_in_temp.cols*scale_IMG,image_in_temp.rows*scale_IMG)); //normlized image
			scale_LM(img_LM_after,&img_LM_after_2, scale_IMG); //縮放移動後的LM
			Point2f src_pts=Point2f((img_LM_after_2[5-1].x+img_LM_after_2[7-1].x)/2,(img_LM_after_2[5-1].y+img_LM_after_2[7-1].y)/2);
			Point2f dst_pts=Point2f(320,160);

			float min_x=image_in_temp_2.cols;
			float min_y=image_in_temp_2.rows;
			float max_x=0;
			float max_y=0;
			for (int i = 0; i < img_LM_after_2.size(); i++)
			{
				min_x=(min_x<img_LM_after_2[i].x)?min_x:img_LM_after_2[i].x;
				min_y=(min_y<img_LM_after_2[i].y)?min_y:img_LM_after_2[i].y;
				max_x=(max_x>img_LM_after_2[i].x)?max_x:img_LM_after_2[i].x;
				max_y=(max_y>img_LM_after_2[i].y)?max_y:img_LM_after_2[i].y;
			}
			min_x=min_x-bound;
			min_y=min_y-bound;
			max_x=max_x+bound;
			max_y=max_y+bound;

			move_LM_point(img_LM_after_2, &img_LM_after_2, dst_pts, src_pts); // 移動測試影像的LM點
			image_in_temp_2(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(image_crop(Rect(dst_pts.x-(src_pts.x-min_x),dst_pts.y-(src_pts.y-min_y),cvRound(max_x-min_x),cvRound(max_y-min_y))));
		}
		if (angle == -90  )
		{
			Point2f Leye=img_LM_after[6-1];
			Point2f Reye=img_LM_after[11-1];
			float eye_length=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
			float norm_length=81;
			float scale_IMG=norm_length/eye_length;
			cv::resize(image_in_temp,image_in_temp_2,Size(image_in_temp.cols*scale_IMG,image_in_temp.rows*scale_IMG)); //normlized image
			scale_LM(img_LM_after,&img_LM_after_2, scale_IMG); //縮放移動後的LM
			Point2f src_pts=Point2f((img_LM_after_2[5-1].x+img_LM_after_2[7-1].x)/2,(img_LM_after_2[5-1].y+img_LM_after_2[7-1].y)/2);
			Point2f dst_pts=Point2f(320,160);

			float min_x=image_in_temp_2.cols;
			float min_y=image_in_temp_2.rows;
			float max_x=0;
			float max_y=0;
			for (int i = 0; i < img_LM_after_2.size(); i++)
			{
				min_x=(min_x<img_LM_after_2[i].x)?min_x:img_LM_after_2[i].x;
				min_y=(min_y<img_LM_after_2[i].y)?min_y:img_LM_after_2[i].y;
				max_x=(max_x>img_LM_after_2[i].x)?max_x:img_LM_after_2[i].x;
				max_y=(max_y>img_LM_after_2[i].y)?max_y:img_LM_after_2[i].y;
			}
			min_x=min_x-bound;
			min_y=min_y-bound;
			max_x=max_x+bound;
			max_y=max_y+bound;

			//Mat mask(image_in_temp_2.rows,image_in_temp_2.cols,CV_8UC1);
			//mask.setTo(0);
			////imshow("mask",mask);waitKey(1);
			////imshow("image_in_temp_2",image_in_temp_2);waitKey(0);
			//std::vector<std::vector<cv::Point> > contours;
			//std::vector<cv::Vec4i> hierarchy;
			//cv::line(mask, Point(img_LM_after_2[8-1].x,10), img_LM_after_2[8-1], cv::Scalar(255), 1, CV_AA, 0);
			//cv::line(mask, img_LM_after_2[8-1], img_LM_after_2[4-1], cv::Scalar(255), 1, CV_AA, 0);
			//cv::line(mask, img_LM_after_2[4-1], img_LM_after_2[3-1], cv::Scalar(255), 1, CV_AA, 0);
			//cv::line(mask, img_LM_after_2[3-1], img_LM_after_2[2-1], cv::Scalar(255), 1, CV_AA, 0);
			//cv::line(mask, img_LM_after_2[2-1], img_LM_after_2[1-1], cv::Scalar(255), 1, CV_AA, 0);
			//cv::line(mask, img_LM_after_2[1-1], img_LM_after_2[10-1], cv::Scalar(255), 1, CV_AA, 0);
			//cv::line(mask, img_LM_after_2[10-1], img_LM_after_2[12-1], cv::Scalar(255), 1, CV_AA, 0);
			//cv::line(mask, img_LM_after_2[12-1], img_LM_after_2[15-1], cv::Scalar(255), 1, CV_AA, 0);
			//cv::line(mask, img_LM_after_2[15-1], img_LM_after_2[16-1], cv::Scalar(255), 1, CV_AA, 0);
			//cv::line(mask, img_LM_after_2[16-1], Point(img_LM_after_2[16-1].x,image_in_temp_2.rows-10), cv::Scalar(255), 1, CV_AA, 0);
			//cv::line(mask, Point(img_LM_after_2[16-1].x,image_in_temp_2.rows-10), Point(image_in_temp_2.cols-10,image_in_temp_2.rows-10), cv::Scalar(255), 1, CV_AA, 0);
			//cv::line(mask, Point(image_in_temp_2.cols-10,image_in_temp_2.rows-10), Point(image_in_temp_2.cols-10,10), cv::Scalar(255), 1, CV_AA, 0);
			//cv::line(mask, Point(image_in_temp_2.cols-10,10), Point(img_LM_after_2[8-1].x,10), cv::Scalar(255), 1, CV_AA, 0);
			////imshow("mask",mask);waitKey(0);
			//cv::findContours(mask, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); //找輪廓
			//cv::drawContours(mask, contours, -1, cv::Scalar(255), -1, CV_AA, hierarchy, 0); //畫輪廓
			////imshow("mask",mask);waitKey(0);
			//for (int i = 0; i < mask.rows; i++)
			//{
			//	for (int j = 0; j < mask.cols; j++)
			//	{
			//		if (mask.at<uchar>(i,j) == 0)
			//		{
			//			image_in_temp_2.at<Vec3b>(i,j)[2]=164;
			//			image_in_temp_2.at<Vec3b>(i,j)[1]=156;
			//			image_in_temp_2.at<Vec3b>(i,j)[0]=160;
			//		} 
			//	}
			//}
			//imshow("image_in_temp_2",image_in_temp_2);waitKey(0);

			move_LM_point(img_LM_after_2, &img_LM_after_2, dst_pts, src_pts); // 移動測試影像的LM點
			image_in_temp_2(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(image_crop(Rect(dst_pts.x-(src_pts.x-min_x),dst_pts.y-(src_pts.y-min_y),cvRound(max_x-min_x),cvRound(max_y-min_y))));
		}
	}
	img_test=image_crop;
	crop_image_test(angle,image_crop, img_LM_after_2, &test_crop);
	//show(image_crop, img_LM_after_2);


	*LM_out=img_LM_after_2;
	*image_out=test_crop;
}

// align image //
void align_test_img(float p_angle, Mat img_in, vector<Point2f> LM_in, Mat move_m, Mat model_LM_in, Mat* img_out, vector<Point2f>* LM_out, Point3f angla_m)
{
	// draw test image LM //
	Mat show_LM=img_in.clone();
	vector<Point2f> LM_out_temp;

	Mat ViewMatrix;
	Mat ProjMatrix;
	matrix_set(&ViewMatrix,&ProjMatrix);
	project_matrix_t=ProjMatrix.t();
	Mat rotate_matrix_x,rotate_matrix_y,rotate_matrix_z;
	float Pitch_f=0.0;
	float Yaw_f=p_angle;
	float Roll_f=0.0;
	double error_num;

	Pitch_d=angla_m.x;
	Yaw_d=angla_m.y;
	Roll_d=angla_m.z;
	R_matrix(Pitch_d, Yaw_d, Roll_d, &rotate_matrix_x, &rotate_matrix_y, &rotate_matrix_z);
	Mat point_lm_gallery;
	point_lm_gallery=ViewMatrix*ProjMatrix*move_m.inv()*rotate_matrix_z*rotate_matrix_y*rotate_matrix_x*move_m*model_LM_in;
	move_to_ori_T=move_m.t();// use in Render//
	Mat move_matrix_inv=move_m.inv();
	move_to_ori_T_inv=move_matrix_inv.t();// use in Render//
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
	glutMainLoopEvent();
	glutPostRedisplay(); 

	//imshow("final_pic_flip",final_pic_flip);waitKey(0);

	for (int i = 0; i < point_lm_gallery.cols; i++)
	{
		LM_out_temp.push_back(Point2f(point_lm_gallery.at<float>(0,i),point_lm_gallery.at<float>(1,i)));
	}
	for (int i = 0; i < LM_out_temp.size(); i++)
	{
		if (i==14-1 || i==15-1 || i==16-1 || i==17-1 || i==22-1 || i==23-1 || i==24-1 || i==25-1)
		{
			for (int j = 0; j < 100; j++)
			{
				float comp_1=final_pic_flip.at<Vec3b>(LM_out_temp[i].y,LM_out_temp[i].x-j)[2];
				float comp_2=final_pic_flip.at<Vec3b>(LM_out_temp[i].y,LM_out_temp[i].x-j)[1];
				float comp_3=final_pic_flip.at<Vec3b>(LM_out_temp[i].y,LM_out_temp[i].x-j)[0];
				//cout<<comp_1<<" "<<comp_2<<" "<<comp_3<<" "<<endl;
				if (comp_1 == 0 && comp_2 == 0 && comp_3 == 0)
				{
					//circle(imd,LM_out_temp[i],3,CV_RGB(255,255,0),-1);
					LM_out_temp[i].x=LM_out_temp[i].x-j;
					break;
				}
			}
		}
		else if (i==18-1 || i==19-1 || i==20-1 || i==21-1 || i==26-1 || i==27-1 || i==28-1 || i==29-1)
		{
			for (int j = 0; j < 100; j++)
			{
				float comp_1=final_pic_flip.at<Vec3b>(LM_out_temp[i].y,LM_out_temp[i].x+j)[2];
				float comp_2=final_pic_flip.at<Vec3b>(LM_out_temp[i].y,LM_out_temp[i].x+j)[1];
				float comp_3=final_pic_flip.at<Vec3b>(LM_out_temp[i].y,LM_out_temp[i].x+j)[0];
				//cout<<comp_1<<" "<<comp_2<<" "<<endl;
				if (comp_1 == 0 && comp_2 == 0 && comp_3 == 0)
				{
					//circle(imd,LM_out_temp[i],3,CV_RGB(255,255,0),-1);
					LM_out_temp[i].x=LM_out_temp[i].x+j;
					break;
				}
			}
		}
	}
	error_cal_c(p_angle, show_LM, LM_in, LM_out_temp, &LM_out_temp , &error_num);
	//error_cal(p_angle, show_LM, LM_in, LM_out_temp, &LM_out_temp , &error_num);
	cout<<error_num<<endl;
	//imshow("final_pic_flip",final_pic_flip);waitKey(0);

	Mat image_Reg_out=final_pic_flip.clone();
	Mat align_image=image_Reg_out.clone();align_image.setTo(0);
	for (int i = 0; i < align_image.rows; i++)
	{
		for (int j = 0; j < align_image.cols; j++)
		{
			if (image_Reg_out.at<Vec3b>(i,j)[0]<=10 && image_Reg_out.at<Vec3b>(i,j)[1]<=10 && image_Reg_out.at<Vec3b>(i,j)[2]<=10)
			{
				//align_image.at<Vec3b>(i,j)[2]=image_MPIE.at<Vec3b>(i,j)[2];
				//align_image.at<Vec3b>(i,j)[1]=image_MPIE.at<Vec3b>(i,j)[1];
				//align_image.at<Vec3b>(i,j)[0]=image_MPIE.at<Vec3b>(i,j)[0];

				align_image.at<Vec3b>(i,j)[2]=164;
				align_image.at<Vec3b>(i,j)[1]=156;
				align_image.at<Vec3b>(i,j)[0]=160;
			} 
			else
			{
				align_image.at<Vec3b>(i,j)[2]=image_Reg_out.at<Vec3b>(i,j)[2];
				align_image.at<Vec3b>(i,j)[1]=image_Reg_out.at<Vec3b>(i,j)[1];
				align_image.at<Vec3b>(i,j)[0]=image_Reg_out.at<Vec3b>(i,j)[0];
			}
		}
	}

	Mat reg_crop;
	crop_image_reg(p_angle,align_image, show_LM, LM_in, &reg_crop);

	*img_out=reg_crop.clone();
	*LM_out=LM_out_temp;
}
// align image //
void align_test_img_auto(float p_angle, Mat img_in, vector<Point2f> LM_in, Mat move_m, Mat model_LM_in, Mat* img_out, vector<Point2f>* LM_out, Point3f angla_m)
{
	Mat reg_crop;

	float range_l;
	float range_r;
	float range_d;
	if (p_angle<=15 && p_angle>=-15)
	{
		range_l=-5.0;
		range_r=5.0;
		range_d=5.0;
	}
	if (p_angle==30 || p_angle==-30)
	{
		range_l=-15.0;
		range_r=15.0;
		range_d=15.0;
	}
	if (p_angle==45 || p_angle==-45)
	{
		range_l=-5.0;
		range_r=5.0;
		range_d=5.0;
	}
	else if (p_angle < -45)
	{
		range_l=-15.0;
		range_r=0.0;
		range_d=5.0;
	}
	else if (p_angle > 45)
	{
		range_l=0.0;
		range_r=15.0;
		range_d=5.0;
	}

	// draw test image LM //
	Mat show_LM=img_in.clone();
	vector<Point2f> LM_out_temp;

	Mat ViewMatrix;
	Mat ProjMatrix;
	matrix_set(&ViewMatrix,&ProjMatrix);
	project_matrix_t=ProjMatrix.t();
	Mat rotate_matrix_x,rotate_matrix_y,rotate_matrix_z;

	float Pitch_f=0.0;
	float Yaw_f=0.0;
	float Roll_f=0.0;
	double error_f=10000.0;

	for (float Pitch = -5.0; Pitch <= 5.0; Pitch=Pitch+2.0)
	{
		for (float Roll = -5.0; Roll <= 5.0; Roll=Roll+2.0)
		{
			for (float Yaw=p_angle+range_l ; Yaw<=p_angle+range_r ; Yaw=Yaw+range_d)
			{
				cout<<"Pitch : "<<Pitch<<" Yaw : "<<Yaw<<" Roll : "<<Roll<<endl;
				Pitch_d=Pitch;
				Yaw_d=Yaw;
				Roll_d=Roll;
				R_matrix(Pitch, Yaw, Roll, &rotate_matrix_x, &rotate_matrix_y, &rotate_matrix_z);
				Mat point_lm_gallery;
				point_lm_gallery=ViewMatrix*ProjMatrix*move_m.inv()*rotate_matrix_z*rotate_matrix_y*rotate_matrix_x*move_m*model_LM_in;
				move_to_ori_T=move_m.t();// use in Render//
				Mat move_matrix_inv=move_m.inv();
				move_to_ori_T_inv=move_matrix_inv.t();// use in Render//
				glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
				glutMainLoopEvent();
				glutPostRedisplay(); 

				//imshow("shoe",final_pic_flip);waitKey(0);

				vector<Point2f> LM_out_temp;
				for (int i = 0; i < point_lm_gallery.cols; i++)
				{
					LM_out_temp.push_back(Point2f(point_lm_gallery.at<float>(0,i),point_lm_gallery.at<float>(1,i)));
				}
				Mat imd=final_pic_flip.clone();
				for (int i = 0; i < LM_out_temp.size(); i++)
				{
					circle(imd,LM_out_temp[i],3,CV_RGB(255,0,0),-1);
					//imshow("imd",imd);waitKey(0);
				}
				for (int i = 12; i < LM_out_temp.size(); i++)
				{
					circle(imd,LM_out_temp[i],3,CV_RGB(0,0,255),-1);
					//imshow("imd",imd);waitKey(0);
				}
				//imshow("imd",imd);waitKey(0);
				imd.setTo(0);
				imd=final_pic_flip.clone();
				for (int i = 0; i < LM_out_temp.size(); i++)
				{
					if (i==14-1 || i==15-1 || i==16-1 || i==17-1 || i==22-1 || i==23-1 || i==24-1 || i==25-1)
					{
						for (int j = 0; j < 100; j++)
						{
							float comp_1=final_pic_flip.at<Vec3b>(LM_out_temp[i].y,LM_out_temp[i].x-j)[2];
							float comp_2=final_pic_flip.at<Vec3b>(LM_out_temp[i].y,LM_out_temp[i].x-j)[1];
							float comp_3=final_pic_flip.at<Vec3b>(LM_out_temp[i].y,LM_out_temp[i].x-j)[0];
							//cout<<comp_1<<" "<<comp_2<<" "<<comp_3<<" "<<endl;
							if (comp_1 == 0 && comp_2 == 0 && comp_3 == 0)
							{
								//circle(imd,LM_out_temp[i],3,CV_RGB(255,255,0),-1);
								LM_out_temp[i].x=LM_out_temp[i].x-j;
								break;
							}
						}
					}
					else if (i==18-1 || i==19-1 || i==20-1 || i==21-1 || i==26-1 || i==27-1 || i==28-1 || i==29-1)
					{
						for (int j = 0; j < 100; j++)
						{
							float comp_1=final_pic_flip.at<Vec3b>(LM_out_temp[i].y,LM_out_temp[i].x+j)[2];
							float comp_2=final_pic_flip.at<Vec3b>(LM_out_temp[i].y,LM_out_temp[i].x+j)[1];
							float comp_3=final_pic_flip.at<Vec3b>(LM_out_temp[i].y,LM_out_temp[i].x+j)[0];
							//cout<<comp_1<<" "<<comp_2<<" "<<endl;
							if (comp_1 == 0 && comp_2 == 0 && comp_3 == 0)
							{
								//circle(imd,LM_out_temp[i],3,CV_RGB(255,255,0),-1);
								LM_out_temp[i].x=LM_out_temp[i].x+j;
								break;
							}
						}
					}
				}
				for (int i = 0; i < LM_out_temp.size(); i++)
				{
					circle(imd,LM_out_temp[i],3,CV_RGB(255,0,0),-1);
					//imshow("imd",imd);waitKey(0);
				}
				for (int i = 12; i < LM_out_temp.size(); i++)
				{
					circle(imd,LM_out_temp[i],3,CV_RGB(0,0,255),-1);
					//imshow("imd",imd);waitKey(0);
				}
				//imshow("imd",imd);waitKey(0);


				double err_cal;
				error_cal_c(p_angle, show_LM, LM_in, LM_out_temp, &LM_out_temp , &err_cal);
				cout<<"error : "<<err_cal<<endl;

				Pitch_f=(err_cal<error_f)?Pitch:Pitch_f;
				Yaw_f=(err_cal<error_f)?Yaw:Yaw_f;
				Roll_f=(err_cal<error_f)?Roll:Roll_f;
				if (err_cal<error_f)
				{
					Mat image_Reg_out=final_pic_flip.clone();
					Mat align_image=image_Reg_out.clone();align_image.setTo(0);
					for (int i = 0; i < align_image.rows; i++)
					{
						for (int j = 0; j < align_image.cols; j++)
						{
							if (image_Reg_out.at<Vec3b>(i,j)[0]<=10 && image_Reg_out.at<Vec3b>(i,j)[1]<=10 && image_Reg_out.at<Vec3b>(i,j)[2]<=10)
							{
								//align_image.at<Vec3b>(i,j)[2]=image_MPIE.at<Vec3b>(i,j)[2];
								//align_image.at<Vec3b>(i,j)[1]=image_MPIE.at<Vec3b>(i,j)[1];
								//align_image.at<Vec3b>(i,j)[0]=image_MPIE.at<Vec3b>(i,j)[0];

								align_image.at<Vec3b>(i,j)[2]=164;
								align_image.at<Vec3b>(i,j)[1]=156;
								align_image.at<Vec3b>(i,j)[0]=160;
							} 
							else
							{
								align_image.at<Vec3b>(i,j)[2]=image_Reg_out.at<Vec3b>(i,j)[2];
								align_image.at<Vec3b>(i,j)[1]=image_Reg_out.at<Vec3b>(i,j)[1];
								align_image.at<Vec3b>(i,j)[0]=image_Reg_out.at<Vec3b>(i,j)[0];
							}
						}
					}
					crop_image_reg(p_angle,align_image, show_LM, LM_in, &reg_crop);
				}
				error_f=(err_cal<error_f)?err_cal:error_f;
			}
		}
	}

	cout<<" 粗估 1 "<<endl;
	cout<<"Pitch : "<<Pitch_f<<" Yaw : "<<Yaw_f<<" Roll : "<<Roll_f<<endl;
	cout<<"error : "<<error_f<<endl;

	float Pitch_f2=0.0;
	float Yaw_f2=0.0;
	float Roll_f2=0.0;
	double error_f2=10000.0;

	for (float Pitch = Pitch_f-1.0; Pitch <= Pitch_f+1.0; Pitch=Pitch+1.0)
	{
		for (float Roll = Roll_f-1.0; Roll <= Roll_f+1.0; Roll=Roll+1.0)
		{
			for (float Yaw=Yaw_f-2.0 ; Yaw<=Yaw_f+2.0 ; Yaw=Yaw+1.0)
			{
				cout<<"Pitch : "<<Pitch<<" Yaw : "<<Yaw<<" Roll : "<<Roll<<endl;
				Pitch_d=Pitch;
				Yaw_d=Yaw;
				Roll_d=Roll;
				R_matrix(Pitch, Yaw, Roll, &rotate_matrix_x, &rotate_matrix_y, &rotate_matrix_z);
				Mat point_lm_gallery;
				point_lm_gallery=ViewMatrix*ProjMatrix*move_m.inv()*rotate_matrix_z*rotate_matrix_y*rotate_matrix_x*move_m*model_LM_in;
				move_to_ori_T=move_m.t();// use in Render//
				Mat move_matrix_inv=move_m.inv();
				move_to_ori_T_inv=move_matrix_inv.t();// use in Render//
				glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
				glutMainLoopEvent();
				glutPostRedisplay(); 

				vector<Point2f> LM_out_temp;
				for (int i = 0; i < point_lm_gallery.cols; i++)
				{
					LM_out_temp.push_back(Point2f(point_lm_gallery.at<float>(0,i),point_lm_gallery.at<float>(1,i)));
				}
				Mat imd=final_pic_flip.clone();
				for (int i = 0; i < LM_out_temp.size(); i++)
				{
					circle(imd,LM_out_temp[i],3,CV_RGB(255,0,0),-1);
					//imshow("imd",imd);waitKey(0);
				}
				for (int i = 12; i < LM_out_temp.size(); i++)
				{
					circle(imd,LM_out_temp[i],3,CV_RGB(0,0,255),-1);
					//imshow("imd",imd);waitKey(0);
				}
				//imshow("imd",imd);waitKey(0);
				for (int i = 0; i < LM_out_temp.size(); i++)
				{
					if (i==14-1 || i==15-1 || i==16-1 || i==17-1 || i==22-1 || i==23-1 || i==24-1 || i==25-1)
					{
						for (int j = 0; j < 100; j++)
						{
							float comp_1=final_pic_flip.at<Vec3b>(LM_out_temp[i].y,LM_out_temp[i].x-j)[2];
							float comp_2=final_pic_flip.at<Vec3b>(LM_out_temp[i].y,LM_out_temp[i].x-j)[1];
							float comp_3=final_pic_flip.at<Vec3b>(LM_out_temp[i].y,LM_out_temp[i].x-j)[0];
							//cout<<comp_1<<" "<<comp_2<<" "<<comp_3<<" "<<endl;
							if (comp_1 == 0 && comp_2 == 0 && comp_3 == 0)
							{
								LM_out_temp[i].x=LM_out_temp[i].x-j;
								break;
							}
						}
					}
					else if (i==18-1 || i==19-1 || i==20-1 || i==21-1 || i==26-1 || i==27-1 || i==28-1 || i==29-1)
					{
						for (int j = 0; j < 100; j++)
						{
							float comp_1=final_pic_flip.at<Vec3b>(LM_out_temp[i].y,LM_out_temp[i].x+j)[2];
							float comp_2=final_pic_flip.at<Vec3b>(LM_out_temp[i].y,LM_out_temp[i].x+j)[1];
							float comp_3=final_pic_flip.at<Vec3b>(LM_out_temp[i].y,LM_out_temp[i].x+j)[0];
							//cout<<comp_1<<" "<<comp_2<<" "<<endl;
							if (comp_1 == 0 && comp_2 == 0 && comp_3 == 0)
							{
								LM_out_temp[i].x=LM_out_temp[i].x+j;
								break;
							}
						}
					}
				}
				imd.setTo(0);
				imd=final_pic_flip.clone();
				for (int i = 0; i < LM_out_temp.size(); i++)
				{
					circle(imd,LM_out_temp[i],3,CV_RGB(255,0,0),-1);
					//imshow("imd",imd);waitKey(0);
				}
				for (int i = 12; i < LM_out_temp.size(); i++)
				{
					circle(imd,LM_out_temp[i],3,CV_RGB(0,0,255),-1);
					//imshow("imd",imd);waitKey(0);
				}
				//imshow("imd",imd);waitKey(0);


				double err_cal;
				error_cal_c(p_angle, show_LM, LM_in, LM_out_temp, &LM_out_temp , &err_cal);
				cout<<"error : "<<err_cal<<endl;

				Pitch_f2=(err_cal<error_f2)?Pitch:Pitch_f2;
				Yaw_f2=(err_cal<error_f2)?Yaw:Yaw_f2;
				Roll_f2=(err_cal<error_f2)?Roll:Roll_f2;

				if (err_cal<error_f2)
				{
					Mat image_Reg_out=final_pic_flip.clone();
					Mat align_image=image_Reg_out.clone();align_image.setTo(0);
					for (int i = 0; i < align_image.rows; i++)
					{
						for (int j = 0; j < align_image.cols; j++)
						{
							if (image_Reg_out.at<Vec3b>(i,j)[0]<=10 && image_Reg_out.at<Vec3b>(i,j)[1]<=10 && image_Reg_out.at<Vec3b>(i,j)[2]<=10)
							{
								//align_image.at<Vec3b>(i,j)[2]=image_MPIE.at<Vec3b>(i,j)[2];
								//align_image.at<Vec3b>(i,j)[1]=image_MPIE.at<Vec3b>(i,j)[1];
								//align_image.at<Vec3b>(i,j)[0]=image_MPIE.at<Vec3b>(i,j)[0];

								align_image.at<Vec3b>(i,j)[2]=164;
								align_image.at<Vec3b>(i,j)[1]=156;
								align_image.at<Vec3b>(i,j)[0]=160;
							} 
							else
							{
								align_image.at<Vec3b>(i,j)[2]=image_Reg_out.at<Vec3b>(i,j)[2];
								align_image.at<Vec3b>(i,j)[1]=image_Reg_out.at<Vec3b>(i,j)[1];
								align_image.at<Vec3b>(i,j)[0]=image_Reg_out.at<Vec3b>(i,j)[0];
							}
						}
					}
					crop_image_reg(p_angle,align_image, show_LM, LM_in, &reg_crop);
				}
				error_f2=(err_cal<error_f2)?err_cal:error_f2;
			}
		}
	}
	cout<<" 細估 "<<endl;
	cout<<"Pitch : "<<Pitch_f2<<" Yaw : "<<Yaw_f2<<" Roll : "<<Roll_f2<<endl;
	cout<<"error : "<<error_f2<<endl;

	Pitch_d=Pitch_f2;
	Yaw_d=Yaw_f2;
	Roll_d=Roll_f2;
	error_comp=error_f2;

	*img_out=reg_crop.clone();
	*LM_out=LM_out_temp;
}
void error_cal_c(float p_angle, Mat img_in, vector<Point2f> LM_in, vector<Point2f> model_in, vector<Point2f>* model_out, double* error_num)
{
	Point2f Leye;
	Point2f Reye;
	Mat image_show=final_pic_flip.clone();
	double error_temp=0.0;
	vector<Point2f> LM_in_temp_draw;
	vector<Point2f> model_in_temp_draw;


	if (p_angle>=-15.0 && p_angle<=15.0)
	{
		Leye=Point2f((LM_in[20-1].x+LM_in[23-1].x)/2,(LM_in[20-1].y+LM_in[23-1].y)/2);
		Reye=Point2f((LM_in[26-1].x+LM_in[29-1].x)/2,(LM_in[26-1].y+LM_in[29-1].y)/2);
		float src_slope=(Leye.y-Reye.y)/(Leye.x-Reye.x);
		float eye_length_src=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
		Point2f src_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);

		Leye=Point2f((model_in[5-1].x+model_in[6-1].x)/2,(model_in[5-1].y+model_in[6-1].y)/2);
		Reye=Point2f((model_in[7-1].x+model_in[8-1].x)/2,(model_in[7-1].y+model_in[8-1].y)/2);
		float dst_slope=(Leye.y-Reye.y)/(Leye.x-Reye.x);
		float eye_length_dst=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
		Point2f dst_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);

		move_img(image_show, &image_show, src_pts, dst_pts); // move the image to eyes center
		move_LM_point(model_in, &model_in, src_pts, dst_pts); // 移動測試影像的LM點
		//imshow("image_show",image_show);waitKey(0);

		//float theate;
		//find_angle(dst_slope, src_slope, &theate);
		//Point2f center_point; // find the center point of landmark
		//find_LM_center(model_in, &center_point); 
		//Mat Rotate_Matrix=getRotationMatrix2D(center_point, theate, 1.0); // calculate the rotate matrix [2x3]
		//rotate_LM(model_in, &model_in, theate); 
		//cv::warpAffine(image_show, image_show, Rotate_Matrix, image_show.size()); // rotate the ori. image 
		//imshow("image_show",image_show);waitKey(0);

		float scale_IMG=eye_length_src/eye_length_dst;
		//cv::resize(image_show,image_show,Size(image_show.cols*scale_IMG,image_show.rows*scale_IMG)); //normlized image
		//scale_LM(model_in,&model_in, scale_IMG); //縮放移動後的LM
		fix_scale(scale_IMG, image_show, model_in, p_angle, img_in,  &image_show, &model_in);
		//imshow("image_show",image_show);waitKey(0);

		Leye=Point2f((model_in[5-1].x+model_in[6-1].x)/2,(model_in[5-1].y+model_in[6-1].y)/2);
		Reye=Point2f((model_in[7-1].x+model_in[8-1].x)/2,(model_in[7-1].y+model_in[8-1].y)/2);
		dst_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);

		move_img(image_show, &image_show, src_pts, dst_pts); // move the image to eyes center
		move_LM_point(model_in, &model_in, src_pts, dst_pts); // 移動測試影像的LM點
		//imshow("image_show",image_show);waitKey(0);

		int LM_in_index[]={11,12,13,14,20,23,26,29,32,35,38,41,50,51,52,55,56};
		vector<int> LM_in_number_index(LM_in_index, LM_in_index + sizeof(LM_in_index)/sizeof(LM_in_index[0]));
		vector<Point2f> LM_in_temp;
		for (int i = 0; i < LM_in_number_index.size(); i++)
		{
			LM_in_temp.push_back(LM_in[LM_in_number_index[i]-1]);
		}

		int model_in_index[]={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,19};
		vector<int> model_in_number_index(model_in_index, model_in_index + sizeof(model_in_index)/sizeof(model_in_index[0]));
		vector<Point2f> model_in_temp;
		for (int i = 0; i < model_in_number_index.size(); i++)
		{
			model_in_temp.push_back(model_in[model_in_number_index[i]-1]);
		}

		for (int i = 0; i < model_in_temp.size(); i++)
		{
			float x=model_in_temp[i].x-LM_in_temp[i].x;
			float y=model_in_temp[i].y-LM_in_temp[i].y;
			error_temp=error_temp+sqrt(x*x+y*y);
		}
		//cout<<"1 : "<<error_temp<<endl;
		error_temp=error_temp/model_in_temp.size();
		//cout<<"2 : "<<error_temp<<endl;

		//show_m(image_show, LM_in_temp, model_in_temp);

		//Mat imd_t=img_test.clone();
		//show_m_2(imd_t, LM_in_temp, model_in_temp);
		//show_m(image_show, LM_in_temp, model_in_temp);

		final_pic_flip.setTo(0);
		int width_c=(image_show.cols<final_pic_flip.cols)?image_show.cols:final_pic_flip.cols;
		int height_c=(image_show.rows<final_pic_flip.rows)?image_show.rows:final_pic_flip.rows;
		image_show(Rect(0,0,width_c,height_c)).copyTo(final_pic_flip(Rect(0,0,width_c,height_c)));

		LM_in_temp_draw=LM_in_temp;
		model_in_temp_draw=model_in_temp;
	}
	if (p_angle==-30.0)
	{
		Leye=Point2f((LM_in[20-1].x+LM_in[23-1].x)/2,(LM_in[20-1].y+LM_in[23-1].y)/2);
		Reye=Point2f((LM_in[26-1].x+LM_in[29-1].x)/2,(LM_in[26-1].y+LM_in[29-1].y)/2);
		float src_slope=(Leye.y-Reye.y)/(Leye.x-Reye.x);
		float eye_length_src=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
		Point2f src_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);

		Leye=Point2f((model_in[5-1].x+model_in[6-1].x)/2,(model_in[5-1].y+model_in[6-1].y)/2);
		Reye=Point2f((model_in[7-1].x+model_in[8-1].x)/2,(model_in[7-1].y+model_in[8-1].y)/2);
		float dst_slope=(Leye.y-Reye.y)/(Leye.x-Reye.x);
		float eye_length_dst=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
		Point2f dst_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);

		move_img(image_show, &image_show, src_pts, dst_pts); // move the image to eyes center
		move_LM_point(model_in, &model_in, src_pts, dst_pts); // 移動測試影像的LM點
		//imshow("image_show",image_show);waitKey(0);

		//float theate;
		//find_angle(dst_slope, src_slope, &theate);
		//Point2f center_point; // find the center point of landmark
		//find_LM_center(model_in, &center_point); 
		//Mat Rotate_Matrix=getRotationMatrix2D(center_point, theate, 1.0); // calculate the rotate matrix [2x3]
		//rotate_LM(model_in, &model_in, theate); 
		//cv::warpAffine(image_show, image_show, Rotate_Matrix, image_show.size()); // rotate the ori. image 
		//imshow("image_show",image_show);waitKey(0);

		float scale_IMG=eye_length_src/eye_length_dst;
		//cv::resize(image_show,image_show,Size(image_show.cols*scale_IMG,image_show.rows*scale_IMG)); //normlized image
		//scale_LM(model_in,&model_in, scale_IMG); //縮放移動後的LM
		fix_scale(scale_IMG, image_show, model_in, p_angle, img_in,  &image_show, &model_in);
		//imshow("image_show",image_show);waitKey(0);

		Leye=Point2f((model_in[5-1].x+model_in[6-1].x)/2,(model_in[5-1].y+model_in[6-1].y)/2);
		Reye=Point2f((model_in[7-1].x+model_in[8-1].x)/2,(model_in[7-1].y+model_in[8-1].y)/2);
		dst_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);

		move_img(image_show, &image_show, src_pts, dst_pts); // move the image to eyes center
		move_LM_point(model_in, &model_in, src_pts, dst_pts); // 移動測試影像的LM點
		//imshow("image_show",image_show);waitKey(0);

		int LM_in_index[]={11,12,13,14,20,23,26,29,32,35,38,41,50,51,52,53,62,61,60};
		vector<int> LM_in_number_index(LM_in_index, LM_in_index + sizeof(LM_in_index)/sizeof(LM_in_index[0]));
		vector<Point2f> LM_in_temp;
		for (int i = 0; i < LM_in_number_index.size(); i++)
		{
			LM_in_temp.push_back(LM_in[LM_in_number_index[i]-1]);
		}

		int model_in_index[]={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,22,23,24};
		vector<int> model_in_number_index(model_in_index, model_in_index + sizeof(model_in_index)/sizeof(model_in_index[0]));
		vector<Point2f> model_in_temp;
		for (int i = 0; i < model_in_number_index.size(); i++)
		{
			model_in_temp.push_back(model_in[model_in_number_index[i]-1]);
		}

		for (int i = 0; i < model_in_temp.size(); i++)
		{
			float x=model_in_temp[i].x-LM_in_temp[i].x;
			float y=model_in_temp[i].y-LM_in_temp[i].y;
			error_temp=error_temp+sqrt(x*x+y*y);
		}
		//cout<<"1 : "<<error_temp<<endl;
		error_temp=error_temp/model_in_temp.size();
		//cout<<"2 : "<<error_temp<<endl;

		//show_m(image_show, LM_in_temp, model_in_temp);

		//Mat imd_t=img_test.clone();
		//show_m_2(imd_t, LM_in_temp, model_in_temp);
		//show_m(image_show, LM_in_temp, model_in_temp);

		final_pic_flip.setTo(0);
		int width_c=(image_show.cols<final_pic_flip.cols)?image_show.cols:final_pic_flip.cols;
		int height_c=(image_show.rows<final_pic_flip.rows)?image_show.rows:final_pic_flip.rows;
		image_show(Rect(0,0,width_c,height_c)).copyTo(final_pic_flip(Rect(0,0,width_c,height_c)));

		LM_in_temp_draw=LM_in_temp;
		model_in_temp_draw=model_in_temp;
	}
	if (p_angle==30.0)
	{
		Leye=Point2f((LM_in[20-1].x+LM_in[23-1].x)/2,(LM_in[20-1].y+LM_in[23-1].y)/2);
		Reye=Point2f((LM_in[26-1].x+LM_in[29-1].x)/2,(LM_in[26-1].y+LM_in[29-1].y)/2);
		float src_slope=(Leye.y-Reye.y)/(Leye.x-Reye.x);
		float eye_length_src=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
		Point2f src_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);

		Leye=Point2f((model_in[5-1].x+model_in[6-1].x)/2,(model_in[5-1].y+model_in[6-1].y)/2);
		Reye=Point2f((model_in[7-1].x+model_in[8-1].x)/2,(model_in[7-1].y+model_in[8-1].y)/2);
		float dst_slope=(Leye.y-Reye.y)/(Leye.x-Reye.x);
		float eye_length_dst=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
		Point2f dst_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);

		move_img(image_show, &image_show, src_pts, dst_pts); // move the image to eyes center
		move_LM_point(model_in, &model_in, src_pts, dst_pts); // 移動測試影像的LM點
		//imshow("image_show",image_show);waitKey(0);

		//float theate;
		//find_angle(dst_slope, src_slope, &theate);
		//Point2f center_point; // find the center point of landmark
		//find_LM_center(model_in, &center_point); 
		//Mat Rotate_Matrix=getRotationMatrix2D(center_point, theate, 1.0); // calculate the rotate matrix [2x3]
		//rotate_LM(model_in, &model_in, theate); 
		//cv::warpAffine(image_show, image_show, Rotate_Matrix, image_show.size()); // rotate the ori. image 
		//imshow("image_show",image_show);waitKey(0);

		float scale_IMG=eye_length_src/eye_length_dst;
		//cv::resize(image_show,image_show,Size(image_show.cols*scale_IMG,image_show.rows*scale_IMG)); //normlized image
		//scale_LM(model_in,&model_in, scale_IMG); //縮放移動後的LM
		fix_scale(scale_IMG, image_show, model_in, p_angle, img_in,  &image_show, &model_in);
		//imshow("image_show",image_show);waitKey(0);

		Leye=Point2f((model_in[5-1].x+model_in[6-1].x)/2,(model_in[5-1].y+model_in[6-1].y)/2);
		Reye=Point2f((model_in[7-1].x+model_in[8-1].x)/2,(model_in[7-1].y+model_in[8-1].y)/2);
		dst_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);

		move_img(image_show, &image_show, src_pts, dst_pts); // move the image to eyes center
		move_LM_point(model_in, &model_in, src_pts, dst_pts); // 移動測試影像的LM點
		//imshow("image_show",image_show);waitKey(0);

		int LM_in_index[]={11,12,13,14,20,23,26,29,32,35,38,41,50,55,56,57,63,64,65};
		vector<int> LM_in_number_index(LM_in_index, LM_in_index + sizeof(LM_in_index)/sizeof(LM_in_index[0]));
		vector<Point2f> LM_in_temp;
		for (int i = 0; i < LM_in_number_index.size(); i++)
		{
			LM_in_temp.push_back(LM_in[LM_in_number_index[i]-1]);
		}

		int model_in_index[]={1,2,3,4,5,6,7,8,9,10,11,12,13,18,19,20,26,27,28};
		vector<int> model_in_number_index(model_in_index, model_in_index + sizeof(model_in_index)/sizeof(model_in_index[0]));
		vector<Point2f> model_in_temp;
		for (int i = 0; i < model_in_number_index.size(); i++)
		{
			model_in_temp.push_back(model_in[model_in_number_index[i]-1]);
		}

		for (int i = 0; i < model_in_temp.size(); i++)
		{
			float x=model_in_temp[i].x-LM_in_temp[i].x;
			float y=model_in_temp[i].y-LM_in_temp[i].y;
			error_temp=error_temp+sqrt(x*x+y*y);
		}
		//cout<<"1 : "<<error_temp<<endl;
		error_temp=error_temp/model_in_temp.size();
		//cout<<"2 : "<<error_temp<<endl;

		//show_m(image_show, LM_in_temp, model_in_temp);

		//Mat imd_t=img_test.clone();
		//show_m_2(imd_t, LM_in_temp, model_in_temp);
		//show_m(image_show, LM_in_temp, model_in_temp);

		final_pic_flip.setTo(0);
		int width_c=(image_show.cols<final_pic_flip.cols)?image_show.cols:final_pic_flip.cols;
		int height_c=(image_show.rows<final_pic_flip.rows)?image_show.rows:final_pic_flip.rows;
		image_show(Rect(0,0,width_c,height_c)).copyTo(final_pic_flip(Rect(0,0,width_c,height_c)));

		LM_in_temp_draw=LM_in_temp;
		model_in_temp_draw=model_in_temp;
	}
	if (p_angle==-45)
	{
		//show_m(image_show, LM_in, model_in);

		Leye=LM_in[38-1];
		Reye=LM_in[29-1];
		float src_slope=(Leye.y-Reye.y)/(Leye.x-Reye.x);
		//float eye_length_src=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
		float eye_length_src=abs(Leye.y-Reye.y);
		//Point2f src_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);
		Point2f src_pts=Point2f((LM_in[35-1].x+LM_in[41-1].x+LM_in[11-1].x)/3,(LM_in[35-1].y+LM_in[41-1].y+LM_in[11-1].y)/3);

		Leye=model_in[11-1];
		Reye=model_in[8-1];
		float dst_slope=(Leye.y-Reye.y)/(Leye.x-Reye.x);
		//float eye_length_dst=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
		float eye_length_dst=abs(Leye.y-Reye.y);
		//Point2f dst_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);
		Point2f dst_pts=Point2f((model_in[10-1].x+model_in[12-1].x+model_in[1-1].x)/3,(model_in[10-1].y+model_in[12-1].y+model_in[1-1].y)/3);

		move_img(image_show, &image_show, src_pts, dst_pts); // move the image to eyes center
		move_LM_point(model_in, &model_in, src_pts, dst_pts); // 移動測試影像的LM點
		//show_m(image_show, LM_in, model_in);
		//imshow("image_show",image_show);waitKey(0);


		//float theate;
		//find_angle(dst_slope, src_slope, &theate);
		//Point2f center_point; // find the center point of landmark
		//find_LM_center(model_in, &center_point); 
		//Mat Rotate_Matrix=getRotationMatrix2D(center_point, theate, 1.0); // calculate the rotate matrix [2x3]
		//rotate_LM(model_in, &model_in, theate); 
		//cv::warpAffine(image_show, image_show, Rotate_Matrix, image_show.size()); // rotate the ori. image 
		//imshow("image_show",image_show);waitKey(0);
		//show_m(image_show, LM_in, model_in);

		float scale_IMG=eye_length_src/eye_length_dst;
		//cv::resize(image_show,image_show,Size(image_show.cols*scale_IMG,image_show.rows*scale_IMG)); //normlized image
		//scale_LM(model_in,&model_in, scale_IMG); //縮放移動後的LM
		//imshow("image_show",image_show);waitKey(0);
		fix_scale(scale_IMG, image_show, model_in, p_angle, img_in,  &image_show, &model_in);
		//imshow("image_show",image_show);waitKey(0);
		//show_m(image_show, LM_in, model_in);

		Leye=model_in[11-1];
		Reye=model_in[8-1];
		//dst_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);
		dst_pts=Point2f((model_in[10-1].x+model_in[12-1].x+model_in[1-1].x)/3,(model_in[10-1].y+model_in[12-1].y+model_in[1-1].y)/3);

		move_img(image_show, &image_show, src_pts, dst_pts); // move the image to eyes center
		move_LM_point(model_in, &model_in, src_pts, dst_pts); // 移動測試影像的LM點
		//show_m(image_show, LM_in, model_in);
		//imshow("image_show",image_show);waitKey(0);

		//int LM_in_index[]={14,29,35,38,41,26,11,32};
		int LM_in_index[]={14,29,35,38,41,26,11,32,50,51,20,23};
		vector<int> LM_in_number_index(LM_in_index, LM_in_index + sizeof(LM_in_index)/sizeof(LM_in_index[0]));
		vector<Point2f> LM_in_temp;
		for (int i = 0; i < LM_in_number_index.size(); i++)
		{
			LM_in_temp.push_back(LM_in[LM_in_number_index[i]-1]);
		}

		//int model_in_index[]={4,8,10,11,12,7,1,9};
		int model_in_index[]={4,8,10,11,12,7,1,9,13,15,5,6};
		vector<int> model_in_number_index(model_in_index, model_in_index + sizeof(model_in_index)/sizeof(model_in_index[0]));
		vector<Point2f> model_in_temp;
		for (int i = 0; i < model_in_number_index.size(); i++)
		{
			model_in_temp.push_back(model_in[model_in_number_index[i]-1]);
		}

		for (int i = 0; i < model_in_temp.size(); i++)
		{
			float x=model_in_temp[i].x-LM_in_temp[i].x;
			float y=model_in_temp[i].y-LM_in_temp[i].y;
			error_temp=error_temp+sqrt(x*x+y*y);
		}
		//cout<<"1 : "<<error_temp<<endl;
		error_temp=error_temp/model_in_temp.size();
		//cout<<"2 : "<<error_temp<<endl;

		final_pic_flip.setTo(0);
		int width_c=(image_show.cols<final_pic_flip.cols)?image_show.cols:final_pic_flip.cols;
		int height_c=(image_show.rows<final_pic_flip.rows)?image_show.rows:final_pic_flip.rows;
		image_show(Rect(0,0,width_c,height_c)).copyTo(final_pic_flip(Rect(0,0,width_c,height_c)));

		LM_in_temp_draw=LM_in_temp;
		model_in_temp_draw=model_in_temp;
	}
	if (p_angle==45)
	{
		Leye=LM_in[32-1];
		Reye=LM_in[20-1];
		float src_slope=(Leye.y-Reye.y)/(Leye.x-Reye.x);
		//float eye_length_src=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
		float eye_length_src=abs(Leye.y-Reye.y);
		Point2f src_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);

		Leye=model_in[9-1];
		Reye=model_in[5-1];
		float dst_slope=(Leye.y-Reye.y)/(Leye.x-Reye.x);
		//float eye_length_dst=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
		float eye_length_dst=abs(Leye.y-Reye.y);
		Point2f dst_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);

		move_img(image_show, &image_show, src_pts, dst_pts); // move the image to eyes center
		move_LM_point(model_in, &model_in, src_pts, dst_pts); // 移動測試影像的LM點
		//imshow("image_show",image_show);waitKey(0);

		//float theate;
		//find_angle(dst_slope, src_slope, &theate);
		//Point2f center_point; // find the center point of landmark
		//find_LM_center(model_in, &center_point); 
		//Mat Rotate_Matrix=getRotationMatrix2D(center_point, theate, 1.0); // calculate the rotate matrix [2x3]
		//rotate_LM(model_in, &model_in, theate); 
		//cv::warpAffine(image_show, image_show, Rotate_Matrix, image_show.size()); // rotate the ori. image 
		//imshow("image_show",image_show);waitKey(0);

		float scale_IMG=eye_length_src/eye_length_dst;
		//cv::resize(image_show,image_show,Size(image_show.cols*scale_IMG,image_show.rows*scale_IMG)); //normlized image
		//scale_LM(model_in,&model_in, scale_IMG); //縮放移動後的LM
		fix_scale(scale_IMG, image_show, model_in, p_angle, img_in,  &image_show, &model_in);
		//imshow("image_show",image_show);waitKey(0);

		Leye=model_in[9-1];
		Reye=model_in[5-1];
		dst_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);

		move_img(image_show, &image_show, src_pts, dst_pts); // move the image to eyes center
		move_LM_point(model_in, &model_in, src_pts, dst_pts); // 移動測試影像的LM點
		//imshow("image_show",image_show);waitKey(0);

		int LM_in_index[]={14,20,35,32,41,23,11,38,50,55,56};
		vector<int> LM_in_number_index(LM_in_index, LM_in_index + sizeof(LM_in_index)/sizeof(LM_in_index[0]));
		vector<Point2f> LM_in_temp;
		for (int i = 0; i < LM_in_number_index.size(); i++)
		{
			LM_in_temp.push_back(LM_in[LM_in_number_index[i]-1]);
		}

		int model_in_index[]={4,5,10,9,12,6,1,11,13,18,19};
		vector<int> model_in_number_index(model_in_index, model_in_index + sizeof(model_in_index)/sizeof(model_in_index[0]));
		vector<Point2f> model_in_temp;
		for (int i = 0; i < model_in_number_index.size(); i++)
		{
			model_in_temp.push_back(model_in[model_in_number_index[i]-1]);
		}

		for (int i = 0; i < model_in_temp.size(); i++)
		{
			float x=model_in_temp[i].x-LM_in_temp[i].x;
			float y=model_in_temp[i].y-LM_in_temp[i].y;
			error_temp=error_temp+sqrt(x*x+y*y);
		}
		//cout<<"1 : "<<error_temp<<endl;
		error_temp=error_temp/model_in_temp.size();
		//cout<<"2 : "<<error_temp<<endl;

		//show_m(image_show, LM_in_temp, model_in_temp);

		final_pic_flip.setTo(0);
		int width_c=(image_show.cols<final_pic_flip.cols)?image_show.cols:final_pic_flip.cols;
		int height_c=(image_show.rows<final_pic_flip.rows)?image_show.rows:final_pic_flip.rows;
		image_show(Rect(0,0,width_c,height_c)).copyTo(final_pic_flip(Rect(0,0,width_c,height_c)));

		LM_in_temp_draw=LM_in_temp;
		model_in_temp_draw=model_in_temp;
	}
	if (p_angle==-60 || p_angle==-75 || p_angle==-90)
	{
		Leye=LM_in[6-1];
		Reye=LM_in[11-1];
		//cout<<Leye<<endl;
		//cout<<Reye<<endl;

		if (Leye.x == Reye.x)
		{
			Leye.x=Leye.x-5;
		}
		float src_slope=(Leye.y-Reye.y)/(Leye.x-Reye.x);
		//float eye_length_src=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
		float eye_length_src=abs(Leye.y-Reye.y);
		//Point2f src_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);
		Point2f src_pts=Point2f((LM_in[4-1].x+LM_in[10-1].x+LM_in[12-1].x+LM_in[16-1].x)/4,(LM_in[4-1].y+LM_in[10-1].y+LM_in[12-1].y+LM_in[16-1].y)/4);
		//Point2f src_pts=LM_in[4-1];

		Leye=model_in[8-1];
		Reye=model_in[11-1];
		//cout<<Leye<<endl;
		//cout<<Reye<<endl;
		if (Leye.x == Reye.x)
		{
			Leye.x=Leye.x-5;
		}
		float dst_slope=(Leye.y-Reye.y)/(Leye.x-Reye.x);
		//float eye_length_dst=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
		float eye_length_dst=abs(Leye.y-Reye.y);
		//Point2f dst_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);
		Point2f dst_pts=Point2f((model_in[1-1].x+model_in[10-1].x+model_in[12-1].x+model_in[14-1].x)/4,(model_in[1-1].y+model_in[10-1].y+model_in[12-1].y+model_in[14-1].y)/4);
		//Point2f dst_pts=model_in[1-1];

		move_img(image_show, &image_show, src_pts, dst_pts); // move the image to eyes center
		move_LM_point(model_in, &model_in, src_pts, dst_pts); // 移動測試影像的LM點
		//imshow("image_show",image_show);waitKey(0);

		//float theate;
		//find_angle(dst_slope, src_slope, &theate);
		////cout<<theate<<endl;
		//Point2f center_point; // find the center point of landmark
		//find_LM_center(model_in, &center_point); 
		//Mat Rotate_Matrix=getRotationMatrix2D(center_point, theate, 1.0); // calculate the rotate matrix [2x3]
		//rotate_LM(model_in, &model_in, theate); 
		//cv::warpAffine(image_show, image_show, Rotate_Matrix, image_show.size()); // rotate the ori. image 
		//imshow("image_show",image_show);waitKey(0);

		float scale_IMG=eye_length_src/eye_length_dst;
		//cv::resize(image_show,image_show,Size(image_show.cols*scale_IMG,image_show.rows*scale_IMG)); //normlized image
		//scale_LM(model_in,&model_in, scale_IMG); //縮放移動後的LM
		fix_scale(scale_IMG, image_show, model_in, p_angle, img_in,  &image_show, &model_in);
		//imshow("image_show",image_show);waitKey(0);

		Leye=model_in[1-1];
		Reye=model_in[14-1];
		//dst_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);
		dst_pts=Point2f((model_in[1-1].x+model_in[10-1].x+model_in[12-1].x+model_in[14-1].x)/4,(model_in[1-1].y+model_in[10-1].y+model_in[12-1].y+model_in[14-1].y)/4);
		//dst_pts=model_in[1-1];

		move_img(image_show, &image_show, src_pts, dst_pts); // move the image to eyes center
		move_LM_point(model_in, &model_in, src_pts, dst_pts); // 移動測試影像的LM點
		//imshow("image_show",image_show);waitKey(0);

		//int LM_in_index[]={3,10,12,16,17,4,6,11};
		int LM_in_index[]={3,17,10,11,12,16,4};
		//int LM_in_index[]={3,10,12,16,17,4};this ori
		//int LM_in_index[]={3,10,12,4,6};//c_u
		vector<int> LM_in_number_index(LM_in_index, LM_in_index + sizeof(LM_in_index)/sizeof(LM_in_index[0]));
		vector<Point2f> LM_in_temp;
		for (int i = 0; i < LM_in_number_index.size(); i++)
		{
			LM_in_temp.push_back(LM_in[LM_in_number_index[i]-1]);
		}

		//int model_in_index[]={4,10,12,14,13,1,8,11};
		int model_in_index[]={4,8,10,11,12,14,1};
		//int model_in_index[]={4,10,12,14,13,1};this ori
		//int model_in_index[]={4,10,12,1,8};//c_u
		vector<int> model_in_number_index(model_in_index, model_in_index + sizeof(model_in_index)/sizeof(model_in_index[0]));
		vector<Point2f> model_in_temp;
		for (int i = 0; i < model_in_number_index.size(); i++)
		{
			model_in_temp.push_back(model_in[model_in_number_index[i]-1]);
		}


		for (int i = 0; i < model_in_temp.size(); i++)
		{
			float x=model_in_temp[i].x-LM_in_temp[i].x;
			float y=model_in_temp[i].y-LM_in_temp[i].y;
			error_temp=error_temp+sqrt(x*x+y*y);
		}
		//cout<<"1 : "<<error_temp<<endl;
		error_temp=error_temp/model_in_temp.size();
		//cout<<"2 : "<<error_temp<<endl;

		Mat imd_t=img_test.clone();
		show_m_2(imd_t, LM_in_temp, model_in_temp);
		show_m(image_show, LM_in_temp, model_in_temp);

		final_pic_flip.setTo(0);
		int width_c=(image_show.cols<final_pic_flip.cols)?image_show.cols:final_pic_flip.cols;
		int height_c=(image_show.rows<final_pic_flip.rows)?image_show.rows:final_pic_flip.rows;
		image_show(Rect(0,0,width_c,height_c)).copyTo(final_pic_flip(Rect(0,0,width_c,height_c)));

		LM_in_temp_draw=LM_in_temp;
		model_in_temp_draw=model_in_temp;
	}
	if (p_angle==60 || p_angle==75 || p_angle==90)
	{
		Leye=LM_in[6-1];
		Reye=LM_in[11-1];
		//cout<<Leye<<endl;
		//cout<<Reye<<endl;

		if (Leye.x == Reye.x)
		{
			Leye.x=Leye.x+5;
		}
		float src_slope=(Leye.y-Reye.y)/(Leye.x-Reye.x);
		//float eye_length_src=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
		float eye_length_src=abs(Leye.y-Reye.y);
		//Point2f src_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);
		Point2f src_pts=Point2f((LM_in[4-1].x+LM_in[10-1].x+LM_in[12-1].x+LM_in[16-1].x)/4,(LM_in[4-1].y+LM_in[10-1].y+LM_in[12-1].y+LM_in[16-1].y)/4);

		Leye=model_in[5-1];
		Reye=model_in[9-1];
		if (Leye.x == Reye.x)
		{
			Leye.x=Leye.x+5;
		}
		//cout<<Leye<<endl;
		//cout<<Reye<<endl;
		float dst_slope=(Leye.y-Reye.y)/(Leye.x-Reye.x);
		//float eye_length_dst=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
		float eye_length_dst=abs(Leye.y-Reye.y);
		//Point2f dst_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);
		Point2f dst_pts=Point2f((model_in[1-1].x+model_in[10-1].x+model_in[12-1].x+model_in[18-1].x)/4,(model_in[1-1].y+model_in[10-1].y+model_in[12-1].y+model_in[18-1].y)/4);


		move_img(image_show, &image_show, src_pts, dst_pts); // move the image to eyes center
		move_LM_point(model_in, &model_in, src_pts, dst_pts); // 移動測試影像的LM點
		//imshow("image_show",image_show);waitKey(0);

		//float theate;
		//find_angle(dst_slope, src_slope, &theate);
		////cout<<theate<<endl;
		//Point2f center_point; // find the center point of landmark
		//find_LM_center(model_in, &center_point); 
		//Mat Rotate_Matrix=getRotationMatrix2D(center_point, theate, 1.0); // calculate the rotate matrix [2x3]
		//rotate_LM(model_in, &model_in, theate); 
		//cv::warpAffine(image_show, image_show, Rotate_Matrix, image_show.size()); // rotate the ori. image 
		//imshow("image_show",image_show);waitKey(0);

		float scale_IMG=eye_length_src/eye_length_dst;
		//cv::resize(image_show,image_show,Size(image_show.cols*scale_IMG,image_show.rows*scale_IMG)); //normlized image
		//scale_LM(model_in,&model_in, scale_IMG); //縮放移動後的LM
		fix_scale(scale_IMG, image_show, model_in, p_angle, img_in,  &image_show, &model_in);
		//imshow("image_show",image_show);waitKey(0);

		Leye=model_in[1-1];
		Reye=model_in[18-1];
		//dst_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);
		dst_pts=Point2f((model_in[1-1].x+model_in[10-1].x+model_in[12-1].x+model_in[18-1].x)/4,(model_in[1-1].y+model_in[10-1].y+model_in[12-1].y+model_in[18-1].y)/4);

		move_img(image_show, &image_show, src_pts, dst_pts); // move the image to eyes center
		move_LM_point(model_in, &model_in, src_pts, dst_pts); // 移動測試影像的LM點
		//imshow("image_show",image_show);waitKey(0);

		int LM_in_index[]={3,10,12,16,17,4};
		vector<int> LM_in_number_index(LM_in_index, LM_in_index + sizeof(LM_in_index)/sizeof(LM_in_index[0]));
		vector<Point2f> LM_in_temp;
		for (int i = 0; i < LM_in_number_index.size(); i++)
		{
			LM_in_temp.push_back(LM_in[LM_in_number_index[i]-1]);
		}

		int model_in_index[]={4,10,12,18,13,1};
		vector<int> model_in_number_index(model_in_index, model_in_index + sizeof(model_in_index)/sizeof(model_in_index[0]));
		vector<Point2f> model_in_temp;
		for (int i = 0; i < model_in_number_index.size(); i++)
		{
			model_in_temp.push_back(model_in[model_in_number_index[i]-1]);
		}

		for (int i = 0; i < model_in_temp.size(); i++)
		{
			float x=model_in_temp[i].x-LM_in_temp[i].x;
			float y=model_in_temp[i].y-LM_in_temp[i].y;
			error_temp=error_temp+sqrt(x*x+y*y);
		}
		//cout<<"1 : "<<error_temp<<endl;
		error_temp=error_temp/model_in_temp.size();
		//cout<<"2 : "<<error_temp<<endl;

		//show_m(image_show, LM_in_temp, model_in_temp);

		final_pic_flip.setTo(0);
		int width_c=(image_show.cols<final_pic_flip.cols)?image_show.cols:final_pic_flip.cols;
		int height_c=(image_show.rows<final_pic_flip.rows)?image_show.rows:final_pic_flip.rows;
		image_show(Rect(0,0,width_c,height_c)).copyTo(final_pic_flip(Rect(0,0,width_c,height_c)));

		LM_in_temp_draw=LM_in_temp;
		model_in_temp_draw=model_in_temp;
	}

	//Mat final_show=final_pic_flip.clone();
	//for (int i = 0; i < LM_in_temp_draw.size(); i++)
	//{
	//	circle(final_show,LM_in_temp_draw[i],5,CV_RGB(255,255,0),-1);
	//}
	//for (int i = 0; i < model_in_temp_draw.size(); i++)
	//{
	//	circle(final_show,model_in_temp_draw[i],3,CV_RGB(255,0,0),-1);
	//}
	//imshow("final_show",final_show);waitKey(0);

	*model_out=model_in;
	*error_num=error_temp;
}

void error_cal(float p_angle, Mat img_in, vector<Point2f> LM_in, vector<Point2f> model_in, vector<Point2f>* model_out, double* error_num)
{
	Point2f Leye;
	Point2f Reye;
	Mat image_show=final_pic_flip.clone();
	double error_temp=0.0;
	vector<Point2f> LM_in_temp_draw;
	vector<Point2f> model_in_temp_draw;


	if (p_angle>=-30.0 && p_angle<=30.0)
	{
		Leye=Point2f((LM_in[20-1].x+LM_in[23-1].x)/2,(LM_in[20-1].y+LM_in[23-1].y)/2);
		Reye=Point2f((LM_in[26-1].x+LM_in[29-1].x)/2,(LM_in[26-1].y+LM_in[29-1].y)/2);
		float src_slope=(Leye.y-Reye.y)/(Leye.x-Reye.x);
		float eye_length_src=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
		Point2f src_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);

		Leye=Point2f((model_in[5-1].x+model_in[6-1].x)/2,(model_in[5-1].y+model_in[6-1].y)/2);
		Reye=Point2f((model_in[7-1].x+model_in[8-1].x)/2,(model_in[7-1].y+model_in[8-1].y)/2);
		float dst_slope=(Leye.y-Reye.y)/(Leye.x-Reye.x);
		float eye_length_dst=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
		Point2f dst_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);

		move_img(image_show, &image_show, src_pts, dst_pts); // move the image to eyes center
		move_LM_point(model_in, &model_in, src_pts, dst_pts); // 移動測試影像的LM點
		//imshow("image_show",image_show);waitKey(0);

		float theate;
		find_angle(dst_slope, src_slope, &theate);
		Point2f center_point; // find the center point of landmark
		find_LM_center(model_in, &center_point); 
		Mat Rotate_Matrix=getRotationMatrix2D(center_point, theate, 1.0); // calculate the rotate matrix [2x3]
		rotate_LM(model_in, &model_in, theate); 
		cv::warpAffine(image_show, image_show, Rotate_Matrix, image_show.size()); // rotate the ori. image 
		//imshow("image_show",image_show);waitKey(0);

		float scale_IMG=eye_length_src/eye_length_dst;
		//cv::resize(image_show,image_show,Size(image_show.cols*scale_IMG,image_show.rows*scale_IMG)); //normlized image
		//scale_LM(model_in,&model_in, scale_IMG); //縮放移動後的LM
		fix_scale(scale_IMG, image_show, model_in, p_angle, img_in,  &image_show, &model_in);
		//imshow("image_show",image_show);waitKey(0);

		Leye=Point2f((model_in[5-1].x+model_in[6-1].x)/2,(model_in[5-1].y+model_in[6-1].y)/2);
		Reye=Point2f((model_in[7-1].x+model_in[8-1].x)/2,(model_in[7-1].y+model_in[8-1].y)/2);
		dst_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);

		move_img(image_show, &image_show, src_pts, dst_pts); // move the image to eyes center
		move_LM_point(model_in, &model_in, src_pts, dst_pts); // 移動測試影像的LM點
		//imshow("image_show",image_show);waitKey(0);

		int LM_in_index[]={14,20,23,26,29,32,35,38,41};
		vector<int> LM_in_number_index(LM_in_index, LM_in_index + sizeof(LM_in_index)/sizeof(LM_in_index[0]));
		vector<Point2f> LM_in_temp;
		for (int i = 0; i < LM_in_number_index.size(); i++)
		{
			LM_in_temp.push_back(LM_in[LM_in_number_index[i]-1]);
		}

		int model_in_index[]={4,5,6,7,8,9,10,11,12};
		vector<int> model_in_number_index(model_in_index, model_in_index + sizeof(model_in_index)/sizeof(model_in_index[0]));
		vector<Point2f> model_in_temp;
		for (int i = 0; i < model_in_number_index.size(); i++)
		{
			model_in_temp.push_back(model_in[model_in_number_index[i]-1]);
		}

#if err_type
		get_error(LM_in_temp, model_in_temp, &error_temp);
#else
		for (int i = 0; i < model_in_temp.size(); i++)
		{
			float x=model_in_temp[i].x-LM_in_temp[i].x;
			float y=model_in_temp[i].y-LM_in_temp[i].y;
			error_temp=error_temp+sqrt(x*x+y*y);
		}
		//cout<<"1 : "<<error_temp<<endl;
		error_temp=error_temp/model_in_temp.size();
		//cout<<"2 : "<<error_temp<<endl;
#endif

		//show_m(image_show, LM_in_temp, model_in_temp);

		final_pic_flip.setTo(0);
		int width_c=(image_show.cols<final_pic_flip.cols)?image_show.cols:final_pic_flip.cols;
		int height_c=(image_show.rows<final_pic_flip.rows)?image_show.rows:final_pic_flip.rows;
		image_show(Rect(0,0,width_c,height_c)).copyTo(final_pic_flip(Rect(0,0,width_c,height_c)));

		LM_in_temp_draw=LM_in_temp;
		model_in_temp_draw=model_in_temp;
	}
	if (p_angle==-45)
	{
		//show_m(image_show, LM_in, model_in);

		Leye=LM_in[38-1];
		Reye=LM_in[29-1];
		float src_slope=(Leye.y-Reye.y)/(Leye.x-Reye.x);
		float eye_length_src=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
		Point2f src_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);

		Leye=model_in[11-1];
		Reye=model_in[8-1];
		float dst_slope=(Leye.y-Reye.y)/(Leye.x-Reye.x);
		float eye_length_dst=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
		Point2f dst_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);

		move_img(image_show, &image_show, src_pts, dst_pts); // move the image to eyes center
		move_LM_point(model_in, &model_in, src_pts, dst_pts); // 移動測試影像的LM點
		//show_m(image_show, LM_in, model_in);
		//imshow("image_show",image_show);waitKey(0);


		float theate;
		find_angle(dst_slope, src_slope, &theate);
		Point2f center_point; // find the center point of landmark
		find_LM_center(model_in, &center_point); 
		Mat Rotate_Matrix=getRotationMatrix2D(center_point, theate, 1.0); // calculate the rotate matrix [2x3]
		rotate_LM(model_in, &model_in, theate); 
		cv::warpAffine(image_show, image_show, Rotate_Matrix, image_show.size()); // rotate the ori. image 
		//imshow("image_show",image_show);waitKey(0);
		//show_m(image_show, LM_in, model_in);

		float scale_IMG=eye_length_src/eye_length_dst;
		//cv::resize(image_show,image_show,Size(image_show.cols*scale_IMG,image_show.rows*scale_IMG)); //normlized image
		//scale_LM(model_in,&model_in, scale_IMG); //縮放移動後的LM
		//imshow("image_show",image_show);waitKey(0);
		fix_scale(scale_IMG, image_show, model_in, p_angle, img_in,  &image_show, &model_in);
		//imshow("image_show",image_show);waitKey(0);
		//show_m(image_show, LM_in, model_in);

		Leye=model_in[11-1];
		Reye=model_in[8-1];
		dst_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);

		move_img(image_show, &image_show, src_pts, dst_pts); // move the image to eyes center
		move_LM_point(model_in, &model_in, src_pts, dst_pts); // 移動測試影像的LM點
		//show_m(image_show, LM_in, model_in);
		//imshow("image_show",image_show);waitKey(0);

		//int LM_in_index[]={14,29,35,38,41,26,11,32};
		int LM_in_index[]={14,29,35,38,41,26,11,32,50,51,52};
		vector<int> LM_in_number_index(LM_in_index, LM_in_index + sizeof(LM_in_index)/sizeof(LM_in_index[0]));
		vector<Point2f> LM_in_temp;
		for (int i = 0; i < LM_in_number_index.size(); i++)
		{
			LM_in_temp.push_back(LM_in[LM_in_number_index[i]-1]);
		}

		//int model_in_index[]={4,8,10,11,12,7,1,9};
		int model_in_index[]={4,8,10,11,12,7,1,9,13,14,15};
		vector<int> model_in_number_index(model_in_index, model_in_index + sizeof(model_in_index)/sizeof(model_in_index[0]));
		vector<Point2f> model_in_temp;
		for (int i = 0; i < model_in_number_index.size(); i++)
		{
			model_in_temp.push_back(model_in[model_in_number_index[i]-1]);
		}

#if err_type
		get_error(LM_in_temp, model_in_temp, &error_temp);
#else
		for (int i = 0; i < model_in_temp.size(); i++)
		{
			float x=model_in_temp[i].x-LM_in_temp[i].x;
			float y=model_in_temp[i].y-LM_in_temp[i].y;
			error_temp=error_temp+sqrt(x*x+y*y);
		}
		//cout<<"1 : "<<error_temp<<endl;
		error_temp=error_temp/model_in_temp.size();
		//cout<<"2 : "<<error_temp<<endl;
#endif

		//show_m(image_show, LM_in_temp, model_in_temp);

		final_pic_flip.setTo(0);
		int width_c=(image_show.cols<final_pic_flip.cols)?image_show.cols:final_pic_flip.cols;
		int height_c=(image_show.rows<final_pic_flip.rows)?image_show.rows:final_pic_flip.rows;
		image_show(Rect(0,0,width_c,height_c)).copyTo(final_pic_flip(Rect(0,0,width_c,height_c)));

		LM_in_temp_draw=LM_in_temp;
		model_in_temp_draw=model_in_temp;
	}
	if (p_angle==45)
	{
		Leye=LM_in[32-1];
		Reye=LM_in[20-1];
		float src_slope=(Leye.y-Reye.y)/(Leye.x-Reye.x);
		float eye_length_src=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
		Point2f src_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);

		Leye=model_in[9-1];
		Reye=model_in[5-1];
		float dst_slope=(Leye.y-Reye.y)/(Leye.x-Reye.x);
		float eye_length_dst=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
		Point2f dst_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);

		move_img(image_show, &image_show, src_pts, dst_pts); // move the image to eyes center
		move_LM_point(model_in, &model_in, src_pts, dst_pts); // 移動測試影像的LM點
		//imshow("image_show",image_show);waitKey(0);

		float theate;
		find_angle(dst_slope, src_slope, &theate);
		Point2f center_point; // find the center point of landmark
		find_LM_center(model_in, &center_point); 
		Mat Rotate_Matrix=getRotationMatrix2D(center_point, theate, 1.0); // calculate the rotate matrix [2x3]
		rotate_LM(model_in, &model_in, theate); 
		cv::warpAffine(image_show, image_show, Rotate_Matrix, image_show.size()); // rotate the ori. image 
		//imshow("image_show",image_show);waitKey(0);

		float scale_IMG=eye_length_src/eye_length_dst;
		//cv::resize(image_show,image_show,Size(image_show.cols*scale_IMG,image_show.rows*scale_IMG)); //normlized image
		//scale_LM(model_in,&model_in, scale_IMG); //縮放移動後的LM
		fix_scale(scale_IMG, image_show, model_in, p_angle, img_in,  &image_show, &model_in);
		//imshow("image_show",image_show);waitKey(0);

		Leye=model_in[9-1];
		Reye=model_in[5-1];
		dst_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);

		move_img(image_show, &image_show, src_pts, dst_pts); // move the image to eyes center
		move_LM_point(model_in, &model_in, src_pts, dst_pts); // 移動測試影像的LM點
		//imshow("image_show",image_show);waitKey(0);

		int LM_in_index[]={14,20,35,32,41,23,11,38,50,55,56};
		vector<int> LM_in_number_index(LM_in_index, LM_in_index + sizeof(LM_in_index)/sizeof(LM_in_index[0]));
		vector<Point2f> LM_in_temp;
		for (int i = 0; i < LM_in_number_index.size(); i++)
		{
			LM_in_temp.push_back(LM_in[LM_in_number_index[i]-1]);
		}

		int model_in_index[]={4,5,10,9,12,6,1,11,13,18,19};
		vector<int> model_in_number_index(model_in_index, model_in_index + sizeof(model_in_index)/sizeof(model_in_index[0]));
		vector<Point2f> model_in_temp;
		for (int i = 0; i < model_in_number_index.size(); i++)
		{
			model_in_temp.push_back(model_in[model_in_number_index[i]-1]);
		}

#if err_type
		get_error(LM_in_temp, model_in_temp, &error_temp);
#else
		for (int i = 0; i < model_in_temp.size(); i++)
		{
			float x=model_in_temp[i].x-LM_in_temp[i].x;
			float y=model_in_temp[i].y-LM_in_temp[i].y;
			error_temp=error_temp+sqrt(x*x+y*y);
		}
		//cout<<"1 : "<<error_temp<<endl;
		error_temp=error_temp/model_in_temp.size();
		//cout<<"2 : "<<error_temp<<endl;
#endif

		//show_m(image_show, LM_in_temp, model_in_temp);

		final_pic_flip.setTo(0);
		int width_c=(image_show.cols<final_pic_flip.cols)?image_show.cols:final_pic_flip.cols;
		int height_c=(image_show.rows<final_pic_flip.rows)?image_show.rows:final_pic_flip.rows;
		image_show(Rect(0,0,width_c,height_c)).copyTo(final_pic_flip(Rect(0,0,width_c,height_c)));

		LM_in_temp_draw=LM_in_temp;
		model_in_temp_draw=model_in_temp;
	}
	if (p_angle==-60 || p_angle==-75 || p_angle==-90)
	{
		Leye=LM_in[11-1];
		Reye=LM_in[6-1];
		//cout<<Leye<<endl;
		//cout<<Reye<<endl;

		if (Leye.x == Reye.x)
		{
			Leye.x=Leye.x-5;
		}
		float src_slope=(Leye.y-Reye.y)/(Leye.x-Reye.x);
		float eye_length_src=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
		Point2f src_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);

		Leye=model_in[11-1];
		Reye=model_in[8-1];
		//cout<<Leye<<endl;
		//cout<<Reye<<endl;
		if (Leye.x == Reye.x)
		{
			Leye.x=Leye.x-5;
		}
		float dst_slope=(Leye.y-Reye.y)/(Leye.x-Reye.x);
		float eye_length_dst=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
		Point2f dst_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);

		move_img(image_show, &image_show, src_pts, dst_pts); // move the image to eyes center
		move_LM_point(model_in, &model_in, src_pts, dst_pts); // 移動測試影像的LM點
		//imshow("image_show",image_show);waitKey(0);

		//float theate;
		//find_angle(dst_slope, src_slope, &theate);
		////cout<<theate<<endl;
		//Point2f center_point; // find the center point of landmark
		//find_LM_center(model_in, &center_point); 
		//Mat Rotate_Matrix=getRotationMatrix2D(center_point, theate, 1.0); // calculate the rotate matrix [2x3]
		//rotate_LM(model_in, &model_in, theate); 
		//cv::warpAffine(image_show, image_show, Rotate_Matrix, image_show.size()); // rotate the ori. image 
		//imshow("image_show",image_show);waitKey(0);

		float scale_IMG=eye_length_src/eye_length_dst;
		//cv::resize(image_show,image_show,Size(image_show.cols*scale_IMG,image_show.rows*scale_IMG)); //normlized image
		//scale_LM(model_in,&model_in, scale_IMG); //縮放移動後的LM
		fix_scale(scale_IMG, image_show, model_in, p_angle, img_in,  &image_show, &model_in);
		//imshow("image_show",image_show);waitKey(0);

		Leye=model_in[11-1];
		Reye=model_in[8-1];
		dst_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);

		move_img(image_show, &image_show, src_pts, dst_pts); // move the image to eyes center
		move_LM_point(model_in, &model_in, src_pts, dst_pts); // 移動測試影像的LM點
		//imshow("image_show",image_show);waitKey(0);

		int LM_in_index[]={3,6,10,11,12,16,4};
		//int LM_in_index[]={3,6,10,4};//c_u
		vector<int> LM_in_number_index(LM_in_index, LM_in_index + sizeof(LM_in_index)/sizeof(LM_in_index[0]));
		vector<Point2f> LM_in_temp;
		for (int i = 0; i < LM_in_number_index.size(); i++)
		{
			LM_in_temp.push_back(LM_in[LM_in_number_index[i]-1]);
		}

		int model_in_index[]={4,8,10,11,12,14,1};
		//int model_in_index[]={4,8,10,1};//c_u
		vector<int> model_in_number_index(model_in_index, model_in_index + sizeof(model_in_index)/sizeof(model_in_index[0]));
		vector<Point2f> model_in_temp;
		for (int i = 0; i < model_in_number_index.size(); i++)
		{
			model_in_temp.push_back(model_in[model_in_number_index[i]-1]);
		}

#if err_type
		get_error(LM_in_temp, model_in_temp, &error_temp);
#else
		for (int i = 0; i < model_in_temp.size(); i++)
		{
			float x=model_in_temp[i].x-LM_in_temp[i].x;
			float y=model_in_temp[i].y-LM_in_temp[i].y;
			error_temp=error_temp+sqrt(x*x+y*y);
		}
		//cout<<"1 : "<<error_temp<<endl;
		error_temp=error_temp/model_in_temp.size();
		//cout<<"2 : "<<error_temp<<endl;
#endif
		Mat imd_t=img_test.clone();
		show_m_2(imd_t, LM_in_temp, model_in_temp);
		show_m(image_show, LM_in_temp, model_in_temp);

		final_pic_flip.setTo(0);
		int width_c=(image_show.cols<final_pic_flip.cols)?image_show.cols:final_pic_flip.cols;
		int height_c=(image_show.rows<final_pic_flip.rows)?image_show.rows:final_pic_flip.rows;
		image_show(Rect(0,0,width_c,height_c)).copyTo(final_pic_flip(Rect(0,0,width_c,height_c)));

		LM_in_temp_draw=LM_in_temp;
		model_in_temp_draw=model_in_temp;
	}
	if (p_angle==60 || p_angle==75 || p_angle==90)
	{
		Leye=LM_in[11-1];
		Reye=LM_in[6-1];
		//cout<<Leye<<endl;
		//cout<<Reye<<endl;

		if (Leye.x == Reye.x)
		{
			Leye.x=Leye.x+5;
		}
		float src_slope=(Leye.y-Reye.y)/(Leye.x-Reye.x);
		float eye_length_src=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
		Point2f src_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);

		Leye=model_in[9-1];
		Reye=model_in[5-1];
		if (Leye.x == Reye.x)
		{
			Leye.x=Leye.x+5;
		}
		//cout<<Leye<<endl;
		//cout<<Reye<<endl;
		float dst_slope=(Leye.y-Reye.y)/(Leye.x-Reye.x);
		float eye_length_dst=sqrt((Leye.x-Reye.x)*(Leye.x-Reye.x)+(Leye.y-Reye.y)*(Leye.y-Reye.y));
		Point2f dst_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);

		move_img(image_show, &image_show, src_pts, dst_pts); // move the image to eyes center
		move_LM_point(model_in, &model_in, src_pts, dst_pts); // 移動測試影像的LM點
		//imshow("image_show",image_show);waitKey(0);

		//float theate;
		//find_angle(dst_slope, src_slope, &theate);
		////cout<<theate<<endl;
		//Point2f center_point; // find the center point of landmark
		//find_LM_center(model_in, &center_point); 
		//Mat Rotate_Matrix=getRotationMatrix2D(center_point, theate, 1.0); // calculate the rotate matrix [2x3]
		//rotate_LM(model_in, &model_in, theate); 
		//cv::warpAffine(image_show, image_show, Rotate_Matrix, image_show.size()); // rotate the ori. image 
		//imshow("image_show",image_show);waitKey(0);

		float scale_IMG=eye_length_src/eye_length_dst;
		//cv::resize(image_show,image_show,Size(image_show.cols*scale_IMG,image_show.rows*scale_IMG)); //normlized image
		//scale_LM(model_in,&model_in, scale_IMG); //縮放移動後的LM
		fix_scale(scale_IMG, image_show, model_in, p_angle, img_in,  &image_show, &model_in);
		//imshow("image_show",image_show);waitKey(0);

		Leye=model_in[9-1];
		Reye=model_in[5-1];
		dst_pts=Point2f((Leye.x+Reye.x)/2,(Leye.y+Reye.y)/2);

		move_img(image_show, &image_show, src_pts, dst_pts); // move the image to eyes center
		move_LM_point(model_in, &model_in, src_pts, dst_pts); // 移動測試影像的LM點
		//imshow("image_show",image_show);waitKey(0);

		int LM_in_index[]={3,6,10,11,12,16,4};
		vector<int> LM_in_number_index(LM_in_index, LM_in_index + sizeof(LM_in_index)/sizeof(LM_in_index[0]));
		vector<Point2f> LM_in_temp;
		for (int i = 0; i < LM_in_number_index.size(); i++)
		{
			LM_in_temp.push_back(LM_in[LM_in_number_index[i]-1]);
		}

		int model_in_index[]={4,5,10,9,12,18,1};
		vector<int> model_in_number_index(model_in_index, model_in_index + sizeof(model_in_index)/sizeof(model_in_index[0]));
		vector<Point2f> model_in_temp;
		for (int i = 0; i < model_in_number_index.size(); i++)
		{
			model_in_temp.push_back(model_in[model_in_number_index[i]-1]);
		}

#if err_type
		get_error(LM_in_temp, model_in_temp, &error_temp);
		//cout<<error_temp<<endl;
#else
		for (int i = 0; i < model_in_temp.size(); i++)
		{
			float x=model_in_temp[i].x-LM_in_temp[i].x;
			float y=model_in_temp[i].y-LM_in_temp[i].y;
			error_temp=error_temp+sqrt(x*x+y*y);
		}
		//cout<<"1 : "<<error_temp<<endl;
		error_temp=error_temp/model_in_temp.size();
		//cout<<"2 : "<<error_temp<<endl;
#endif

		//show_m(image_show, LM_in_temp, model_in_temp);

		final_pic_flip.setTo(0);
		int width_c=(image_show.cols<final_pic_flip.cols)?image_show.cols:final_pic_flip.cols;
		int height_c=(image_show.rows<final_pic_flip.rows)?image_show.rows:final_pic_flip.rows;
		image_show(Rect(0,0,width_c,height_c)).copyTo(final_pic_flip(Rect(0,0,width_c,height_c)));

		LM_in_temp_draw=LM_in_temp;
		model_in_temp_draw=model_in_temp;
	}

	//Mat final_show=final_pic_flip.clone();
	//for (int i = 0; i < LM_in_temp_draw.size(); i++)
	//{
	//	circle(final_show,LM_in_temp_draw[i],5,CV_RGB(255,255,0),-1);
	//}
	//for (int i = 0; i < model_in_temp_draw.size(); i++)
	//{
	//	circle(final_show,model_in_temp_draw[i],3,CV_RGB(255,0,0),-1);
	//}
	//imshow("final_show",final_show);waitKey(0);

	*model_out=model_in;
	*error_num=error_temp;
}
void find_angle(float src_slope, float dst_slope, float* theate)
{
	//cout<<src_slope<<endl;
	//cout<<dst_slope<<endl;

	double angle_1=(src_slope-dst_slope)/(1+src_slope*dst_slope);
	double angle_2=-(src_slope-dst_slope)/(1+src_slope*dst_slope);

	//cout<<angle_1<<endl;
	//cout<<angle_2<<endl;

	double angle_arct_1=atan(angle_1);
	double angle_arct_2=atan(angle_2);

	//cout<<angle_arct_1<<endl;
	//cout<<angle_arct_2<<endl;

	angle_arct_1=angle_arct_1*180/3.141592;
	angle_arct_2=angle_arct_2*180/3.141592;

	if (src_slope > dst_slope)
	{
		*theate=(angle_arct_1>angle_arct_2)?angle_arct_1:angle_arct_2;
	} 
	else
	{
		*theate=(angle_arct_1<angle_arct_2)?angle_arct_1:angle_arct_2;
	}
}
void get_error(vector<Point2f> target_LM, vector<Point2f> Reg_LM, double* err_num)
{
	Mat ideal_H=Mat::eye(3,3,CV_64FC1);
	Mat H=findHomography(Reg_LM,target_LM,CV_RANSAC, 3);
	//Mat H=findHomography(Reg_LM,target_LM,CV_LMEDS );

	//cout<<ideal_H<<endl;
	cout<<H<<endl;

	double error_temp=0.0;
	for (int i = 0; i < H.rows; i++)
	{
		for (int j = 0; j < H.cols; j++)
		{
			error_temp = error_temp +abs(H.at<double>(i,j)-ideal_H.at<double>(i,j));
			//cout<<abs(H.at<double>(i,j)-ideal_H.at<double>(i,j))<<endl;
		}
	}
	//cout<<error_temp<<endl;

	//H=findHomography(Reg_LM,target_LM,CV_LMEDS);
	//for (int i = 0; i < H.rows; i++)
	//{
	//	for (int j = 0; j < H.cols; j++)
	//	{
	//		error_temp = error_temp +abs(H.at<double>(i,j)-ideal_H.at<double>(i,j));
	//		//cout<<abs(H.at<double>(i,j)-ideal_H.at<double>(i,j))<<endl;
	//	}
	//}

	//system("pause");
	*err_num=error_temp;
}

// crop image //
void crop_image_test(float p_angle, Mat image_in, vector<Point2f> LM_in, Mat* image_out)
{
	int min_x=image_in.cols;
	int min_y=image_in.rows;
	int max_x=0;
	int max_y=0;

	Mat image_temp;

	float bound=5.0;

	if (p_angle == -15 || p_angle == -30 || p_angle == 0 || p_angle == 30 || p_angle == 15)
	{
		//int LM_in_index[]={20,23,26,29,32,35,38,41,50,67,14,6,7,8,9,10};
		int LM_in_index[]={20,23,26,29,32,35,38,41,67,14,1,2,3,4,5,6,7,8,9,10,50,51,55};
		vector<int> LM_in_number_index(LM_in_index, LM_in_index + sizeof(LM_in_index)/sizeof(LM_in_index[0]));
		vector<Point2f> LM_in_temp;
		for (int i = 0; i < LM_in_number_index.size(); i++)
		{
			LM_in_temp.push_back(LM_in[LM_in_number_index[i]-1]);
		}

		for (int i = 0; i < LM_in_temp.size(); i++)
		{
			min_x=(min_x<LM_in_temp[i].x)?min_x:LM_in_temp[i].x;
			min_y=(min_y<LM_in_temp[i].y)?min_y:LM_in_temp[i].y;
			max_x=(max_x>LM_in_temp[i].x)?max_x:LM_in_temp[i].x;
			max_y=(max_y>LM_in_temp[i].y)?max_y:LM_in_temp[i].y;
		}

		//int min_x_z=min_x;
		//int min_y_z=min_y;
		//int max_x_z=max_x;
		//int max_y_z=max_y;
		min_x=min_x-bound;
		min_y=min_y-bound;
		max_x=max_x+bound;
		max_y=max_y+bound;

		//Mat image_temp(cvRound(max_y-min_y),cvRound(max_x-min_x),CV_8UC3);
		image_in(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(image_temp);

		//for (int i = 0; i < LM_in_temp.size(); i++)
		//{
		//	circle(image_in,LM_in_temp[i],1,CV_RGB(255,255,0),-1);
		//}
		//rectangle(image_in,Rect(min_x_z,min_y_z,cvRound(max_x_z-min_x_z),cvRound(max_y_z-min_y_z)), CV_RGB(255,0,0), 1, 8,0 );
		//min_x_z=min_x_z-5;
		//min_y_z=min_y_z-5;
		//max_x_z=max_x_z+5;
		//max_y_z=max_y_z+5;
		//rectangle(image_in,Rect(min_x_z,min_y_z,cvRound(max_x_z-min_x_z),cvRound(max_y_z-min_y_z)), CV_RGB(255,0,255), 1, 8,0 );
		//min_x_z=min_x_z-5;
		//min_y_z=min_y_z-5;
		//max_x_z=max_x_z+5;
		//max_y_z=max_y_z+5;
		//rectangle(image_in,Rect(min_x_z,min_y_z,cvRound(max_x_z-min_x_z),cvRound(max_y_z-min_y_z)), CV_RGB(0,255,0), 1, 8,0 );
		//min_x_z=min_x_z-5;
		//min_y_z=min_y_z-5;
		//max_x_z=max_x_z+5;
		//max_y_z=max_y_z+5;
		//rectangle(image_in,Rect(min_x_z,min_y_z,cvRound(max_x_z-min_x_z),cvRound(max_y_z-min_y_z)), CV_RGB(0,255,255), 1, 8,0 );
		//imshow("image_in",image_in);waitKey(0);
	}
	if (p_angle == -45)
	{
		//int LM_in_index[]={20,23,26,29,32,35,38,41,50,67,14,6,7,8,9,10};
		//int LM_in_index[]={20,23,26,29,32,35,38,41,67,14,6,7,8,9,10};
		int LM_in_index[]={20,23,26,29,32,35,38,41,67,14,6,7,8,9,10,50,51,52,55,56};
		vector<int> LM_in_number_index(LM_in_index, LM_in_index + sizeof(LM_in_index)/sizeof(LM_in_index[0]));
		vector<Point2f> LM_in_temp;
		for (int i = 0; i < LM_in_number_index.size(); i++)
		{
			LM_in_temp.push_back(LM_in[LM_in_number_index[i]-1]);
		}

		for (int i = 0; i < LM_in_temp.size(); i++)
		{
			min_x=(min_x<LM_in_temp[i].x)?min_x:LM_in_temp[i].x;
			min_y=(min_y<LM_in_temp[i].y)?min_y:LM_in_temp[i].y;
			max_x=(max_x>LM_in_temp[i].x)?max_x:LM_in_temp[i].x;
			max_y=(max_y>LM_in_temp[i].y)?max_y:LM_in_temp[i].y;
		}

		min_x=min_x-bound;
		min_y=min_y-bound;
		max_x=max_x+bound;
		max_y=max_y+bound;

		//Mat image_temp(cvRound(max_y-min_y),cvRound(max_x-min_x),CV_8UC3);
		image_in(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(image_temp);
		//show(image_in, LM_in_temp);
	}
	if (p_angle == 45)
	{
		//int LM_in_index[]={20,23,26,29,32,35,38,41,50,67,14,6,7,8,9,10};
		int LM_in_index[]={20,23,26,29,32,35,38,41,67,14,1,2,3,4,5,50,51,52,55,56};
		vector<int> LM_in_number_index(LM_in_index, LM_in_index + sizeof(LM_in_index)/sizeof(LM_in_index[0]));
		vector<Point2f> LM_in_temp;
		for (int i = 0; i < LM_in_number_index.size(); i++)
		{
			LM_in_temp.push_back(LM_in[LM_in_number_index[i]-1]);
		}

		for (int i = 0; i < LM_in_temp.size(); i++)
		{
			min_x=(min_x<LM_in_temp[i].x)?min_x:LM_in_temp[i].x;
			min_y=(min_y<LM_in_temp[i].y)?min_y:LM_in_temp[i].y;
			max_x=(max_x>LM_in_temp[i].x)?max_x:LM_in_temp[i].x;
			max_y=(max_y>LM_in_temp[i].y)?max_y:LM_in_temp[i].y;
		}

		min_x=min_x-bound;
		min_y=min_y-bound;
		max_x=max_x+bound;
		max_y=max_y+bound;

		//Mat image_temp(cvRound(max_y-min_y),cvRound(max_x-min_x),CV_8UC3);
		image_in(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(image_temp);
	}
	if (p_angle == -60 || p_angle == -75 || p_angle == -90)
	{
		int LM_in_index[]={1,2,3,4,5,6,7,8,9,10,11,12,15,16,17};
		//int LM_in_index[]={1,2,3,4,5,6,7,8,9,10,11,12};
		//int LM_in_index[]={1,2,3,4,5,6,7,8,9};
		vector<int> LM_in_number_index(LM_in_index, LM_in_index + sizeof(LM_in_index)/sizeof(LM_in_index[0]));
		vector<Point2f> LM_in_temp;
		for (int i = 0; i < LM_in_number_index.size(); i++)
		{
			LM_in_temp.push_back(LM_in[LM_in_number_index[i]-1]);
		}

		for (int i = 0; i < LM_in_temp.size(); i++)
		{
			min_x=(min_x<LM_in_temp[i].x)?min_x:LM_in_temp[i].x;
			min_y=(min_y<LM_in_temp[i].y)?min_y:LM_in_temp[i].y;
			max_x=(max_x>LM_in_temp[i].x)?max_x:LM_in_temp[i].x;
			max_y=(max_y>LM_in_temp[i].y)?max_y:LM_in_temp[i].y;
		}

		min_x=min_x-bound;
		min_y=min_y-bound;
		max_x=max_x+bound;
		max_y=max_y+bound;

		//Mat image_temp(cvRound(max_y-min_y),cvRound(max_x-min_x),CV_8UC3);
		image_in(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(image_temp);

		//int LM_in_index_u[]={1,2,3,4,5,6,7,8,9};
		//vector<int> LM_in_number_index_u(LM_in_index_u, LM_in_index_u + sizeof(LM_in_index_u)/sizeof(LM_in_index_u[0]));
		//LM_in_temp.clear();
		//for (int i = 0; i < LM_in_number_index_u.size(); i++)
		//{
		//	LM_in_temp.push_back(LM_in[LM_in_number_index_u[i]-1]);
		//}
		//min_x=image_in.cols;min_y=image_in.rows;
		//max_x=0;max_y=0;
		//for (int i = 0; i < LM_in_temp.size(); i++)
		//{
		//	min_x=(min_x<LM_in_temp[i].x)?min_x:LM_in_temp[i].x;
		//	min_y=(min_y<LM_in_temp[i].y)?min_y:LM_in_temp[i].y;
		//	max_x=(max_x>LM_in_temp[i].x)?max_x:LM_in_temp[i].x;
		//	max_y=(max_y>LM_in_temp[i].y)?max_y:LM_in_temp[i].y;
		//}
		//min_x=min_x-bound;
		//min_y=min_y-bound;
		//max_x=max_x+bound;
		//max_y=max_y+bound;
		//image_in(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(test_c_u);

		//int LM_in_index_b[]={1,2,3,4,10,11,12};
		//vector<int> LM_in_number_index_b(LM_in_index_b, LM_in_index_b + sizeof(LM_in_index_b)/sizeof(LM_in_index_b[0]));
		//LM_in_temp.clear();
		//for (int i = 0; i < LM_in_number_index_b.size(); i++)
		//{
		//	LM_in_temp.push_back(LM_in[LM_in_number_index_b[i]-1]);
		//}
		//min_x=image_in.cols;min_y=image_in.rows;
		//max_x=0;max_y=0;
		//for (int i = 0; i < LM_in_temp.size(); i++)
		//{
		//	min_x=(min_x<LM_in_temp[i].x)?min_x:LM_in_temp[i].x;
		//	min_y=(min_y<LM_in_temp[i].y)?min_y:LM_in_temp[i].y;
		//	max_x=(max_x>LM_in_temp[i].x)?max_x:LM_in_temp[i].x;
		//	max_y=(max_y>LM_in_temp[i].y)?max_y:LM_in_temp[i].y;
		//}
		//min_x=min_x-bound;
		//min_y=min_y-bound;
		//max_x=max_x+bound;
		//max_y=max_y+bound;
		//image_in(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(test_c_b);
	}
	if (p_angle == 60 || p_angle == 75 || p_angle == 90)
	{
		int LM_in_index[]={1,2,3,4,5,6,7,8,9,10,11,12,15,16,17};
		//int LM_in_index[]={1,2,3,4,5,6,7,8,9,10,11,12};
		//int LM_in_index[]={1,2,3,4,5,6,7,8,9};
		vector<int> LM_in_number_index(LM_in_index, LM_in_index + sizeof(LM_in_index)/sizeof(LM_in_index[0]));
		vector<Point2f> LM_in_temp;
		for (int i = 0; i < LM_in_number_index.size(); i++)
		{
			LM_in_temp.push_back(LM_in[LM_in_number_index[i]-1]);
		}

		for (int i = 0; i < LM_in_temp.size(); i++)
		{
			min_x=(min_x<LM_in_temp[i].x)?min_x:LM_in_temp[i].x;
			min_y=(min_y<LM_in_temp[i].y)?min_y:LM_in_temp[i].y;
			max_x=(max_x>LM_in_temp[i].x)?max_x:LM_in_temp[i].x;
			max_y=(max_y>LM_in_temp[i].y)?max_y:LM_in_temp[i].y;
		}

		min_x=min_x-bound;
		min_y=min_y-bound;
		max_x=max_x+bound;
		max_y=max_y+bound;

		//Mat image_temp(cvRound(max_y-min_y),cvRound(max_x-min_x),CV_8UC3);
		image_in(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(image_temp);

		//int LM_in_index_u[]={1,2,3,4,5,6,7,8,9};
		//vector<int> LM_in_number_index_u(LM_in_index_u, LM_in_index_u + sizeof(LM_in_index_u)/sizeof(LM_in_index_u[0]));
		//LM_in_temp.clear();
		//for (int i = 0; i < LM_in_number_index_u.size(); i++)
		//{
		//	LM_in_temp.push_back(LM_in[LM_in_number_index_u[i]-1]);
		//}
		//min_x=image_in.cols;min_y=image_in.rows;
		//max_x=0;max_y=0;
		//for (int i = 0; i < LM_in_temp.size(); i++)
		//{
		//	min_x=(min_x<LM_in_temp[i].x)?min_x:LM_in_temp[i].x;
		//	min_y=(min_y<LM_in_temp[i].y)?min_y:LM_in_temp[i].y;
		//	max_x=(max_x>LM_in_temp[i].x)?max_x:LM_in_temp[i].x;
		//	max_y=(max_y>LM_in_temp[i].y)?max_y:LM_in_temp[i].y;
		//}
		//min_x=min_x-bound;
		//min_y=min_y-bound;
		//max_x=max_x+bound;
		//max_y=max_y+bound;
		//image_in(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(test_c_u);

		//int LM_in_index_b[]={1,2,3,4,10,11,12,15,16,17,18};
		//vector<int> LM_in_number_index_b(LM_in_index_b, LM_in_index_b + sizeof(LM_in_index_b)/sizeof(LM_in_index_b[0]));
		//LM_in_temp.clear();
		//for (int i = 0; i < LM_in_number_index_b.size(); i++)
		//{
		//	LM_in_temp.push_back(LM_in[LM_in_number_index_b[i]-1]);
		//}
		//min_x=image_in.cols;min_y=image_in.rows;
		//max_x=0;max_y=0;
		//for (int i = 0; i < LM_in_temp.size(); i++)
		//{
		//	min_x=(min_x<LM_in_temp[i].x)?min_x:LM_in_temp[i].x;
		//	min_y=(min_y<LM_in_temp[i].y)?min_y:LM_in_temp[i].y;
		//	max_x=(max_x>LM_in_temp[i].x)?max_x:LM_in_temp[i].x;
		//	max_y=(max_y>LM_in_temp[i].y)?max_y:LM_in_temp[i].y;
		//}
		//min_x=min_x-bound;
		//min_y=min_y-bound;
		//max_x=max_x+bound;
		//max_y=max_y+bound;
		//image_in(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(test_c_b);
	}

	//imshow("image_temp",image_temp);waitKey(0);

	*image_out=image_temp;
}
void crop_image_reg(float p_angle, Mat image_in, Mat testImg, vector<Point2f> LM_in, Mat* image_out)
{
	int min_x=image_in.cols;
	int min_y=image_in.rows;
	int max_x=0;
	int max_y=0;

	Mat image_temp;
	Mat image_serach;
	Mat resultImg;
	float bound=10.0;
	double minVal,maxVal; 
	Point minLoc,maxLoc;

	if (p_angle == -15 || p_angle == -30 || p_angle == 0 || p_angle == 30 || p_angle == 15)
	{
		//int LM_in_index[]={20,23,26,29,32,35,38,41,50,67,14,6,7,8,9,10};
		int LM_in_index[]={20,23,26,29,32,35,38,41,67,14,1,2,3,4,5,6,7,8,9,10,50,51,55};
		vector<int> LM_in_number_index(LM_in_index, LM_in_index + sizeof(LM_in_index)/sizeof(LM_in_index[0]));
		vector<Point2f> LM_in_temp;
		for (int i = 0; i < LM_in_number_index.size(); i++)
		{
			LM_in_temp.push_back(LM_in[LM_in_number_index[i]-1]);
		}

		for (int i = 0; i < LM_in_temp.size(); i++)
		{
			min_x=(min_x<LM_in_temp[i].x)?min_x:LM_in_temp[i].x;
			min_y=(min_y<LM_in_temp[i].y)?min_y:LM_in_temp[i].y;
			max_x=(max_x>LM_in_temp[i].x)?max_x:LM_in_temp[i].x;
			max_y=(max_y>LM_in_temp[i].y)?max_y:LM_in_temp[i].y;
		}

		min_x=min_x-bound;
		min_y=min_y-bound;
		max_x=max_x+bound;
		max_y=max_y+bound;

		//Mat image_temp(cvRound(max_y-min_y),cvRound(max_x-min_x),CV_8UC3);
		//image_in(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(image_temp);
		image_in(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(image_serach);
		matchTemplate(image_serach, testImg, resultImg, CV_TM_SQDIFF_NORMED);
		minMaxLoc(resultImg, &minVal, &maxVal, &minLoc, &maxLoc);
		//rectangle(image_serach, minLoc, Point(minLoc.x+testImg.cols , minLoc.y+testImg.rows), CV_RGB(255,0,0), 3);
		//imshow("image_serach",image_serach);
		//imshow("testImg",testImg);waitKey(0);
		image_serach(Rect(minLoc.x,minLoc.y,testImg.cols,testImg.rows)).copyTo(image_temp);
	}
	if (p_angle == -45)
	{
		//int LM_in_index[]={20,23,26,29,32,35,38,41,50,67,14,6,7,8,9,10};
		//int LM_in_index[]={20,23,26,29,32,35,38,41,67,14,6,7,8,9,10};
		int LM_in_index[]={20,23,26,29,32,35,38,41,67,14,6,7,8,9,10,50,51,52,55,56};
		vector<int> LM_in_number_index(LM_in_index, LM_in_index + sizeof(LM_in_index)/sizeof(LM_in_index[0]));
		vector<Point2f> LM_in_temp;
		for (int i = 0; i < LM_in_number_index.size(); i++)
		{
			LM_in_temp.push_back(LM_in[LM_in_number_index[i]-1]);
		}

		for (int i = 0; i < LM_in_temp.size(); i++)
		{
			min_x=(min_x<LM_in_temp[i].x)?min_x:LM_in_temp[i].x;
			min_y=(min_y<LM_in_temp[i].y)?min_y:LM_in_temp[i].y;
			max_x=(max_x>LM_in_temp[i].x)?max_x:LM_in_temp[i].x;
			max_y=(max_y>LM_in_temp[i].y)?max_y:LM_in_temp[i].y;
		}

		min_x=min_x-bound;
		min_y=min_y-bound;
		max_x=max_x+bound;
		max_y=max_y+bound;

		//Mat image_temp(cvRound(max_y-min_y),cvRound(max_x-min_x),CV_8UC3);
		//image_in(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(image_temp);
		image_in(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(image_serach);
		matchTemplate(image_serach, testImg, resultImg, CV_TM_SQDIFF_NORMED);
		minMaxLoc(resultImg, &minVal, &maxVal, &minLoc, &maxLoc);
		//rectangle(image_serach, minLoc, Point(minLoc.x+testImg.cols , minLoc.y+testImg.rows), CV_RGB(255,0,0), 3);
		//imshow("image_serach",image_serach);
		//imshow("testImg",testImg);waitKey(0);
		image_serach(Rect(minLoc.x,minLoc.y,testImg.cols,testImg.rows)).copyTo(image_temp);

	}
	if (p_angle == 45)
	{
		//int LM_in_index[]={20,23,26,29,32,35,38,41,50,67,14,6,7,8,9,10};
		int LM_in_index[]={20,23,26,29,32,35,38,41,67,14,1,2,3,4,5,50,51,52,55,56};
		vector<int> LM_in_number_index(LM_in_index, LM_in_index + sizeof(LM_in_index)/sizeof(LM_in_index[0]));
		vector<Point2f> LM_in_temp;
		for (int i = 0; i < LM_in_number_index.size(); i++)
		{
			LM_in_temp.push_back(LM_in[LM_in_number_index[i]-1]);
		}

		for (int i = 0; i < LM_in_temp.size(); i++)
		{
			min_x=(min_x<LM_in_temp[i].x)?min_x:LM_in_temp[i].x;
			min_y=(min_y<LM_in_temp[i].y)?min_y:LM_in_temp[i].y;
			max_x=(max_x>LM_in_temp[i].x)?max_x:LM_in_temp[i].x;
			max_y=(max_y>LM_in_temp[i].y)?max_y:LM_in_temp[i].y;
		}

		min_x=min_x-bound;
		min_y=min_y-bound;
		max_x=max_x+bound;
		max_y=max_y+bound;

		//Mat image_temp(cvRound(max_y-min_y),cvRound(max_x-min_x),CV_8UC3);
		//image_in(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(image_temp);
		image_in(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(image_serach);
		matchTemplate(image_serach, testImg, resultImg, CV_TM_SQDIFF_NORMED);
		minMaxLoc(resultImg, &minVal, &maxVal, &minLoc, &maxLoc);
		//rectangle(image_serach, minLoc, Point(minLoc.x+testImg.cols , minLoc.y+testImg.rows), CV_RGB(255,0,0), 3);
		//imshow("image_serach",image_serach);
		//imshow("testImg",testImg);waitKey(0);
		image_serach(Rect(minLoc.x,minLoc.y,testImg.cols,testImg.rows)).copyTo(image_temp);
	}
	if (p_angle == -60 || p_angle == -75 || p_angle == -90)
	{
		int LM_in_index[]={1,2,3,4,5,6,7,8,9,10,11,12,15,16,17};
		//int LM_in_index[]={1,2,3,4,5,6,7,8,9,10,11,12};
		//int LM_in_index[]={1,2,3,4,5,6,7,8,9};
		vector<int> LM_in_number_index(LM_in_index, LM_in_index + sizeof(LM_in_index)/sizeof(LM_in_index[0]));
		vector<Point2f> LM_in_temp;
		for (int i = 0; i < LM_in_number_index.size(); i++)
		{
			LM_in_temp.push_back(LM_in[LM_in_number_index[i]-1]);
		}

		for (int i = 0; i < LM_in_temp.size(); i++)
		{
			min_x=(min_x<LM_in_temp[i].x)?min_x:LM_in_temp[i].x;
			min_y=(min_y<LM_in_temp[i].y)?min_y:LM_in_temp[i].y;
			max_x=(max_x>LM_in_temp[i].x)?max_x:LM_in_temp[i].x;
			max_y=(max_y>LM_in_temp[i].y)?max_y:LM_in_temp[i].y;
		}

		min_x=min_x-bound;
		min_y=min_y-bound;
		max_x=max_x+bound;
		max_y=max_y+bound;

		//Mat image_temp(cvRound(max_y-min_y),cvRound(max_x-min_x),CV_8UC3);
		//image_in(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(image_temp);
		image_in(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(image_serach);
		matchTemplate(image_serach, testImg, resultImg, CV_TM_SQDIFF_NORMED);
		minMaxLoc(resultImg, &minVal, &maxVal, &minLoc, &maxLoc);
		//rectangle(image_serach, minLoc, Point(minLoc.x+testImg.cols , minLoc.y+testImg.rows), CV_RGB(255,0,0), 3);
		//imshow("image_serach",image_serach);
		//imshow("testImg",testImg);waitKey(0);
		image_serach(Rect(minLoc.x,minLoc.y,testImg.cols,testImg.rows)).copyTo(image_temp);
		//matchTemplate(image_serach, test_c_u, resultImg, CV_TM_SQDIFF_NORMED);
		//minMaxLoc(resultImg, &minVal, &maxVal, &minLoc, &maxLoc);
		//image_serach(Rect(minLoc.x,minLoc.y,test_c_u.cols,test_c_u.rows)).copyTo(reg_c_u);
		//matchTemplate(image_serach, test_c_b, resultImg, CV_TM_SQDIFF_NORMED);
		//minMaxLoc(resultImg, &minVal, &maxVal, &minLoc, &maxLoc);
		//image_serach(Rect(minLoc.x,minLoc.y,test_c_b.cols,test_c_b.rows)).copyTo(reg_c_b);
	}
	if (p_angle == 60 || p_angle == 75 || p_angle == 90)
	{
		int LM_in_index[]={1,2,3,4,5,6,7,8,9,10,11,12,15,16,17};
		//int LM_in_index[]={1,2,3,4,5,6,7,8,9,10,11,12};
		//int LM_in_index[]={1,2,3,4,5,6,7,8,9};
		vector<int> LM_in_number_index(LM_in_index, LM_in_index + sizeof(LM_in_index)/sizeof(LM_in_index[0]));
		vector<Point2f> LM_in_temp;
		for (int i = 0; i < LM_in_number_index.size(); i++)
		{
			LM_in_temp.push_back(LM_in[LM_in_number_index[i]-1]);
		}

		for (int i = 0; i < LM_in_temp.size(); i++)
		{
			min_x=(min_x<LM_in_temp[i].x)?min_x:LM_in_temp[i].x;
			min_y=(min_y<LM_in_temp[i].y)?min_y:LM_in_temp[i].y;
			max_x=(max_x>LM_in_temp[i].x)?max_x:LM_in_temp[i].x;
			max_y=(max_y>LM_in_temp[i].y)?max_y:LM_in_temp[i].y;
		}

		min_x=min_x-bound;
		min_y=min_y-bound;
		max_x=max_x+bound;
		max_y=max_y+bound;

		//Mat image_temp(cvRound(max_y-min_y),cvRound(max_x-min_x),CV_8UC3);
		//image_in(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(image_temp);
		image_in(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(image_serach);
		matchTemplate(image_serach, testImg, resultImg, CV_TM_SQDIFF_NORMED);
		minMaxLoc(resultImg, &minVal, &maxVal, &minLoc, &maxLoc);
		//rectangle(image_serach, minLoc, Point(minLoc.x+testImg.cols , minLoc.y+testImg.rows), CV_RGB(255,0,0), 3);
		//imshow("image_serach",image_serach);
		//imshow("testImg",testImg);waitKey(0);
		image_serach(Rect(minLoc.x,minLoc.y,testImg.cols,testImg.rows)).copyTo(image_temp);
		//matchTemplate(image_serach, test_c_u, resultImg, CV_TM_SQDIFF_NORMED);
		//minMaxLoc(resultImg, &minVal, &maxVal, &minLoc, &maxLoc);
		//image_serach(Rect(minLoc.x,minLoc.y,test_c_u.cols,test_c_u.rows)).copyTo(reg_c_u);
		//matchTemplate(image_serach, test_c_b, resultImg, CV_TM_SQDIFF_NORMED);
		//minMaxLoc(resultImg, &minVal, &maxVal, &minLoc, &maxLoc);
		//image_serach(Rect(minLoc.x,minLoc.y,test_c_b.cols,test_c_b.rows)).copyTo(reg_c_b);
	}

	//imshow("image_temp",image_temp);waitKey(0);

	*image_out=image_temp;
}

// scale issue //
void match_scale(Mat testImg, Mat model_Image)
{
	imshow("testImg",testImg);
	imshow("model_Image",model_Image);
	waitKey(1);

	Mat resultImg;
	matchTemplate(model_Image, testImg, resultImg, CV_TM_SQDIFF_NORMED);
	imshow("resultImg",resultImg);
	waitKey(0);

	double minVal,maxVal; 
	Point minLoc,maxLoc;
	minMaxLoc(resultImg, &minVal, &maxVal, &minLoc, &maxLoc);
	cout<<minVal<<endl;
	cout<<minLoc<<endl;
	cout<<maxVal<<endl;
	cout<<maxLoc<<endl;
	rectangle(model_Image, minLoc, Point(minLoc.x+testImg.cols , minLoc.y+testImg.rows), CV_RGB(255,0,0), 3);
	rectangle(model_Image, maxLoc, Point(maxLoc.x+testImg.cols , maxLoc.y+testImg.rows), CV_RGB(0,0,255), 3);
	imshow("model_Image",model_Image);
	waitKey(0);

}
void fix_scale(float ori_scale, Mat oriImg, vector<Point2f> oriLM, float angle, Mat templte,  Mat* Img_out, vector<Point2f>* LM_out)
{
	float p_angle=angle;
	//float bound=30.0;
	vector<Point2f> oriLM_temp;
	vector<Point2f> LM_in_temp;

	//imshow("oriImg",oriImg);
	//imshow("templte",templte);waitKey(1);

	double score=1000.0;
	float scale_f;
	Mat img_f;
	vector<Point2f> LM_f;

	//cout<<ori_scale<<endl;
	for (float i = ori_scale-0.02; i <= ori_scale+0.02; i=i+0.01)
	{
		//cout<<"scale "<<i<<endl;
		Mat image_serach;
		Mat resultImg;
		double minVal,maxVal; 
		Point minLoc,maxLoc;

		Mat scaledImg;
		cv::resize(oriImg,scaledImg,Size(oriImg.cols*i,oriImg.rows*i)); //normlized image
		scale_LM(oriLM,&oriLM_temp, i); //縮放移動後的LM
		//imshow("scaledImg",scaledImg);waitKey(0);

		int min_x=scaledImg.cols;
		int min_y=scaledImg.rows;
		int max_x=0;
		int max_y=0;
		if (p_angle == -15 || p_angle == -30 || p_angle == 0 || p_angle == 30 || p_angle == 15)
		{
			int LM_in_index[]={1,2,3,4,5,6,7,8,9,10,11,12};
			vector<int> LM_in_number_index(LM_in_index, LM_in_index + sizeof(LM_in_index)/sizeof(LM_in_index[0]));
			//vector<Point2f> LM_in_temp;
			for (int i = 0; i < LM_in_number_index.size(); i++)
			{
				LM_in_temp.push_back(oriLM_temp[LM_in_number_index[i]-1]);
			}
		}
		if (p_angle == -45)
		{
			int LM_in_index[]={1,2,3,4,7,8,10,11,12};
			vector<int> LM_in_number_index(LM_in_index, LM_in_index + sizeof(LM_in_index)/sizeof(LM_in_index[0]));
			//vector<Point2f> LM_in_temp;
			for (int i = 0; i < LM_in_number_index.size(); i++)
			{
				LM_in_temp.push_back(oriLM_temp[LM_in_number_index[i]-1]);
			}
		}
		if (p_angle == 45)
		{
			int LM_in_index[]={1,2,3,4,5,6,9,10,12};
			vector<int> LM_in_number_index(LM_in_index, LM_in_index + sizeof(LM_in_index)/sizeof(LM_in_index[0]));
			//vector<Point2f> LM_in_temp;
			for (int i = 0; i < LM_in_number_index.size(); i++)
			{
				LM_in_temp.push_back(oriLM_temp[LM_in_number_index[i]-1]);
			}
		}
		if (p_angle == -60 || p_angle == -75 || p_angle == -90)
		{
			int LM_in_index[]={1,2,3,4,7,8,10,11,12};
			vector<int> LM_in_number_index(LM_in_index, LM_in_index + sizeof(LM_in_index)/sizeof(LM_in_index[0]));
			//vector<Point2f> LM_in_temp;
			for (int i = 0; i < LM_in_number_index.size(); i++)
			{
				LM_in_temp.push_back(oriLM_temp[LM_in_number_index[i]-1]);
			}
		}
		if (p_angle == 60 || p_angle == 75 || p_angle == 90)
		{
			int LM_in_index[]={1,2,3,4,7,8,10,11,12};
			vector<int> LM_in_number_index(LM_in_index, LM_in_index + sizeof(LM_in_index)/sizeof(LM_in_index[0]));
			//vector<Point2f> LM_in_temp;
			for (int i = 0; i < LM_in_number_index.size(); i++)
			{
				LM_in_temp.push_back(oriLM_temp[LM_in_number_index[i]-1]);
			}
		}

		for (int i = 0; i < LM_in_temp.size(); i++)
		{
			min_x=(min_x<LM_in_temp[i].x)?min_x:LM_in_temp[i].x;
			min_y=(min_y<LM_in_temp[i].y)?min_y:LM_in_temp[i].y;
			max_x=(max_x>LM_in_temp[i].x)?max_x:LM_in_temp[i].x;
			max_y=(max_y>LM_in_temp[i].y)?max_y:LM_in_temp[i].y;
		}

		//cout<<min_x<<" "<<min_y<<" "<<max_x<<" "<<max_y<<endl;

		float bound=5.0;
		float bound_X=0.0;
		int compare_x=cvRound(max_x-min_x);
		while (templte.cols>compare_x)
		{
			compare_x=compare_x+bound;
			bound_X=bound_X+bound;
		}
		float bound_Y=0.0;
		int compare_y=cvRound(max_y-min_y);
		while (templte.rows>compare_y)
		{
			compare_y=compare_y+bound;
			bound_Y=bound_Y+bound;
		}

		//bound_X=cvRound(templte.cols-cvRound(max_x-min_x));
		//bound_Y=cvRound(templte.rows-cvRound(max_y-min_y));
		//cout<<bound_X<<" "<<bound_Y<<endl;
		min_x=min_x-bound_X;
		min_y=min_y-bound_Y;
		max_x=max_x+bound_X;
		max_y=max_y+bound_Y;

		//cout<<min_x<<" "<<min_y<<" "<<max_x<<" "<<max_y<<endl;
		//imshow("scaledImg",scaledImg);waitKey(1);
		scaledImg(Rect(min_x,min_y,cvRound(max_x-min_x),cvRound(max_y-min_y))).copyTo(image_serach);
		//imshow("image_serach",image_serach);waitKey(0);
		//resultImg.release();
		cv::matchTemplate(image_serach, templte, resultImg, CV_TM_SQDIFF_NORMED);
		//cv::matchTemplate(image_serach, templte, resultImg, CV_TM_CCOEFF_NORMED);
		//normalize( resultImg, resultImg, 0, 1, NORM_MINMAX, -1, Mat() );
		cv::minMaxLoc(resultImg, &minVal, &maxVal, &minLoc, &maxLoc);
		//cout<<"Min value "<<minVal<<endl;
		//cout<<"Max value "<<maxVal<<endl;
		//normalize( resultImg, resultImg, 0, 1, NORM_MINMAX, -1, Mat() );
		//imshow("resultImg",resultImg);waitKey(1);
		//rectangle(image_serach, minLoc, Point(minLoc.x+templte.cols , minLoc.y+templte.rows), CV_RGB(255,0,0), 3);
		//rectangle(image_serach, maxLoc, Point(maxLoc.x+templte.cols , maxLoc.y+templte.rows), CV_RGB(255,0,0), 3);
		//imshow("image_serach",image_serach);waitKey(0);

		scale_f=(minVal<score)?i:scale_f;
		img_f=(minVal<score)?scaledImg:img_f;
		LM_f=(minVal<score)?oriLM_temp:LM_f;
		score=(minVal<score)?minVal:score;
	}

	*Img_out=img_f;
	*LM_out=LM_f;
}