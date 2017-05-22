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

// Chehra landmark // 臉部內輪廓
#define use_Chehra 1
// Lab face landmark // 臉部外輪廓
#define use_Lab_LM_model 1

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
Mat model_x_d,model_y_d,model_z_d;
Mat mask_d,Reg_img_d;
void Initialize(int argc, char* argv[]);

void RenderFunction(void);
void Draw_Model(Mat mask, Mat model_x, Mat model_y, Mat model_z, Mat img);
void matrix_set(Mat* view,Mat* project);
void model_gravityPoint(Mat mask, Mat model_x, Mat model_y, Mat model_z, Mat *move_matrix);

//////////////////////////////////////////
///*          Sub function            *///
//////////////////////////////////////////
// 前處理 //
void xml_read_name(string Path_name,vector<string> *image_name);
void xml_read_pts(string Path_name,vector<string> image_name,vector<vector<Point2f>> *model_LM_all);
void xml_read_name_MPIE(string Path_name,vector<string> *image_name);
void xml_read_pts_MPIE(string Path_name,vector<string> image_name,vector<vector<Point2f>> *model_LM_all);
void FRGC_img_show(string FRGC_model_image_path,vector<string> FRGC_image_name,vector<vector<Point2f>> model_LM_all);
void MPIE_img_show(string MPIE_file_image_path,vector<string> MPIE_image_name,vector<vector<Point2f>> MPIE_LM_all);

// 模型選擇 //
void model_choose_LMset(vector<vector<Point2f>> LM_in, vector<vector<Point2f>>* LM_out);
void draw_point(vector<vector<Point2f>> LM_in);
void model_choose(vector<vector<Point2f>> model_LM_all,vector<Point2f> MPIE_LM_each,vector<int>* choose_num, int* min_num);
void angle_calculate(vector<Point2f> LM_in, vector<float> *angle_out);
void draw_point_afterModelChoose(vector<Point2f> FRGC_1,vector<Point2f> FRGC_2,vector<Point2f> MPIE);

// 模型變形 //
void model_transform(vector<Point2f> model_LM, vector<Point2f> img_LM, Mat input_img, string FRGC_model_image_path, string FRGC_image_name, Mat ViewMatrix, Mat ProjMatrix, Mat *model_x_out, Mat *model_y_out, Mat *model_z_out, Mat *mask_out, Mat *img_out,vector<Point2f> *Reg_LM_out);
void model_transform_LMset(vector<Point2f> LM_in, vector<Point2f> *LM_out, Mat ViewMatrix, Mat ProjMatrix);
void volume_spline_interpolation(vector<Point2f> src, vector<Point2f> dst, Mat *coefficient, float *k_num);
float radial_basis_Fun(vector<Point2f> LM, int row_s, int col_s, float k);
float radial_basis_Fun_check(vector<Point2f> src, int row_s, int col_s, float k);
void model_transform_cood(Mat mask, Mat model_x, Mat model_y, Mat *model_x_out, Mat *model_y_out, vector<Point2f> src, Mat coefficient, float k_num);
float trans_cood(Mat model_x,Mat model_y, vector<Point2f> src, Mat coefficient, int row_s, int col_s, int dims, float k_num);
float radial_basis_Fun2(Mat model_x, Mat model_y,vector<Point2f> LM, int row_s, int col_s, int nums, float k_num);
void Create_face_mask(vector<Point2f> input_point, Mat &out_mask);
void Create_Leye_mask(vector<Point2f> input_point, Mat &out_mask);
void Create_Reye_mask(vector<Point2f> input_point, Mat &out_mask);

// 模型重建 //
// Recovering Light //
void calculate_Light(Mat image_i,Mat albedo_ref,Mat normal_x,Mat normal_y,Mat normal_z, Mat affine_Z_mask, Mat* L);
// Recovering Depth //
cv::Mat fspecialLoG(int WinSize, double sigma);
void calculate_Depth(Mat image_i,Mat albedo_ref,Mat model_z, Mat affine_Z_mask, Mat L, Mat h,Mat model_x,Mat model_y,Mat* out_z);
void calculate_Depth_iter(Mat image_i,Mat albedo_ref,Mat model_z, Mat affine_Z_mask, Mat L, Mat h,Mat model_x,Mat model_y,Mat* out_z);
void calculate_albedo(Mat image_i, Mat albedo_ref,Mat model_z, Mat affine_Z_mask, Mat L, Mat h, Mat *albedo_out);
void calculate_albedo_iter(Mat image_i, Mat albedo_ref, Mat model_z, Mat affine_Z_mask, Mat L, Mat h, Mat *albedo_out);
void calculate_normal(Mat affine_Z_mask, Mat model_x, Mat model_y, Mat model_z,Mat *normal_x_out,Mat *normal_y_out,Mat *normal_z_out);

// 模型儲存 //
void write_model(string savePath, Mat msak, Mat img, Mat model_x, Mat model_y, Mat model_z);
void write_model_LM(string savePath, vector<Point2f> Reg_LM, Mat model_x, Mat model_y, Mat model_z);
void write_model_old(string savePath, Mat msak, Mat img, Mat model_x, Mat model_y, Mat model_z);
void write_model_LM_old(string savePath, vector<Point2f> Reg_LM, Mat model_x, Mat model_y, Mat model_z);

// 舊方法 //
// warp sub function //
void set_persp_aff(vector<cv::Point2f> ref_pt, vector<vector<cv::Point2f>>* out_pt, string name, int* affine_num);
void aff_warp(vector<cv::Point2f> src_pnt, vector<cv::Point2f> dst_pnt,cv::Mat img_in,cv::Mat *img_out,cv::Mat *mask_out);

string data_file_title="../../using_data_utech/";//工作路徑
float lambda_record=0.0;
float pic=0.0;
int main(int argc, char* argv[])
{
	// set Chehra LM //
	// 設置 Chehra LM 使用時的參數 //
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
	string regFile_str=data_file_title+"Chehra-model/Chehra_t1.0.model";
	string fdFile_str=data_file_title+"Chehra-model/haarcascade_frontalface_alt_tree.xml";
	//strcpy(regFile,"../../using_data/Chehra-model/Chehra_t1.0.model");
	//strcpy(fdFile,"../../using_data/Chehra-model/haarcascade_frontalface_alt_tree.xml");
	strcpy(regFile,regFile_str.c_str());
	strcpy(fdFile,fdFile_str.c_str());

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

	/*選擇需要的 model
	AM: asian man
	AF: asian female
	CM: occident man
	CF: occident female
	*/
	string FRGC_model_file_name="FRGC-model-AM";		//模型種類
	string FRGC_image_name="04461d294";					//模型名稱
	string FRGC_model_image_path = data_file_title+FRGC_model_file_name+"/";		//模型種類路徑
	string FRGC_path=FRGC_model_image_path+FRGC_image_name+"/"+FRGC_image_name+"-ori.ppm";		//種類路徑內的之特定檔案路徑
	Mat FRGC_image=imread(FRGC_path);					//讀取特定檔案的路徑
	Mat FRGC_image_d;									//FRGC_image_d 供後續儲存灰階圖
	Mat FRGC_image_t;
	Mat FRGC_image_D=FRGC_image.clone();
	cvtColor(FRGC_image,FRGC_image_d,CV_BGR2GRAY);		//將讀入的圖片轉為灰階存入 FRGC_image_d

	ChehraObj.Reinitialize();							//初始化 ChehraObj (因為此為偵測影像用, 而我們是輸入圖片, 若重複執行時下張圖片無連續性)
	vector<Point2f> model_LM_each;
	if (ChehraObj.TrackFrame(FRGC_image_d,fcheck_interval,fcheck_score_treshold,fcheck_fail_treshold,face_cascade) == 0)
	{
		Chehra_Plot(FRGC_image,ChehraObj._bestFaceShape,ChehraObj._bestEyeShape,&model_LM_each);
		// LM detect progess //
#if use_Lab_LM_model
		vector<Data_bs>bs;vector<Data_bs> top;int point;
		float scale_LM_img=1.0;
		resize(FRGC_image,FRGC_image_t, Size(cvRound(scale_LM_img*FRGC_image.cols),cvRound(scale_LM_img*FRGC_image.rows)), scale_LM_img, scale_LM_img ,INTER_LINEAR );
		char *modelname_Char = new char[face_LM_model_path.length() + 1];
		strcpy(modelname_Char, face_LM_model_path.c_str());
		detector.detect(bs,FRGC_image_t ,detector.model_ ,modelname_Char);
		delete [] modelname_Char;
		detector.clipboxes(FRGC_image_t.rows,FRGC_image_t.cols,bs);
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
	}
	//在原始模型圖上畫出所有 landmark 點
	for (vector<Point2f>::iterator LM_pt=model_LM_each.begin(); LM_pt!=model_LM_each.end(); ++LM_pt)
	{
		circle(FRGC_image_D,*LM_pt,1,CV_RGB(0,255,0),-1);
	}
	imshow("FRGC_image_final",FRGC_image_D);
	waitKey(1);

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

	int inputKey;
	bool p_flag = false; //pick the model//

	while(1)					//主迴圈
	{
		Mat test_img;
		Mat test_img_d;
		Mat test_img_t;
		src.retrieve(test_img);
		Mat test_img_D=test_img.clone();
		cvtColor(test_img,test_img_d,CV_BGR2GRAY);
		ChehraObj.Reinitialize();
		vector<Point2f> test_LM_each;
		if (ChehraObj.TrackFrame(test_img_d,fcheck_interval,fcheck_score_treshold,fcheck_fail_treshold,face_cascade) == 0)
		{
			Chehra_Plot(test_img,ChehraObj._bestFaceShape,ChehraObj._bestEyeShape,&test_LM_each);
			// LM detect progess //
#if use_Lab_LM_model
			vector<Data_bs>bs;vector<Data_bs> top;int point;
			float scale_LM_img=0.4;
			resize(test_img,test_img_t, Size(cvRound(scale_LM_img*test_img.cols),cvRound(scale_LM_img*test_img.rows)), scale_LM_img, scale_LM_img ,INTER_LINEAR );
			char *modelname_Char = new char[face_LM_model_path.length() + 1];
			strcpy(modelname_Char, face_LM_model_path.c_str());
			detector.detect(bs,test_img_t ,detector.model_ ,modelname_Char);
			delete [] modelname_Char;
			detector.clipboxes(test_img_t.rows,test_img_t.cols,bs);
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
					test_LM_each.push_back(cvPoint( cvRound(((top[0].xy[n][0]+top[0].xy[n][2])/2)/scale_LM_img) ,cvRound(((top[0].xy[n][1]+top[0].xy[n][3])/2)/scale_LM_img)));
				}
				//Set_the_LM(model_LM_each,&model_LM_each);
			}
			bs.clear();
			top.clear();
			vector<Data_bs>().swap(bs);
			vector<Data_bs>().swap(top);
#endif
		}
		//在拍攝之測試圖像 landmark 點上畫圈
		for (vector<Point2f>::iterator LM_pt=test_LM_each.begin();LM_pt!=test_LM_each.end();++LM_pt)
		{
			circle(test_img_D,*LM_pt,1,CV_RGB(0,255,0),-1);
		}
		imshow("test_img_D",test_img_D);
		inputKey=waitKey(1);


		if (p_flag)			//按下 p 開始進行 3D重建
		{
			p_flag = !p_flag;
			pic=pic+1;					//計算本次執行至此圖形重建數量

			//重建//
			// OpenGL 投影矩陣設置 //
			//Initialize(argc, argv);// 初始化 glut //
			Mat ViewMatrix;
			Mat ProjMatrix;
			matrix_set(&ViewMatrix,&ProjMatrix);
			project_matrix_t=ProjMatrix.t();

			// 模型變形 // & // 模型重建 //
			vector<Point2f> Reg_LM_out;
			model_transform(model_LM_each, test_LM_each, test_img, FRGC_model_image_path, FRGC_image_name, ViewMatrix, ProjMatrix, &model_x_d, &model_y_d, &model_z_d, &mask_d, &Reg_img_d, &Reg_LM_out);
			
			Mat move_matrix,move_matrix_inv;
			model_gravityPoint(mask_d, model_x_d, model_y_d, model_z_d, &move_matrix);
			move_to_ori_T=move_matrix.t();
			move_matrix_inv=move_matrix.inv();
			move_to_ori_T_inv=move_matrix_inv.t();

			cout<<"reconstruction finish !"<<endl;

			// 模型儲存 // 
			//lambda_record=2.0;
			string lambda_str;
			stringstream float2string; //float to string
			float2string << lambda_record;
			float2string >> lambda_str;
			string pic_no;
			stringstream float2string2; //float to string
			float2string2 << pic;
			float2string2 >> pic_no;

			string save_model_title=data_file_title+"model_rec_test/"+pic_no+"/";			  //輸出檔案路徑
			//string save_model_title=data_file_title+"model_rec_test/FRGC_2_FRGC/"+pic_no+"/";
			//string save_model_title=data_file_title+"model_rec_test/MPIE_2_FRGC/"+pic_no+"/";
			_mkdir(save_model_title.c_str());

			string save_depth_w = save_model_title +pic_no+"_lambda_"+lambda_str+".ply";	       //輸出 3D 模型
			write_model_old(save_depth_w, mask_d, Reg_img_d, model_x_d, model_y_d, model_z_d);

			string save_depth_l = save_model_title +pic_no+"_lambda_"+lambda_str+"_lm.ply";			//
			write_model_LM_old(save_depth_l, Reg_LM_out, model_x_d, model_y_d, model_z_d);

			string save_depth_m = save_model_title +pic_no+"_lambda_"+lambda_str+"_m.png";			//輸出 mask 的圖像
			imwrite(save_depth_m,mask_d);

			string save_depth_i = save_model_title +pic_no+"_lambda_"+lambda_str+"_i.png";			//輸出測試圖像含 landmark
			imwrite(save_depth_i,test_img_D);

			string save_depth_f = save_model_title +pic_no+"_lambda_"+lambda_str+"_f.png";			//輸出模型圖像含 landmark
			imwrite(save_depth_f,FRGC_image_D);

			float2string.str(""); //再次使用前須請空內容
			float2string.clear(); //再次使用前須請空內容yLeaks();
			float2string2.str(""); //再次使用前須請空內容
			float2string2.clear(); //再次使用前須請空內容yLeaks();
			cout<<"save model finish !"<<endl;
			destroyAllWindows();
		}

		if(inputKey == VK_ESCAPE) {break;}
		else if(char(inputKey) == 'p' || char(inputKey) == 'P') {p_flag = !p_flag;} 
	}

	
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
	glutInitWindowSize(CurrentWidth, CurrentHeight);
	glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA);
	
	//根據已設定好的 glut (如尺寸,color,depth) 向window要求建立一個視窗，接著若失敗則退出程式
	WindowHandle = glutCreateWindow("MPIE Data Example");
	if(WindowHandle < 1) {	fprintf(stderr,"ERROR: Could not create a new rendering window.\n");exit(EXIT_FAILURE);	}
	
	//glutReshapeFunc(ResizeFunction); //設定視窗 大小若改變，則跳到"ResizeFunction"這個函數處理應變事項
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
	//glRotatef(roangles,0,1,0);
	//glScalef(0.5,0.5,0.5);
	glMultMatrixf((float*)move_to_ori_T.data);
	//glCallLists(1,GL_UNSIGNED_BYTE,&myLists[Registration_num_count]);
	Draw_Model(mask_d, model_x_d, model_y_d, model_z_d, Reg_img_d);
	glPopMatrix();

	Mat final_pic(480, 640, CV_8UC3);
	Mat final_pic_flip(480, 640, CV_8UC3);
	glReadPixels(0,0,CurrentWidth,CurrentHeight,GL_BGR,GL_UNSIGNED_BYTE,final_pic.data);
	flip(final_pic,final_pic_flip,0);
	imshow("final_pic_flip",final_pic_flip);	
	waitKey(1);

	glFlush();
	glutSwapBuffers();
}
void Draw_Model(Mat mask, Mat model_x, Mat model_y, Mat model_z, Mat img)
{
	glPointSize(2);
	glEnable(GL_DEPTH_TEST);
	glBegin(GL_POINTS);
	for (int i=0;i<mask.rows;i++)
	{
		for (int j=0;j<mask.cols;j++)
		{
			if (mask.at<uchar>(i,j)!=0)
			{
				glColor3f(img.at<Vec3b>(i,j)[2]/255.0,img.at<Vec3b>(i,j)[1]/255.0,img.at<Vec3b>(i,j)[0]/255.0);
				glVertex3f(model_x.at<float>(i,j),model_y.at<float>(i,j),model_z.at<float>(i,j));
			}
		}
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

	float Zmax=-5000.0;
	float Zmin=5000.0;
	float F_ten=2/(Zmax-Zmin);
	float F_fourTeen=-(Zmax+Zmin)/(Zmax-Zmin);
	Mat project_matrix(4,4,CV_32FC1);
	project_matrix.at<float>(0,0)=0.0054;  
	project_matrix.at<float>(0,1)=-1.9066*pow(10.0,-6.0);  
	project_matrix.at<float>(0,2)=5.9645*pow(10,-7.0);  
	project_matrix.at<float>(0,3)=0.0014;
	project_matrix.at<float>(1,0)=7.9908*pow(10.0,-6.0);
	project_matrix.at<float>(1,1)=0.0074;
	project_matrix.at<float>(1,2)=-2.0505*pow(10,-4.0);
	project_matrix.at<float>(1,3)=-0.3964;
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

//////////////////////////////////////////
///*          Sub function            *///
//////////////////////////////////////////
void xml_read_name(string Path_name,vector<string> *image_name)
{
	vector<string> temp_image_name; 
	FileStorage FS_NT_R;
	FS_NT_R.open(Path_name, FileStorage::READ);
	FS_NT_R["FRGC_Name_Tar"] >> temp_image_name;
	FS_NT_R.release();
	FileStorage FS_LDT_R;
	*image_name=temp_image_name;
}
void xml_read_pts(string Path_name,vector<string> image_name,vector<vector<Point2f>> *model_LM_all)
{
	vector<vector<Point2f>> temp_model_LM_all;
	FileStorage FS_LDT_R;
	FS_LDT_R.open(Path_name, FileStorage::READ);
	for (int i=0;i<image_name.size();i++)
	{
		vector<Point2f> temp;
		stringstream int2str;
		string num;
		int2str << i;
		int2str >> num;
		string label="FRGC_LMPT_Data_Tar_"+num;
		FS_LDT_R[label] >> temp;
		temp_model_LM_all.push_back(temp);
	}
	FS_LDT_R.release();
	*model_LM_all=temp_model_LM_all;
}
void xml_read_name_MPIE(string Path_name,vector<string> *image_name)
{
	vector<string> temp_image_name; 
	FileStorage FS_NT_R;
	FS_NT_R.open(Path_name, FileStorage::READ);
	FS_NT_R["MPIE_Name_Tar"] >> temp_image_name;
	FS_NT_R.release();
	FileStorage FS_LDT_R;
	*image_name=temp_image_name;
}
void xml_read_pts_MPIE(string Path_name,vector<string> image_name,vector<vector<Point2f>> *model_LM_all)
{
	vector<vector<Point2f>> temp_model_LM_all;
	FileStorage FS_LDT_R;
	FS_LDT_R.open(Path_name, FileStorage::READ);
	for (int i=0;i<image_name.size();i++)
	{
		vector<Point2f> temp;
		stringstream int2str;
		string num;
		int2str << i;
		int2str >> num;
		string label="MPIE_LMPT_Data_Tar_"+num;
		FS_LDT_R[label] >> temp;
		temp_model_LM_all.push_back(temp);
	}
	FS_LDT_R.release();
	*model_LM_all=temp_model_LM_all;
}
void FRGC_img_show(string FRGC_model_image_path,vector<string> FRGC_image_name,vector<vector<Point2f>> model_LM_all)
{
	for (int i=0; i<FRGC_image_name.size();i++)
	{
		Mat model_ori_img;
		Mat model_ori_img_d;
		int inputKey;

		string img_load_path=FRGC_model_image_path+FRGC_image_name[i]+"/"+FRGC_image_name[i]+"-ori.ppm";
		model_ori_img=imread(img_load_path);
		model_ori_img.copyTo(model_ori_img_d);

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
		cv::imshow("model_ori_img_d",model_ori_img_d);inputKey=waitKey(0);
	}
}
void MPIE_img_show(string MPIE_file_image_path,vector<string> MPIE_image_name,vector<vector<Point2f>> MPIE_LM_all)
{
	Mat model_ori_img_mpie;
	Mat model_ori_img_d_mpie;
	int inputKey_mpie;
	for (int i=0; i<MPIE_image_name.size();i++)
	{
		string img_load_path=MPIE_file_image_path+MPIE_image_name[i];

		model_ori_img_mpie=imread(img_load_path);
		model_ori_img_mpie.copyTo(model_ori_img_d_mpie);

		int point_number=1;
		for (vector<Point2f>::iterator LM_pt=MPIE_LM_all[i].begin();LM_pt!=MPIE_LM_all[i].end();++LM_pt)
		{
			circle(model_ori_img_d_mpie,*LM_pt,1,CV_RGB(0,255,0),-1);
			Point2f temp=*LM_pt;
			string point_number_str;
			stringstream int2string;
			int2string<<point_number;
			int2string>>point_number_str;
			putText(model_ori_img_d_mpie,point_number_str,cv::Point2f(temp.x-5,temp.y-5),FONT_HERSHEY_PLAIN , 1, CV_RGB(0,0,255), 1);
			int2string.str("");
			int2string.clear();
			point_number=point_number+1;
		}
		imshow("model_ori_img_d",model_ori_img_d_mpie);inputKey_mpie=waitKey(0);
	}
}

// 模型選擇 //
void model_choose_LMset(vector<vector<Point2f>> LM_in, vector<vector<Point2f>>* LM_out)
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
	int point_F_contour[]={54,53,52,51,50,55,56,57,58};
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
void draw_point_afterModelChoose(vector<Point2f> FRGC_1,vector<Point2f> FRGC_2,vector<Point2f> MPIE)
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

	for(int k=0;k<MPIE.size();k++)
	{
		circle(img,Point2f(MPIE[k].x+300,MPIE[k].y+300),1,colors[0],-1);
	}
	imshow("point",img);waitKey(0);
	for(int k=0;k<FRGC_1.size();k++)
	{
		circle(img,Point2f(FRGC_1[k].x+300,FRGC_1[k].y+300),1,colors[2],-1);
	}
	imshow("point",img);waitKey(0);
	for(int k=0;k<FRGC_2.size();k++)
	{
		circle(img,Point2f(FRGC_2[k].x+300,FRGC_2[k].y+300),1,colors[4],-1);
	}
	imshow("point",img);waitKey(0);
}

// 模型變形 //
void model_transform(vector<Point2f> model_LM, vector<Point2f> img_LM, Mat input_img, string FRGC_model_image_path, string FRGC_image_name, Mat ViewMatrix, Mat ProjMatrix, Mat *model_x_out, Mat *model_y_out, Mat *model_z_out, Mat *mask_out, Mat *img_out, vector<Point2f> *Reg_LM_out)
{
	// check FRGC model LM 座標 與 回推的座標 //
	string FRGC_path=FRGC_model_image_path+FRGC_image_name+"/"+FRGC_image_name+"-ori.ppm";
	Mat FRGC_image=imread(FRGC_path);
	//imshow("FRGC_image_final",FRGC_image);waitKey(1);

	// rough align test img to FRGC model image //
	Mat Reg_img_temp=input_img.clone();
	Reg_img_temp.setTo(0);

	Point2f dst_point;
	find_LM_center(img_LM, &dst_point); // 找尋測試影像中心點
	Point2f src_point;
	find_LM_center(model_LM, &src_point); // 找尋FRGC LM 中心點

	/*******************************MPIE 製作*****************************************/
	//===============處理平移與旋轉================//
	move_img(input_img, &Reg_img_temp, src_point, dst_point); // 移動測試影像至FRGC的測試影像位置
	vector<Point2f> Reg_LM_each_move;
	move_LM_point(img_LM, &Reg_LM_each_move, src_point, dst_point); // 移動測試影像的LM點
	float Rotate_angle;
	find_theate(Reg_LM_each_move, model_LM, &Rotate_angle); // 尋找測試影像與FRGC影像之間的角度
	Point2f dst_center_point;
	find_LM_center(Reg_LM_each_move, &dst_center_point); // 尋找移動後測試影像的LM點中心點
	Mat Rotate_Matrix=getRotationMatrix2D(dst_center_point, Rotate_angle, 1.0); // 計算旋轉矩陣[2X3]
	cv::warpAffine(Reg_img_temp, Reg_img_temp, Rotate_Matrix, Reg_img_temp.size()); // 旋轉測試影像
	rotate_LM(Reg_LM_each_move,&Reg_LM_each_move, Rotate_angle); //旋轉移動後的LM
	//=================================================/

	//===============處理放大縮小================//
	float scale_img;
	find_scale(Reg_LM_each_move,model_LM, &scale_img); //求取放大倍率
	cv::resize(Reg_img_temp,Reg_img_temp,Size(Reg_img_temp.cols*scale_img,Reg_img_temp.rows*scale_img)); //縮放測試影像
	scale_LM(Reg_LM_each_move,&Reg_LM_each_move, scale_img); //縮放移動後的LM
	find_LM_center(Reg_LM_each_move, &dst_center_point); // 尋找移動後測試影像的LM點中心點
	move_img(Reg_img_temp, &Reg_img_temp, src_point, dst_center_point); // 移動測試影像至FRGC的測試影像位置
	move_LM_point(Reg_LM_each_move, &Reg_LM_each_move, src_point, dst_center_point); // 移動測試影像的LM點
	//==========================================//

	// 刮圖640*480 //  跟模型同大小
	Mat Reg_img_temp_crop=Reg_img_temp(Rect(0,0,640,480)).clone();

	// MPIE 及 FRGC 圖像確認//
	imshow("MPIE",Reg_img_temp_crop);
	/************************************************************************/

	//==========================合成圖片處理======================//
	Mat merge_2=0.5*FRGC_image+0.5*Reg_img_temp_crop;		//將兩張MPIE, FRGC 合成為一張圖

	// LM點位確認 //              在合成圖像 landmark 上畫出綠圈
	for (vector<Point2f>::iterator LM_pt=Reg_LM_each_move.begin();LM_pt!=Reg_LM_each_move.end();++LM_pt)
	{
		circle(merge_2,*LM_pt,1,CV_RGB(0,255,0),-1);
	}
	//              在合成圖像 landmark 上畫出藍綠圈
	for (vector<Point2f>::iterator LM_pt=model_LM.begin();LM_pt!=model_LM.end();++LM_pt)
	{
		circle(merge_2,*LM_pt,1,CV_RGB(0,255,255),-1);
	}
	imshow("merge_2",merge_2);
	waitKey(0);
	//==================================================================//

	//讀取 Model 的 x y z r g b nx ny nz//
	string ply_model_path=FRGC_model_image_path+FRGC_image_name+"/"+FRGC_image_name+"-no-Color-t.ply";			//模型的單色點雲圖
	string file_open_FRGC_index=FRGC_model_image_path+FRGC_image_name+"/"+FRGC_image_name+"-index.txt";			//
	string file_open_FRGC_model_mask=FRGC_model_image_path+FRGC_image_name+"/"+FRGC_image_name+"-mask.ppm";		//模型的 mask
	Mat model_x,model_y,model_z;
	Mat model_r,model_g,model_b;
	Mat model_nx,model_ny,model_nz;
	Read_PLY(ply_model_path, file_open_FRGC_index,file_open_FRGC_model_mask,&model_x,&model_y,&model_z,&model_r,&model_g,&model_b,&model_nx,&model_ny,&model_nz);

	// reconstruction //
	vector<Point2f> model_LM_after_2;
	vector<Point2f> img_LM_after_2;
	Set_the_LM(model_LM,&model_LM_after_2);
	Set_the_LM(Reg_LM_each_move,&img_LM_after_2);

	vector<cv::Point2f> src_pt=img_LM_after_2;
	vector<cv::Point2f> dst_pt=model_LM_after_2;

	int aff_num=0;
	vector<vector<cv::Point2f>> aff_src_pt;
	vector<vector<cv::Point2f>> aff_dst_pt;
	string title=data_file_title+"old_rec_data/";
	string warp_txt_path=title+"Warp-txt/persp_v5.txt";					//三角形 mask 的總數以及每個區域的三個點
	set_persp_aff(src_pt,&aff_src_pt,warp_txt_path,&aff_num);			//
	set_persp_aff(dst_pt,&aff_dst_pt,warp_txt_path,&aff_num);


	//======開新圖像並設為全黑=========//
	Mat_<float> colorWarp_temp(480,640);
	colorWarp_temp.setTo(0);
	cv::Mat_<uchar> affine_Z_mask_t(480,640);
	affine_Z_mask_t.setTo(0);
	cv::Mat_<uchar> temp_mask(480,640);
	temp_mask.setTo((0));

	vector<Mat> crop_img_s;
	vector<Mat> crop_img_s_after;							//儲存
	split(Reg_img_temp_crop,crop_img_s);					//將多通道的圖像分離成三個通道 (存到一個 Mat 型態的向量)
	split(Reg_img_temp_crop,crop_img_s_after);				//Reg_img_temp_crop 是與模型圖片同大小的側視圖像
	crop_img_s_after[0].setTo(0);
	crop_img_s_after[1].setTo(0);
	crop_img_s_after[2].setTo(0);
	//==================================//

	for(int i=0; i<aff_num; i++)							//跑完所有點組
	{
		//三個通道依序做
		aff_warp(aff_src_pt[i], aff_dst_pt[i], crop_img_s[0], &colorWarp_temp,&temp_mask);  //第一個通道圖像送入 aff_warp, 得到 mask 
		colorWarp_temp.copyTo(crop_img_s_after[0],temp_mask);				//temp_mask 遮罩 colorWarp_temp 存入 crop_img_s_after[0]
		colorWarp_temp.setTo(0);
		//imshow("channel 1",crop_img_s_after[0]);
		aff_warp(aff_src_pt[i], aff_dst_pt[i], crop_img_s[1], &colorWarp_temp,&temp_mask);
		colorWarp_temp.copyTo(crop_img_s_after[1],temp_mask);
		//imshow("channel 2",crop_img_s_after[1]);
		colorWarp_temp.setTo(0);
		aff_warp(aff_src_pt[i], aff_dst_pt[i], crop_img_s[2], &colorWarp_temp,&temp_mask);
		colorWarp_temp.copyTo(crop_img_s_after[2],temp_mask);
		//imshow("channel 3",crop_img_s_after[2]);waitKey(0);
		colorWarp_temp.setTo(0);

		affine_Z_mask_t=affine_Z_mask_t+temp_mask;						//傳回的 mask 加總
		//imshow("temp_mask",temp_mask);waitKey(0);						//temp_mask: 此次迴圈的 mask
		//imshow("affine_Z_mask_t",affine_Z_mask_t);waitKey(0);			//affine_Z_mask_t:　迴圈到目前為止 mask 的累加
		temp_mask.setTo((0));
	}
	Mat crop_img_after;
	merge(crop_img_s_after,crop_img_after);								//將分割的 Mat vector crop_img_s_after 合併成 crop_img_after
	imshow("merge_img",crop_img_after);waitKey(0);
	//============以上 2D 重建===========//

	for (int i=0;i<affine_Z_mask_t.rows;i++)
		for (int j=0;j<affine_Z_mask_t.cols;j++)
			if (affine_Z_mask_t.at<uchar>(i,j)!=0)
			{
				if (model_x.at<float>(i,j)==0)
					affine_Z_mask_t.at<uchar>(i,j)=0;
				else if(model_y.at<float>(i,j)==0)
					affine_Z_mask_t.at<uchar>(i,j)=0;
				else if(model_z.at<float>(i,j)==0)
					affine_Z_mask_t.at<uchar>(i,j)=0;
			}
	//imshow("???", affine_Z_mask_t);
	//imshow("crop_img_after1",crop_img_after);waitKey(0);
	crop_img_after.copyTo(crop_img_after,affine_Z_mask_t);   //做這件事幹嘛???
	//imshow("crop_img_after2",crop_img_after);waitKey(0);

	// Recovering Light /
	Mat albode_ref;
	FRGC_image.copyTo(albode_ref,affine_Z_mask_t);
	Mat L(4,1,CV_32FC1);
	calculate_Light(crop_img_after, albode_ref,model_nx,model_ny,model_nz,affine_Z_mask_t, &L);

	// Recovering Depth //
	Mat h=fspecialLoG(3, 3);
	Mat Depth_recover=model_z.clone(); Depth_recover.setTo(0);
	calculate_Depth(crop_img_after,albode_ref, model_z, affine_Z_mask_t, L, h, model_x, model_y, &Depth_recover);
	Mat FRGC_mask_temp=imread(file_open_FRGC_model_mask,0);
	Mat Leye_mask=FRGC_mask_temp.clone();Leye_mask.setTo(0);
	Mat Reye_mask=FRGC_mask_temp.clone();Reye_mask.setTo(0);
	Create_Leye_mask(Reg_LM_each_move,Leye_mask);
	Create_Reye_mask(Reg_LM_each_move,Reye_mask);

	for(int m=0; m<affine_Z_mask_t.rows; m++)
	{
		for(int n=0; n<affine_Z_mask_t.cols; n++)
		{
			if(affine_Z_mask_t.at<uchar>(m,n)!=0)
			{
				if(Leye_mask.at<uchar>(m,n)!=0){Depth_recover.at<float>(m,n)=model_z.at<float>(m,n);continue;}
				if(Reye_mask.at<uchar>(m,n)!=0){Depth_recover.at<float>(m,n)=model_z.at<float>(m,n);continue;}
				Depth_recover.at<float>(m,n)=Depth_recover.at<float>(m,n)+model_z.at<float>(m,n);
			}
		}
	}
	model_z.setTo(0);
	model_z=Depth_recover.clone();
	Mat albedo_rec;
	calculate_albedo(crop_img_after, albode_ref, model_z,affine_Z_mask_t, L, h, &albedo_rec);
	//imshow("albedo_rec",albedo_rec);waitKey(1);
	Mat normal_x,normal_y,normal_z;
	calculate_normal(affine_Z_mask_t, model_x, model_y, model_z, &normal_x, &normal_y,&normal_z);
		// Depth warp //
	Set_the_LM(model_LM,&model_LM);
	Set_the_LM(Reg_LM_each_move,&Reg_LM_each_move);
	vector<cv::Point2f> src_pt_2=model_LM;
	vector<cv::Point2f> dst_pt_2=Reg_LM_each_move;
		//將ref_pt的點依分好的點位做區塊,存至aff_src_pt//
	int aff_num_2=0;
	vector<vector<cv::Point2f>> aff_src_pt_2;
	vector<vector<cv::Point2f>> aff_dst_pt_2;
	string title_2=data_file_title+"old_rec_data/";
	string warp_txt_path_2=title_2+"Warp-txt/persp_v5.txt";
	set_persp_aff(src_pt_2,&aff_src_pt_2,warp_txt_path_2,&aff_num_2);
	set_persp_aff(dst_pt_2,&aff_dst_pt_2,warp_txt_path_2,&aff_num_2);
	Mat xWarp_temp=model_z.clone();xWarp_temp.setTo(0);
	Mat xWarp_all=model_z.clone();xWarp_all.setTo(0);
	Mat yWarp_temp=model_z.clone();yWarp_temp.setTo(0);
	Mat yWarp_all=model_z.clone();yWarp_all.setTo(0);
	Mat zWarp_temp=model_z.clone();zWarp_temp.setTo(0);
	Mat zWarp_all=model_z.clone();zWarp_all.setTo(0);
	Mat nxWarp_temp=model_z.clone();nxWarp_temp.setTo(0);
	Mat nxWarp_all=model_z.clone();nxWarp_all.setTo(0);
	Mat nyWarp_temp=model_z.clone();nyWarp_temp.setTo(0);
	Mat nyWarp_all=model_z.clone();nyWarp_all.setTo(0);
	Mat nzWarp_temp=model_z.clone();nzWarp_temp.setTo(0);
	Mat nzWarp_all=model_z.clone();zWarp_all.setTo(0);



	cv::Mat_<uchar> affine_Z_mask(480,640);
	affine_Z_mask.setTo(0);
	//cv::Mat_<uchar> temp_mask(480,640);
	temp_mask.setTo((0));
	for(int i=0; i<aff_num_2; i++)			//所有點組跑完
	{
		aff_warp(aff_src_pt_2[i], aff_dst_pt_2[i], model_x, &xWarp_temp,&temp_mask);
		xWarp_temp.copyTo(xWarp_all,temp_mask);
		xWarp_temp.setTo(0);
		aff_warp(aff_src_pt_2[i], aff_dst_pt_2[i], model_y, &yWarp_temp,&temp_mask);
		yWarp_temp.copyTo(yWarp_all,temp_mask);
		yWarp_temp.setTo(0);
		aff_warp(aff_src_pt_2[i], aff_dst_pt_2[i], model_z, &zWarp_temp,&temp_mask);
		zWarp_temp.copyTo(zWarp_all,temp_mask);
		zWarp_temp.setTo(0);
		affine_Z_mask=affine_Z_mask+temp_mask;
		/*imshow("temp_mask",temp_mask);waitKey(0);
		imshow("affine_Z_mask",affine_Z_mask);waitKey(0);*/
		temp_mask.setTo((0));
	}

	*model_x_out=xWarp_all;
	*model_y_out=yWarp_all;
	*model_z_out=zWarp_all;
	*mask_out=affine_Z_mask;
	*img_out=Reg_img_temp_crop;
	*Reg_LM_out=Reg_LM_each_move;
}

void model_transform_LMset(vector<Point2f> LM_in, vector<Point2f> *LM_out, Mat ViewMatrix, Mat ProjMatrix)
{
	vector<Point2f> temp_LM;
	for (int i=0;i<LM_in.size();i++)
	{
		Mat temp(4,1,CV_32FC1);
		Mat ProjMatrix_t=ProjMatrix.t();
		temp.at<float>(0,0)=LM_in[i].y;
		temp.at<float>(1,0)=LM_in[i].x;
		temp.at<float>(2,0)=1.0;
		temp.at<float>(3,0)=1.0;
		temp=ProjMatrix_t.inv()*ViewMatrix.inv()*temp;
		temp_LM.push_back(Point2f(temp.at<float>(0,0),temp.at<float>(1,0)));
	}
	*LM_out=temp_LM;
}
void volume_spline_interpolation(vector<Point2f> src, vector<Point2f> dst, Mat *coefficient, float *k_num)
{
	
	vector<float> err_count;
	vector<float> k_count;
	vector<Mat> coeff_count;
	for(float k=0.5;k<=1.0;k=k+0.1)
	{
		Mat A(src.size()+3,src.size()+3,CV_32FC1);
		A.setTo(0);
		for (int i=0;i<src.size();i++)
		{
			for (int j=0;j<src.size();j++)
			{
				A.at<float>(i,j)=radial_basis_Fun(src, i, j, k);
			}
		}
		for (int i=0;i<src.size();i++)
		{
			A.at<float>(i,src.size())=1.0;
			A.at<float>(i,src.size()+1)=src[i].x;
			A.at<float>(i,src.size()+2)=src[i].y;
		}
		for (int i=0;i<src.size();i++)
		{
			A.at<float>(src.size(),i)=1.0;
			A.at<float>(src.size()+1,i)=src[i].x;
			A.at<float>(src.size()+2,i)=src[i].y;
		}

		Mat b(src.size()+3,2,CV_32FC1);
		b.setTo(0);
		for (int i=0;i<dst.size();i++)
		{
			b.at<float>(i,0)=dst[i].x;
			b.at<float>(i,1)=dst[i].y;
		}

		Mat x;
		solve(A, b, x, DECOMP_SVD);

		float ans_x=0.0;
		float ans_y=0.0;
		float error=0.0;
		float scale=10.0;
		for (int i=0;i<dst.size();i++)
		{
			for (int j=0;j<src.size();j++)
			{
				ans_x=ans_x+x.at<float>(j,0)*radial_basis_Fun_check(src, i, j, k);
				ans_y=ans_y+x.at<float>(j,1)*radial_basis_Fun_check(src, i, j, k);
			}
			ans_x=ans_x+x.at<float>(src.size(),0);
			ans_x=ans_x+x.at<float>(src.size()+1,0)*src[i].x;
			ans_x=ans_x+x.at<float>(src.size()+2,0)*src[i].y;
			ans_y=ans_y+x.at<float>(src.size(),1);
			ans_y=ans_y+x.at<float>(src.size()+1,1)*src[i].x;
			ans_y=ans_y+x.at<float>(src.size()+2,1)*src[i].y;

			error=error+sqrt((ans_x*scale-dst[i].x*scale)*(ans_x*scale-dst[i].x*scale)+(ans_y*scale-dst[i].y*scale)*(ans_y*scale-dst[i].y*scale));
		}
		//cout<<error<<endl;
		
		
		err_count.push_back(error);
		k_count.push_back(k);
		coeff_count.push_back(x);
	}

	vector<float> err_count_compare=err_count;
	std::sort(err_count.begin(),err_count.end(),less<float>());

	vector<int> num;
	for (int i = 0; i < err_count.size(); i++)
	{
		for (int k=0; k<err_count_compare.size(); k++)
		{
			if (err_count[i]==err_count_compare[k])
			{
				num.push_back(k);
			}
		}
	}

	cout<<"k of volume spline"<<k_count[num[0]]<<endl;

	*coefficient=coeff_count[num[0]];
	*k_num=k_count[num[0]];
}
float radial_basis_Fun(vector<Point2f> LM, int row_s, int col_s, float k)
{
	float r=sqrt((LM[row_s].x-LM[col_s].x)*(LM[row_s].x-LM[col_s].x)+(LM[row_s].y-LM[col_s].y)*(LM[row_s].y-LM[col_s].y));
	//float k=2.0;
	float ans=exp(-k*r);
	return ans;
}
float radial_basis_Fun_check(vector<Point2f> src, int row_s, int col_s, float k)
{
	float r=sqrt((src[row_s].x-src[col_s].x)*(src[row_s].x-src[col_s].x)+(src[row_s].y-src[col_s].y)*(src[row_s].y-src[col_s].y));
	//float k=2.0;
	float ans=exp(-k*r);
	return ans;
}
void model_transform_cood(Mat mask, Mat model_x, Mat model_y, Mat *model_x_out, Mat *model_y_out, vector<Point2f> src, Mat coefficient, float k_num)
{
	Mat model_x_temp=model_x.clone(); model_x_temp.setTo(0);
	Mat model_y_temp=model_y.clone(); model_y_temp.setTo(0);
	for (int i=0;i<mask.rows;i++)
	{
		for (int j=0;j<mask.cols;j++)
		{
			if (mask.at<uchar>(i,j)!=0)
			{
				model_x_temp.at<float>(i,j)=trans_cood(model_x, model_y, src, coefficient, i, j, 0, k_num);
				model_y_temp.at<float>(i,j)=trans_cood(model_x, model_y, src, coefficient, i, j, 1, k_num);
			}
		}
	}

	*model_x_out=model_x_temp;
	*model_y_out=model_y_temp;
}
float trans_cood(Mat model_x, Mat model_y, vector<Point2f> src, Mat coefficient, int row_s, int col_s, int dims, float k_num)
{
	float ans=0.0;
	for (int i=0;i<src.size();i++)
	{
		ans=ans+coefficient.at<float>(i,dims)*radial_basis_Fun2(model_x, model_y,src, row_s, col_s, i, k_num);
	}
	ans=ans+coefficient.at<float>(src.size(),dims);
	ans=ans+coefficient.at<float>(src.size()+1,dims)*model_x.at<float>(row_s,col_s);
	ans=ans+coefficient.at<float>(src.size()+2,dims)*model_y.at<float>(row_s,col_s);

	return ans;
}
float radial_basis_Fun2(Mat model_x, Mat model_y,vector<Point2f> LM, int row_s, int col_s, int nums, float k_num)
{
	float r=sqrt((model_x.at<float>(row_s,col_s)-LM[nums].x)*(model_x.at<float>(row_s,col_s)-LM[nums].x)+(model_y.at<float>(row_s,col_s)-LM[nums].y)*(model_y.at<float>(row_s,col_s)-LM[nums].y));
	//float k=2.0;
	float ans=exp(-k_num*r);
	return ans;
}
void Create_face_mask(vector<Point2f> input_point, Mat &out_mask)
{
	int Mask_point_glass_mid[]={1,2,3,4,7,8,9,10,58,57,56,55,50,51,52,53,54};
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
void Create_Leye_mask(vector<Point2f> input_point, Mat &out_mask)
{
	int Mask_point_glass_mid[]={20,21,22,23,24,25};
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
void Create_Reye_mask(vector<Point2f> input_point, Mat &out_mask)
{
	int Mask_point_glass_mid[]={26,27,28,29,30,31};
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

// 模型重建 //
void calculate_Light(Mat image_i,Mat albedo_ref,Mat normal_x,Mat normal_y,Mat normal_z, Mat affine_Z_mask, Mat* L)
{
	//imshow("image_i",image_i);
	//imshow("albedo_ref",albedo_ref);
	//imshow("normal_x",normal_x);
	//imshow("normal_y",normal_y);
	//imshow("normal_z",normal_z);
	//imshow("affine_Z_mask",affine_Z_mask);
	//waitKey(1);

	// 拉直affine_Z_mask //
	Mat affine_Z_mask_line=affine_Z_mask.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	int noneZeroNum=cv::countNonZero(affine_Z_mask);

	// b matrix //
	Mat b(noneZeroNum,1,CV_32FC1);
	Mat image_i_HSV;
	cvtColor(image_i,image_i_HSV,CV_BGR2HSV);
	vector<Mat> image_i_HSV_s;
	split(image_i_HSV,image_i_HSV_s);
	Mat image_i_V_line=image_i_HSV_s[2].reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	int b_postion=0;
	for (int i=0;i<affine_Z_mask_line.rows;i++)
	{
		if (affine_Z_mask_line.at<uchar>(i,0)!=0)
		{
			//b.at<float>(b_postion,0)=(float)image_i_V_line.at<uchar>(i,0);
			b.at<float>(b_postion,0)=(float)image_i_V_line.at<uchar>(i,0)/255.0; //normalized 0~1
			b_postion=b_postion+1;
		}
	}

	// A matrix //
	Mat A(noneZeroNum,4,CV_32FC1);
	Mat albedo_ref_HSV;
	Mat albedo_ref_V_line;
	if (albedo_ref.channels()==3)
	{
		cvtColor(albedo_ref,albedo_ref_HSV,CV_BGR2HSV);
		vector<Mat> albedo_ref_HSV_s;
		split(albedo_ref_HSV,albedo_ref_HSV_s);
		albedo_ref_V_line=albedo_ref_HSV_s[2].reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	}
	else
	{
		albedo_ref_V_line=albedo_ref.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	}
	Mat normal_x_line=normal_x.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat normal_y_line=normal_y.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat normal_z_line=normal_z.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	int A_postion=0;
	float a0=3.141592;
	float a1=2*3.141592/sqrt(3.0);
	float c0=1/sqrt(4.0*3.141592);
	float c1=sqrt(3.0)/sqrt(4.0*3.141592);
	for (int i=0;i<affine_Z_mask_line.rows;i++)
	{
		if (affine_Z_mask_line.at<uchar>(i,0)!=0)
		{
			//A.at<float>(A_postion,0)=(float)albedo_ref_V_line.at<uchar>(i,0)*a0*c0;
			//A.at<float>(A_postion,1)=(float)albedo_ref_V_line.at<uchar>(i,0)*a1*c1*normal_x_line.at<float>(i,0);
			//A.at<float>(A_postion,2)=(float)albedo_ref_V_line.at<uchar>(i,0)*a1*c1*normal_y_line.at<float>(i,0);
			//A.at<float>(A_postion,3)=(float)albedo_ref_V_line.at<uchar>(i,0)*a1*c1*normal_z_line.at<float>(i,0);

			//cout<<A.at<float>(A_postion,0)<<endl;
			//cout<<A.at<float>(A_postion,1)<<endl;
			//cout<<A.at<float>(A_postion,2)<<endl;
			//cout<<A.at<float>(A_postion,3)<<endl;

			//normalized 0~1
			A.at<float>(A_postion,0)=(float)albedo_ref_V_line.at<uchar>(i,0)*a0*c0/255.0;
			A.at<float>(A_postion,1)=(float)albedo_ref_V_line.at<uchar>(i,0)*a1*c1*normal_x_line.at<float>(i,0)/255.0;
			A.at<float>(A_postion,2)=(float)albedo_ref_V_line.at<uchar>(i,0)*a1*c1*normal_y_line.at<float>(i,0)/255.0;
			A.at<float>(A_postion,3)=(float)albedo_ref_V_line.at<uchar>(i,0)*a1*c1*normal_z_line.at<float>(i,0)/255.0;

			A_postion=A_postion+1;
			//cout<<A_postion<<endl;
		}
	}

	// calculate Ax=b by opencv SVD //
	Mat x_to_solve(4,1,CV_32FC1);
	//double t = cv::getTickCount();
	cv::solve(A,b,x_to_solve,DECOMP_SVD);
	//t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
	//std::cout << "Recover Lightingcoef.: " << t <<"s"<< std::endl;

	// output //
	*L=x_to_solve;
}
cv::Mat fspecialLoG(int WinSize, double sigma)
{
	Mat h(WinSize,WinSize,CV_32FC1);
	float sum=0;
	for (int i=0;i<WinSize;i++)
	{
		for (int j=0;j<WinSize;j++)
		{
			float x=(i-(WinSize-1)/2)*(i-(WinSize-1)/2);
			float y=(j-(WinSize-1)/2)*(j-(WinSize-1)/2);
			//float a_part=(x+y)/(2*sigma*sigma);
			//float b_part=exp(-(a_part));
			//h.at<float>(i,j)=-((1-a_part)*b_part)/(3.1415*sigma*sigma*sigma*sigma);
			h.at<float>(i,j)=exp(-(x+y)/(2*sigma*sigma));
			sum=sum+h.at<float>(i,j);
		}
	}

	h=h/sum;

	return h;
}
void calculate_Depth(Mat image_i,Mat albedo_ref,Mat model_z, Mat affine_Z_mask, Mat L, Mat h,Mat model_x,Mat model_y, Mat* out_z)
{
	float lambda=30.0;
	float sigma=3.0;
	lambda_record=lambda;
	
	// boundary mask //
	Mat boundary_L=affine_Z_mask.clone(); boundary_L.setTo(0);// 左邊界
	Mat boundary_R=affine_Z_mask.clone(); boundary_R.setTo(0);// 右邊界
	Mat boundary_B=affine_Z_mask.clone(); boundary_B.setTo(0);// 下邊界
	Mat boundary_U=affine_Z_mask.clone(); boundary_U.setTo(0);// 上邊界
	Mat boundary_UL=affine_Z_mask.clone(); boundary_UL.setTo(0);// 上左邊界
	Mat boundary_UR=affine_Z_mask.clone(); boundary_UR.setTo(0);// 上右邊界
	Mat boundary_BL=affine_Z_mask.clone(); boundary_BL.setTo(0);// 下左邊界
	Mat boundary_BR=affine_Z_mask.clone(); boundary_BR.setTo(0);// 下右邊界

	Mat data_count_number=model_z.clone(); data_count_number.setTo(0);// 計算affine_Z_mask 內model_z有值的數量
	int count_number=1;
	for (int i = 0; i < model_z.rows; i++)
	{
		for (int j = 0; j < model_z.cols; j++)
		{
			if (affine_Z_mask.at<uchar>(i,j)!=0)
			{
				if (model_z.at<float>(i,j)!=0)
				{
					data_count_number.at<float>(i,j)=count_number;
					count_number=count_number+1;
					if (model_z.at<float>(i-1,j-1)==0)
						boundary_UL.at<uchar>(i,j)=255;
					if (model_z.at<float>(i-1,j)==0)
						boundary_U.at<uchar>(i,j)=255;
					if (model_z.at<float>(i-1,j+1)==0)
						boundary_UR.at<uchar>(i,j)=255;
					if (model_z.at<float>(i,j-1)==0)
						boundary_L.at<uchar>(i,j)=255;
					if (model_z.at<float>(i,j+1)==0)
						boundary_R.at<uchar>(i,j)=255;
					if (model_z.at<float>(i+1,j-1)==0)
						boundary_BL.at<uchar>(i,j)=255;
					if (model_z.at<float>(i+1,j)==0)
						boundary_B.at<uchar>(i,j)=255;
					if (model_z.at<float>(i+1,j+1)==0)
						boundary_BR.at<uchar>(i,j)=255;	
				}
			}
		}
	}
	// 拉直 boundary mask //
	Mat boundary_L_line=boundary_L.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat boundary_R_line=boundary_R.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat boundary_B_line=boundary_B.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat boundary_U_line=boundary_U.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat boundary_UL_line=boundary_UL.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat boundary_UR_line=boundary_UR.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat boundary_BL_line=boundary_BL.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat boundary_BR_line=boundary_BR.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);

	// p=Z(x+1,y)-Z(x,y) //
	// q=Z(x,y+1)-Z(x,y) //
	// Nref = sqrt(p*p+q*q+1) //
	Mat p_normal=model_z.clone(); p_normal.setTo(0);
	Mat q_normal=model_z.clone(); q_normal.setTo(0);
	Mat Nref=model_z.clone(); Nref.setTo(0);
	for (int i = 0; i < model_z.rows; i++)
	{
		for (int j = 0; j < model_z.cols; j++)
			if (affine_Z_mask.at<uchar>(i,j)!=0)
				if (model_z.at<float>(i,j)!=0)
				{
					if (boundary_R.at<uchar>(i,j)==0)
						p_normal.at<float>(i,j)=model_z.at<float>(i,j+1)-model_z.at<float>(i,j);
					else
						p_normal.at<float>(i,j)=model_z.at<float>(i,j-1)-model_z.at<float>(i,j);
					if (boundary_B.at<uchar>(i,j)==0)
						q_normal.at<float>(i,j)=model_z.at<float>(i+1,j)-model_z.at<float>(i,j);
					else
						q_normal.at<float>(i,j)=model_z.at<float>(i-1,j)-model_z.at<float>(i,j);

					Nref.at<float>(i,j)=sqrt(p_normal.at<float>(i,j)*p_normal.at<float>(i,j)+q_normal.at<float>(i,j)*q_normal.at<float>(i,j)+1);
				}
	}
	Mat Nref_line=Nref.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);

	/*vector<vector<float>> Z_depth;
	for (int i = 0; i < model_z.rows; i++)
	{
		for (int j = 0; j < model_z.cols; j++)
		{
			if (affine_Z_mask.at<uchar>(i,j)!=0)
			{
				vector<float> temp;
				if (boundary_UL.at<uchar>(i,j)==0)
				{
					cout<<"1_v :"<<i-1<<" "<<j-1<<" "<<model_z.at<float>(i-1,j-1)<<endl;
					cout<<"1_L :"<<i-1<<" "<<j-1<<" "<<data_count_number.at<float>(i-1,j-1)<<endl;
				}
				if (boundary_U.at<uchar>(i,j)==0)
				{
					cout<<"2_v :"<<i-1<<" "<<j<<" "<<model_z.at<float>(i-1,j)<<endl;
					cout<<"2_L :"<<i-1<<" "<<j<<" "<<data_count_number.at<float>(i-1,j)<<endl;
				}
				if (boundary_UR.at<uchar>(i,j)==0)
				{
					cout<<"3_v :"<<i-1<<" "<<j+1<<" "<<model_z.at<float>(i-1,j+1)<<endl;
					cout<<"3_L :"<<i-1<<" "<<j+1<<" "<<data_count_number.at<float>(i-1,j+1)<<endl;
				}
				if (boundary_L.at<uchar>(i,j)==0)
				{
					cout<<"4_v :"<<i<<" "<<j-1<<" "<<model_z.at<float>(i,j-1)<<endl;
					cout<<"4_L :"<<i<<" "<<j-1<<" "<<data_count_number.at<float>(i,j-1)<<endl;
				}
				
				cout<<"5_v :"<<i<<" "<<j<<" "<<model_z.at<float>(i,j)<<endl;
				cout<<"5_L :"<<i<<" "<<j<<" "<<data_count_number.at<float>(i,j)<<endl;
				if (boundary_R.at<uchar>(i,j)==0)
				{
					cout<<"6_v :"<<i<<" "<<j+1<<" "<<model_z.at<float>(i,j+1)<<endl;
					cout<<"6_L :"<<i<<" "<<j+1<<" "<<data_count_number.at<float>(i,j+1)<<endl;
				}
				if (boundary_BL.at<uchar>(i,j)==0)
				{
					cout<<"7_v :"<<i+1<<" "<<j-1<<" "<<model_z.at<float>(i+1,j-1)<<endl;
					cout<<"7_L :"<<i+1<<" "<<j-1<<" "<<data_count_number.at<float>(i+1,j-1)<<endl;
				}
				if (boundary_B.at<uchar>(i,j)==0)
				{
					cout<<"8_v :"<<i+1<<" "<<j<<" "<<model_z.at<float>(i+1,j)<<endl;
					cout<<"8_L :"<<i+1<<" "<<j<<" "<<data_count_number.at<float>(i+1,j)<<endl;
				}
				if (boundary_BR.at<uchar>(i,j)==0)
				{
					cout<<"9_v :"<<i+1<<" "<<j+1<<" "<<model_z.at<float>(i+1,j+1)<<endl;
					cout<<"9_L :"<<i+1<<" "<<j+1<<" "<<data_count_number.at<float>(i+1,j+1)<<endl;
				}
				waitKey(0);
			}
		}
	}*/

	// A matrix sparse eigen//
	int noneZeroNum=cv::countNonZero(affine_Z_mask);
	Mat affine_Z_mask_line=affine_Z_mask.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);

	Mat albedo_ref_HSV;
	Mat albedo_ref_HSV_s_gauss;
	Mat albedo_ref_V_line;
	if (albedo_ref.channels()==3)
	{
		cvtColor(albedo_ref,albedo_ref_HSV,CV_BGR2HSV);
		vector<Mat> albedo_ref_HSV_s;
		split(albedo_ref_HSV,albedo_ref_HSV_s);
		albedo_ref_HSV_s_gauss=albedo_ref_HSV_s[2].clone();
		GaussianBlur(albedo_ref_HSV_s[2], albedo_ref_HSV_s_gauss, Size(3,3), sigma, sigma, BORDER_DEFAULT );
		//Mat albedo_ref_V_line=albedo_ref_HSV_s[2].reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
		albedo_ref_V_line=albedo_ref_HSV_s_gauss.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	}
	else
	{
		albedo_ref_HSV_s_gauss=albedo_ref.clone();
		GaussianBlur(albedo_ref, albedo_ref_HSV_s_gauss, Size(3,3), sigma, sigma, BORDER_DEFAULT );
		albedo_ref_V_line=albedo_ref_HSV_s_gauss.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	}

	Eigen::SparseMatrix<float> A_sparse(noneZeroNum,noneZeroNum);
	A_sparse.setZero();
	typedef Eigen::Triplet<float> T;
	vector< T > tripletList;
	for (int i = 0; i < model_z.rows; i++)
	{
		for (int j = 0; j < model_z.cols; j++)
		{
			if (affine_Z_mask.at<uchar>(i,j)!=0)
			{
				if (false)//data_count_number.at<float>(i,j)-1==0 false
				{
					float D1=1.0;
					int row_location = data_count_number.at<float>(i,j)-1;
					int col_location = data_count_number.at<float>(i,j)-1;
					tripletList.push_back(T(row_location,col_location,D1));
				} 
				else
				{
					//float D1=(float)albedo_ref_HSV_s[2].at<uchar>(i,j)*(-(L.at<float>(1,0)+L.at<float>(2,0)))/Nref.at<float>(i,j);
					float D1=(float)albedo_ref_HSV_s_gauss.at<uchar>(i,j)*(-(L.at<float>(1,0)+L.at<float>(2,0)))/Nref.at<float>(i,j);
					float R5=lambda*(1-h.at<float>(1,1));
					int row_location = data_count_number.at<float>(i,j)-1;
					int col_location = data_count_number.at<float>(i,j)-1;
					tripletList.push_back(T(row_location,col_location,D1+R5));
					if (data_count_number.at<float>(i,j+1)!=0) //D2
					{
						//float D2=(float)albedo_ref_HSV_s[2].at<uchar>(i,j)*(L.at<float>(1,0))/Nref.at<float>(i,j);
						float D2=(float)albedo_ref_HSV_s_gauss.at<uchar>(i,j)*(L.at<float>(1,0))/Nref.at<float>(i,j);
						float R6=lambda*(0-h.at<float>(1,2));
						int row_location = data_count_number.at<float>(i,j)-1;
						int col_location = data_count_number.at<float>(i,j+1)-1;
						tripletList.push_back(T(row_location,col_location,D2+R6));
					}
					if (data_count_number.at<float>(i+1,j)!=0) //D3
					{
						//float D3=(float)albedo_ref_HSV_s[2].at<uchar>(i,j)*(L.at<float>(2,0))/Nref.at<float>(i,j);
						float D3=(float)albedo_ref_HSV_s_gauss.at<uchar>(i,j)*(L.at<float>(2,0))/Nref.at<float>(i,j);
						float R8=lambda*(0-h.at<float>(2,1));
						int row_location = data_count_number.at<float>(i,j)-1;
						int col_location = data_count_number.at<float>(i+1,j)-1;
						tripletList.push_back(T(row_location,col_location,D3+R8));
					}
					//if (data_count_number.at<float>(i-1,j-1)!=0) //R1
					//{
					//	float R1=lambda*(0-h.at<float>(0,0));
					//	int row_location = data_count_number.at<float>(i,j)-1;
					//	int col_location = data_count_number.at<float>(i-1,j-1)-1;
					//	tripletList.push_back(T(row_location,col_location,R1));
					//}
					//if (data_count_number.at<float>(i-1,j)!=0) //R2
					//{
					//	float R2=lambda*(0-h.at<float>(0,1));
					//	int row_location = data_count_number.at<float>(i,j)-1;
					//	int col_location = data_count_number.at<float>(i-1,j)-1;
					//	tripletList.push_back(T(row_location,col_location,R2));
					//}
					//if (data_count_number.at<float>(i-1,j+1)!=0) //R3
					//{
					//	float R3=lambda*(0-h.at<float>(0,2));
					//	int row_location = data_count_number.at<float>(i,j)-1;
					//	int col_location = data_count_number.at<float>(i-1,j+1)-1;
					//	tripletList.push_back(T(row_location,col_location,R3));
					//}
					//if (data_count_number.at<float>(i,j-1)!=0) //R4
					//{
					//	float R4=lambda*(0-h.at<float>(0,2));
					//	int row_location = data_count_number.at<float>(i,j)-1;
					//	int col_location = data_count_number.at<float>(i,j-1)-1;
					//	tripletList.push_back(T(row_location,col_location,R4));
					//}
					//if (data_count_number.at<float>(i+1,j-1)!=0) //R7
					//{
					//	float R7=lambda*(0-h.at<float>(0,2));
					//	int row_location = data_count_number.at<float>(i,j)-1;
					//	int col_location = data_count_number.at<float>(i+1,j-1)-1;
					//	tripletList.push_back(T(row_location,col_location,R7));
					//}
					//if (data_count_number.at<float>(i+1,j+1)!=0) //R9
					//{
					//	float R9=lambda*(0-h.at<float>(0,2));
					//	int row_location = data_count_number.at<float>(i,j)-1;
					//	int col_location = data_count_number.at<float>(i+1,j+1)-1;
					//	tripletList.push_back(T(row_location,col_location,R9));
					//}
				}
			}
		}
	}
	A_sparse.setFromTriplets(tripletList.begin(),tripletList.end());

	// b matrix eigen // 
	Mat model_z_G=model_z.clone();
	GaussianBlur(model_z, model_z_G, Size(3,3), sigma, sigma, BORDER_DEFAULT );
	Mat model_z_line=model_z.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat model_z_G_line=model_z_G.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat image_i_HSV;
	cvtColor(image_i,image_i_HSV,CV_BGR2HSV);
	vector<Mat> image_i_HSV_s;
	split(image_i_HSV,image_i_HSV_s);
	Mat image_i_HSV_s_gauss=image_i_HSV_s[2].clone();
	GaussianBlur(image_i_HSV_s[2], image_i_HSV_s_gauss, Size(3,3), sigma, sigma, BORDER_DEFAULT );
	Mat image_i_V_line=image_i_HSV_s_gauss.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Eigen::VectorXf b_sparse(noneZeroNum);
	int b_postion=0;
	for (int i=0;i<affine_Z_mask_line.rows;i++)
	{
		if (affine_Z_mask_line.at<uchar>(i,0)!=0)
		{
			if (false)//b_postion==0 false
			{
				float b_val=model_z_line.at<float>(i,0);
				b_sparse(b_postion)=b_val;
				b_postion=b_postion+1;
			} 
			else
			{
				float b_val=(float)image_i_V_line.at<uchar>(i,0)-(float)albedo_ref_V_line.at<uchar>(i,0)*(L.at<float>(0,0)-(L.at<float>(3,0)/Nref_line.at<float>(i,0)));
				float R_val=lambda*(model_z_line.at<float>(i,0)-model_z_G_line.at<float>(i,0));
				b_sparse(b_postion)=b_val+R_val;
				b_postion=b_postion+1;
			}
			
		}
	}

	//  x matrix eigen // 
	Eigen::VectorXf x_sparse(noneZeroNum);
	A_sparse.makeCompressed();
	Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
	double t = cv::getTickCount();
	solver.compute(A_sparse);
	t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
	//std::cout << "solver.compute(A_sparse) : " << t <<"s"<< std::endl;
	if(solver.info()!=Eigen::Success) {
		// decomposition failed
		cout<<"solver.compute(A_sparse) : fail"<<endl;
	}
	double t_2 = cv::getTickCount();
	x_sparse = solver.solve(b_sparse);
	t_2 = ((double)cv::getTickCount() - t_2)/cv::getTickFrequency();
	//std::cout << "solver.solve(b_sparse) : " << t_2 <<"s"<< std::endl;
	if(solver.info()!=Eigen::Success) {
		// solving failed
		cout<<"solving failed"<<endl;
	}
	//std::cout << "#iterations:     " << solver.iterations() << std::endl;
	//std::cout << "estimated error: " << solver.error()      << std::endl;

	//for (int i=0;i<noneZeroNum;i++)
	//{
	//	cout<<x_sparse(i)<<endl;
	//	waitKey(0);
	//}

	Mat Depth_recover=model_z.clone(); Depth_recover.setTo(0);
	int k=0;
	for (int i=0;i<model_z.rows; i++)
	{
		for (int j=0;j<model_z.cols; j++)
		{
			if (affine_Z_mask.at<uchar>(i,j)!=0)
			{
				Depth_recover.at<float>(i,j)=x_sparse(k);
				k++;
			}
		}
	}

	*out_z=Depth_recover;
}
void calculate_Depth_iter(Mat image_i,Mat albedo_ref,Mat model_z, Mat affine_Z_mask, Mat L, Mat h,Mat model_x,Mat model_y, Mat* out_z)
{	
	// boundary mask //
	Mat boundary_L=affine_Z_mask.clone(); boundary_L.setTo(0);// 左邊界
	Mat boundary_R=affine_Z_mask.clone(); boundary_R.setTo(0);// 右邊界
	Mat boundary_B=affine_Z_mask.clone(); boundary_B.setTo(0);// 下邊界
	Mat boundary_U=affine_Z_mask.clone(); boundary_U.setTo(0);// 上邊界
	Mat boundary_UL=affine_Z_mask.clone(); boundary_UL.setTo(0);// 上左邊界
	Mat boundary_UR=affine_Z_mask.clone(); boundary_UR.setTo(0);// 上右邊界
	Mat boundary_BL=affine_Z_mask.clone(); boundary_BL.setTo(0);// 下左邊界
	Mat boundary_BR=affine_Z_mask.clone(); boundary_BR.setTo(0);// 下右邊界

	Mat data_count_number=model_z.clone(); data_count_number.setTo(0);// 計算affine_Z_mask 內model_z有值的數量
	int count_number=1;
	for (int i = 0; i < model_z.rows; i++)
	{
		for (int j = 0; j < model_z.cols; j++)
		{
			if (affine_Z_mask.at<uchar>(i,j)!=0)
			{
				if (model_z.at<float>(i,j)!=0)
				{
					data_count_number.at<float>(i,j)=count_number;
					count_number=count_number+1;
					if (model_z.at<float>(i-1,j-1)==0)
						boundary_UL.at<uchar>(i,j)=255;
					if (model_z.at<float>(i-1,j)==0)
						boundary_U.at<uchar>(i,j)=255;
					if (model_z.at<float>(i-1,j+1)==0)
						boundary_UR.at<uchar>(i,j)=255;
					if (model_z.at<float>(i,j-1)==0)
						boundary_L.at<uchar>(i,j)=255;
					if (model_z.at<float>(i,j+1)==0)
						boundary_R.at<uchar>(i,j)=255;
					if (model_z.at<float>(i+1,j-1)==0)
						boundary_BL.at<uchar>(i,j)=255;
					if (model_z.at<float>(i+1,j)==0)
						boundary_B.at<uchar>(i,j)=255;
					if (model_z.at<float>(i+1,j+1)==0)
						boundary_BR.at<uchar>(i,j)=255;	
				}
			}
		}
	}
	// 拉直 boundary mask //
	Mat boundary_L_line=boundary_L.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat boundary_R_line=boundary_R.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat boundary_B_line=boundary_B.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat boundary_U_line=boundary_U.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat boundary_UL_line=boundary_UL.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat boundary_UR_line=boundary_UR.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat boundary_BL_line=boundary_BL.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat boundary_BR_line=boundary_BR.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);

	// p=Z(x+1,y)-Z(x,y) //
	// q=Z(x,y+1)-Z(x,y) //
	// Nref = sqrt(p*p+q*q+1) //
	Mat p_normal=model_z.clone(); p_normal.setTo(0);
	Mat q_normal=model_z.clone(); q_normal.setTo(0);
	Mat Nref=model_z.clone(); Nref.setTo(0);
	for (int i = 0; i < model_z.rows; i++)
	{
		for (int j = 0; j < model_z.cols; j++)
			if (affine_Z_mask.at<uchar>(i,j)!=0)
				if (model_z.at<float>(i,j)!=0)
				{
					if (boundary_R.at<uchar>(i,j)==0)
						p_normal.at<float>(i,j)=model_z.at<float>(i,j+1)-model_z.at<float>(i,j);
					else
						p_normal.at<float>(i,j)=model_z.at<float>(i,j-1)-model_z.at<float>(i,j);
					if (boundary_B.at<uchar>(i,j)==0)
						q_normal.at<float>(i,j)=model_z.at<float>(i+1,j)-model_z.at<float>(i,j);
					else
						q_normal.at<float>(i,j)=model_z.at<float>(i-1,j)-model_z.at<float>(i,j);

					Nref.at<float>(i,j)=sqrt(p_normal.at<float>(i,j)*p_normal.at<float>(i,j)+q_normal.at<float>(i,j)*q_normal.at<float>(i,j)+1);
				}
	}
	Mat Nref_line=Nref.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);

	/*vector<vector<float>> Z_depth;
	for (int i = 0; i < model_z.rows; i++)
	{
		for (int j = 0; j < model_z.cols; j++)
		{
			if (affine_Z_mask.at<uchar>(i,j)!=0)
			{
				vector<float> temp;
				if (boundary_UL.at<uchar>(i,j)==0)
				{
					cout<<"1_v :"<<i-1<<" "<<j-1<<" "<<model_z.at<float>(i-1,j-1)<<endl;
					cout<<"1_L :"<<i-1<<" "<<j-1<<" "<<data_count_number.at<float>(i-1,j-1)<<endl;
				}
				if (boundary_U.at<uchar>(i,j)==0)
				{
					cout<<"2_v :"<<i-1<<" "<<j<<" "<<model_z.at<float>(i-1,j)<<endl;
					cout<<"2_L :"<<i-1<<" "<<j<<" "<<data_count_number.at<float>(i-1,j)<<endl;
				}
				if (boundary_UR.at<uchar>(i,j)==0)
				{
					cout<<"3_v :"<<i-1<<" "<<j+1<<" "<<model_z.at<float>(i-1,j+1)<<endl;
					cout<<"3_L :"<<i-1<<" "<<j+1<<" "<<data_count_number.at<float>(i-1,j+1)<<endl;
				}
				if (boundary_L.at<uchar>(i,j)==0)
				{
					cout<<"4_v :"<<i<<" "<<j-1<<" "<<model_z.at<float>(i,j-1)<<endl;
					cout<<"4_L :"<<i<<" "<<j-1<<" "<<data_count_number.at<float>(i,j-1)<<endl;
				}
				
				cout<<"5_v :"<<i<<" "<<j<<" "<<model_z.at<float>(i,j)<<endl;
				cout<<"5_L :"<<i<<" "<<j<<" "<<data_count_number.at<float>(i,j)<<endl;
				if (boundary_R.at<uchar>(i,j)==0)
				{
					cout<<"6_v :"<<i<<" "<<j+1<<" "<<model_z.at<float>(i,j+1)<<endl;
					cout<<"6_L :"<<i<<" "<<j+1<<" "<<data_count_number.at<float>(i,j+1)<<endl;
				}
				if (boundary_BL.at<uchar>(i,j)==0)
				{
					cout<<"7_v :"<<i+1<<" "<<j-1<<" "<<model_z.at<float>(i+1,j-1)<<endl;
					cout<<"7_L :"<<i+1<<" "<<j-1<<" "<<data_count_number.at<float>(i+1,j-1)<<endl;
				}
				if (boundary_B.at<uchar>(i,j)==0)
				{
					cout<<"8_v :"<<i+1<<" "<<j<<" "<<model_z.at<float>(i+1,j)<<endl;
					cout<<"8_L :"<<i+1<<" "<<j<<" "<<data_count_number.at<float>(i+1,j)<<endl;
				}
				if (boundary_BR.at<uchar>(i,j)==0)
				{
					cout<<"9_v :"<<i+1<<" "<<j+1<<" "<<model_z.at<float>(i+1,j+1)<<endl;
					cout<<"9_L :"<<i+1<<" "<<j+1<<" "<<data_count_number.at<float>(i+1,j+1)<<endl;
				}
				waitKey(0);
			}
		}
	}*/

	// 最佳化lambda值嘗試 //
	//float lambda=3.0;
	float sigma=3.0;
	vector<float> error_compare;
	vector<Mat> Depth_recover_v;
	vector<float> lambda_v;

	for (float lambda=1.0; lambda < 30.0; lambda++)
	{
		vector<float> img_i;
		vector<float> model_v;
		vector<float> rec_z;
		lambda_v.push_back(lambda);

		// A matrix sparse eigen//
		int noneZeroNum=cv::countNonZero(affine_Z_mask);
		Mat affine_Z_mask_line=affine_Z_mask.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);

		Mat albedo_ref_HSV;
		cvtColor(albedo_ref,albedo_ref_HSV,CV_BGR2HSV);
		vector<Mat> albedo_ref_HSV_s;
		split(albedo_ref_HSV,albedo_ref_HSV_s);
		Mat albedo_ref_HSV_s_gauss=albedo_ref_HSV_s[2].clone();
		GaussianBlur(albedo_ref_HSV_s[2], albedo_ref_HSV_s_gauss, Size(3,3), sigma, sigma, BORDER_DEFAULT );
		//Mat albedo_ref_V_line=albedo_ref_HSV_s[2].reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
		Mat albedo_ref_V_line=albedo_ref_HSV_s_gauss.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);

		Eigen::SparseMatrix<float> A_sparse(noneZeroNum,noneZeroNum);
		A_sparse.setZero();
		typedef Eigen::Triplet<float> T;
		vector< T > tripletList;
		for (int i = 0; i < model_z.rows; i++)
		{
			for (int j = 0; j < model_z.cols; j++)
			{
				if (affine_Z_mask.at<uchar>(i,j)!=0)
				{
					if (false)//data_count_number.at<float>(i,j)-1==0 false
					{
						float D1=1.0;
						int row_location = data_count_number.at<float>(i,j)-1;
						int col_location = data_count_number.at<float>(i,j)-1;
						tripletList.push_back(T(row_location,col_location,D1));
					} 
					else
					{
						//float D1=(float)albedo_ref_HSV_s[2].at<uchar>(i,j)*(-(L.at<float>(1,0)+L.at<float>(2,0)))/Nref.at<float>(i,j);
						float D1=(float)albedo_ref_HSV_s_gauss.at<uchar>(i,j)*(-(L.at<float>(1,0)+L.at<float>(2,0)))/Nref.at<float>(i,j);
						float R5=lambda*(1-h.at<float>(h.rows/2,h.cols/2));
						int row_location = data_count_number.at<float>(i,j)-1;
						int col_location = data_count_number.at<float>(i,j)-1;
						tripletList.push_back(T(row_location,col_location,D1+R5));
						model_v.push_back(D1+R5);
						if (data_count_number.at<float>(i,j+1)!=0) //D2
						{
							//float D2=(float)albedo_ref_HSV_s[2].at<uchar>(i,j)*(L.at<float>(1,0))/Nref.at<float>(i,j);
							float D2=(float)albedo_ref_HSV_s_gauss.at<uchar>(i,j)*(L.at<float>(1,0))/Nref.at<float>(i,j);
							float R6=lambda*(0-h.at<float>(h.rows/2,h.cols/2+1));
							int row_location = data_count_number.at<float>(i,j)-1;
							int col_location = data_count_number.at<float>(i,j+1)-1;
							tripletList.push_back(T(row_location,col_location,D2+R6));
						}
						if (data_count_number.at<float>(i+1,j)!=0) //D3
						{
							//float D3=(float)albedo_ref_HSV_s[2].at<uchar>(i,j)*(L.at<float>(2,0))/Nref.at<float>(i,j);
							float D3=(float)albedo_ref_HSV_s_gauss.at<uchar>(i,j)*(L.at<float>(2,0))/Nref.at<float>(i,j);
							float R8=lambda*(0-h.at<float>(h.rows/2+1,h.cols/2));
							int row_location = data_count_number.at<float>(i,j)-1;
							int col_location = data_count_number.at<float>(i+1,j)-1;
							tripletList.push_back(T(row_location,col_location,D3+R8));
						}
						//if (data_count_number.at<float>(i-1,j-1)!=0) //R1
						//{
						//	float R1=lambda*(0-h.at<float>(h.rows/2-1,h.cols/2-1));
						//	int row_location = data_count_number.at<float>(i,j)-1;
						//	int col_location = data_count_number.at<float>(i-1,j-1)-1;
						//	tripletList.push_back(T(row_location,col_location,R1));
						//}
						//if (data_count_number.at<float>(i-1,j)!=0) //R2
						//{
						//	float R2=lambda*(0-h.at<float>(h.rows/2-1,h.cols/2));
						//	int row_location = data_count_number.at<float>(i,j)-1;
						//	int col_location = data_count_number.at<float>(i-1,j)-1;
						//	tripletList.push_back(T(row_location,col_location,R2));
						//}
						//if (data_count_number.at<float>(i-1,j+1)!=0) //R3
						//{
						//	float R3=lambda*(0-h.at<float>(h.rows/2-1,h.cols/2+1));
						//	int row_location = data_count_number.at<float>(i,j)-1;
						//	int col_location = data_count_number.at<float>(i-1,j+1)-1;
						//	tripletList.push_back(T(row_location,col_location,R3));
						//}
						//if (data_count_number.at<float>(i,j-1)!=0) //R4
						//{
						//	float R4=lambda*(0-h.at<float>(h.rows/2,h.cols/2-1));
						//	int row_location = data_count_number.at<float>(i,j)-1;
						//	int col_location = data_count_number.at<float>(i,j-1)-1;
						//	tripletList.push_back(T(row_location,col_location,R4));
						//}
						//if (data_count_number.at<float>(i+1,j-1)!=0) //R7
						//{
						//	float R7=lambda*(0-h.at<float>(h.rows/2+1,h.cols/2-1));
						//	int row_location = data_count_number.at<float>(i,j)-1;
						//	int col_location = data_count_number.at<float>(i+1,j-1)-1;
						//	tripletList.push_back(T(row_location,col_location,R7));
						//}
						//if (data_count_number.at<float>(i+1,j+1)!=0) //R9
						//{
						//	float R9=lambda*(0-h.at<float>(h.rows/2+1,h.cols/2+1));
						//	int row_location = data_count_number.at<float>(i,j)-1;
						//	int col_location = data_count_number.at<float>(i+1,j+1)-1;
						//	tripletList.push_back(T(row_location,col_location,R9));
						//}
					}
				}
			}
		}
		A_sparse.setFromTriplets(tripletList.begin(),tripletList.end());

		// b matrix eigen // 
		Mat model_z_G=model_z.clone();
		GaussianBlur(model_z, model_z_G, Size(3,3), sigma, sigma, BORDER_DEFAULT );
		Mat model_z_line=model_z.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
		Mat model_z_G_line=model_z_G.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
		Mat image_i_HSV;
		cvtColor(image_i,image_i_HSV,CV_BGR2HSV);
		vector<Mat> image_i_HSV_s;
		split(image_i_HSV,image_i_HSV_s);
		Mat image_i_HSV_s_gauss=image_i_HSV_s[2].clone();
		GaussianBlur(image_i_HSV_s[2], image_i_HSV_s_gauss, Size(3,3), sigma, sigma, BORDER_DEFAULT );
		Mat image_i_V_line=image_i_HSV_s_gauss.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
		Eigen::VectorXf b_sparse(noneZeroNum);
		int b_postion=0;
		for (int i=0;i<affine_Z_mask_line.rows;i++)
		{
			if (affine_Z_mask_line.at<uchar>(i,0)!=0)
			{
				if (false)//b_postion==0 false
				{
					float b_val=model_z_line.at<float>(i,0);
					b_sparse(b_postion)=b_val;
					b_postion=b_postion+1;
				} 
				else
				{
					float b_val=(float)image_i_V_line.at<uchar>(i,0)-(float)albedo_ref_V_line.at<uchar>(i,0)*(L.at<float>(0,0)-(L.at<float>(3,0)/Nref_line.at<float>(i,0)));
					float R_val=lambda*(model_z_line.at<float>(i,0)-model_z_G_line.at<float>(i,0));
					b_sparse(b_postion)=b_val+R_val;
					b_postion=b_postion+1;
					img_i.push_back(b_val+R_val);
				}
			
			}
		}

		//  x matrix eigen // 
		Eigen::VectorXf x_sparse(noneZeroNum);
		A_sparse.makeCompressed();
		Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
		double t = cv::getTickCount();
		solver.compute(A_sparse);
		t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
		std::cout << "solver.compute(A_sparse) : " << t <<"s"<< std::endl;
		if(solver.info()!=Eigen::Success) {
			// decomposition failed
			cout<<"solver.compute(A_sparse) : fail"<<endl;
		}
		double t_2 = cv::getTickCount();
		x_sparse = solver.solve(b_sparse);
		t_2 = ((double)cv::getTickCount() - t_2)/cv::getTickFrequency();
		std::cout << "solver.solve(b_sparse) : " << t_2 <<"s"<< std::endl;
		if(solver.info()!=Eigen::Success) {
			// solving failed
			cout<<"solving failed"<<endl;
		}
		//std::cout << "#iterations:     " << solver.iterations() << std::endl;
		//std::cout << "estimated error: " << solver.error()      << std::endl;

		//for (int i=0;i<noneZeroNum;i++)
		//{
		//	cout<<x_sparse(i)<<endl;
		//	waitKey(0);
		//}

		Mat Depth_recover=model_z.clone(); Depth_recover.setTo(0);
		int k=0;
		for (int i=0;i<model_z.rows; i++)
		{
			for (int j=0;j<model_z.cols; j++)
			{
				if (affine_Z_mask.at<uchar>(i,j)!=0)
				{
					Depth_recover.at<float>(i,j)=x_sparse(k);
					rec_z.push_back(x_sparse(k));
					k++;
				}
			}
		}
		Depth_recover_v.push_back(Depth_recover);

		float error=0.0;
		for (int e=0; e<img_i.size(); e++)
		{
			float error_temp=abs(img_i[e]-model_v[e]*rec_z[e]);
			error=error+error_temp;
			//cout<<img_i[e]<<endl;
			//cout<<model_v[e]<<endl;
			//cout<<rec_z[e]<<endl;
			//cout<<model_v[e]*rec_z[e]<<endl;
			//cout<<error_temp<<endl;
			//system("pause");
		}
		error=error/img_i.size();
		error_compare.push_back(error);

		string flie_no="02463"; //02463
		string pic_no="02463d216"; //02463d216
		string lambda_str;
		stringstream float2string; //string to float
		float2string << lambda;
		float2string >> lambda_str;
		string save_model_title=data_file_title+"model_rec_test/FRGC_2_FRGC_each/"+pic_no+"/";
		//string save_model_title=data_file_title+"model_rec_test/MPIE_2_FRGC/"+pic_no+"/";
		_mkdir(save_model_title.c_str());
		string save_depth_w = save_model_title +pic_no+"_lambda_"+lambda_str+".ply";
		Mat model_z_temp=model_z+Depth_recover;
		write_model(save_depth_w, affine_Z_mask, image_i, model_x, model_y, model_z_temp);
		string save_depth_e = save_model_title +pic_no+"_lambda_"+lambda_str+"_e.ply";
		write_model(save_depth_e, affine_Z_mask, image_i, model_x, model_y, Depth_recover);
		float2string.str(""); //再次使用前須請空內容
		float2string.clear(); //再次使用前須請空內容yLeaks();
	}

	vector<float> s_dis_compare=error_compare;
	std::sort(error_compare.begin(),error_compare.end(),less<float>());

	vector<int> num;
	for (int i = 0; i < error_compare.size(); i++)
	{
		for (int k=0; k<s_dis_compare.size(); k++)
		{
			if (error_compare[i]==s_dis_compare[k])
			{
				num.push_back(k);
			}
		}
	}

	//for(int k=0;k<num.size();k++)
	//{
	//	cout<<num[k]<<endl;
	//	cout<<error_compare[0]<<endl;
	//	cout<<s_dis_comparelambda_v<<endl;
	//	system("pause");
	//}
	lambda_record=lambda_v[num[0]];
	//cout<<"lambda : "<<lambda_v[num[0]]<<endl;
	*out_z=Depth_recover_v[num[0]];
}
void calculate_albedo(Mat image_i, Mat albedo_ref, Mat model_z, Mat affine_Z_mask, Mat L, Mat h, Mat *albedo_out)
{
	float lambda_2=30.0;
	float sigma=3.0;
	
	// boundary mask //
	Mat boundary_L=affine_Z_mask.clone(); boundary_L.setTo(0);// 左邊界
	Mat boundary_R=affine_Z_mask.clone(); boundary_R.setTo(0);// 右邊界
	Mat boundary_B=affine_Z_mask.clone(); boundary_B.setTo(0);// 下邊界
	Mat boundary_U=affine_Z_mask.clone(); boundary_U.setTo(0);// 上邊界
	Mat boundary_UL=affine_Z_mask.clone(); boundary_UL.setTo(0);// 上左邊界
	Mat boundary_UR=affine_Z_mask.clone(); boundary_UR.setTo(0);// 上右邊界
	Mat boundary_BL=affine_Z_mask.clone(); boundary_BL.setTo(0);// 下左邊界
	Mat boundary_BR=affine_Z_mask.clone(); boundary_BR.setTo(0);// 下右邊界

	Mat data_count_number=model_z.clone(); data_count_number.setTo(0);// 計算affine_Z_mask 內model_z有值的數量
	int count_number=1;
	for (int i = 0; i < model_z.rows; i++)
	{
		for (int j = 0; j < model_z.cols; j++)
		{
			if (affine_Z_mask.at<uchar>(i,j)!=0)
			{
				if (model_z.at<float>(i,j)!=0)
				{
					data_count_number.at<float>(i,j)=count_number;
					count_number=count_number+1;
					if (model_z.at<float>(i-1,j-1)==0)
						boundary_UL.at<uchar>(i,j)=255;
					if (model_z.at<float>(i-1,j)==0)
						boundary_U.at<uchar>(i,j)=255;
					if (model_z.at<float>(i-1,j+1)==0)
						boundary_UR.at<uchar>(i,j)=255;
					if (model_z.at<float>(i,j-1)==0)
						boundary_L.at<uchar>(i,j)=255;
					if (model_z.at<float>(i,j+1)==0)
						boundary_R.at<uchar>(i,j)=255;
					if (model_z.at<float>(i+1,j-1)==0)
						boundary_BL.at<uchar>(i,j)=255;
					if (model_z.at<float>(i+1,j)==0)
						boundary_B.at<uchar>(i,j)=255;
					if (model_z.at<float>(i+1,j+1)==0)
						boundary_BR.at<uchar>(i,j)=255;	
				}
			}
		}
	}
	// 拉直 boundary mask //
	Mat boundary_L_line=boundary_L.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat boundary_R_line=boundary_R.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat boundary_B_line=boundary_B.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat boundary_U_line=boundary_U.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat boundary_UL_line=boundary_UL.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat boundary_UR_line=boundary_UR.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat boundary_BL_line=boundary_BL.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat boundary_BR_line=boundary_BR.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);

	Mat p_normal=model_z.clone(); p_normal.setTo(0);
	Mat q_normal=model_z.clone(); q_normal.setTo(0);
	Mat Nref=model_z.clone(); Nref.setTo(0);
	for (int i = 0; i < model_z.rows; i++)
	{
		for (int j = 0; j < model_z.cols; j++)
			if (affine_Z_mask.at<uchar>(i,j)!=0)
				if (model_z.at<float>(i,j)!=0)
				{
					if (boundary_R.at<uchar>(i,j)==0)
						p_normal.at<float>(i,j)=model_z.at<float>(i,j+1)-model_z.at<float>(i,j);
					else
						p_normal.at<float>(i,j)=model_z.at<float>(i,j-1)-model_z.at<float>(i,j);
					if (boundary_B.at<uchar>(i,j)==0)
						q_normal.at<float>(i,j)=model_z.at<float>(i+1,j)-model_z.at<float>(i,j);
					else
						q_normal.at<float>(i,j)=model_z.at<float>(i-1,j)-model_z.at<float>(i,j);

					Nref.at<float>(i,j)=sqrt(p_normal.at<float>(i,j)*p_normal.at<float>(i,j)+q_normal.at<float>(i,j)*q_normal.at<float>(i,j)+1);
				}
	}
	Mat Nref_line=Nref.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);

	// A matrix sparse eigen//
	int noneZeroNum=cv::countNonZero(affine_Z_mask);
	Mat affine_Z_mask_line=affine_Z_mask.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);

	float a0=3.141592;
	float a1=2*3.141592/sqrt(3.0);
	float c0=1/sqrt(4.0*3.141592);
	float c1=sqrt(3.0)/sqrt(4.0*3.141592);
	Eigen::SparseMatrix<float> A_sparse(noneZeroNum,noneZeroNum);
	A_sparse.setZero();
	typedef Eigen::Triplet<float> T;
	vector< T > tripletList;
	for (int i = 0; i < model_z.rows; i++)
	{
		for (int j = 0; j < model_z.cols; j++)
		{
			if (affine_Z_mask.at<uchar>(i,j)!=0)
			{
				if (false)//data_count_number.at<float>(i,j)-1==0 false
				{
					float D1=1.0;
					int row_location = data_count_number.at<float>(i,j)-1;
					int col_location = data_count_number.at<float>(i,j)-1;
					tripletList.push_back(T(row_location,col_location,D1));
				} 
				else
				{
					float D1=L.at<float>(0,0)*c0*a0+L.at<float>(1,0)*c1*a1*p_normal.at<float>(i,j)+L.at<float>(2,0)*c1*a1*q_normal.at<float>(i,j)-L.at<float>(3,0)*c1*a1;
					float R5=lambda_2*(1-h.at<float>(1,1));
					int row_location = data_count_number.at<float>(i,j)-1;
					int col_location = data_count_number.at<float>(i,j)-1;
					tripletList.push_back(T(row_location,col_location,D1+R5));
					if (data_count_number.at<float>(i,j+1)!=0) //D2
					{
						float R6=lambda_2*(0-h.at<float>(1,2));
						int row_location = data_count_number.at<float>(i,j)-1;
						int col_location = data_count_number.at<float>(i,j+1)-1;
						tripletList.push_back(T(row_location,col_location,R6));
					}
					if (data_count_number.at<float>(i+1,j)!=0) //D3
					{
						float R8=lambda_2*(0-h.at<float>(2,1));
						int row_location = data_count_number.at<float>(i,j)-1;
						int col_location = data_count_number.at<float>(i+1,j)-1;
						tripletList.push_back(T(row_location,col_location,R8));
					}
					//if (data_count_number.at<float>(i-1,j-1)!=0) //R1
					//{
					//	float R1=lambda_2*(0-h.at<float>(0,0));
					//	int row_location = data_count_number.at<float>(i,j)-1;
					//	int col_location = data_count_number.at<float>(i-1,j-1)-1;
					//	tripletList.push_back(T(row_location,col_location,R1));
					//}
					if (data_count_number.at<float>(i-1,j)!=0) //R2
					{
						float R2=lambda_2*(0-h.at<float>(0,1));
						int row_location = data_count_number.at<float>(i,j)-1;
						int col_location = data_count_number.at<float>(i-1,j)-1;
						tripletList.push_back(T(row_location,col_location,R2));
					}
					//if (data_count_number.at<float>(i-1,j+1)!=0) //R3
					//{
					//	float R3=lambda_2*(0-h.at<float>(0,2));
					//	int row_location = data_count_number.at<float>(i,j)-1;
					//	int col_location = data_count_number.at<float>(i-1,j+1)-1;
					//	tripletList.push_back(T(row_location,col_location,R3));
					//}
					if (data_count_number.at<float>(i,j-1)!=0) //R4
					{
						float R4=lambda_2*(0-h.at<float>(0,2));
						int row_location = data_count_number.at<float>(i,j)-1;
						int col_location = data_count_number.at<float>(i,j-1)-1;
						tripletList.push_back(T(row_location,col_location,R4));
					}
					//if (data_count_number.at<float>(i+1,j-1)!=0) //R7
					//{
					//	float R7=lambda_2*(0-h.at<float>(0,2));
					//	int row_location = data_count_number.at<float>(i,j)-1;
					//	int col_location = data_count_number.at<float>(i+1,j-1)-1;
					//	tripletList.push_back(T(row_location,col_location,R7));
					//}
					//if (data_count_number.at<float>(i+1,j+1)!=0) //R9
					//{
					//	float R9=lambda_2*(0-h.at<float>(0,2));
					//	int row_location = data_count_number.at<float>(i,j)-1;
					//	int col_location = data_count_number.at<float>(i+1,j+1)-1;
					//	tripletList.push_back(T(row_location,col_location,R9));
					//}
				}
			}
		}
	}
	A_sparse.setFromTriplets(tripletList.begin(),tripletList.end());

	// b matrix eigen // 
	Mat albedo_ref_HSV;
	cvtColor(albedo_ref,albedo_ref_HSV,CV_BGR2HSV);
	vector<Mat> albedo_ref_HSV_s;
	split(albedo_ref_HSV,albedo_ref_HSV_s);
	Mat albedo_ref_HSV_s_gauss=albedo_ref_HSV_s[2].clone();
	GaussianBlur(albedo_ref_HSV_s[2], albedo_ref_HSV_s_gauss, Size(3,3), sigma, sigma, BORDER_DEFAULT );
	Mat albedo_ref_V_line=albedo_ref_HSV_s[2].reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat albedo_ref_V_G_line=albedo_ref_HSV_s_gauss.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat image_i_HSV;
	cvtColor(image_i,image_i_HSV,CV_BGR2HSV);
	vector<Mat> image_i_HSV_s;
	split(image_i_HSV,image_i_HSV_s);
	Mat image_i_HSV_s_gauss=image_i_HSV_s[2].clone();
	GaussianBlur(image_i_HSV_s[2], image_i_HSV_s_gauss, Size(3,3), sigma, sigma, BORDER_DEFAULT );
	Mat image_i_V_line=image_i_HSV_s_gauss.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Eigen::VectorXf b_sparse(noneZeroNum);
	int b_postion=0;
	for (int i=0;i<affine_Z_mask_line.rows;i++)
	{
		if (affine_Z_mask_line.at<uchar>(i,0)!=0)
		{
			if (false)//b_postion==0 false
			{
				float b_val=albedo_ref_V_line.at<uchar>(i,0);
				b_sparse(b_postion)=b_val;
				b_postion=b_postion+1;
			} 
			else
			{
				float b_val=(float)image_i_V_line.at<uchar>(i,0);
				float R_val=lambda_2*(albedo_ref_V_line.at<uchar>(i,0)-albedo_ref_V_G_line.at<uchar>(i,0));
				b_sparse(b_postion)=b_val+R_val;
				b_postion=b_postion+1;
			}

		}
	}

	//  x matrix eigen // 
	Eigen::VectorXf x_sparse(noneZeroNum);
	A_sparse.makeCompressed();
	Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
	double t = cv::getTickCount();
	solver.compute(A_sparse);
	t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
	//std::cout << "solver.compute(A_sparse) : " << t <<"s"<< std::endl;
	if(solver.info()!=Eigen::Success) {
		// decomposition failed
		cout<<"solver.compute(A_sparse) : fail"<<endl;
	}
	double t_2 = cv::getTickCount();
	x_sparse = solver.solve(b_sparse);
	t_2 = ((double)cv::getTickCount() - t_2)/cv::getTickFrequency();
	//std::cout << "solver.solve(b_sparse) : " << t_2 <<"s"<< std::endl;
	if(solver.info()!=Eigen::Success) {
		// solving failed
		cout<<"solving failed"<<endl;
	}

	//imshow("image_i_HSV_s[2]",image_i_HSV_s[2]);
	//imshow("albedo_ref_HSV_s[2]",albedo_ref_HSV_s[2]);waitKey(1);
	Mat albedo_recover=albedo_ref_HSV_s[2].clone(); //albedo_recover.setTo(0);
	int k=0;
	for (int i=0;i<model_z.rows; i++)
	{
		for (int j=0;j<model_z.cols; j++)
		{
			if (affine_Z_mask.at<uchar>(i,j)!=0)
			{
				//cout<<(float)albedo_recover.at<uchar>(i,j)<<endl;
				albedo_recover.at<uchar>(i,j)=albedo_ref_HSV_s[2].at<uchar>(i,j)+x_sparse(k);
				//cout<<x_sparse(k)<<endl;
				//cout<<(float)albedo_recover.at<uchar>(i,j)<<endl;
				//system("pause");
				k++;
			}
		}
	}

	//*albedo_out=albedo_recover+albedo_ref_HSV_s[2];
	*albedo_out=albedo_recover;
}
void calculate_albedo_iter(Mat image_i, Mat albedo_ref, Mat model_z, Mat affine_Z_mask, Mat L, Mat h, Mat *albedo_out)
{
	// boundary mask //
	Mat boundary_L=affine_Z_mask.clone(); boundary_L.setTo(0);// 左邊界
	Mat boundary_R=affine_Z_mask.clone(); boundary_R.setTo(0);// 右邊界
	Mat boundary_B=affine_Z_mask.clone(); boundary_B.setTo(0);// 下邊界
	Mat boundary_U=affine_Z_mask.clone(); boundary_U.setTo(0);// 上邊界
	Mat boundary_UL=affine_Z_mask.clone(); boundary_UL.setTo(0);// 上左邊界
	Mat boundary_UR=affine_Z_mask.clone(); boundary_UR.setTo(0);// 上右邊界
	Mat boundary_BL=affine_Z_mask.clone(); boundary_BL.setTo(0);// 下左邊界
	Mat boundary_BR=affine_Z_mask.clone(); boundary_BR.setTo(0);// 下右邊界

	Mat data_count_number=model_z.clone(); data_count_number.setTo(0);// 計算affine_Z_mask 內model_z有值的數量
	int count_number=1;
	for (int i = 0; i < model_z.rows; i++)
	{
		for (int j = 0; j < model_z.cols; j++)
		{
			if (affine_Z_mask.at<uchar>(i,j)!=0)
			{
				if (model_z.at<float>(i,j)!=0)
				{
					data_count_number.at<float>(i,j)=count_number;
					count_number=count_number+1;
					if (model_z.at<float>(i-1,j-1)==0)
						boundary_UL.at<uchar>(i,j)=255;
					if (model_z.at<float>(i-1,j)==0)
						boundary_U.at<uchar>(i,j)=255;
					if (model_z.at<float>(i-1,j+1)==0)
						boundary_UR.at<uchar>(i,j)=255;
					if (model_z.at<float>(i,j-1)==0)
						boundary_L.at<uchar>(i,j)=255;
					if (model_z.at<float>(i,j+1)==0)
						boundary_R.at<uchar>(i,j)=255;
					if (model_z.at<float>(i+1,j-1)==0)
						boundary_BL.at<uchar>(i,j)=255;
					if (model_z.at<float>(i+1,j)==0)
						boundary_B.at<uchar>(i,j)=255;
					if (model_z.at<float>(i+1,j+1)==0)
						boundary_BR.at<uchar>(i,j)=255;	
				}
			}
		}
	}
	// 拉直 boundary mask //
	Mat boundary_L_line=boundary_L.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat boundary_R_line=boundary_R.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat boundary_B_line=boundary_B.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat boundary_U_line=boundary_U.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat boundary_UL_line=boundary_UL.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat boundary_UR_line=boundary_UR.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat boundary_BL_line=boundary_BL.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
	Mat boundary_BR_line=boundary_BR.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);

	Mat p_normal=model_z.clone(); p_normal.setTo(0);
	Mat q_normal=model_z.clone(); q_normal.setTo(0);
	Mat Nref=model_z.clone(); Nref.setTo(0);
	for (int i = 0; i < model_z.rows; i++)
	{
		for (int j = 0; j < model_z.cols; j++)
			if (affine_Z_mask.at<uchar>(i,j)!=0)
				if (model_z.at<float>(i,j)!=0)
				{
					if (boundary_R.at<uchar>(i,j)==0)
						p_normal.at<float>(i,j)=model_z.at<float>(i,j+1)-model_z.at<float>(i,j);
					else
						p_normal.at<float>(i,j)=model_z.at<float>(i,j-1)-model_z.at<float>(i,j);
					if (boundary_B.at<uchar>(i,j)==0)
						q_normal.at<float>(i,j)=model_z.at<float>(i+1,j)-model_z.at<float>(i,j);
					else
						q_normal.at<float>(i,j)=model_z.at<float>(i-1,j)-model_z.at<float>(i,j);

					Nref.at<float>(i,j)=sqrt(p_normal.at<float>(i,j)*p_normal.at<float>(i,j)+q_normal.at<float>(i,j)*q_normal.at<float>(i,j)+1);
				}
	}
	Mat Nref_line=Nref.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);

	
	int noneZeroNum=cv::countNonZero(affine_Z_mask);
	Mat affine_Z_mask_line=affine_Z_mask.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);

	float a0=3.141592;
	float a1=2*3.141592/sqrt(3.0);
	float c0=1/sqrt(4.0*3.141592);
	float c1=sqrt(3.0)/sqrt(4.0*3.141592);

	
	float sigma=11.0;
	for (float lambda_2=1.0;lambda_2<=100.0;lambda_2++)
	{
		cout<<"lambda_2 : "<<lambda_2<<endl;
		// A matrix sparse eigen//
		Eigen::SparseMatrix<float> A_sparse(noneZeroNum,noneZeroNum);
		A_sparse.setZero();
		typedef Eigen::Triplet<float> T;
		vector< T > tripletList;
		for (int i = 0; i < model_z.rows; i++)
		{
			for (int j = 0; j < model_z.cols; j++)
			{
				if (affine_Z_mask.at<uchar>(i,j)!=0)
				{
					if (false)//data_count_number.at<float>(i,j)-1==0 false
					{
						float D1=1.0;
						int row_location = data_count_number.at<float>(i,j)-1;
						int col_location = data_count_number.at<float>(i,j)-1;
						tripletList.push_back(T(row_location,col_location,D1));
					} 
					else
					{
						float D1=L.at<float>(0,0)*c0*a0+L.at<float>(1,0)*c1*a1*p_normal.at<float>(i,j)+L.at<float>(2,0)*c1*a1*q_normal.at<float>(i,j)-L.at<float>(3,0)*c1*a1;
						float R5=lambda_2*(1-h.at<float>(1,1));
						int row_location = data_count_number.at<float>(i,j)-1;
						int col_location = data_count_number.at<float>(i,j)-1;
						tripletList.push_back(T(row_location,col_location,D1+R5));
						if (data_count_number.at<float>(i,j+1)!=0) //D2
						{
							float R6=lambda_2*(0-h.at<float>(1,2));
							int row_location = data_count_number.at<float>(i,j)-1;
							int col_location = data_count_number.at<float>(i,j+1)-1;
							tripletList.push_back(T(row_location,col_location,R6));
						}
						if (data_count_number.at<float>(i+1,j)!=0) //D3
						{
							float R8=lambda_2*(0-h.at<float>(2,1));
							int row_location = data_count_number.at<float>(i,j)-1;
							int col_location = data_count_number.at<float>(i+1,j)-1;
							tripletList.push_back(T(row_location,col_location,R8));
						}
						//if (data_count_number.at<float>(i-1,j-1)!=0) //R1
						//{
						//	float R1=lambda_2*(0-h.at<float>(0,0));
						//	int row_location = data_count_number.at<float>(i,j)-1;
						//	int col_location = data_count_number.at<float>(i-1,j-1)-1;
						//	tripletList.push_back(T(row_location,col_location,R1));
						//}
						if (data_count_number.at<float>(i-1,j)!=0) //R2
						{
							float R2=lambda_2*(0-h.at<float>(0,1));
							int row_location = data_count_number.at<float>(i,j)-1;
							int col_location = data_count_number.at<float>(i-1,j)-1;
							tripletList.push_back(T(row_location,col_location,R2));
						}
						//if (data_count_number.at<float>(i-1,j+1)!=0) //R3
						//{
						//	float R3=lambda_2*(0-h.at<float>(0,2));
						//	int row_location = data_count_number.at<float>(i,j)-1;
						//	int col_location = data_count_number.at<float>(i-1,j+1)-1;
						//	tripletList.push_back(T(row_location,col_location,R3));
						//}
						if (data_count_number.at<float>(i,j-1)!=0) //R4
						{
							float R4=lambda_2*(0-h.at<float>(0,2));
							int row_location = data_count_number.at<float>(i,j)-1;
							int col_location = data_count_number.at<float>(i,j-1)-1;
							tripletList.push_back(T(row_location,col_location,R4));
						}
						//if (data_count_number.at<float>(i+1,j-1)!=0) //R7
						//{
						//	float R7=lambda_2*(0-h.at<float>(0,2));
						//	int row_location = data_count_number.at<float>(i,j)-1;
						//	int col_location = data_count_number.at<float>(i+1,j-1)-1;
						//	tripletList.push_back(T(row_location,col_location,R7));
						//}
						//if (data_count_number.at<float>(i+1,j+1)!=0) //R9
						//{
						//	float R9=lambda_2*(0-h.at<float>(0,2));
						//	int row_location = data_count_number.at<float>(i,j)-1;
						//	int col_location = data_count_number.at<float>(i+1,j+1)-1;
						//	tripletList.push_back(T(row_location,col_location,R9));
						//}
					}
				}
			}
		}
		A_sparse.setFromTriplets(tripletList.begin(),tripletList.end());

		// b matrix eigen // 
		Mat albedo_ref_HSV;
		cvtColor(albedo_ref,albedo_ref_HSV,CV_BGR2HSV);
		vector<Mat> albedo_ref_HSV_s;
		split(albedo_ref_HSV,albedo_ref_HSV_s);
		Mat albedo_ref_HSV_s_gauss=albedo_ref_HSV_s[2].clone();
		GaussianBlur(albedo_ref_HSV_s[2], albedo_ref_HSV_s_gauss, Size(3,3), sigma, sigma, BORDER_DEFAULT );
		Mat albedo_ref_V_line=albedo_ref_HSV_s[2].reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
		Mat albedo_ref_V_G_line=albedo_ref_HSV_s_gauss.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
		Mat image_i_HSV;
		cvtColor(image_i,image_i_HSV,CV_BGR2HSV);
		vector<Mat> image_i_HSV_s;
		split(image_i_HSV,image_i_HSV_s);
		Mat image_i_HSV_s_gauss=image_i_HSV_s[2].clone();
		GaussianBlur(image_i_HSV_s[2], image_i_HSV_s_gauss, Size(3,3), sigma, sigma, BORDER_DEFAULT );
		Mat image_i_V_line=image_i_HSV_s_gauss.reshape(0,affine_Z_mask.rows*affine_Z_mask.cols);
		Eigen::VectorXf b_sparse(noneZeroNum);
		int b_postion=0;
		for (int i=0;i<affine_Z_mask_line.rows;i++)
		{
			if (affine_Z_mask_line.at<uchar>(i,0)!=0)
			{
				if (false)//b_postion==0 false
				{
					float b_val=albedo_ref_V_line.at<uchar>(i,0);
					b_sparse(b_postion)=b_val;
					b_postion=b_postion+1;
				} 
				else
				{
					float b_val=(float)image_i_V_line.at<uchar>(i,0);
					float R_val=lambda_2*(albedo_ref_V_line.at<uchar>(i,0)-albedo_ref_V_G_line.at<uchar>(i,0));
					b_sparse(b_postion)=b_val+R_val;
					b_postion=b_postion+1;
				}

			}
		}

		//  x matrix eigen // 
		Eigen::VectorXf x_sparse(noneZeroNum);
		A_sparse.makeCompressed();
		Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
		double t = cv::getTickCount();
		solver.compute(A_sparse);
		t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
		std::cout << "solver.compute(A_sparse) : " << t <<"s"<< std::endl;
		if(solver.info()!=Eigen::Success) {
			// decomposition failed
			cout<<"solver.compute(A_sparse) : fail"<<endl;
		}
		double t_2 = cv::getTickCount();
		x_sparse = solver.solve(b_sparse);
		t_2 = ((double)cv::getTickCount() - t_2)/cv::getTickFrequency();
		std::cout << "solver.solve(b_sparse) : " << t_2 <<"s"<< std::endl;
		if(solver.info()!=Eigen::Success) {
			// solving failed
			cout<<"solving failed"<<endl;
		}

		imshow("image_i_HSV_s[2]",image_i_HSV_s[2]);
		imshow("albedo_ref_HSV_s[2]",albedo_ref_HSV_s[2]);waitKey(1);
		Mat albedo_recover=albedo_ref_HSV_s[2].clone(); //albedo_recover.setTo(0);
		int k=0;
		for (int i=0;i<model_z.rows; i++)
		{
			for (int j=0;j<model_z.cols; j++)
			{
				if (affine_Z_mask.at<uchar>(i,j)!=0)
				{
					//cout<<(float)albedo_recover.at<uchar>(i,j)<<endl;
					albedo_recover.at<uchar>(i,j)=albedo_ref_HSV_s[2].at<uchar>(i,j)+x_sparse(k);
					//albedo_recover.at<uchar>(i,j)=x_sparse(k);
					//cout<<x_sparse(k)<<endl;
					//cout<<(float)albedo_recover.at<uchar>(i,j)<<endl;
					//system("pause");
					k++;
				}
			}
		}
		imshow("albedo_recover",albedo_recover);waitKey(0);
	}
	

	//*albedo_out=albedo_recover+albedo_ref_HSV_s[2];
	//*albedo_out=albedo_recover;
}
void calculate_normal(Mat affine_Z_mask, Mat model_x, Mat model_y, Mat model_z,Mat *normal_x_out,Mat *normal_y_out,Mat *normal_z_out)
{
	// boundary mask //
	Mat boundary_L=affine_Z_mask.clone(); boundary_L.setTo(0);// 左邊界
	Mat boundary_R=affine_Z_mask.clone(); boundary_R.setTo(0);// 右邊界
	Mat boundary_B=affine_Z_mask.clone(); boundary_B.setTo(0);// 下邊界
	Mat boundary_U=affine_Z_mask.clone(); boundary_U.setTo(0);// 上邊界
	Mat boundary_UL=affine_Z_mask.clone(); boundary_UL.setTo(0);// 上左邊界
	Mat boundary_UR=affine_Z_mask.clone(); boundary_UR.setTo(0);// 上右邊界
	Mat boundary_BL=affine_Z_mask.clone(); boundary_BL.setTo(0);// 下左邊界
	Mat boundary_BR=affine_Z_mask.clone(); boundary_BR.setTo(0);// 下右邊界

	Mat data_count_number=model_z.clone(); data_count_number.setTo(0);// 計算affine_Z_mask 內model_z有值的數量
	int count_number=1;
	for (int i = 0; i < model_z.rows; i++)
	{
		for (int j = 0; j < model_z.cols; j++)
		{
			if (affine_Z_mask.at<uchar>(i,j)!=0)
			{
				if (model_z.at<float>(i,j)!=0)
				{
					data_count_number.at<float>(i,j)=count_number;
					count_number=count_number+1;
					if (model_z.at<float>(i-1,j-1)==0)
						boundary_UL.at<uchar>(i,j)=255;
					if (model_z.at<float>(i-1,j)==0)
						boundary_U.at<uchar>(i,j)=255;
					if (model_z.at<float>(i-1,j+1)==0)
						boundary_UR.at<uchar>(i,j)=255;
					if (model_z.at<float>(i,j-1)==0)
						boundary_L.at<uchar>(i,j)=255;
					if (model_z.at<float>(i,j+1)==0)
						boundary_R.at<uchar>(i,j)=255;
					if (model_z.at<float>(i+1,j-1)==0)
						boundary_BL.at<uchar>(i,j)=255;
					if (model_z.at<float>(i+1,j)==0)
						boundary_B.at<uchar>(i,j)=255;
					if (model_z.at<float>(i+1,j+1)==0)
						boundary_BR.at<uchar>(i,j)=255;	
				}
			}
		}
	}
	
	Mat a(1,3,CV_32FC1); a.setTo(0);
	Mat b(1,3,CV_32FC1); b.setTo(0);
	Mat c(1,3,CV_32FC1); c.setTo(0);
	Mat normal_x=model_x.clone(); normal_x.setTo(0);
	Mat normal_y=model_x.clone(); normal_y.setTo(0);
	Mat normal_z=model_x.clone(); normal_z.setTo(0);
	for (int i = 0; i < affine_Z_mask.rows; i++)
	{
		for (int j = 0; j < affine_Z_mask.cols; j++)
		{
			if (affine_Z_mask.at<uchar>(i,j)!=0)
			{
				if (boundary_R.at<uchar>(i,j)==0)
				{
					b.at<float>(0,0)=model_x.at<float>(i,j+1)-model_x.at<float>(i,j);
					b.at<float>(0,1)=model_y.at<float>(i,j+1)-model_y.at<float>(i,j);
					b.at<float>(0,2)=model_z.at<float>(i,j+1)-model_z.at<float>(i,j);
				}
				else
				{
					b.at<float>(0,0)=model_x.at<float>(i,j-1)-model_x.at<float>(i,j);
					b.at<float>(0,1)=model_y.at<float>(i,j-1)-model_y.at<float>(i,j);
					b.at<float>(0,2)=model_z.at<float>(i,j-1)-model_z.at<float>(i,j);
				}
				if (boundary_B.at<uchar>(i,j)==0)
				{
					a.at<float>(0,0)=model_x.at<float>(i+1,j)-model_x.at<float>(i,j);
					a.at<float>(0,1)=model_y.at<float>(i+1,j)-model_y.at<float>(i,j);
					a.at<float>(0,2)=model_z.at<float>(i+1,j)-model_z.at<float>(i,j);
				}
				else
				{
					a.at<float>(0,0)=model_x.at<float>(i-1,j)-model_x.at<float>(i,j);
					a.at<float>(0,1)=model_y.at<float>(i-1,j)-model_y.at<float>(i,j);
					a.at<float>(0,2)=model_z.at<float>(i-1,j)-model_z.at<float>(i,j);
				}
				c=a.cross(b);
				float length_c=sqrt(c.at<float>(0,0)*c.at<float>(0,0)+c.at<float>(0,1)*c.at<float>(0,1)+c.at<float>(0,2)*c.at<float>(0,2));
				normal_x.at<float>(i,j)=c.at<float>(0,0)/length_c;
				normal_y.at<float>(i,j)=c.at<float>(0,1)/length_c;
				normal_z.at<float>(i,j)=c.at<float>(0,2)/length_c;
			}
		}
	}
	*normal_x_out=normal_x;
	*normal_y_out=normal_y;
	*normal_z_out=normal_z;
}

// 模型儲存 //
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
			if(msak.at<uchar>(m,n)==0) continue;

			fprintf(fs,"%f %f %f %d %d %d\n",model_x.at<float>(m,n), model_y.at<float>(m,n), model_z.at<float>(m,n), img.at<Vec3b>(m,n)[2], img.at<Vec3b>(m,n)[1], img.at<Vec3b>(m,n)[0]);
		}
	}
	fclose(fs);
}
void write_model_LM(string savePath, vector<Point2f> Reg_LM, Mat model_x, Mat model_y, Mat model_z)
{
	FILE *fs;
	fs=fopen(savePath.c_str(),"wt");//fs=fopen(save_name,"wt");
	fprintf(fs,"ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n",7);	
	fprintf(fs,"%f %f %f %d %d %d\n",model_x.at<float>(Reg_LM[20-1].y,Reg_LM[20-1].x), model_y.at<float>(Reg_LM[20-1].y,Reg_LM[20-1].x), model_z.at<float>(Reg_LM[20-1].y,Reg_LM[20-1].x), 0.0, 255.0, 0.0);
	fprintf(fs,"%f %f %f %d %d %d\n",model_x.at<float>(Reg_LM[29-1].y,Reg_LM[29-1].x), model_y.at<float>(Reg_LM[29-1].y,Reg_LM[29-1].x), model_z.at<float>(Reg_LM[29-1].y,Reg_LM[29-1].x), 0.0, 255.0, 0.0);
	fprintf(fs,"%f %f %f %d %d %d\n",model_x.at<float>(Reg_LM[17-1].y,Reg_LM[17-1].x), model_y.at<float>(Reg_LM[17-1].y,Reg_LM[17-1].x), model_z.at<float>(Reg_LM[17-1].y,Reg_LM[17-1].x), 0.0, 255.0, 0.0);
	fprintf(fs,"%f %f %f %d %d %d\n",model_x.at<float>(Reg_LM[32-1].y,Reg_LM[32-1].x), model_y.at<float>(Reg_LM[32-1].y,Reg_LM[32-1].x), model_z.at<float>(Reg_LM[32-1].y,Reg_LM[32-1].x), 0.0, 255.0, 0.0);
	fprintf(fs,"%f %f %f %d %d %d\n",model_x.at<float>(Reg_LM[35-1].y,Reg_LM[35-1].x), model_y.at<float>(Reg_LM[35-1].y,Reg_LM[35-1].x), model_z.at<float>(Reg_LM[35-1].y,Reg_LM[35-1].x), 0.0, 255.0, 0.0);
	fprintf(fs,"%f %f %f %d %d %d\n",model_x.at<float>(Reg_LM[38-1].y,Reg_LM[38-1].x), model_y.at<float>(Reg_LM[38-1].y,Reg_LM[38-1].x), model_z.at<float>(Reg_LM[38-1].y,Reg_LM[38-1].x), 0.0, 255.0, 0.0);
	fprintf(fs,"%f %f %f %d %d %d\n",model_x.at<float>(Reg_LM[41-1].y,Reg_LM[41-1].x), model_y.at<float>(Reg_LM[41-1].y,Reg_LM[41-1].x), model_z.at<float>(Reg_LM[41-1].y,Reg_LM[41-1].x), 0.0, 255.0, 0.0);
	fclose(fs);
}
void write_model_old(string savePath, Mat msak, Mat img, Mat model_x, Mat model_y, Mat model_z)
{
	int count_warp=0;
	count_warp=cv::countNonZero(msak);

	cv::Mat_<cv::Vec2i> table(480,640);
	for (int m=0; m<480; m++)
	{
		for (int n=0; n<640; n++)
		{
			table(m,n)[0]=n;//x-row
			table(m,n)[1]=m;//y-col
		}
	}
	cv::flip(table,table,0);

	float scale=1.5;
	FILE *fs;
	fs=fopen(savePath.c_str(),"wt");//fs=fopen(save_name,"wt");
	fprintf(fs,"ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n",count_warp);	
	for(int m=0; m<msak.rows; m++)
	{
		for(int n=0; n<msak.cols; n++)
		{
			if(msak.at<uchar>(m,n)==0) continue;

			//fprintf(fs,"%f %f %f %d %d %d\n",model_x.at<float>(m,n), model_y.at<float>(m,n), model_z.at<float>(m,n), img.at<Vec3b>(m,n)[2], img.at<Vec3b>(m,n)[1], img.at<Vec3b>(m,n)[0]);
			fprintf(fs,"%f ",(float)(table(m,n)[0]));
			fprintf(fs,"%f ",(float)(table(m,n)[1]));
			fprintf(fs,"%f ",model_z.at<float>(m,n)*scale);
			fprintf(fs,"%d ",(uchar)img.at<Vec3b>(m,n)[2]);
			fprintf(fs,"%d ",(uchar)img.at<Vec3b>(m,n)[1]);
			fprintf(fs,"%d \n",(uchar)img.at<Vec3b>(m,n)[0]);
		}
	}
	fclose(fs);
}
void write_model_LM_old(string savePath, vector<Point2f> Reg_LM, Mat model_x, Mat model_y, Mat model_z)
{
	cv::Mat_<cv::Vec2i> table(480,640);
	for (int m=0; m<480; m++)
	{
		for (int n=0; n<640; n++)
		{
			table(m,n)[0]=n;//x-row
			table(m,n)[1]=m;//y-col
		}
	}
	cv::flip(table,table,0);

	float scale=1.5;

	FILE *fs;
	fs=fopen(savePath.c_str(),"wt");//fs=fopen(save_name,"wt");
	fprintf(fs,"ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n",7);	
	fprintf(fs,"%f %f %f %d %d %d\n",(float)(table(Reg_LM[20-1].y,Reg_LM[20-1].x)[0]),(float)(table(Reg_LM[20-1].y,Reg_LM[20-1].x)[1]),model_z.at<float>(Reg_LM[20-1].y,Reg_LM[20-1].x)*scale, 0, 255, 0);
	fprintf(fs,"%f %f %f %d %d %d\n",(float)(table(Reg_LM[29-1].y,Reg_LM[29-1].x)[0]),(float)(table(Reg_LM[29-1].y,Reg_LM[29-1].x)[1]),model_z.at<float>(Reg_LM[29-1].y,Reg_LM[29-1].x)*scale, 0, 255, 0);
	fprintf(fs,"%f %f %f %d %d %d\n",(float)(table(Reg_LM[17-1].y,Reg_LM[17-1].x)[0]),(float)(table(Reg_LM[17-1].y,Reg_LM[17-1].x)[1]),model_z.at<float>(Reg_LM[17-1].y,Reg_LM[17-1].x)*scale, 0, 255, 0);
	fprintf(fs,"%f %f %f %d %d %d\n",(float)(table(Reg_LM[32-1].y,Reg_LM[32-1].x)[0]),(float)(table(Reg_LM[32-1].y,Reg_LM[32-1].x)[1]),model_z.at<float>(Reg_LM[32-1].y,Reg_LM[32-1].x)*scale, 0, 255, 0);
	fprintf(fs,"%f %f %f %d %d %d\n",(float)(table(Reg_LM[35-1].y,Reg_LM[35-1].x)[0]),(float)(table(Reg_LM[35-1].y,Reg_LM[35-1].x)[1]),model_z.at<float>(Reg_LM[35-1].y,Reg_LM[35-1].x)*scale, 0, 255, 0);
	fprintf(fs,"%f %f %f %d %d %d\n",(float)(table(Reg_LM[38-1].y,Reg_LM[38-1].x)[0]),(float)(table(Reg_LM[38-1].y,Reg_LM[38-1].x)[1]),model_z.at<float>(Reg_LM[38-1].y,Reg_LM[38-1].x)*scale, 0, 255, 0);
	fprintf(fs,"%f %f %f %d %d %d\n",(float)(table(Reg_LM[41-1].y,Reg_LM[41-1].x)[0]),(float)(table(Reg_LM[41-1].y,Reg_LM[41-1].x)[1]),model_z.at<float>(Reg_LM[41-1].y,Reg_LM[41-1].x)*scale, 0, 255, 0);
	fclose(fs);
}

// 舊方法 //
// warp sub function //
void set_persp_aff(vector<cv::Point2f> ref_pt, vector<vector<cv::Point2f>>* out_pt, string name, int* affine_num)
{
	FILE* f=fopen(name.c_str(),"rt");
	int a=0,b=0,c=0;
	int aff_num=0;
	fscanf(f,"%d",&aff_num); //區塊數目存入aff_num

	vector<vector<cv::Point2f>> temp_out;
	for(int j=0;j<aff_num;j++)
	{
		vector<Point2f> temp;
		fscanf(f,"%i %i %i",&a,&b,&c);		//讀取框成三角形的三個點

		temp.push_back(ref_pt[a-1]);
		temp.push_back(ref_pt[b-1]);
		temp.push_back(ref_pt[c-1]);
		temp_out.push_back(temp);
	}
	fclose(f);

	*out_pt=temp_out;						//用來回傳紀錄每組點(三個)的向量
	*affine_num=aff_num;					//用來回傳記錄總共有多少組點
}
void aff_warp(vector<cv::Point2f> src_pnt, vector<cv::Point2f> dst_pnt,cv::Mat img_in,cv::Mat *img_out,cv::Mat *mask_out)
{
	cv::Mat_<float> map_matrix(3,3); 

	//宣告跟輸入圖片同大小的 mask
	cv::Mat TMask(img_in.rows, img_in.cols, CV_8UC1);
	TMask.setTo((0));
	
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::Mat tempWarp;			//儲存經過 warping 的結果
	
	//得到轉換的 affine matrix
	map_matrix = cv::getAffineTransform(src_pnt, dst_pnt); //getAffineTransform 從ref model 的區域到 每個人的區域
	cv::warpAffine(img_in, tempWarp, map_matrix, img_in.size());				//利用得到的 affine matrix 旋轉圖像存入 tempWarp
	
	//===============利用 landmark點 製作 mask===================//
	cv::line(TMask, dst_pnt[0], dst_pnt[1], cv::Scalar(255), 1, CV_AA, 0);		//三線形成三角形
	cv::line(TMask, dst_pnt[1], dst_pnt[2], cv::Scalar(255), 1, CV_AA, 0);
	cv::line(TMask, dst_pnt[2], dst_pnt[0], cv::Scalar(255), 1, CV_AA, 0);
	//==========================================================//
	//imshow("bb", TMask);

	/*
	hierarchy: 儲存拓樸的向量
	contour: 儲存輪廓的向量
	*/

	cv::findContours(TMask, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); //利用 mask 找輪廓
	//imshow("cc", TMask);
	cv::drawContours(TMask, contours, -1, cv::Scalar(155), -1, CV_AA, hierarchy, 0); //利用 mask 填滿輪廓
	//imshow("dd", TMask);
	if (img_in.type()==5)				//Mat::type == 5, CV_32F, 32-bit floating point number (float)
	{
		for (int i=0;i<img_in.rows;i++)
		{
			for (int j=0;j<img_in.rows;j++)
			{
				if (img_in.at<float>(i,j)==0)
				{
					TMask.at<uchar>(i,j)=0;
				}
			}
		}
	}

	/*imshow("mask", TMask);
	imshow("temp", tempWarp);*/
	

	tempWarp.copyTo(*img_out, TMask);		//利用得到的 mask 進行遮罩
	/*imshow("tempWarpemp", *img_out);
	waitKey(0);*/
	*mask_out=TMask;						//回傳 mask
}