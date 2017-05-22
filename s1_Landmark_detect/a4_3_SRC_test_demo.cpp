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
	//string angle="L90_11_0";// F00_05_1 //
	string angle="L15_14_0";
	/*cout<<"input angle : ";
	cin>>angle;*/
	// R90_24_0 R75_01_0 R60_20_0 R45_19_0 R30_04_1 R15_05_0 //
	// L90_11_0 L75_12_0 L60_09_0 L45_08_0 L30_13_0 L15_14_0 //
	string glass_model_type="no_glass_1";    //no_glass_1
	// glass_frame_full glass_frame_half glass_frame_none no_glass //
	string recog_data_sPath_title=data_file_title+"Test_Data_xml"+"/"+light+"/";

	// load test image //
	string test_image_title=data_file_title+"Test_Data_xml"+"/"+light+"/"+angle+"/"+glass_model_type+"/"+"test"+"/";
	vector<string> test_img_name;
	Load_insideFile_name(test_image_title,&test_img_name);

	// data save //
	string x_predict_Path=data_file_title+"Test_Data_xml"+"/"+light+"/"+angle+"/"+glass_model_type+"/"+"x_predict.txt";
	string recog_result_path=data_file_title+"Test_Data_xml"+"/"+light+"/"+angle+"/"+glass_model_type+"/"+"recog_result.txt";
	string src_rank_path=data_file_title+"Test_Data_xml"+"/"+light+"/"+angle+"/"+glass_model_type+"/"+"src_rank.txt"; //add
	FILE *fs;
	fs=fopen(x_predict_Path.c_str(),"wt");
	FILE *fd;
	fd=fopen(recog_result_path.c_str(),"wt");
	FILE *ff;
	ff=fopen(src_rank_path.c_str(),"wt");

	//--Sparse Coefficient
	double sparsity = 0.1; //--create sparse X 
	double tol = 0.000001;//0.01 //--sparse homotopy tolerance(control)
	int iter = 300;//iteration

	double lambda = 0.000001;//--homotopy of lambda(control)
	int maxIter = 300;//--max iter number //80
	int n,m;
	int n_b,m_b;

	double normB, normX0, normA;
	normB  = 0;
	normX0 = 0;

	float FR_SRC = 0;
	float Total_TestNum = 0;
	float Total_TargetNum = 0;
	double oop_sum=0;

	double t = (double)getTickCount();
	for (int i = 0; i < test_img_name.size(); i++) //for (int i = 0; i < test_img_name.size(); i++)
	{
		Total_TestNum=Total_TestNum+1;
		
		// load train data //
		string train_data_title=data_file_title+"Test_Data_xml"+"/"+light+"/"+angle+"/"+glass_model_type+"/"+"train"+"/"+test_img_name[i]+"/";

		string name_xml=train_data_title+"name.xml";
		string data_xml=train_data_title+"data.xml";
		//-----Read the Data from xml
		Mat Data_Tar_regist;
		vector<string> Name_Tar_regist;
		FileStorage FS_NT_test;
		FS_NT_test.open(name_xml, FileStorage::READ);
		FS_NT_test["Name_Tar"] >> Name_Tar_regist;
		FS_NT_test.release();
		FileStorage FS_DT_test;
		FS_DT_test.open(data_xml, FileStorage::READ);
		FS_DT_test["Data_Tar"] >> Data_Tar_regist;
		FS_DT_test.release();

		/*cout<<Data_Tar_regist.cols<<endl;
		system("pause");*/

		//---------------------Construct the Target matrix of A---------------------//
		m = Data_Tar_regist.cols,n = Data_Tar_regist.rows; Total_TargetNum=Data_Tar_regist.rows;
		double *A  = new double[m*n];

		for(int i = 0 ; i < m; i++)
		{
			normA  = 0;
			for(int j = 0 ; j < n; j++)
			{
				A[j*m + i] = Data_Tar_regist.at<float>(j,i); //--put the target data
				normA += A[j*m + i]*A[j*m + i];
			}
			normA = sqrt(normA);
			for(int j = 0 ; j < n; j++)
			{
				A[j*m + i] = (double)(A[j*m + i] / normA);
			}
		}
		

		// load test data //
		string test_data_title=data_file_title+"Test_Data_xml"+"/"+light+"/"+angle+"/"+glass_model_type+"/"+"test"+"/"+test_img_name[i]+"/";

		name_xml=test_data_title+"name.xml";
		data_xml=test_data_title+"data.xml";
		//-----Read the Data from xml
		Mat Data_Tar_test;
		vector<string> Name_Tar_test;
		FS_NT_test.open(name_xml, FileStorage::READ);
		FS_NT_test["Name_Tar"] >> Name_Tar_test;
		FS_NT_test.release();
		FS_DT_test.open(data_xml, FileStorage::READ);
		FS_DT_test["Data_Tar"] >> Data_Tar_test;
		FS_DT_test.release();

		double oop;
		for(int gg=0;gg<Data_Tar_regist.rows; gg++)
		{
			//double base=cv::norm(Data_Tar_test.t(),NORM_L2);
			//double each=cv::norm(Data_Tar_regist.row(gg),NORM_L2);
			oop=cv::norm(Data_Tar_test.t(),Data_Tar_regist.row(gg),NORM_L2);
			cout<<oop<<endl;
		}
		
		//oop_sum=oop_sum+oop;
		//cout<<oop<<endl;
		//system("pause");

		//----B---//
		m_b = Data_Tar_test.rows;
		double *b  = new double[m_b];
		for(int j = 0 ; j < m_b; j++)
		{
			b[j] = Data_Tar_test.at<float>(j,0) ;
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
		//string char_str="test";
		//FR_SRC += SRC_Sort_ShowResult(x_predict,Name_Tar_regist,n,char_str);

		vector<double> predict_value;
		vector<int> location_rank;
		vector<string> name_rank;
		double max_value=-100;
		int location=-100;
		for(int step_i = 0; step_i < n;step_i++)
		{
			location_rank.push_back(step_i);
			name_rank.push_back(Name_Tar_regist[step_i]);

			predict_value.push_back(x_predict[step_i]);
			location=(max_value>x_predict[step_i])?location:step_i;
			max_value=(max_value>x_predict[step_i])?max_value:x_predict[step_i];
			fprintf(fs,"%f ",x_predict[step_i]);
		}
		fprintf(fs,"\n");
		//cout<<location<<endl;
		//cout<<Name_Tar_regist[location]<<endl;
		//cout<<max_value<<endl;

		if (!strcmp(Name_Tar_regist[location].c_str(),Name_Tar_test[0].c_str()))
		{
			FR_SRC=FR_SRC+1;
			//cout<<FR_SRC<<endl;
			fprintf(fd,"%s %s\n",Name_Tar_test[0],Name_Tar_regist[location]);
			//fprintf(ff,"%s %s\n",Name_Tar_test[0],Name_Tar_regist[location]);
			cout<<test_img_name[i]<<" "<<Name_Tar_regist[location]<<endl;
		}
		else
		{
			fprintf(fd,"%s %s\n",Name_Tar_test[0],Name_Tar_regist[location]);
			//fprintf(ff,"%s %s\n",Name_Tar_test[0],Name_Tar_regist[location]);
			cout<<test_img_name[i]<<" "<<Name_Tar_regist[location]<<endl;
		}
		//Release Memory
		delete [] b;
		delete [] x_predict;
		delete [] A;

		vector<double> predict_value_compare=predict_value;
		std::sort(predict_value.begin(),predict_value.end(),less<double>());
		std::reverse(predict_value.begin(),predict_value.end());

		vector<int> location_rank_temp;
		vector<string> name_rank_temp;
		vector<double> new_pre_val;
		//cout << predict_value.size() << " " << predict_value_compare.size() << endl;
		//system("pause");

		for (int kk = 0; kk < predict_value.size(); kk++)
		{
			for (int k=0; k<predict_value_compare.size(); k++)
			{
				if (predict_value[kk]==0)
				{
					continue;
				}
				if (predict_value[kk]==predict_value_compare[k])
				{
					location_rank_temp.push_back(location_rank[k]);
					name_rank_temp.push_back(name_rank[k]);
					new_pre_val.push_back(predict_value_compare[k]);
				}
			}
		}
		
		int rank_num=location_rank_temp.size();
		if (rank_num>5)
		{
			rank_num=5;
		}
		for (int iii = 0; iii < rank_num; iii++)
		{
			if (iii==0)
			{
				fprintf(ff,"%s  %s\n",Name_Tar_test[0],Name_Tar_regist[location]);
			}
			cout<< iii+1 <<" "<<name_rank_temp[iii]<<" "<<new_pre_val[iii]<<endl;
			//fprintf(ff,"%d %s %f\n",iii+1,name_rank_temp[iii],new_pre_val[iii]);
			fprintf(ff,"%d  %s  %f\n",iii+1,name_rank_temp[iii],new_pre_val[iii]);
			//system("pause");
		}
		//system("pause");
		fprintf(ff,"\n");
	}
	t = ((double)getTickCount() - t)/getTickFrequency();
	cout<<"total : "<< t <<" sec."<<endl;
	float result_rate=(FR_SRC/Total_TestNum)*100;

	fprintf(ff,"light : %s angle : %s glass_model_type : %s\n",light,angle,glass_model_type);
	fprintf(ff,"registered people : %f\n",Total_TargetNum);
	fprintf(ff,"tested people : %f\n",Total_TestNum);
	fprintf(ff,"correct people : %f\n",FR_SRC);
	fprintf(ff,"Rate : %f\n",result_rate);
	
	fclose(fs);
	fclose(fd);
	fclose(ff);

	//cout<<oop_sum/9<<endl;

	cout<<"light : "<<light<<" angle : "<<angle<<" glass_model_type : "<<glass_model_type<<endl;
	cout<<"註冊人數 : "<<Total_TargetNum<<endl;
	cout<<"測試人數 : "<<Total_TestNum<<endl;
	cout<<"正確人數 : "<<FR_SRC<<endl;
	cout<<"辨識率 : "<<result_rate<<endl;

	cout << '\a';
	system("pause");
	return 0;
}