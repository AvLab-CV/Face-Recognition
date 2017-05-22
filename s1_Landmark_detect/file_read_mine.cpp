#include "stdlib.h"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void Load_insideFile_name(string input_path, vector<string>* output_path)
{
	vector<string> temp_name_save;
	int testSampleCount = 0; //���դH�Ʋέp
	DIR *DP;
	struct dirent *DirpathP;
	DP = opendir(input_path.c_str());
	while (DirpathP = readdir(DP))
	{
		//�p�G���[�U�����@��  Ū���X�Ӫ��ɮ׷|���I�I
		if( strcmp(DirpathP->d_name, ".") != 0 && strcmp(DirpathP->d_name, "..") != 0 && strcmp(DirpathP->d_name, "Thumbs.db") != 0 )
		{
			testSampleCount=testSampleCount+1;
			//cout<<DirpathP->d_name<<endl;
			temp_name_save.push_back(DirpathP->d_name);
			//system("pause");
		}
	}
	//cout<<"�@"<<testSampleCount<<"�H�i�����"<<endl;

	*output_path=temp_name_save;
}

void Read_PLY(string name_model, string name_index,string name_mask,Mat *Model_x,Mat *Model_y,Mat *Model_z,Mat *Model_r,Mat *Model_g,Mat *Model_b,Mat *Model_nx,Mat *Model_ny,Mat *Model_nz)
{
	string input_str; //��J�r��

	ifstream fin;
	fin.open(name_model);
	//cout<<name_model<<endl;
	if(!fin) { 
		cout << "�L�kŪ���ɮ�\n"; 
		// �䥦�B�z 
	}
	else
	{
		//cout << "Ū���ɮצ��\\n"; 
	}

	// ply ��Ū��
	getline(fin,input_str); //ply
	getline(fin,input_str); //format ascii 1.0
	getline(fin,input_str); //comment VCGLIB generated
	getline(fin,input_str); //element vertex INPUT_NUMBER
	char str_v[100];
	strcpy(str_v, input_str.c_str()); //string to char
	char *delim = " "; //�P�_�r�� �J��ɤ���
	char * pch;
	pch = strtok(str_v,delim);
	vector<string> vectorString;
	while (pch != NULL)
	{
		vectorString.push_back(pch);
		//printf ("%s\n",pch);
		pch = strtok (NULL, delim);
	}
	int vertex_num=0;
	for(int i=2;i<3;i++)
	{
		stringstream string2float; //string to float
		string2float << vectorString[i];
		string2float >> vertex_num;
		//std::cout <<"vertex_num = "<< vertex_num<< std::endl;
		string2float.str(""); //�A���ϥΫe���ЪŤ��e
		string2float.clear(); //�A���ϥΫe���ЪŤ��e
	}
	getline(fin,input_str); //property float x
	getline(fin,input_str); //property float y
	getline(fin,input_str); //property float z
	getline(fin,input_str); //property float nx
	getline(fin,input_str); //property float ny
	getline(fin,input_str); //property float nz
	getline(fin,input_str); //property uchar red
	getline(fin,input_str); //property uchar green
	getline(fin,input_str); //property uchar blue
	getline(fin,input_str); //property uchar alpha
	getline(fin,input_str); //element face 0
	getline(fin,input_str); //property list uchar int vertex_indices
	getline(fin,input_str); //end_header

	// �}�lŪ���I
	int index_vertex_num=0;
	cv::Mat_<float> point(vertex_num,3); //�ŧi�x�s3d point���x�}
	cv::Mat_<float> color(vertex_num,3); //�ŧi�x�scolor���x�}
	cv::Mat_<float> normal(vertex_num,3); //�ŧi�x�scolor���x�}
	while(index_vertex_num<vertex_num)
	{
		getline(fin,input_str); //�o����r��
		char str_v[100];
		strcpy(str_v, input_str.c_str()); //string to char
		char *delim = " "; //�P�_�r�� �J��ɤ���
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
		for(int i=0;i<3;i++) //point location x y z
		{
			float point_of_v;
			stringstream string2float; //string to float
			string2float << vectorString[i];
			string2float >> point_of_v;
			point.at<float>(index_vertex_num,i)=point_of_v;
			//cout<<point[index_vertex_num][i]<<endl;
			string2float.str(""); //�A���ϥΫe���ЪŤ��e
			string2float.clear(); //�A���ϥΫe���ЪŤ��e
		}
		for(int i=3;i<6;i++) //point normal x y z
		{
			float normal_of_v;
			stringstream string2float; //string to float
			string2float << vectorString[i];
			string2float >> normal_of_v;
			normal.at<float>(index_vertex_num,i-3)=normal_of_v;
			//cout<<point[index_vertex_num][i]<<endl;
			string2float.str(""); //�A���ϥΫe���ЪŤ��e
			string2float.clear(); //�A���ϥΫe���ЪŤ��e
		}
		for(int j=6;j<9;j++) //point color R G B
		{
			float color_of_v;
			stringstream string2float; //string to float
			string2float << vectorString[j];
			string2float >> color_of_v;
			color.at<float>(index_vertex_num,j-6)=color_of_v;
			//cout<<color[index_vertex_num][j-3]<<endl;
			string2float.str(""); //�A���ϥΫe���ЪŤ��e
			string2float.clear(); //�A���ϥΫe���ЪŤ��e
		}
		//system("pause");
		index_vertex_num++;
	}
	fin.close();

	// load_index
	FILE *findex;
	findex=fopen(name_index.c_str(),"r");
	cv::Point2f temp;
	vector<cv::Point2f> model_index;
	float model_index_x,model_index_y;
	for (int i=0; i<vertex_num; i++)//Ū�J66��landmark�I,�h��ɹs
	{

		fscanf(findex,"%f %f",&model_index_x,&model_index_y);
		temp=cv::Point2f (model_index_x,model_index_y);
		model_index.push_back(temp);
		//cout<<model_index[i][0]<<" "<<model_index[i][1]<<endl;
		//system("pause");
	}
	fclose(findex);
	//system("pause");

	Mat model_mask=imread(name_mask);
	Mat model_x(model_mask.rows,model_mask.cols,CV_32FC1);model_x.setTo(0);
	Mat model_y(model_mask.rows,model_mask.cols,CV_32FC1);model_y.setTo(0);
	Mat model_z(model_mask.rows,model_mask.cols,CV_32FC1);model_z.setTo(0);
	Mat model_nx(model_mask.rows,model_mask.cols,CV_32FC1);model_nx.setTo(0);
	Mat model_ny(model_mask.rows,model_mask.cols,CV_32FC1);model_ny.setTo(0);
	Mat model_nz(model_mask.rows,model_mask.cols,CV_32FC1);model_nz.setTo(0);
	Mat model_R(model_mask.rows,model_mask.cols,CV_32FC1);model_R.setTo(0);
	Mat model_G(model_mask.rows,model_mask.cols,CV_32FC1);model_G.setTo(0);
	Mat model_B(model_mask.rows,model_mask.cols,CV_32FC1);model_B.setTo(0);
	for (int i=0; i<vertex_num; i++)
	{
		model_x.at<float>(model_index[i].x,model_index[i].y)=point.at<float>(i,0);
		model_y.at<float>(model_index[i].x,model_index[i].y)=point.at<float>(i,1);
		model_z.at<float>(model_index[i].x,model_index[i].y)=point.at<float>(i,2);
		model_R.at<float>(model_index[i].x,model_index[i].y)=color.at<float>(i,0);
		model_G.at<float>(model_index[i].x,model_index[i].y)=color.at<float>(i,1);
		model_B.at<float>(model_index[i].x,model_index[i].y)=color.at<float>(i,2);
		model_nx.at<float>(model_index[i].x,model_index[i].y)=normal.at<float>(i,0);
		model_ny.at<float>(model_index[i].x,model_index[i].y)=normal.at<float>(i,1);
		model_nz.at<float>(model_index[i].x,model_index[i].y)=normal.at<float>(i,2);
	}
	*Model_x=model_x;
	*Model_y=model_y;
	*Model_z=model_z;

	*Model_r=model_R;
	*Model_g=model_G;
	*Model_b=model_B;

	*Model_nx=model_nx;
	*Model_ny=model_ny;
	*Model_nz=model_nz;

}
