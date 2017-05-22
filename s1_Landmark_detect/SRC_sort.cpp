#include <string>
#include <vector>
#include "opencv.hpp"

using namespace std;
using namespace cv;

//Classification Initialization
class myclass
{
public:
	myclass(double a, int b):first(a), second(b){}
	double first;
	int second;
	bool operator < (const myclass &m)const 
	{
		return first > m.first;
	}
};

bool less_second(const myclass & m1, const myclass & m2)
{
	return m1.second < m2.second;
}

int str2num(char *str)
{
    int res = 0;
    while( !( (*str >= '0' && *str <= '9') || (*str == '-') || (*str == '+') ) )
		str++;

    //printf("*str='%c'\n",*str);
    
    if(*str == '-')
    {
        str++;
        while(*str >= '0' && *str <= '9')
        {
                res *= 10;
                res += *str - '0';
                ++str;
        }
        return res*-1;
    }

    if(*str == '+')
    {
        str++;
        while(*str >= '0' && *str <= '9')
        {
                res *= 10;
                res += *str - '0';
                ++str;
        }
        return res;
    }
    while(*str >= '0' && *str <= '9')
    {
        res *= 10;
        res += *str - '0';
        ++str;
    }
    return res;
}

string ConvertToString(int value) 
{
  std::stringstream ss;
  ss << value;
  return ss.str();
}

int SRC_Sort_ShowResult(double*& x_predict,vector<string> &name,int num,string query_ID)
{
	vector< myclass > vect;
	int x_len=sizeof(x_predict)/sizeof(float);

	for(int step_i = 0; step_i < num;step_i++)
	{
		myclass my(x_predict[step_i], step_i);
		vect.push_back(my);
	}

	/*cout<<"before sorted:"<<endl;
	for(int i = 0 ; i < vect.size(); i ++) 
		cout << "("<<vect[i].first<<"," << vect[i].second << ")\n";*/
    sort(vect.begin(), vect.end());
    /*cout<<"after sorted by first:"<<endl;
    for(int i = 0 ; i < vect.size(); i ++)
		cout << "(" << vect[i].first << "," << vect[i].second << ")\n";*/

	//char Path_image[] = "Image\\FERET\\Target\\";
	////char FilePath[100];
	//string FilePath;

	//vector<string> wn;

	//for (int rank_n = 0; rank_n < 3; rank_n++ )
	//{
	//	//sprintf(FilePath,"%s\\%s\\*.jpg",Path_image,name[vect[rank_n].second].c_str);

	//	FilePath = string(Path_image) +"\\"+ name[vect[rank_n].second] +"\\*.jpg";

	//	WIN32_FIND_DATA FILEDATA;
	//	HANDLE HDATA;

	//	if ((HDATA = FindFirstFile(FilePath.c_str(),&FILEDATA)) == INVALID_HANDLE_VALUE)
	//	{
	//		printf("No File Be Found.\n\n");
	//	}
	//	else
	//	{
	//		while (1)
	//		{
	//			string ImagePath_Tar;
	//			ImagePath_Tar = string(Path_image) +"\\"+ name[vect[rank_n].second] +"\\"+ FILEDATA.cFileName;
	//			/*sprintf(ImagePath_Tar,"%s\\%s\\%s",Path_image,name[vect[rank_n].second],FILEDATA.cFileName);*/
	//			Mat Im_show_rank = imread(ImagePath_Tar,0);

	//			/*char windows_name[100];
	//			sprintf(windows_name,"Rank_%d:",rank_n+1);*/
	//			string windows_name;
	//			windows_name = "Rank_" + ConvertToString(rank_n+1) + ":" ;
	//			wn.push_back(windows_name);
	//			imshow(windows_name,Im_show_rank);//waitKey(0);

	//			if (!FindNextFile(HDATA,&FILEDATA))
	//			{
	//				if (GetLastError() == ERROR_NO_MORE_FILES)
	//				{
	//					break;
	//				}
	//			}
	//		}
	//	}
	//	FindClose(HDATA);
	//}

	//int predict_label = Label_Tar.at<int>(vect[0].second,0);

	if ( vect[0].first > 0.0021 )//0.014 0.01 0.009
		//cout << "Query_ID: " << query_ID.substr(3,query_ID.length()-3).c_str();
		//cout<< "Predict_ID: " << name[vect[0].second].substr(3,name[vect[0].second].length()-3) << endl;
		cout<< "Predict_ID: " << name[vect[0].second] << endl;
	else
		//cout << "Query_ID: " << query_ID.substr(3,query_ID.length()-3).c_str();
		cout<< "Predict_ID: Non Registered" << endl;
	printf("------------------------------------------------\n");
	//Mat Predict_img;
	//char Predictfile[50],Predictfile_path[50];
	//string	test_string=name[vect[0].second].substr(3,name[vect[0].second].length());
	//strcpy(Predictfile, test_string.c_str());
	//sprintf(Predictfile_path,"warp_regist\\result_ID_%s.jpg",Predictfile);
	//Predict_img=imread(Predictfile_path,0);
	//imshow("Predict_img",Predict_img);
	//cvDestroyWindow("Predict_img");
	//imshow("Predict_img",Predict_img);
	//string QID[10],PID[10];
	//if ( query_ID.size() == 7 && vect[0].first > 0.014)
	//{
	//	QID[0] = query_ID.c_str()[0];
	//	PID[0] = name[vect[0].second].c_str()[3];
	//}
	//else if ( query_ID.size() == 8 && vect[0].first > 0.014)
	//{
	//	for (int i=0;i<2;i++)
	//	{
	//		QID[i] = query_ID.c_str()[i];
	//		PID[i] = name[vect[0].second].c_str()[i+3];
	//	}
	//}
	//else if ( query_ID.size() == 9 && vect[0].first > 0.014)
	//{
	//	for (int i=0;i<3;i++)
	//	{
	//		QID[i] = query_ID.c_str()[i];
	//		PID[i] = name[vect[0].second].c_str()[i+3];
	//	}
	//}


	////cout << QID->c_str() << PID->c_str();
	//if ( strcmp( QID->c_str(), PID->c_str() )==0 && QID->size() !=0)
	////if ( strcmp( query_ID.c_str(), name[vect[0].second].c_str() )==0 )
	//	return 1;
	//else if ( QID->size() ==0 && PID->size() ==0)
	//	return 0;
	//else
	//	return 0;

	/*waitKey(0);

	for (int rank_n = 0; rank_n < 3; rank_n++ )
	{
		destroyWindow(wn[rank_n]);
	}*/

}


int SRC_Sort_ShowResult2(double*& x_predict,int num)
{
	vector< myclass > vect;
	int x_len=sizeof(x_predict)/sizeof(float);

	for(int step_i = 0; step_i < num;step_i++)
	{
		myclass my(x_predict[step_i], step_i);
		vect.push_back(my);
	}

    sort(vect.begin(), vect.end());

	if ( vect[0].first > 0.014 )
		return vect[0].second;
	else
		return 1000;
}