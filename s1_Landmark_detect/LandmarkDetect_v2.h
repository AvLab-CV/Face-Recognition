#pragma once
#include "stdlib.h"
#include "vector"
//#include "Common.h"
#include <opencv.hpp>

//using namespace ImgUtils;
#define DllExport __declspec(dllexport)

//////////////////////////////////////////////////////////////////////////

class  Data_bs
{
public:
	float s;
	int c;
	float xy[68][4];
	int level;
};

class LandmarkDetector_PCA
{
public:
	class LandmarkModelPCA
	{
	public:
		double *COEFF_;
		double *SCORE_;
		double *feat4_;
		std::vector<double>filt2_;
		double *tsquare_;
		double *latent_;
	};

public:
	LandmarkDetector_PCA();
	virtual ~LandmarkDetector_PCA();

	bool Load_PCA(const char* path);

public:
	LandmarkModelPCA* PCA_;
};



class LandmarkDetector
{
public:
	class  Feature 
	{
	public:
		Feature();
		virtual ~Feature();
	public:
		float *feat[40];
		double scale[40];
		int intervel;
		int imy;
		int imx;
		int pady;
		int padx;
		int dim_r[40];
		int dim_c[40];
	}; //©w¸q pyra

	class Def
	{
	public:
		double* w_;
		double i_;
		std::vector<double> anchor_;
	};

	class Filters
	{
	public:
		int i_;
		double *w_;
	};

	class Components
	{
	public:
		std::vector<double> defid_;
		std::vector<double> filterid_;
		std::vector<double> parent_;
	};

	class LandmarkModel
	{
	public:
		int len_;
		int interval_;
		int sbin_;
		double delta_;
		double thresh_;
		double obj_;	
		int max_width_;
		int max_height_;
		std::vector<Def> defs_; 	
		std::vector<Components> components_; 
		std::vector<Filters> filters_; 
	};
	/////////////////////  detect /////////////////////
	class Data_s
	{
	public:
		int defid_;
		int filterid_;
		int parent_;
		int sizy_;
		int sizx_;
		int filterI_;
		int defI_;
		double* w_;
		int scale;
		int starty;
		int startx;
		int step;
		int level;
		float* score;
		int* Ix;
		int* Iy;
	};

	class Filters_conv
	{
	public:
		double *w_;
	};

	class comp
	{
	public:
		std::vector<Data_s> comp_;
	};

	class	
	{
	public: 
		int dim_r;
		int dim_c;
		//float** r_;
		float* r_; //fconv
	}resp[40];

	class fdim
	{
	public:
		int fd[3];
	};

public:
	LandmarkDetector();
	~LandmarkDetector();

	bool Load(const char* path);
	void detect(std::vector<Data_bs>&,cv::Mat& , LandmarkModel *,char []);
	void detect_L2(std::vector<Data_bs>&,cv::Mat&, LandmarkModel *,char [],int,int);
	//void detect(std::vector<Data_bs>&,ImgInfo<unsigned char>&, LandmarkModel *,char [],float &);
	void detectPCA(std::vector<Data_bs>&,cv::Mat&, LandmarkModel *,double thresh,char [],int,std::vector<double*>,double*);
	void detectPCA_CV(std::vector<Data_bs>&,cv::Mat&, LandmarkModel *,char [],int,std::vector<double*>,cv::Mat,float &);
	void clipboxes(int,int ,std::vector<Data_bs>);
	void nmsface(std::vector<Data_bs>,float overlap, std::vector<Data_bs> &);
	void Clean();
	void delete_model();

private:
	bool ReadModel(void* matVar);
	bool ReadDefs(void* matVar);
	bool Readfilters(void* matVar);
	bool Readcomponents(void* matVar);

	void featpyramid(cv::Mat& , LandmarkModel *);
	//void fconv_MT(thread_data**,float* hog, int* hogdim, int* outdim,Filters_conv *);
	void fconv_25cells(float** ,float* , int* , int* ,std::vector<double*>);
	void fconv_25cells_PCA(float** ,float* , int* , int* ,std::vector<double*>,double*);
	void fconv_25cells_PCA_CV(float** ,float* , int* , int* ,std::vector<double*>,cv::Mat);
	void fconv_17cells(float** ,float* , int* , int* ,std::vector<double*>);
	void fconv_9cells(float** ,float* , int* , int* ,std::vector<double*>);
	void pca_recon(float**,float*,double*,int,int*);
	void modelcomponents(std::vector<comp> &, LandmarkModel *);
	void create_components(std::vector<comp>&, LandmarkModel *,int c,int k);
	void backtrack(float box[], std::vector<int>, std::vector<int>, std::vector<Data_s>,double[],int,int, int* );
	void shiftdt(float** , int** , int**,std::vector<Data_s> &parts,int*);
	void dt1d(float *src, float *dst, int *ptr, int step, int len, double a, double b, int dshift, double dstep);
	//void backtrack(float box[], int  X[], int Y[], std::vector<Data_s>,double[],int,int, int* rootdim,int);

private:
	int numparts;
	int filter_num;
	int posemap[18];
	int FP_level;
	Data_bs bs;
	Data_s* p,child;
	Feature pyra;
	Def x;
	std::vector<double*> filters;

	std::vector<Data_bs> boxes;
	std::vector<int> filter_dim;

public:
	LandmarkModel* model_;
};

float sum(std::vector<float>time);
static void reduce(cv::Mat& dest, cv::Mat& srcimg);
static void resize(cv::Mat& dest, cv::Mat& srcimg); 
//static float* fconv_MT(float* hog, int* hogdim, int* outdim, double* w, int* wdim);
//static float* fconv(float* hog, int* hogdim, int* outdim, double* w, int* wdim);
//static void shiftdt(float** , int** , int** , float* , double* , int ,int , int* ,int);
//static void* ConvProcess_MT(void *thread_arg);
static void ConvProcess_25cells(void *thread_arg,int,int*);
static void ConvProcess_17cells(void *thread_arg,int,int*);
static void ConvProcess_9cells(void *thread_arg,int,int*);
static float* PadArray( float* oldArray, int* dim, int* paddim, int* newdim);
static void features(float** feature ,int* out,float* im , int w,int h , int sbin);
static void MixScore(float &, float* part, int* rootdims);
std::vector<cv::Point2f> LKTrincingProcessLandMark(cv::Mat , cv::Mat ,std::vector<cv::Point2f>LpsPre );
std::vector<cv::Point2f> GPUbrox(cv::Mat , cv::Mat ,std::vector<cv::Point2f>LpsPre ,int);
std::vector<cv::Point2f> GPUTVL1(cv::Mat , cv::Mat ,std::vector<cv::Point2f>LpsPre ,int);
std::vector<cv::Rect> detectAndDisplay(cv:: Mat frame,cv::CascadeClassifier& cascade, cv::CascadeClassifier& nestedCascade );
cv::Rect Face_boundary(std::vector<cv::Point2f> ,int);
//void RecImage(void *p);