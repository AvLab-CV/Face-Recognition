//#include "StdAfx.h"
#include <algorithm>
#include <functional>
//#include <pthread.h>
#include <stdio.h>
#include <time.h>

//#include "ImageProc.h"
#include "LandmarkDetect_v2.h"
#include "matio_private.h"
#include <string>
#include <iostream>
//#include "ImageProc.h"
#include <opencv.hpp>
#include "opencv2/gpu/gpu.hpp"

///Check 內存 ///
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#define new new( _CLIENT_BLOCK, __FILE__, __LINE__)

static inline int square(int x) { return x*x; }

static inline double min(double x, double y) { return (x <= y ? x : y); }
static inline double max(double x, double y) { return (x <= y ? y : x); }

static inline int min(int x, int y) { return (x <= y ? x : y); }
static inline int max(int x, int y) { return (x <= y ? y : x); }

static inline float min(float x, int y) { return (x <= y ? x : y); }
static inline float max(float x, int y) { return (x <= y ? y : x); }

static inline float round(float val){return (int)std::floor((double)val+0.5f); }


using namespace std;
//using namespace cv;

#define DEBUGLOG
#define INF 1E20
#define eps 0.0001

int ccc=20; //25-20=5
int ccc2=5; //25-20+5=10

double uu[9] = {1.0000, 
	0.9397, 
	0.7660, 
	0.500, 
	0.1736, 
	-0.1736, 
	-0.5000, 
	-0.7660, 
	-0.9397};
double vv[9] = {0.0000, 
	0.3420, 
	0.6428, 
	0.8660, 
	0.9848, 
	0.9848, 
	0.8660, 
	0.6428, 
	0.3420};
LandmarkDetector::LandmarkDetector()	:	model_(NULL)
{
	//std::cout << "constructor called.." << std::endl;
	//cout<<"model_ create  : "<< model_<<endl;
	//model_ = new LandmarkModel;
}

LandmarkDetector::~LandmarkDetector()
{
	if(model_ != NULL)
	{
		//std::cout << "destructor called.." << std::endl;
		delete model_;
	}
	//std::cout << "destructor called.." << std::endl;
	//delete model_;
}

LandmarkDetector_PCA::LandmarkDetector_PCA()
	:	PCA_(NULL)
{
}

LandmarkDetector_PCA::~LandmarkDetector_PCA()
{
	if(PCA_ != NULL)
	{
		delete PCA_;
	}
}


LandmarkDetector::Feature::Feature()
{
	for (int i=0;i<40;i++)
	{
		this->dim_c[i] = 0;
		this->dim_r[i] = 0;
		this->feat[i] = NULL;
		this->scale[i]=0;
	}
	this->intervel = 0;
	this->imx=0;
	this->imy=0;
	this->padx=0;
	this->pady=0;
}

LandmarkDetector::Feature::~Feature()
{
	for (int i=0;i<40;i++)
	{
		if(this->feat[i]  != NULL)
		{
			delete [] this->feat[i] ;
			this->feat[i]  = NULL;
		}
	}
}

void LandmarkDetector::Clean()
{
	this->pyra.intervel = 0;
	this->pyra.imx=0;
	this->pyra.imy=0;
	this->pyra.padx=0;
	this->pyra.pady=0;

	for (int i=0; i<this->FP_level; i++)
	{
		if(this->pyra.feat[i]  != NULL)
		{
			delete [] this->pyra.feat[i] ;
			this->pyra.feat[i]  = NULL;
		}

		this->pyra.dim_c[i] = 0;
		this->pyra.dim_r[i] = 0;
		this->pyra.scale[i]=0;
	}


	for (int i=this->FP_level-ccc; i<this->FP_level-ccc+ccc2; i++) //
	{


		if (this->resp[i].dim_c !=NULL)
		{
			this->resp[i].dim_c = NULL;
		}

		if (this->resp[i].dim_r !=NULL)
		{
			this->resp[i].dim_r = NULL;
		}

		if (this->resp[i].r_ !=NULL)
		{
			delete [] this->resp[i].r_;
			this->resp[i].r_ = NULL;
		}
	}
}

void LandmarkDetector::delete_model()
{

	delete model_;
	//_CrtDumpMemoryLeaks();
	cout<<"model_ delete  : "<< model_<<endl;
}

int RetrieveNum(void* in)
{	
	matvar_t* matvar = (matvar_t*)in;
	if ( matvar->rank == 0 )
		return 0;
	int ret = matvar->dims[0];
	for ( int i = 1; i < matvar->rank; i++ )
		ret *= matvar->dims[i];
	return ret;
}

bool RetrieveDouble(void* matvar, double& out, const char* name)
{
	matvar_t* retvar = Mat_VarGetStructFieldByName( (matvar_t*)matvar, name, 0);
	if(retvar && RetrieveNum(retvar) == 1)
	{
		out = *((double*)retvar->data);
#ifdef DEBUGLOG
		//::printf("%s : %f\n", name, out);
#endif
	}
	else
	{
		return false;	
	}

	return true;
}

bool RetrieveDouble(void* matvar, int& out, const char* name)
{
	matvar_t* retvar = Mat_VarGetStructFieldByName( (matvar_t*)matvar, name, 0);
	if(retvar && RetrieveNum(retvar) == 1)
	{
		out = *((double*)retvar->data);
#ifdef DEBUGLOG
		//::printf("%s : %d\n", name, out);
#endif
	}
	else
	{
		return false;	
	}

	return true;
}

bool RetrieveDoubleArray(void* matvar, int& outSize, double** out, const char* name, bool allocate)
{
	matvar_t* retvar = Mat_VarGetStructFieldByName( (matvar_t*)matvar, name, 0);
	if(retvar)
	{
		//::printf("[%d] %s\n", retvar->data_type, retvar->name);
		outSize = RetrieveNum(retvar);
		if(outSize == 0)
			return false;

		if(allocate)
		{
			if(*out != NULL)
			{
				delete [] *out;
			}
			*out = new double[outSize];
		}
		::memcpy(*out, retvar->data, sizeof(double)*outSize);
	}
	else
	{
		return false;
	}
	return true;
}

bool LandmarkDetector::Load(const char* path)
{
	mat_t    *mat;
	matvar_t *matvar;
	mat = Mat_Open( path, MAT_ACC_RDONLY );
	if(!mat)
	{
		printf("Fail to open the file!");
		return false;
	}

	matvar = Mat_VarRead(mat,"model");
	if(!matvar)
	{
		printf("Fail to parse the content!");
		Mat_Close(mat);
		return false;
	}
	else
	{
		// Try to parse "model" in the mat file
		if(!ReadModel(matvar))
		{
			Mat_VarFree(matvar);
			Mat_Close(mat);
			::printf("Parse fail!!\n");
			return false;
		}

        //Mat_VarFree(matvar);    
		Mat_Close(mat);
		return true;
	}

}

bool LandmarkDetector_PCA::Load_PCA(const char* path)
{
	// Initialize the PCA model
	if(PCA_ != NULL)
	{
		delete PCA_;
	}
	PCA_ = new LandmarkModelPCA();

	mat_t    *mat;
	matvar_t *matvar;
	mat = Mat_Open( path, MAT_ACC_RDONLY );
	if(!mat)
	{
		printf("Fail to open the file!");
		return false;
	}

	matvar = Mat_VarRead(mat,"PCA");
	if(!matvar)
	{
		printf("Fail to parse the content!");
		Mat_Close(mat);
		return false;
	}
	else
	{
		matvar_t *retvar;
		matvar_t *matvar_pca = (matvar_t *)matvar;
		double* temp;
		retvar = Mat_VarGetStructFieldByName( (matvar_t*)matvar_pca, "COEFF", 0);
		// Read COEFF //
		if(retvar)
		{
			PCA_->COEFF_ = ((double*)retvar->data);
		}
		else
			return false;
		// Read SCORE //
		retvar = Mat_VarGetStructFieldByName( (matvar_t*)matvar_pca, "SCORE", 0);
		if(retvar)
		{
			PCA_->SCORE_ = ((double*)retvar->data);
		}
		else
			return false;
		// Read feat4 //
		retvar = Mat_VarGetStructFieldByName( (matvar_t*)matvar_pca, "feat4", 0);
		if(retvar)
		{
			PCA_->feat4_ = ((double*)retvar->data);
		}
		else
			return false;
		// Read latent //
		retvar = Mat_VarGetStructFieldByName( (matvar_t*)matvar_pca, "latent", 0);
		if(retvar)
		{
			PCA_->latent_ = ((double*)retvar->data);
		}
		else
			return false;
		// Read tsquare //
		retvar = Mat_VarGetStructFieldByName( (matvar_t*)matvar_pca, "tsquare", 0);
		if(retvar)
		{
			PCA_->tsquare_ = ((double*)retvar->data);
		}
		else
			return false;

		// Read filt2 //
		retvar = Mat_VarGetStructFieldByName( (matvar_t*)matvar_pca, "filt2", 0);
		if(retvar)
		{
			int filtersNum = RetrieveNum(retvar);
			matvar_t **cells = (matvar_t **)retvar->data;
			matvar_t *matvarTmp;
			for ( int i = 0; i < filtersNum; i++ )
			{
				matvarTmp = *cells++;
				PCA_->filt2_.push_back(*(double*)matvarTmp->data);
			}
		}
		else
			return false;
	}
	return true;
}

bool LandmarkDetector::ReadModel(void* in)
{
	//Initialize the internal model
	if(model_ != NULL)
	{
		//cout<<"model_ delete  : "<< model_<<endl;
		delete model_;
	}
	model_ = new LandmarkModel();
	//cout<<"model_ create  : "<< model_<<endl;

	//matvar_t *retvar;
	matvar_t *matvar = (matvar_t *)in;

	if(!RetrieveDouble(matvar, model_->len_, "len"))
		return false;

	if(!RetrieveDouble(matvar, model_->interval_, "interval"))
		return false;	
	// 5 levels for each octave //
	if (model_->interval_!=5)
		model_->interval_=5;
	//printf("interval change to %d\n",model_->interval_);


	if(!RetrieveDouble(matvar, model_->sbin_, "sbin"))
		return false;

	if(!RetrieveDouble(matvar, model_->delta_, "delta"))
		return false;

	if(!RetrieveDouble(matvar, model_->thresh_, "thresh"))
		return false;

	//if(!RetrieveDouble(matvar, model_->obj_, "obj"))   //原作者3個model才有
	//	return false;


	double size[2];
	int useless,max_width_=0,max_height_=0;
	double* sPtr = &size[0];
	if(!RetrieveDoubleArray(matvar, useless, &sPtr, "maxsize", false))
		return false;
	model_->max_width_ = size[0];
	model_->max_height_ = size[1];
	//printf("maxsize=[%d %d]",model_->max_width_,model_->max_height_);

	if(!ReadDefs(matvar))
		return false;

	if(!Readfilters(matvar))
		return false;

	if(!Readcomponents(matvar))
		return false;

	_CrtDumpMemoryLeaks();
	return true;
}

//read defs //
bool LandmarkDetector::ReadDefs(void* in)
{
	matvar_t *retvar;
	matvar_t *matvar = (matvar_t *)in;

	retvar = Mat_VarGetStructFieldByName( (matvar_t*)matvar, "defs", 0);
	if(retvar)
	{
		//std::vector<Def>& refDefs = this->model_->defs_;
		std::vector<Def>& refDefs = model_->defs_;

		vector<Def>().swap(refDefs);//refDefs.clear();
		int defNum = RetrieveNum(retvar);
		refDefs.reserve(defNum);
		Def newDef;
		matvar_t **cells = (matvar_t **)retvar->data;  //def裡的w
		matvar_t *matvarTmp;
		for ( int i = 0; i < defNum; i++ )
		{			
			matvarTmp = *cells++;
			int Num_w = RetrieveNum(matvarTmp);
			for (int ii=0;ii<Num_w;ii++)
			{
				newDef.w_=((double*)matvarTmp->data);
			}
			matvarTmp = *cells++;
			newDef.i_ = *((double*)matvarTmp->data);  //讀取defs裡的i
		
			matvarTmp = *cells++;
			int Num_a = RetrieveNum(matvarTmp);
			for (int ii=0;ii<Num_a;ii++)
			{
				newDef.anchor_.push_back(((double*)matvarTmp->data)[ii]);
			}
			refDefs.push_back(newDef);
			vector<double>().swap(newDef.anchor_);//newDef.anchor_.clear();
		}

		return true;
	}
	else
		return false;
}

//read filters //
bool LandmarkDetector::Readfilters(void* in)
{
	matvar_t *retvar;
	matvar_t *matvar = (matvar_t *)in;
	retvar = Mat_VarGetStructFieldByName( (matvar_t*)matvar, "filters", 0);
	if(retvar)
	{
		//std::vector<Filters>& refFilters = this->model_->filters_;
		std::vector<Filters>& refFilters = model_->filters_;

		vector<Filters>().swap(refFilters);//refFilters.clear();
		int filtersNum = RetrieveNum(retvar);
		refFilters.reserve(filtersNum);
		matvar_t **cells = (matvar_t **)retvar->data;
		matvar_t *matvarTmp;
		for ( int i = 0; i < filtersNum; i++ )
		{
			Filters newFilters;
			matvarTmp = *cells++;
			newFilters.w_ = ((double*)matvarTmp->data);
			this->filter_dim.push_back(matvarTmp->dims[0]*matvarTmp->dims[1]*matvarTmp->dims[2]);

			matvarTmp = *cells++;
			newFilters.i_ = *((double*)matvarTmp->data);

			refFilters.push_back(newFilters);
		}
		
		return true;
	}
	else
		return false;
}

//read components //
bool LandmarkDetector::Readcomponents(void* in)
{
	matvar_t *retvar;
	matvar_t *matvar = (matvar_t *)in;

	retvar = Mat_VarGetStructFieldByName( (matvar_t*)matvar, "components", 0);
	if(retvar)
	{
		//std::vector<Components>& refComponents = this->model_->components_;
		std::vector<Components>& refComponents = model_->components_;

		vector<Components>().swap(refComponents);//refComponents.clear();
		int componentsNum = RetrieveNum(retvar);
		refComponents.reserve(componentsNum);
		Components newComponents;
		matvar_t **cells = (matvar_t **)retvar->data;
		matvar_t *matvarTmp;
		for ( int i = 0; i < componentsNum; i++ )
		{
			matvarTmp = *cells++;
			int Num_1 = RetrieveNum(matvarTmp);
			matvar_t **cells = (matvar_t **)matvarTmp->data;
			matvar_t *matvarTmp;

			for (int ii=0;ii<Num_1;ii++)
			{
				matvarTmp = *cells++;
				newComponents.defid_.push_back(*((double*)matvarTmp->data));

				matvarTmp = *cells++;
				newComponents.filterid_.push_back(*((double*)matvarTmp->data));

				matvarTmp = *cells++;
				newComponents.parent_.push_back(*((double*)matvarTmp->data));
				
			}
			refComponents.push_back(newComponents);
			vector<double>().swap(newComponents.defid_);//newComponents.defid_.clear();
			vector<double>().swap(newComponents.filterid_);//newComponents.filterid_.clear();
			vector<double>().swap(newComponents.parent_);//newComponents.parent_.clear();
		}

		return true;
	}
	else
		return false;
}


//void LandmarkDetector::detect(std::vector<Data_bs>&boxes,ImgInfo<unsigned char>& img ,LandmarkModel *model,char modelname[],float &f_time)
void LandmarkDetector::detect(std::vector<Data_bs>&boxes,cv::Mat& img ,LandmarkModel *model,char modelname[])
{
	//ImgUtils::StopWatch stopwatch;
	// fconv 宣告
	int newdim[3],rootdim[2],N_size[2];
	int f;	
	float XY[500][68][4];
	//int level;
	vector<comp> components;
	vector<float> fconv_time;vector<float> dt_time,bt;
	vector<cv::Mat> splitimg;  
	cv::Mat bimg, gimg,rimg,trimg;

	cv::transpose(img,trimg); //將原影像長*寬轉置為寬*長
	cv::split(trimg,splitimg);

	bimg = splitimg.at(0);  
	gimg = splitimg.at(1);  
	rimg = splitimg.at(2);  

	cv::Mat fimg(bimg.rows*3, bimg.cols, CV_32FC1);
	int k=0; int m=0;
	//cv::Mat fimg=trimg.reshape(1,trimg.rows*3);
	// 將三通道合併成一通道 //
	for (int i=0;i<fimg.rows;i++)
	{
		if ( i>=fimg.rows/3 && i<(fimg.rows/3)*2)
		{
			for (int j=0;j<gimg.cols;j++)
			{
				fimg.at<float>(i,j) = (float) gimg.at<uchar>(k,j);
				//fimg.at<float>(j,i) = (float) gimg.at<uchar>(j,k);
			}
			k++;
		}

		else if (i>=(fimg.rows/3)*2)
		{
			for (int j=0;j<rimg.cols;j++)
			{
				fimg.at<float>(i,j) = (float) rimg.at<uchar>(m,j);
				//fimg.at<float>(j,i) = (float) rimg.at<uchar>(j,m);
			}
			m++;
		}

		else
		for (int j=0;j<fimg.cols;j++)
		{
			fimg.at<float>(i,j) = (float) bimg.at<uchar>(i,j);
			//fimg.at<float>(j,i) = (float) bimg.at<uchar>(j,i);
		}
		//cout<<endl<<endl;
	}
	//for (int i=0;i<fimg.rows;i++)
	//{
	//	for (int j=0;j<fimg.cols;j++)
	//	{
	//		printf("%f ", fimg.at<float>(i,j));
	//		//cout << fimg.at<uchar>(i,j) << " ";
	//	}
	//	cout<<endl<<endl;
	//}

	////feat_time.Start();
	featpyramid(fimg,model);
	////cout << "Feature Consumption : " <<feat_time.Stop()<<endl<<endl;

	////printf("featpyramid preprocessing elapsed : %f\n", feat_time.Stop());

	// inital resp[i] //
	//cout<<"this->FP_level : "<<this->FP_level<<endl;
	for (int i=this->FP_level-ccc; i<this->FP_level-ccc+ccc2; i++) //
	{
		if (this->resp[i].dim_c !=NULL)
		{
			this->resp[i].dim_c = NULL;
		}

		if (this->resp[i].dim_r !=NULL)
		{
			this->resp[i].dim_r = NULL;
		}

		if (this->resp[i].r_ !=NULL)
		{
			this->resp[i].r_ = NULL;
		}
	}


	std::vector<Data_s> parts;float* resp_;
	modelcomponents(components,model);
	for (int c=0; c<model->components_.size(); c++)
	{			
		//pose.Start();
		parts= components[c].comp_ ;
		this->numparts = parts.size();
		//printf("c = %d: ",c);
		for (int rlevel=this->FP_level-ccc; rlevel<this->FP_level-ccc+ccc2;rlevel++) 
		{		
			if ( resp[rlevel].r_ == NULL) 
			{		
				newdim[0]=pyra.dim_c[rlevel];newdim[1]=pyra.dim_r[rlevel]; newdim[2]=32;	
				//temp_ft.Start();
				fconv_25cells(&resp_,pyra.feat[rlevel],&newdim[0],&rootdim[0],filters);
				//fconv_time.push_back(temp_ft.Stop());
				//printf("End of detect : %f\n", temp_ft.Stop());
				resp[rlevel].r_ = resp_; 
				resp[rlevel].dim_c=rootdim[0];resp[rlevel].dim_r=rootdim[1]; //存取resp每層的維度大小
			}

			//Local part scores
			//#pragma omp parallel for num_threads(8)
			for (int k=0;k<numparts;k++)
			{
				//cout << k <<endl;
				f=parts[k].filterid_-1;
				parts[k].score = resp[rlevel].r_+(f*resp[rlevel].dim_r*resp[rlevel].dim_c);
				parts[k].level=rlevel+1;	
			}
			//printf("Local part scores preprocessing elapsed : %f\n", temp_ft.Stop());

			//if (strcmp(modelname,"LM_Model\\multipie_independent.mat") ==1 && (c ==8 || c==11)){
			//	N_size[0] = resp[rlevel].dim_c-1;N_size [1] = resp[rlevel].dim_r-1;}
			//else if(strcmp(modelname,"LM_Model\\multipie_independent.mat") ==1 && c ==13 || c==4){
			//	N_size[0] = resp[rlevel].dim_c+1;N_size [1] = resp[rlevel].dim_r+1;}
			//else{
				N_size[0] = resp[rlevel].dim_c;N_size[1] = resp[rlevel].dim_r;//}


            /// Distance Transform //
			//Walk from leaves to root of tree, passing message to parent
			//Given a 2D array of filter scores 'child', shiftdt() does the following:
			//(1) Apply distance transform
			//(2) Shift by anchor position (child.startxy) of part wrt parent
			//(3) Downsample by child.step

			//int** ixvec = new int*[this->numparts-1];
			//int** iyvec = new int*[this->numparts-1];
			int par,index=0;
			float* vals;
			float* M = new float[N_size[0]*N_size[1]];
			float*tmpM = new float[N_size[0]*N_size[1]];
			int *tmpIy = new int[N_size[0]*N_size[1]];
			int* Ix = new int[N_size[0]*N_size[1]];
			int* Iy = new int[N_size[0]*N_size[1]];	
			float* temp2 = new float[(parts.size()-1)*N_size[1]*N_size[0]];
			int* tempIx = new int[(parts.size()-1)*N_size[1]*N_size[0]];
			int* tempIy = new int[(parts.size()-1)*N_size[1]*N_size[0]];

			for (int i=parts.size()-1;i>0;i--)
			{ 
				::memset(M,0,sizeof(float)*N_size[0]*N_size[1]);
				child = parts[i];
				par=child.parent_-1;
				vals = child.score;
				for (int x = 0; x < N_size[1]; x++)
				{
					dt1d(vals+x*N_size[0], tmpM+x*N_size[0], tmpIy+x*N_size[0], 1, N_size[0], -child.w_[2], -child.w_[3], child.starty-1, child.step);
				}
				for (int y = 0; y < N_size[0]; y++)
				{
					dt1d(tmpM+y, M+y, Ix+y, N_size[0], N_size[1], -child.w_[0], -child.w_[1], child.startx-1, child.step);
				}
				// get argmins and adjust for matlab indexing from 1
				for (int x = 0; x < N_size[1]; x++) 
				{
					for (int y = 0; y < N_size[0]; y++) 
					{
						int p = x*N_size[0]+y;
						Iy[p] = tmpIy[Ix[p]*N_size[0]+y]+1;	
						Ix[p] = Ix[p]+1;
					}
				}		
				for(int kk = 0; kk < N_size[0]*N_size[1]; ++kk) 
				{
					*(temp2 +(index*N_size[0]*N_size[1])+ kk) = *(parts[par].score + kk) + *(M + kk);		
					*(tempIx+(index*N_size[0]*N_size[1])+kk) = *(Ix+kk);
					*(tempIy+(index*N_size[0]*N_size[1])+kk) = *(Iy+kk);
				}
				parts[par].score = &temp2[index*N_size[0]*N_size[1]] ;	//parts(par).score = parts(par).score + msg
				parts[i].Ix = &tempIx[index*N_size[0]*N_size[1]];
				parts[i].Iy = &tempIy[index*N_size[0]*N_size[1]];;

				index++;
			}
			delete [] tmpM;delete [] M;delete [] tmpIy;delete [] Ix;delete [] Iy;

			//////////////////////////////////////////////////////////////////////////

			//shiftdt(&temp2,&tempIx,&tempIy,parts,&N_size[0]);

			//for (int k2=this->numparts-1;k2>0;k2--)
			//{		
			//	float* msg;
			//	child = parts[k2];
			//	int par=child.parent_-1;
			//	temp_dt.Start();
			//	shiftdt(&msg, &ixvec[index_dt], &iyvec[index_dt], child.score,&child.w_[0],child.startx,child.starty,&N_size[0],child.step);
			//	//shiftdt(&msg, &parts[k2].Ix, &parts[k2].Iy, child.score,&child.w_[0],child.startx,child.starty,&N_size[0],child.step);
			//	//cout << temp_dt.Stop()<<endl;
			//	dt_time.push_back(temp_dt.Stop());
			//	parts[k2].Ix=ixvec[index_dt];
			//	parts[k2].Iy=iyvec[index_dt];
			//	//parts[k2].Ix=ixvec;
			//	//parts[k2].Iy=iyvec;
			//	//MixScore(parts[par].score,msg,N_size);
			//	for(int kk = 0; kk < N_size[0]*N_size[1]; ++kk) 
			//	{
			//		*(temp2 +(index_dt*N_size[0]*N_size[1])+ kk) = *(parts[par].score + kk) + *(msg + kk);
			//	}
			//	parts[par].score = &temp2[index_dt*N_size[0]*N_size[1]] ;	//parts(par).score = parts(par).score + msg
			//	delete [] msg;msg = NULL;
			//	index_dt++;
			//}
			//printf("levels %d DT preprocessing elapsed : %f\n", rlevel,sum(dt_time));
			//vector<float>().swap(dt_time);
			// Add bias to root score
			//////////////////////////////////////////////////////////////////////////  rscore = parts(1).score + parts(1).w;  //////////////////////////////////////////////////////////////////////////
			float* rscore = new float[N_size[0]*N_size[1]];
			for(int jj = 0; jj != N_size[1]*N_size[0]; ++jj) {
					*(rscore+jj ) = *(parts[0].score + jj) + *(parts[0].w_);} 

			//int X[600],Y[600],index=0;
			vector<int> Y;vector<int> X;
			//#pragma omp parallel for num_threads(10)
			for(int jj = 0; jj < N_size[1]; ++jj) 
			{
				for(int kk = 0; kk < N_size[0]; ++kk) 
				{
					if ( *(rscore+(jj * N_size[0]) + kk )  >= model->thresh_ )
					{
						Y.push_back(kk+1); //列
						X.push_back(jj+1); //行
					}
				}
			}

			if (X.empty() == false)
			//if (X[0] != NULL)
			{
				//temp_bt.Start();
				backtrack(XY[0][0],X,Y,parts, pyra.scale,pyra.padx,pyra.pady ,&N_size[0]);
				//bt.push_back(temp_bt.Stop());
				//cout << temp_bt.Stop()<<"  ";
				//backtrack(XY[0][0],X,Y,parts, pyra.scale,pyra.padx,pyra.pady ,&back_dim[0], index);
			}
			// Walk back down tree following pointers
			for (int iii=0 ; iii<X.size() ; iii++)
			{
				int x = X[iii]-1; int y = Y[iii]-1;				
				bs.c=c;
				bs.s=*(rscore + (x * N_size[0]) + y);
				bs.level = rlevel;
				memcpy( bs.xy,XY[iii],sizeof(float)*numparts*4);
				boxes.push_back(bs);
			}
			vector<int>().swap(X);vector<int>().swap(Y);
			delete [] rscore;rscore = NULL;
			delete [] temp2;temp2 = NULL;
			delete [] tempIx;tempIx = NULL;
			delete [] tempIy;tempIy = NULL;
			//for(int np = 0; np < this->numparts-1; np++)
			//{
			//	delete [] ixvec[np];
			//	delete [] iyvec[np];
			//	//delete [] temp2[np];
			//}
			//delete [] ixvec;ixvec=NULL;
			//delete [] iyvec;iyvec=NULL;
			//delete [] temp2;temp2=NULL;
			//fconv_time.push_back(temp_ft.Stop());
		}
		//parts.clear();
		vector<Data_s>().swap(parts);
		vector<double*>().swap(filters);
		//cout <<sum(dt_time) <<"    "; //cout << "bt_time : "<<sum(bt) <<" ";
		//cout << pose.Stop()<<endl;
		vector<float>().swap(dt_time);//vector<float>().swap(bt);
		//printf("c = %d done\n",c);
	}
	//f_time =  sum(fconv_time);
	vector<comp>().swap(components);
	Clean();
	_CrtDumpMemoryLeaks();
	//boxes.resize(cnt);
}

void LandmarkDetector::detect_L2(std::vector<Data_bs>&boxes,cv::Mat& img ,LandmarkModel *model,char modelname[],int L1_Pose,int L1_Level)
{
	//ImgUtils::StopWatch stopwatch;
	// fconv 宣告
	int newdim[3],rootdim[2],N_size[2];
	int f;	
	float XY[600][68][4];
	//int level;
	//ImgUtils::StopWatch temp_ft;ImgUtils::StopWatch temp_dt;
	//ImgUtils::StopWatch feat_time;
	//ImgUtils::StopWatch pose,temp_bt;
	vector<comp> components;
	vector<float> fconv_time,dt_time,bt;
	vector<int>For_Level,For_Pose;
	//ImgInfo<unsigned char> splitimg , trimg;
	//trimg.AllocateImage(img.height_, img.width_, 3);
	//Operation::Transpose(trimg, img);
	//splitimg.AllocateImage(trimg.width_, trimg.height_ * 3);
	//Operation::SplitChannel(splitimg, trimg);

	//ImgInfo<float> fimg, rimg, gimg, bimg;
	//fimg.AllocateImage(splitimg.width_, splitimg.height_, splitimg.channel_);
	//Operation::Cast(fimg, splitimg);
	//float* data = fimg.data_;
	//int offset = fimg.width_step_ * img.width_;
	//bimg.SetImage(img.height_, img.width_, 1, fimg.width_step_, data);
	//data += offset;
	//gimg.SetImage(img.height_, img.width_, 1, fimg.width_step_, data);
	//data += offset;
	//rimg.SetImage(img.height_, img.width_, 1, fimg.width_step_, data);

	vector<cv::Mat> splitimg;  
	cv::Mat bimg, gimg,rimg,trimg;

	cv::transpose(img,trimg); //將原影像長*寬轉置為寬*長
	cv::split(trimg,splitimg);

	bimg = splitimg.at(0);  
	gimg = splitimg.at(1);  
	rimg = splitimg.at(2);  

	cv::Mat fimg(bimg.rows*3, bimg.cols, CV_32FC1);
	int k=0; int m=0;
	//cv::Mat fimg=trimg.reshape(1,trimg.rows*3);
	// 將三通道合併成一通道 //
	for (int i=0;i<fimg.rows;i++)
	{
		if ( i>=fimg.rows/3 && i<(fimg.rows/3)*2)
		{
			for (int j=0;j<gimg.cols;j++)
			{
				fimg.at<float>(i,j) = (float) gimg.at<uchar>(k,j);
				//fimg.at<float>(j,i) = (float) gimg.at<uchar>(j,k);
			}
			k++;
		}

		else if (i>=(fimg.rows/3)*2)
		{
			for (int j=0;j<rimg.cols;j++)
			{
				fimg.at<float>(i,j) = (float) rimg.at<uchar>(m,j);
				//fimg.at<float>(j,i) = (float) rimg.at<uchar>(j,m);
			}
			m++;
		}

		else
			for (int j=0;j<fimg.cols;j++)
			{
				fimg.at<float>(i,j) = (float) bimg.at<uchar>(i,j);
				//fimg.at<float>(j,i) = (float) bimg.at<uchar>(j,i);
			}
			//cout<<endl<<endl;
	}

	//feat_time.Start();
	featpyramid(fimg,model);
	//cout << "Feature Consumption : " <<feat_time.Stop()<<endl<<endl;

	//printf("featpyramid preprocessing elapsed : %f\n", feat_time.Stop());

	// inital resp[i] //
	for (int i=this->FP_level-ccc; i<this->FP_level-ccc+ccc2; i++) //
	{
		if (this->resp[i].dim_c !=NULL)
		{
			this->resp[i].dim_c = NULL;
		}

		if (this->resp[i].dim_r !=NULL)
		{
			this->resp[i].dim_r = NULL;
		}

		if (this->resp[i].r_ !=NULL)
		{
			this->resp[i].r_ = NULL;
		}
	}

	// Level 搜尋 //
	/*if ( L1_Level < 6 ) 
	{
		for (int i=5;i<8;i++)
		{
			For_Level.push_back(i);
		}
	}
	else 
	{
		for (int i=L1_Level-1;i<L1_Level+2;i++)
		{
			For_Level.push_back(i);
		}
	}
*/
	// Pose 搜尋 //
	//if ( L1_Pose <= 0) 
	//{
	//	for (int i=0;i<3;i++)
	//	{
	//		For_Pose.push_back(i);
	//	}
	//}
	//else if ( L1_Pose>=12 )
	//{
	//	for (int i=12;i>9;i--)
	//	{
	//		For_Pose.push_back(i);
	//	}
	//}
	//else 
	//{
	//	if (model->components_.size() == 18)
	//	{
	//		for (int i=L1_Pose-1;i<L1_Pose+2;i++)
	//		{
	//			For_Pose.push_back(i);
	//		}
	//	}
	//	else
	//	{
	//		for (int i=L1_Pose-1;i<L1_Pose+2;i++)
	//		{
	//			For_Pose.push_back(i);
	//		}
	//	}
	//}
	//
	//int Pose_L2[1]={7};
	std::vector<Data_s> parts;float* resp_;
	modelcomponents(components,model);
	//for (int c = For_Pose[0]; c < For_Pose[2]; c++)
	int c=6;
/*	for (int c = Pose_L2[0]; c < Pose_L2[1]; c++)
	{	*/		
		//pose.Start();
		parts= components[c].comp_ ;
		this->numparts = parts.size();
		//printf("c = %d: ",c);
		//for (int rlevel = For_Level[0]; rlevel<For_Level[2];rlevel++) //this->FP_level
		for (int rlevel = this->FP_level-ccc; rlevel<this->FP_level-ccc+ccc2;rlevel++) //this->FP_level
		{		
			if ( resp[rlevel].r_ == NULL) 
			{		
				newdim[0]=pyra.dim_c[rlevel];newdim[1]=pyra.dim_r[rlevel]; newdim[2]=32;	
				//temp_ft.Start();
				fconv_25cells(&resp_,pyra.feat[rlevel],&newdim[0],&rootdim[0],filters);
				//fconv_time.push_back(temp_ft.Stop());
				//printf("End of detect : %f\n", temp_ft.Stop());
				resp[rlevel].r_ = resp_; 
				resp[rlevel].dim_c=rootdim[0];resp[rlevel].dim_r=rootdim[1]; //存取resp每層的維度大小
			}

			//Local part scores
			//#pragma omp parallel for num_threads(8)
			for (int k=0;k<numparts;k++)
			{
				//cout << k <<endl;
				f=parts[k].filterid_-1;
				parts[k].score = resp[rlevel].r_+(f*resp[rlevel].dim_r*resp[rlevel].dim_c);
				parts[k].level=rlevel+1;	
			}
			//printf("Local part scores preprocessing elapsed : %f\n", temp_ft.Stop());

			//if (strcmp(modelname,"LM_Model\\multipie_independent.mat") ==1 && (c ==8 || c==11)){
			//	N_size[0] = resp[rlevel].dim_c-1;N_size [1] = resp[rlevel].dim_r-1;}
			//else if(strcmp(modelname,"LM_Model\\multipie_independent.mat") ==1 && (c ==13||c==4)){
			//	N_size[0] = resp[rlevel].dim_c+1;N_size [1] = resp[rlevel].dim_r+1;}
			//else{
			N_size[0] = resp[rlevel].dim_c;N_size[1] = resp[rlevel].dim_r;//}


			/// Distance Transform //
			//Walk from leaves to root of tree, passing message to parent
			//Given a 2D array of filter scores 'child', shiftdt() does the following:
			//(1) Apply distance transform
			//(2) Shift by anchor position (child.startxy) of part wrt parent
			//(3) Downsample by child.step

			//int** ixvec = new int*[this->numparts-1];
			//int** iyvec = new int*[this->numparts-1];
			int par,index=0;
			float* vals;
			float* M = new float[N_size[0]*N_size[1]];
			float*tmpM = new float[N_size[0]*N_size[1]];
			int *tmpIy = new int[N_size[0]*N_size[1]];
			int* Ix = new int[N_size[0]*N_size[1]];
			int* Iy = new int[N_size[0]*N_size[1]];	
			float* temp2 = new float[(parts.size()-1)*N_size[1]*N_size[0]];
			int* tempIx = new int[(parts.size()-1)*N_size[1]*N_size[0]];
			int* tempIy = new int[(parts.size()-1)*N_size[1]*N_size[0]];

			//temp_dt.Start();
			for (int i=parts.size()-1;i>0;i--)
			{ 
				::memset(M,0,sizeof(float)*N_size[0]*N_size[1]);
				child = parts[i];
				par=child.parent_-1;
				vals = child.score;
				for (int x = 0; x < N_size[1]; x++)
				{
					dt1d(vals+x*N_size[0], tmpM+x*N_size[0], tmpIy+x*N_size[0], 1, N_size[0], -child.w_[2], -child.w_[3], child.starty-1, child.step);
				}
				for (int y = 0; y < N_size[0]; y++)
				{
					dt1d(tmpM+y, M+y, Ix+y, N_size[0], N_size[1], -child.w_[0], -child.w_[1], child.startx-1, child.step);
				}
				// get argmins and adjust for matlab indexing from 1
				for (int x = 0; x < N_size[1]; x++) 
				{
					for (int y = 0; y < N_size[0]; y++) 
					{
						int p = x*N_size[0]+y;
						Iy[p] = tmpIy[Ix[p]*N_size[0]+y]+1;	
						Ix[p] = Ix[p]+1;
					}
				}		
				for(int kk = 0; kk < N_size[0]*N_size[1]; ++kk) 
				{
					*(temp2 +(index*N_size[0]*N_size[1])+ kk) = *(parts[par].score + kk) + *(M + kk);		
					*(tempIx+(index*N_size[0]*N_size[1])+kk) = *(Ix+kk);
					*(tempIy+(index*N_size[0]*N_size[1])+kk) = *(Iy+kk);
				}
				parts[par].score = &temp2[index*N_size[0]*N_size[1]] ;	//parts(par).score = parts(par).score + msg
				parts[i].Ix = &tempIx[index*N_size[0]*N_size[1]];
				parts[i].Iy = &tempIy[index*N_size[0]*N_size[1]];;

				index++;
			}
			//dt_time.push_back(temp_dt.Stop());
			delete [] tmpM;delete [] M;delete [] tmpIy;delete [] Ix;delete [] Iy;

			//////////////////////////////////////////////////////////////////////////

			//shiftdt(&temp2,&tempIx,&tempIy,parts,&N_size[0]);

			//for (int k2=this->numparts-1;k2>0;k2--)
			//{		
			//	float* msg;
			//	child = parts[k2];
			//	int par=child.parent_-1;
			//	temp_dt.Start();
			//	shiftdt(&msg, &ixvec[index_dt], &iyvec[index_dt], child.score,&child.w_[0],child.startx,child.starty,&N_size[0],child.step);
			//	//shiftdt(&msg, &parts[k2].Ix, &parts[k2].Iy, child.score,&child.w_[0],child.startx,child.starty,&N_size[0],child.step);
			//	//cout << temp_dt.Stop()<<endl;
			//	dt_time.push_back(temp_dt.Stop());
			//	parts[k2].Ix=ixvec[index_dt];
			//	parts[k2].Iy=iyvec[index_dt];
			//	//parts[k2].Ix=ixvec;
			//	//parts[k2].Iy=iyvec;
			//	//MixScore(parts[par].score,msg,N_size);
			//	for(int kk = 0; kk < N_size[0]*N_size[1]; ++kk) 
			//	{
			//		*(temp2 +(index_dt*N_size[0]*N_size[1])+ kk) = *(parts[par].score + kk) + *(msg + kk);
			//	}
			//	parts[par].score = &temp2[index_dt*N_size[0]*N_size[1]] ;	//parts(par).score = parts(par).score + msg
			//	delete [] msg;msg = NULL;
			//	index_dt++;
			//}
			//printf("levels %d DT preprocessing elapsed : %f\n", rlevel,sum(dt_time));
			//vector<float>().swap(dt_time);
			// Add bias to root score
			//////////////////////////////////////////////////////////////////////////  rscore = parts(1).score + parts(1).w;  //////////////////////////////////////////////////////////////////////////
			float* rscore = new float[N_size[0]*N_size[1]];
			for(int jj = 0; jj != N_size[1]*N_size[0]; ++jj) {
				*(rscore+jj ) = *(parts[0].score + jj) + *(parts[0].w_);} 

			//int X[600],Y[600],index=0;
			vector<int> Y;vector<int> X;
			//#pragma omp parallel for num_threads(10)
			for(int jj = 0; jj < N_size[1]; ++jj) 
			{
				for(int kk = 0; kk < N_size[0]; ++kk) 
				{
					if ( *(rscore+(jj * N_size[0]) + kk )  >= model->thresh_ )
					{
						Y.push_back(kk+1); //列
						X.push_back(jj+1); //行
					}
				}
			}

			if (X.empty() == false)
				//if (X[0] != NULL)
			{
				//temp_bt.Start();
				backtrack(XY[0][0],X,Y,parts, pyra.scale,pyra.padx,pyra.pady ,&N_size[0]);
				//bt.push_back(temp_bt.Stop());
				//cout << temp_bt.Stop()<<"  ";
				//backtrack(XY[0][0],X,Y,parts, pyra.scale,pyra.padx,pyra.pady ,&back_dim[0], index);
			}
			// Walk back down tree following pointers
			for (int iii=0 ; iii<X.size() ; iii++)
			{
				int x = X[iii]-1; int y = Y[iii]-1;				
				bs.c=c;
				bs.s=*(rscore + (x * N_size[0]) + y);
				bs.level = rlevel;
				memcpy( bs.xy,XY[iii],sizeof(float)*numparts*4);
				boxes.push_back(bs);
			}
			vector<int>().swap(X);vector<int>().swap(Y);
			delete [] rscore;rscore = NULL;
			delete [] temp2;temp2 = NULL;
			delete [] tempIx;tempIx = NULL;
			delete [] tempIy;tempIy = NULL;
			//for(int np = 0; np < this->numparts-1; np++)
			//{
			//	delete [] ixvec[np];
			//	delete [] iyvec[np];
			//	//delete [] temp2[np];
			//}
			//delete [] ixvec;ixvec=NULL;
			//delete [] iyvec;iyvec=NULL;
			//delete [] temp2;temp2=NULL;
			//fconv_time.push_back(temp_ft.Stop());
		}
		//parts.clear();
		vector<Data_s>().swap(parts);
		vector<double*>().swap(filters);
		//cout <<sum(dt_time) <<"    "; //cout << "bt_time : "<<sum(bt) <<" ";
		//cout << pose.Stop()<<endl;
		vector<float>().swap(dt_time);//vector<float>().swap(bt);
		//printf("c = %d done\n",c);
	//}
	//f_time =  sum(fconv_time);
	vector<comp>().swap(components);
	Clean();
	//_CrtDumpMemoryLeaks();
	//boxes.resize(cnt);
}



//void LandmarkDetector::detectPCA(std::vector<Data_bs>&boxes,cv::Mat& img ,LandmarkModel *model,double thresh,char modelname[],int kk4,vector<double*>filter_4_pca,double*SCORE4)
//{
//	// fconv 宣告
//	int newdim[3],rootdim[2],rdim[3]; 
//	rdim[2] = 32;
//	//int level;
//	//ImgUtils::StopWatch temp_ft;ImgUtils::StopWatch temp_dt;
//	//ImgUtils::StopWatch feat_time;
//	vector<comp> components;
//	//vector<float> fconv_time;vector<float> dt_time;
//
//	ImgInfo<unsigned char> splitimg , trimg;
//	trimg.AllocateImage(img.height_, img.width_, 3);
//	Operation::Transpose(trimg, img);
//	splitimg.AllocateImage(trimg.width_, trimg.height_ * 3);
//	Operation::SplitChannel(splitimg, trimg);
//
//	ImgInfo<float> fimg, rimg, gimg, bimg;
//	fimg.AllocateImage(splitimg.width_, splitimg.height_, splitimg.channel_);
//	Operation::Cast(fimg, splitimg);
//	float* data = fimg.data_;
//	int offset = fimg.width_step_ * img.width_;
//	bimg.SetImage(img.height_, img.width_, 1, fimg.width_step_, data);
//	data += offset;
//	gimg.SetImage(img.height_, img.width_, 1, fimg.width_step_, data);
//	data += offset;
//	rimg.SetImage(img.height_, img.width_, 1, fimg.width_step_, data);
//
//	//feat_time.Start();
//	featpyramid(fimg,model);
//	//printf("featpyramid preprocessing elapsed : %f\n", feat_time.Stop());
//
//	std::vector<Data_s> parts;float *resp_4_temp,*resp_;
//	modelcomponents(components,model);
//
//	//filter_num = filters.size();
//	//cv::Mat filter_matrix(filter_num,5*5*32,CV_32FC1);
//	//for (int i=0;i<filter_num;i++)
//	//{
//	//	for (int j=0;j<5*5*32;j++)
//	//	{
//	//		filter_matrix.at<float>(i,j) = *(filters[i]+j);
//	//	}
//	//}
//	//for (int i=0;i<filter_num;i++)
//	//{
//	//	for (int j=0;j<5*5*32;j++)
//	//	{
//	//		cout << filter_matrix.at<float>(i,j) <<" ";
//	//	}
//	//}
//	//cv::PCA *pca = new cv::PCA(filter_matrix, cv::Mat(), CV_PCA_DATA_AS_ROW,60);
//
//	//for (int i=0;i<800;i++)
//	//{
//	//	cout << pca->eigenvalues.at<float>(i,0) <<endl;
//	//}
//
//
//	for (int i=5; i<this->FP_level; i++) //
//	{
//		if (this->resp[i].dim_c !=NULL)
//		{
//			this->resp[i].dim_c = NULL;
//		}
//
//		if (this->resp[i].dim_r !=NULL)
//		{
//			this->resp[i].dim_r = NULL;
//		}
//
//		if (this->resp[i].r_ !=NULL)
//		{
//			this->resp[i].r_ = NULL;
//		}
//	}
//
//
//	//this->filter_num = model->filters_.size();
//	//for (int c=3; c<=14; c++)
//	
//	for (int c=0; c<model->components_.size(); c++)
//	{			
//		pose.Start();
//		parts= components[c].comp_ ;
//		this->numparts = parts.size();
//		printf("c = %d: ",c);
//		//#pragma omp parallel for num_threads(10)
//		for (int rlevel=5; rlevel<11; rlevel++) //11 this->FP_level
//		{
//			//printf("%d ",rlevel);
//			//float** resp_ = new float*[filter_num];
//			if ( resp[rlevel].r_ == NULL ) 
//			{	
//				newdim[0]=pyra.dim_c[rlevel];newdim[1]=pyra.dim_r[rlevel]; newdim[2]=32;	
//				stopwatch.Start();
//				fconv_25cells_PCA(&resp_,pyra.feat[rlevel],&newdim[0],&rootdim[0], filter_4_pca,SCORE4);//filter_4_pca
//				//fconv_25cells(&resp_,pyra.feat[rlevel],&newdim[0],&rootdim[0],filters);
//				printf("%f  ", stopwatch.Stop());
//				//stopwatch.Start();
//				//pca_recon(&resp_4,resp_4_temp,SCORE4,filter_4_pca.size(),&rootdim[0]);
//				//printf("%f  ", stopwatch.Stop());
//				//}
//				//resp[rlevel].r_ = resp_4_temp;
//				resp[rlevel].r_ = resp_; // will be cancer
//				resp[rlevel].dim_c=rootdim[0];resp[rlevel].dim_r=rootdim[1]; //存取resp每層的維度大小
//				//delete [] resp_4_temp;resp_4_temp=NULL;
//			}
//			//temp_ft.Start();
//			//Local part scores
//
//			for (int k=0;k<numparts;k++)
//			{
//				//printf("%d ",k);
//				int f=parts[k].filterid_-1;
//				//level=rlevel-parts[k].scale*model->interval_-1;
//				parts[k].score = resp[rlevel].r_+ (f*resp[rlevel].dim_r*resp[rlevel].dim_c);
//				parts[k].level=rlevel+1;	
//			}
//			//printf("Local part scores preprocessing elapsed : %f\n", temp_ft.Stop());
//
//			//Walk from leaves to root of tree, passing message to parent
//			//Given a 2D array of filter scores 'child', shiftdt() does the following:
//			//(1) Apply distance transform
//			//(2) Shift by anchor position (child.startxy) of part wrt parent
//			//(3) Downsample by child.step
//			int N_size[2];
//
//			if (strcmp(modelname,"model_multipie_independent") ==1 && (c ==8 || c==11)){
//				N_size[0] = resp[rlevel].dim_c-1;N_size [1] = resp[rlevel].dim_r-1;}
//			else if(strcmp(modelname,"model_multipie_independent") ==1 && (c ==13||c==4)){
//				N_size[0] = resp[rlevel].dim_c+1;N_size [1] = resp[rlevel].dim_r+1;}
//			else{
//				N_size[0] = resp[rlevel].dim_c;N_size[1] = resp[rlevel].dim_r;}
//
//			int par,index=0;
//			float* vals;
//			float* M = new float[N_size[0]*N_size[1]];
//			float*tmpM = new float[N_size[0]*N_size[1]];
//			int *tmpIy = new int[N_size[0]*N_size[1]];
//			int* Ix = new int[N_size[0]*N_size[1]];
//			int* Iy = new int[N_size[0]*N_size[1]];	
//			float* temp2 = new float[(parts.size()-1)*N_size[1]*N_size[0]];
//			int* tempIx = new int[(parts.size()-1)*N_size[1]*N_size[0]];
//			int* tempIy = new int[(parts.size()-1)*N_size[1]*N_size[0]];
//
//			//temp_dt.Start();
//			for (int i=parts.size()-1;i>0;i--)
//			{ 
//				child = parts[i];
//				par=child.parent_-1;
//				vals = child.score;
//				for (int x = 0; x < N_size[1]; x++)
//				{
//					//cout << x<<endl;
//					dt1d(vals+x*N_size[0], tmpM+x*N_size[0], tmpIy+x*N_size[0], 1, N_size[0], -child.w_[2], -child.w_[3], child.starty-1, child.step);
//				}
//				for (int y = 0; y < N_size[0]; y++)
//				{
//					//cout << **(Mptr + (index*dims[0]*dims[1])+y) << "  ";
//					dt1d(tmpM+y, M+y, Ix+y, N_size[0], N_size[1], -child.w_[0], -child.w_[1], child.startx-1, child.step);
//				}
//				// get argmins and adjust for matlab indexing from 1
//				for (int x = 0; x < N_size[1]; x++) 
//				{
//					for (int y = 0; y < N_size[0]; y++) 
//					{
//						int p = x*N_size[0]+y;
//						Iy[p] = tmpIy[Ix[p]*N_size[0]+y]+1;	
//						Ix[p] = Ix[p]+1;
//					}
//				}		
//				for(int kk = 0; kk < N_size[0]*N_size[1]; ++kk) 
//				{
//					*(temp2 +(index*N_size[0]*N_size[1])+ kk) = *(parts[par].score + kk) + *(M + kk);		
//					*(tempIx+(index*N_size[0]*N_size[1])+kk) = *(Ix+kk);
//					*(tempIy+(index*N_size[0]*N_size[1])+kk) = *(Iy+kk);
//				}
//				parts[par].score = &temp2[index*N_size[0]*N_size[1]] ;	//parts(par).score = parts(par).score + msg
//				parts[i].Ix = &tempIx[index*N_size[0]*N_size[1]];
//				parts[i].Iy = &tempIy[index*N_size[0]*N_size[1]];;
//
//				index++;
//			}
//			delete [] tmpM;delete [] M;delete [] tmpIy;delete [] Ix;delete [] Iy;
//			//int** ixvec = new int*[this->numparts-1];
//			//int** iyvec = new int*[this->numparts-1];
//			//int index_dt=0;
//			////temp_dt.Start();
//			//float* temp2 = new float[(this->numparts-1)*N_size[0]*N_size[1]];
//			//for (int k2=this->numparts-1;k2>0;k2--)
//			//{	
//			//	float* msg;
//			//	child = parts[k2];
//			//	int par=child.parent_-1;
//			//	shiftdt(&msg, &ixvec[index_dt], &iyvec[index_dt], child.score, 
//			//		&child.w_[0],child.startx,child.starty,&N_size[0],child.step);
//			//	parts[k2].Ix=ixvec[index_dt];
//			//	parts[k2].Iy=iyvec[index_dt];
//			//	//MixScore(parts[par].score,msg,&N_size[0]);
//			//	//*parts[par].score=*parts[par].score+*msg;
//			//	for(int jj = 0; jj != N_size[1]; ++jj) 
//			//	{
//			//		for(int kk = 0; kk != N_size[0]; ++kk) 
//			//		{
//			//			*(temp2+ (index_dt*N_size[0]*N_size[1])+ (jj * N_size[0]) + kk) = *(parts[par].score+ (jj * N_size[0]) + kk) + *(msg + (jj * N_size[0]) + kk);
//			//		}
//			//	}
//			//	parts[par].score = &temp2[index_dt*N_size[0]*N_size[1]] ;	//parts(par).score = parts(par).score + msg
//			//	delete [] msg;msg = NULL;
//			//	index_dt++;
//			//}
//			
//			//printf("DT preprocessing elapsed : %f\n", temp_dt.Stop());
//
//			// Add bias to root score
//			float* rscore = new float[N_size[0]*N_size[1]];
//			for(int jj = 0; jj != N_size[1]*N_size[0]; ++jj) {
//				*(rscore+jj ) = *(parts[0].score + jj) + *(parts[0].w_);}  //rscore = parts(1).score + parts(1).w;
//
//			//int X[600],Y[600],index=0;
//			vector<int> Y;vector<int> X;
//			//#pragma omp parallel //for num_threads(10)
//			for(int jj = 0; jj != N_size[1]; ++jj) 
//			{
//				//printf("%d ",jj);
//				for(int kk = 0; kk != N_size[0]; ++kk) 
//				{
//					if ( *(rscore+(jj * N_size[0]) + kk )  >= model->thresh_ )
//					{
//						//Y[index]=kk+1;
//						//X[index]=jj+1;
//						Y.push_back(kk+1); //列
//						X.push_back(jj+1); //行
//						//index++;
//					}
//				}
//			}
//			float XY[600][68][4];
//			int back_dim[2];
//			back_dim[0] = resp[rlevel].dim_c;back_dim[1] = resp[rlevel].dim_r;
//
//			if (X.empty() == false)
//			//if (X[0] != NULL)
//			{
//				backtrack(XY[0][0],X,Y,parts, pyra.scale,pyra.padx,pyra.pady ,&back_dim[0]);
//				//backtrack(XY[0][0],X,Y,parts, pyra.scale,pyra.padx,pyra.pady ,&back_dim[0], index);
//			}
//			// Walk back down tree following pointers
//			for (int iii=0 ; iii<X.size() ; iii++)
//			{
//				int x = X[iii]-1; int y = Y[iii]-1;				
//				bs.c=c;
//				bs.s=*(rscore + (x * N_size[0]) + y);
//				bs.level = rlevel;
//				memcpy( bs.xy,XY[iii],sizeof(float)*numparts*4);
//				boxes.push_back(bs);
//			}
//			vector<int>().swap(X);vector<int>().swap(Y);//X.clear();Y.clear();
//			delete [] rscore;rscore = NULL;
//			delete [] temp2;temp2 = NULL;
//			delete [] tempIx;tempIx = NULL;
//			delete [] tempIy;tempIy = NULL;
//
//			//for(int np = 0; np < this->numparts-1; np++)
//			//{
//			//	delete [] ixvec[np];
//			//	delete [] iyvec[np];
//			//}
//			//delete [] ixvec;ixvec=NULL;
//			//delete [] iyvec;iyvec=NULL;
//			
//			//fconv_time.push_back(temp_ft.Stop());
//			//dt_time.push_back(temp_dt.Stop());
//		}
//		//parts.clear();
//		vector<Data_s>().swap(parts);
//		vector<double*>().swap(filters);
//		cout << pose.Stop()<<endl;
//		//printf("c = %d done\n",c);
//	}
//	//printf("total fconv_time:%f\n",sum(fconv_time));
//	//printf("total shiftdt_time:%f\n",sum(dt_time));
//	//vector<float>().swap(fconv_time);vector<float>().swap(dt_time);
//	vector<comp>().swap(components);
//	Clean();
//	_CrtDumpMemoryLeaks();
//	//boxes.resize(cnt);
//}
//
//void LandmarkDetector::detectPCA_CV(std::vector<Data_bs>&boxes,cv::Mat& img ,LandmarkModel *model,char modelname[],int kk4,vector<double*>filter_pca,cv::Mat coeffs,float &f_time)
//{
//	// fconv 宣告
//	int newdim[3],rootdim[2],rdim[3]; 
//	rdim[2] = 32;
//	//int level;
//	//ImgUtils::StopWatch temp_ft;ImgUtils::StopWatch temp_dt;
//	//ImgUtils::StopWatch feat_time;
//	vector<comp> components;
//	vector<float> fconv_time;vector<float> dt_time;
//
//	ImgInfo<unsigned char> splitimg , trimg;
//	trimg.AllocateImage(img.height_, img.width_, 3);
//	Operation::Transpose(trimg, img);
//	splitimg.AllocateImage(trimg.width_, trimg.height_ * 3);
//	Operation::SplitChannel(splitimg, trimg);
//
//	ImgInfo<float> fimg, rimg, gimg, bimg;
//	fimg.AllocateImage(splitimg.width_, splitimg.height_, splitimg.channel_);
//	Operation::Cast(fimg, splitimg);
//	float* data = fimg.data_;
//	int offset = fimg.width_step_ * img.width_;
//	bimg.SetImage(img.height_, img.width_, 1, fimg.width_step_, data);
//	data += offset;
//	gimg.SetImage(img.height_, img.width_, 1, fimg.width_step_, data);
//	data += offset;
//	rimg.SetImage(img.height_, img.width_, 1, fimg.width_step_, data);
//
//	//feat_time.Start();
//	featpyramid(fimg,model);
//	//printf("featpyramid preprocessing elapsed : %f\n", feat_time.Stop());
//
//	std::vector<Data_s> parts;float *resp_4_temp,*resp_;
//	modelcomponents(components,model);
//
//	for (int i=5; i<this->FP_level; i++) //
//	{
//		if (this->resp[i].dim_c !=NULL)
//		{
//			this->resp[i].dim_c = NULL;
//		}
//
//		if (this->resp[i].dim_r !=NULL)
//		{
//			this->resp[i].dim_r = NULL;
//		}
//
//		if (this->resp[i].r_ !=NULL)
//		{
//			this->resp[i].r_ = NULL;
//		}
//	}
//
//
//	//this->filter_num = model->filters_.size();
//	//for (int c=3; c<=14; c++)
//	
//	for (int c=0; c<model->components_.size(); c++)
//	{			
//		//pose.Start();
//		parts= components[c].comp_ ;
//		this->numparts = parts.size();
//		//printf("c = %d: ",c);
//		//#pragma omp parallel for num_threads(10)
//		for (int rlevel=5; rlevel<this->FP_level; rlevel++) //11 
//		{
//			//printf("%d ",rlevel);
//			//float** resp_ = new float*[filter_num];
//			if ( resp[rlevel].r_ == NULL ) 
//			{	
//				newdim[0]=pyra.dim_c[rlevel];newdim[1]=pyra.dim_r[rlevel]; newdim[2]=32;	
//				stopwatch.Start();
//				fconv_25cells_PCA_CV(&resp_,pyra.feat[rlevel],&newdim[0],&rootdim[0], filter_pca,coeffs);//filter_4_pca
//				//fconv_25cells(&resp_,pyra.feat[rlevel],&newdim[0],&rootdim[0],filters);
//				fconv_time.push_back(stopwatch.Stop());
//				//printf("%f  ", stopwatch.Stop());
//				//stopwatch.Start();
//				//pca_recon(&resp_4,resp_4_temp,SCORE4,filter_4_pca.size(),&rootdim[0]);
//				//printf("%f  ", stopwatch.Stop());
//				//}
//				//resp[rlevel].r_ = resp_4_temp;
//				resp[rlevel].r_ = resp_; // will be cancer
//				resp[rlevel].dim_c=rootdim[0];resp[rlevel].dim_r=rootdim[1]; //存取resp每層的維度大小
//				//delete [] resp_4_temp;resp_4_temp=NULL;
//			}
//			//temp_ft.Start();
//			//Local part scores
//
//			for (int k=0;k<numparts;k++)
//			{
//				//printf("%d ",k);
//				int f=parts[k].filterid_-1;
//				//level=rlevel-parts[k].scale*model->interval_-1;
//				parts[k].score = resp[rlevel].r_+ (f*resp[rlevel].dim_r*resp[rlevel].dim_c);
//				parts[k].level=rlevel+1;	
//			}
//			//printf("Local part scores preprocessing elapsed : %f\n", temp_ft.Stop());
//
//			//Walk from leaves to root of tree, passing message to parent
//			//Given a 2D array of filter scores 'child', shiftdt() does the following:
//			//(1) Apply distance transform
//			//(2) Shift by anchor position (child.startxy) of part wrt parent
//			//(3) Downsample by child.step
//			int N_size[2];
//
//			if (strcmp(modelname,"model_multipie_independent") ==1 && (c ==8 || c==11)){
//				N_size[0] = resp[rlevel].dim_c-1;N_size [1] = resp[rlevel].dim_r-1;}
//			else if(strcmp(modelname,"model_multipie_independent") ==1 && (c ==13||c==4)){
//				N_size[0] = resp[rlevel].dim_c+1;N_size [1] = resp[rlevel].dim_r+1;}
//			else{
//				N_size[0] = resp[rlevel].dim_c;N_size[1] = resp[rlevel].dim_r;}
//
//			int par,index=0;
//			float* vals;
//			float* M = new float[N_size[0]*N_size[1]];
//			float*tmpM = new float[N_size[0]*N_size[1]];
//			int *tmpIy = new int[N_size[0]*N_size[1]];
//			int* Ix = new int[N_size[0]*N_size[1]];
//			int* Iy = new int[N_size[0]*N_size[1]];	
//			float* temp2 = new float[(parts.size()-1)*N_size[1]*N_size[0]];
//			int* tempIx = new int[(parts.size()-1)*N_size[1]*N_size[0]];
//			int* tempIy = new int[(parts.size()-1)*N_size[1]*N_size[0]];
//
//			//temp_dt.Start();
//			for (int i=parts.size()-1;i>0;i--)
//			{ 
//				child = parts[i];
//				par=child.parent_-1;
//				vals = child.score;
//				for (int x = 0; x < N_size[1]; x++)
//				{
//					//cout << x<<endl;
//					dt1d(vals+x*N_size[0], tmpM+x*N_size[0], tmpIy+x*N_size[0], 1, N_size[0], -child.w_[2], -child.w_[3], child.starty-1, child.step);
//				}
//				for (int y = 0; y < N_size[0]; y++)
//				{
//					//cout << **(Mptr + (index*dims[0]*dims[1])+y) << "  ";
//					dt1d(tmpM+y, M+y, Ix+y, N_size[0], N_size[1], -child.w_[0], -child.w_[1], child.startx-1, child.step);
//				}
//				// get argmins and adjust for matlab indexing from 1
//				for (int x = 0; x < N_size[1]; x++) 
//				{
//					for (int y = 0; y < N_size[0]; y++) 
//					{
//						int p = x*N_size[0]+y;
//						Iy[p] = tmpIy[Ix[p]*N_size[0]+y]+1;	
//						Ix[p] = Ix[p]+1;
//					}
//				}		
//				for(int kk = 0; kk < N_size[0]*N_size[1]; ++kk) 
//				{
//					*(temp2 +(index*N_size[0]*N_size[1])+ kk) = *(parts[par].score + kk) + *(M + kk);		
//					*(tempIx+(index*N_size[0]*N_size[1])+kk) = *(Ix+kk);
//					*(tempIy+(index*N_size[0]*N_size[1])+kk) = *(Iy+kk);
//				}
//				parts[par].score = &temp2[index*N_size[0]*N_size[1]] ;	//parts(par).score = parts(par).score + msg
//				parts[i].Ix = &tempIx[index*N_size[0]*N_size[1]];
//				parts[i].Iy = &tempIy[index*N_size[0]*N_size[1]];;
//
//				index++;
//			}
//			delete [] tmpM;delete [] M;delete [] tmpIy;delete [] Ix;delete [] Iy;
//			//int** ixvec = new int*[this->numparts-1];
//			//int** iyvec = new int*[this->numparts-1];
//			//int index_dt=0;
//			////temp_dt.Start();
//			//float* temp2 = new float[(this->numparts-1)*N_size[0]*N_size[1]];
//			//for (int k2=this->numparts-1;k2>0;k2--)
//			//{	
//			//	float* msg;
//			//	child = parts[k2];
//			//	int par=child.parent_-1;
//			//	shiftdt(&msg, &ixvec[index_dt], &iyvec[index_dt], child.score, 
//			//		&child.w_[0],child.startx,child.starty,&N_size[0],child.step);
//			//	parts[k2].Ix=ixvec[index_dt];
//			//	parts[k2].Iy=iyvec[index_dt];
//			//	//MixScore(parts[par].score,msg,&N_size[0]);
//			//	//*parts[par].score=*parts[par].score+*msg;
//			//	for(int jj = 0; jj != N_size[1]; ++jj) 
//			//	{
//			//		for(int kk = 0; kk != N_size[0]; ++kk) 
//			//		{
//			//			*(temp2+ (index_dt*N_size[0]*N_size[1])+ (jj * N_size[0]) + kk) = *(parts[par].score+ (jj * N_size[0]) + kk) + *(msg + (jj * N_size[0]) + kk);
//			//		}
//			//	}
//			//	parts[par].score = &temp2[index_dt*N_size[0]*N_size[1]] ;	//parts(par).score = parts(par).score + msg
//			//	delete [] msg;msg = NULL;
//			//	index_dt++;
//			//}
//			
//			//printf("DT preprocessing elapsed : %f\n", temp_dt.Stop());
//
//			// Add bias to root score
//			float* rscore = new float[N_size[0]*N_size[1]];
//			for(int jj = 0; jj != N_size[1]*N_size[0]; ++jj) {
//				*(rscore+jj ) = *(parts[0].score + jj) + *(parts[0].w_);}  //rscore = parts(1).score + parts(1).w;
//
//			//int X[600],Y[600],index=0;
//			vector<int> Y;vector<int> X;
//			//#pragma omp parallel //for num_threads(10)
//			for(int jj = 0; jj != N_size[1]; ++jj) 
//			{
//				//printf("%d ",jj);
//				for(int kk = 0; kk != N_size[0]; ++kk) 
//				{
//					if ( *(rscore+(jj * N_size[0]) + kk )  >= model->thresh_ )
//					{
//						//Y[index]=kk+1;
//						//X[index]=jj+1;
//						Y.push_back(kk+1); //列
//						X.push_back(jj+1); //行
//						//index++;
//					}
//				}
//			}
//			float XY[600][68][4];
//			int back_dim[2];
//			back_dim[0] = resp[rlevel].dim_c;back_dim[1] = resp[rlevel].dim_r;
//
//			if (X.empty() == false)
//			//if (X[0] != NULL)
//			{
//				backtrack(XY[0][0],X,Y,parts, pyra.scale,pyra.padx,pyra.pady ,&back_dim[0]);
//				//backtrack(XY[0][0],X,Y,parts, pyra.scale,pyra.padx,pyra.pady ,&back_dim[0], index);
//			}
//			// Walk back down tree following pointers
//			for (int iii=0 ; iii<X.size() ; iii++)
//			{
//				int x = X[iii]-1; int y = Y[iii]-1;				
//				bs.c=c;
//				bs.s=*(rscore + (x * N_size[0]) + y);
//				bs.level = rlevel;
//				memcpy( bs.xy,XY[iii],sizeof(float)*numparts*4);
//				boxes.push_back(bs);
//			}
//			vector<int>().swap(X);vector<int>().swap(Y);//X.clear();Y.clear();
//			delete [] rscore;rscore = NULL;
//			delete [] temp2;temp2 = NULL;
//			delete [] tempIx;tempIx = NULL;
//			delete [] tempIy;tempIy = NULL;
//
//			//for(int np = 0; np < this->numparts-1; np++)
//			//{
//			//	delete [] ixvec[np];
//			//	delete [] iyvec[np];
//			//}
//			//delete [] ixvec;ixvec=NULL;
//			//delete [] iyvec;iyvec=NULL;
//			
//			//fconv_time.push_back(temp_ft.Stop());
//			//dt_time.push_back(temp_dt.Stop());
//		}
//		//parts.clear();
//		vector<Data_s>().swap(parts);
//		vector<double*>().swap(filters);
//		//cout << pose.Stop()<<endl;
//		//printf("c = %d done\n",c);
//	}
//	printf("total fconv_time:%f\n",sum(fconv_time));
//	//printf("total shiftdt_time:%f\n",sum(dt_time));
//	//vector<float>().swap(fconv_time);vector<float>().swap(dt_time);
//	f_time =  sum(fconv_time);
//	vector<comp>().swap(components);
//	Clean();
//	_CrtDumpMemoryLeaks();
//	//boxes.resize(cnt);
//}


void LandmarkDetector::modelcomponents(std::vector<comp> &components,LandmarkModel *model)
{
	for (int c=0;c<model->components_.size();c++)
	{
		int k=model->components_[c].defid_.size();
		create_components(components,model,c,k);
	}

	for (int i=0;i<model->filters_.size();i++)
	{
		filters.push_back( model->filters_[i].w_ );
	}

	//for (int i=0;i<filters.size();i++)
	//{
	//	for (int j=0;j<32;j++)
	//	{
	//		for (int k=0;k<4;k++)
	//		{
	//			for (int s=0;s<4;s++)
	//			{
	//				cout << *( filters[i] + j*4*4+k*4+s )<<"  ";
	//			}
	//		}
	//		cout << endl;
	//	}
	//}
}

void LandmarkDetector::create_components(std::vector<comp>&components,LandmarkModel *model,int c,int k1)
{
	comp a;Data_s p;
	for (int k=0;k<k1;k++)
	{
		p.defid_=model->components_[c].defid_[k];
		p.filterid_=model->components_[c].filterid_[k];
		p.parent_=model->components_[c].parent_[k];
		x.i_=model->filters_[p.filterid_-1].i_;
		p.sizy_=model->max_height_;p.sizx_=model->max_width_;int foo=32;
		p.filterI_=x. i_;
		x.w_=model->defs_[p.defid_-1].w_;
		x.i_=model->defs_[p.defid_-1].i_;
		x.anchor_=model->defs_[p.defid_-1].anchor_;
		p.defI_=x.i_;
		p.w_=x.w_;
		double par=p.parent_-1;
		double ax=x.anchor_[0];double ay=x.anchor_[1];double ds=x.anchor_[2];
		if (par>0)
		{
			p.scale = ds+a.comp_[par].scale;
		}
		else
		{
			p.scale=0;
		}
		// amount of (virtual) padding to hallucinate
		double step=std::pow(2.0,ds);
		float virtpady=(step-1)*pyra.pady;
		float virtpadx=(step-1)*pyra.padx;
		// starting points (simulates additional padding at finer scales)
		p.starty=ay-virtpady;
		p.startx=ax-virtpadx;
		p.step=step;p.level=0;p.score=0;p.Ix=0;p.Iy=0;
		a.comp_.push_back(p);
	}
	components.push_back(a);
}

struct thread_data 
{
	float *A;
	double *B;
	float *C;
	int *A_dims;
	int *B_dims;
	int *C_dims;
};///function fconv

//void  LandmarkDetector::fconv_MT(float** resp_,float* hog, int* hogdim, int* outdim, Filters_conv* filter)
//{
//	thread_data* td = new thread_data[filter_num];
//	pthread_t* ts =new pthread_t[filter_num];
//	int rdim[3];
//	resp_=&td[0].C;
//	for (int f1=0;f1<filter_num;f1++)
//	{
//		td[f1].A_dims = hogdim;
//		td[f1].A = hog;
//
//		rdim[0]=(int)sqrt((float)this->filter_dim[f1]/32);rdim[1]=rdim[0];rdim[2]=32;
//
//		td[f1].B_dims = rdim;
//		td[f1].B = filter[f1].w_;
//
//		// compute size of output
//		outdim[0] = td[f1].A_dims[0] - td[f1].B_dims[0] + 1;
//		outdim[1] = td[f1].A_dims[1] - td[f1].B_dims[1] + 1;
//		outdim[2] = 1;
//		td[f1].C_dims = outdim;
//		td[f1].C = new float[outdim[0]*outdim[1]];
//		::memset(td[f1].C, 0, sizeof(float)*outdim[0]*outdim[1]);
//		if (pthread_create(&ts[f1], NULL, ConvProcess_MT, (void *)&td[f1]))
//		{
//			printf("Error creating thread");  
//		}
//	}
//	//void *status;
//	//for (int f1=0;f1<filter_num;f1++)
//	//{
//	//	pthread_join(ts[f1], &status);
//	//}
//	//return td[0].C;
//}

void  LandmarkDetector::fconv_25cells_PCA(float** resp_,float* hog, int* hogdim, int* outdim,std::vector<double*> filters,double* SCORE)
{
	int rdim[3];
	thread_data td;
	filter_num = filters.size();
	//filter_num = model_->filters_.size();
	td.C = new float[filter_num* (hogdim[0]-(int)sqrt((float)this->filter_dim[0]/32)+1) * (hogdim[1]-(int)sqrt((float)this->filter_dim[0]/32)+1)];
	::memset(td.C, 0, sizeof(float)*filter_num* (hogdim[0]-(int)sqrt((float)this->filter_dim[0]/32)+1) * (hogdim[1]-(int)sqrt((float)this->filter_dim[0]/32)+1));
	//*resp_=td.C;
	td.A_dims = hogdim;
	td.A = hog;
	#pragma omp parallel for num_threads(8)
	for (int f1=0;f1<filter_num;f1++)
	{	
		rdim[0]=(int)sqrt((float)this->filter_dim[f1]/32);rdim[1]=rdim[0];rdim[2]=32;
		td.B_dims = rdim;
		td.B = filters[f1];
		//td.B = filters[f1].w_;
		// compute size of output
		outdim[0] = td.A_dims[0] - td.B_dims[0] + 1;
		outdim[1] = td.A_dims[1] - td.B_dims[1] + 1;
		td.C_dims = outdim;
		ConvProcess_25cells((void *)&td,f1,outdim);
		//printf("%d ",f1);
		//delete td.C;
	}
	float *b = new float[filter_num*outdim[0]*outdim[1]];
	::memset(b,0,sizeof(float)*filter_num*outdim[0]*outdim[1]);
	float *c= new float[outdim[0]*outdim[1]*model_->filters_.size()]; // c = a*b;
	::memset(c,0,sizeof(float)*outdim[0]*outdim[1]*model_->filters_.size());
	float *out_temp= new float[model_->filters_.size()*outdim[0]*outdim[1]];
	::memset(out_temp,0,sizeof(float)*model_->filters_.size()*outdim[0]*outdim[1]);

	//#pragma omp parallel for num_threads(10)
	for (int i=0;i<filter_num;i++)
	{
		int flag = 0;
		for (int j=0;j<outdim[1];j++)
		{
			for (int k=0;k<outdim[0];k++)
			{
				*( b+flag*filter_num+i ) = *( td.C+i*outdim[1]*outdim[0]+j*outdim[0]+k );
				//cout << *( in+i*indim[1]*indim[0]+j*indim[0]+k )<< "  ";
				flag++;
			}
		}
		//cout << endl<<endl;
	}

	float temp_c=0;  int flag = 0; int flag2 = 0;
	//#pragma omp parallel for num_threads(10)
	for (int i=0;i<model_->filters_.size();i++)
	{

		for (int j=0;j<outdim[0]*outdim[1];j++)
		{		
			float sum=0;
			for (int k=0;k<filter_num;k++)
			{
				temp_c = *( SCORE+k*model_->filters_.size() + i )**( b+j*filter_num+k );
				sum += temp_c;
				//cout <<*( a+k*model_->filters_.size() + i )<< "  ";
			}
			*( c+j*model_->filters_.size()+flag )  = sum;
			//cout <<*( c+i*model_->filters_.size()+j )<< "  ";
		}
		flag++;
		//cout << endl<<endl;
	}

	delete [] td.C;
	delete [] b;


	flag = -1;
	for (int i=0;i<model_->filters_.size();i++)
	{
		flag++; int flag2=0;
		for (int j=0;j<outdim[1];j++)
		{		
			for (int k=0;k<outdim[0];k++)
			{
				*( out_temp+i*outdim[1]*outdim[0] + j*outdim[0] + k ) = *( c+flag2*model_->filters_.size() + flag );
				flag2++;
				//cout <<*( a+k*model_->filters_.size() + i )<< "  ";
			}
		}
		//cout << endl<<endl;
	}
	*resp_ = out_temp;
	delete [] c;
	//return td.C;
}

void  LandmarkDetector::fconv_25cells_PCA_CV(float** resp_,float* hog, int* hogdim, int* outdim,std::vector<double*> filters,cv::Mat SCORE)
{
	int rdim[3];
	thread_data td;
	filter_num = filters.size();
	//filter_num = model_->filters_.size();
	td.C = new float[filter_num* (hogdim[0]-(int)sqrt((float)this->filter_dim[0]/32)+1) * (hogdim[1]-(int)sqrt((float)this->filter_dim[0]/32)+1)];
	::memset(td.C, 0, sizeof(float)*filter_num* (hogdim[0]-(int)sqrt((float)this->filter_dim[0]/32)+1) * (hogdim[1]-(int)sqrt((float)this->filter_dim[0]/32)+1));
	//*resp_=td.C;
	td.A_dims = hogdim;
	td.A = hog;
	#pragma omp parallel for num_threads(8)
	for (int f1=0;f1<filter_num;f1++)
	{	
		rdim[0]=(int)sqrt((float)this->filter_dim[f1]/32);rdim[1]=rdim[0];rdim[2]=32;
		td.B_dims = rdim;
		td.B = filters[f1];
		//td.B = filters[f1].w_;
		// compute size of output
		outdim[0] = td.A_dims[0] - td.B_dims[0] + 1;
		outdim[1] = td.A_dims[1] - td.B_dims[1] + 1;
		td.C_dims = outdim;
		ConvProcess_25cells((void *)&td,f1,outdim);
		//printf("%d ",f1);
		//delete td.C;
	}

	cv::Mat b;
	b.create(filter_num, outdim[0]*outdim[1], SCORE.type());
	//float *b = new float[filter_num*outdim[0]*outdim[1]];
	//::memset(b,0,sizeof(float)*filter_num*outdim[0]*outdim[1]);

	//float *c= new float[model_->filters_.size()*outdim[0]*outdim[1]]; // c = a*b;
	//::memset(c,0,sizeof(float)*outdim[0]*outdim[1]*model_->filters_.size());

	float *out_temp= new float[model_->filters_.size()*outdim[0]*outdim[1]];
	::memset(out_temp,0,sizeof(float)*model_->filters_.size()*outdim[0]*outdim[1]);

	//#pragma omp parallel for num_threads(10)
	for (int i=0;i<filter_num;i++)
	{
		int flag = 0;
		for (int j=0;j<outdim[1];j++)
		{
			for (int k=0;k<outdim[0];k++)
			{
				b.at<float>(i,flag) = *( td.C+i*outdim[1]*outdim[0]+j*outdim[0]+k );
				//*( b+flag*filter_num+i ) = *( td.C+i*outdim[1]*outdim[0]+j*outdim[0]+k );
				//cout << *( td.C+i*outdim[1]*outdim[0]+j*outdim[0]+k )<< "  ";
				flag++;
			}
		}
		//cout << endl<<endl;
	}
	cv::Mat c = SCORE * b;
	//for (int i=0;i<model_->filters_.size();i++)
	//{
	//	for (int j=0;j<outdim[0]*outdim[1];j++)
	//	{		
	//		float temp_c=0;
	//		//#pragma omp parallel for num_threads(10)
	//		for (int k=0;k<filter_num;k++)
	//		{
	//			//temp_c += SCORE.dot(b);
	//			
	//			//cout <<*( td.C+k*outdim[0]*outdim[1]+flag )<< "  ";
	//			temp_c +=  SCORE.at<float>(i,k)* b1.at<float>(j,k);
	//			//temp_c +=  SCORE.at<float>(i,k)**( b+j*filter_num+k );
	//			//sum += temp_c;
	//		}
	//		*( c+j*model_->filters_.size()+flag )  = temp_c;
	//		//cout <<*( c+i*model_->filters_.size()+j )<< "  ";
	//	}
	//	flag++;
	//	//cout << endl<<endl;
	//}

	delete [] td.C;
	//delete [] b;


	int flag = -1;
	for (int i=0;i<model_->filters_.size();i++)
	{
		flag++; int flag2=0;
		for (int j=0;j<outdim[1];j++)
		{		
			for (int k=0;k<outdim[0];k++)
			{
				*( out_temp+i*outdim[1]*outdim[0] + j*outdim[0] + k ) = c.at<float>(flag,flag2);//*( c+flag2*model_->filters_.size() + flag );
				flag2++;
				//cout <<*( a+k*model_->filters_.size() + i )<< "  ";
			}
		}
		//cout << endl<<endl;
	}
	*resp_ = out_temp;
	//delete [] c;
	//return td.C;
}
void  LandmarkDetector::fconv_25cells(float** resp_,float* hog, int* hogdim, int* outdim,std::vector<double*> filters)
{
	int rdim[3];
	thread_data td;
	filter_num = filters.size();
	//filter_num = model_->filters_.size();
	td.C = new float[filter_num* (hogdim[0]-(int)sqrt((float)this->filter_dim[0]/32)+1) * (hogdim[1]-(int)sqrt((float)this->filter_dim[0]/32)+1)];
	::memset(td.C, 0, sizeof(float)*filter_num* (hogdim[0]-(int)sqrt((float)this->filter_dim[0]/32)+1) * (hogdim[1]-(int)sqrt((float)this->filter_dim[0]/32)+1));
	*resp_=td.C;		
	td.A_dims = hogdim;
	td.A = hog;
	#pragma omp parallel for num_threads(8)  //OpenMP 不適用在Part filter 的size有變化的model下
	for (int f1=0;f1<filter_num;f1++)
	{	
		rdim[0]=(int)sqrt((float)this->filter_dim[f1]/32);rdim[1]=rdim[0];rdim[2]=32;
		td.B_dims = rdim;
		td.B = filters[f1];
		//td.B = filters[f1].w_;
		// compute size of output
		outdim[0] = td.A_dims[0] - td.B_dims[0] + 1;
		outdim[1] = td.A_dims[1] - td.B_dims[1] + 1;
		td.C_dims = outdim;
		ConvProcess_25cells((void *)&td,f1,outdim);
		//printf("%d ",f1);
		//delete td.C;
	}
	//return td.C;
}

void  LandmarkDetector::fconv_17cells(float** resp_,float* hog, int* hogdim, int* outdim,std::vector<double*>)
{
	int rdim[3];
	thread_data td;
	//filter_num = filter.size();
	filter_num = model_->filters_.size();
	td.C = new float[filter_num* (hogdim[0]-(int)sqrt((float)this->filter_dim[0]/32)+1) * (hogdim[1]-(int)sqrt((float)this->filter_dim[0]/32)+1)];
	::memset(td.C, 0, sizeof(float)*filter_num* (hogdim[0]-(int)sqrt((float)this->filter_dim[0]/32)+1) * (hogdim[1]-(int)sqrt((float)this->filter_dim[0]/32)+1));
	*resp_=td.C;
	#pragma omp parallel for num_threads(10)
	for (int f1=0;f1<filter_num;f1++)
	{	
		td.A_dims = hogdim;
		td.A = hog;
		rdim[0]=(int)sqrt((float)this->filter_dim[f1]/32);rdim[1]=rdim[0];rdim[2]=32;
		td.B_dims = rdim;
		double* temp_filter = new double[17*32];
		::memset(temp_filter, 0, sizeof(double)*17*32);
		int step = 0;
		for (int i=0;i<32;i++)
		{
			for (int j=0;j<5;j++)
			{
				for (int k=0;k<5;k++)
				{
					if ( ((j==0 || j==4) && (k == 0 || k == 2 || k == 4)) || ( (j==1 || j==3) && (k==1 || k==2 || k==3)) || j == 2 )
					{
						*(temp_filter+step) = *(model_->filters_[f1].w_+i*5*5+j*5+k);
						//cout << *(temp_filter+step) <<" ";
						step++;
					}

				}
			}

		}
		td.B = temp_filter;
		//
		// compute size of output
		outdim[0] = td.A_dims[0] - td.B_dims[0] + 1;
		outdim[1] = td.A_dims[1] - td.B_dims[1] + 1;
		td.C_dims = outdim;
		ConvProcess_17cells((void *)&td,f1,outdim);
		//printf("%d ",f1);
		delete [] temp_filter;temp_filter = NULL;
	}
	//return td.C;
}

void  LandmarkDetector::fconv_9cells(float** resp_,float* hog, int* hogdim, int* outdim,std::vector<double*> filters)
{
	int rdim[3];
	thread_data td;
	//filter_num = filter.size();
	filter_num = model_->filters_.size();
	td.C = new float[filter_num* (hogdim[0]-(int)sqrt((float)this->filter_dim[0]/32)+1) * (hogdim[1]-(int)sqrt((float)this->filter_dim[0]/32)+1)];
	::memset(td.C, 0, sizeof(float)*filter_num* (hogdim[0]-(int)sqrt((float)this->filter_dim[0]/32)+1) * (hogdim[1]-(int)sqrt((float)this->filter_dim[0]/32)+1));
	*resp_=td.C;
	#pragma omp parallel for num_threads(10)
	for (int f1=0;f1<filter_num;f1++)
	{	
		td.A_dims = hogdim;
		td.A = hog;

	/*	for (int i=0;i<32;i++)
		{
			for (int j=0;j<hogdim[1];j++)
			{
				for (int k=0;k<hogdim[0];k++)
				{
					cout<< *hog << " ";
					*(hog++);
				}
				cout<<endl;
			}
		}*/

		rdim[0]=(int)sqrt((float)this->filter_dim[f1]/32);rdim[1]=rdim[0];rdim[2]=32;
		td.B_dims = rdim;
		double* temp_filter = new double[9*32];
		::memset(temp_filter, 0, sizeof(double)*9*32);
		int step = 0;
		for (int i=0;i<32;i++)
		{
			for (int j=0;j<5;j++)
			{
				for (int k=0;k<5;k++)
				{
					if (k == 2 || j == 2)
					{
						*(temp_filter+step) = *(model_->filters_[f1].w_+i*5*5+j*5+k);
						//cout << *(temp_filter+step) <<" ";
						step++;
					}

				}
			}

		}
		td.B = temp_filter;
		//
		// compute size of output
		outdim[0] = td.A_dims[0] - td.B_dims[0] + 1;
		outdim[1] = td.A_dims[1] - td.B_dims[1] + 1;
		td.C_dims = outdim;
		ConvProcess_9cells((void *)&td,f1,outdim);
		//printf("%d ",f1);
		delete [] temp_filter;temp_filter = NULL;
	}
	//return td.C;
}

//void* ConvProcess_MT(void *thread_arg)
//{
//	thread_data *args = (thread_data *)thread_arg;
//	float *A = args->A;
//	double *B = args->B;
//	float *C = args->C;
//	int *A_dims = args->A_dims;
//	int *B_dims = args->B_dims;
//	int *C_dims = args->C_dims;
//	int num_features = A_dims[2];
//
//	for (int f = 0; f < num_features; f++) 
//	{
//		float *dst = C;
//		float *A_src = A + f*A_dims[0]*A_dims[1];      
//		double *B_src = B + f*B_dims[0]*B_dims[1];
//		for (int x = 0; x < C_dims[1]; x++) 
//		{
//			for (int y = 0; y < C_dims[0]; y++) 
//			{
//				double val = 0;
//				for (int xp = 0; xp < B_dims[1]; xp++) 
//				{
//					float *A_off = A_src + (x+xp)*A_dims[0] + y;
//					double *B_off = B_src + xp*B_dims[0];
//					switch(B_dims[0]) 
//					{
//					case 20: val += A_off[19] * B_off[19];
//					case 19: val += A_off[18] * B_off[18];
//					case 18: val += A_off[17] * B_off[17];
//					case 17: val += A_off[16] * B_off[16];
//					case 16: val += A_off[15] * B_off[15];
//					case 15: val += A_off[14] * B_off[14];
//					case 14: val += A_off[13] * B_off[13];
//					case 13: val += A_off[12] * B_off[12];
//					case 12: val += A_off[11] * B_off[11];
//					case 11: val += A_off[10] * B_off[10];
//					case 10: val += A_off[9] * B_off[9];
//					case 9: val += A_off[8] * B_off[8];
//					case 8: val += A_off[7] * B_off[7];
//					case 7: val += A_off[6] * B_off[6];
//					case 6: val += A_off[5] * B_off[5];
//					case 5: val += A_off[4] * B_off[4];
//					case 4: val += A_off[3] * B_off[3];
//					case 3: val += A_off[2] * B_off[2];
//					case 2: val += A_off[1] * B_off[1];
//					case 1: val += A_off[0] * B_off[0];
//					break;
//					default:	    	      
//						for (int yp = 0; yp < B_dims[0]; yp++) 
//						{
//							val += *(A_off++) * *(B_off++);
//						}
//					}
//				}
//				//*(dst+x*C_dims[0]+y) += val;
//				*(dst++) += val;
//			}
//		}
//	}
//	return NULL;
//};

void ConvProcess_25cells(void *thread_arg,int f1,int* outdim)
{
	thread_data *args = (thread_data *)thread_arg;
	float *A = args->A;
	double *B = args->B;
	float *C = args->C+f1*outdim[1]*outdim[0];
	int *A_dims = args->A_dims;
	int *B_dims = args->B_dims;
	int *C_dims = args->C_dims;
	int num_features = A_dims[2];

	for (int f = 0; f < num_features; f++) 
	{
		float *dst = C;
		float *A_src = A + f*A_dims[0]*A_dims[1];  //image Feature 32層中的哪一層  (二維陣列)
		double *B_src = B + f*B_dims[0]*B_dims[1]; //filter(i) 32層中的哪一層   (二維陣列)

		for (int x = 0; x < C_dims[1]; x++) //Filter response size(w-filter.w+1)
		{
			for (int y = 0; y < C_dims[0]; y++)  //Filter response size(h-filter.h+1)
			{
				double val = 0;
				for (int xp = 0; xp < B_dims[1]; xp++) //filter size(filter.w)
				{
					float *A_off = A_src + (x+xp)*A_dims[0] + y; //將指標位置指到A_src第一個位置
					double *B_off = B_src+ xp*B_dims[0];		 //將指標位置指到B_src第一個位置
					switch(B_dims[0]) // 判斷filter 大小
					{
					case 20: val += A_off[19] * B_off[19];
					case 19: val += A_off[18] * B_off[18];
					case 18: val += A_off[17] * B_off[17];
					case 17: val += A_off[16] * B_off[16];
					case 16: val += A_off[15] * B_off[15];
					case 15: val += A_off[14] * B_off[14];
					case 14: val += A_off[13] * B_off[13];
					case 13: val += A_off[12] * B_off[12];
					case 12: val += A_off[11] * B_off[11];
					case 11: val += A_off[10] * B_off[10];
					case 10: val += A_off[9] * B_off[9];
					case 9: val += A_off[8] * B_off[8];
					case 8: val += A_off[7] * B_off[7];
					case 7: val += A_off[6] * B_off[6];
					case 6: val += A_off[5] * B_off[5];
					case 5: val += A_off[4] * B_off[4];
					case 4: val += A_off[3] * B_off[3];
					case 3: val += A_off[2] * B_off[2];
					case 2: val += A_off[1] * B_off[1];
					case 1: val += A_off[0] * B_off[0];
						break;
					default:	    	      
						for (int yp = 0; yp < B_dims[0]; yp++) 
						{
							val += *(A_off++) * *(B_off++);
						}
					}
				}
				//*(dst+x*C_dims[0]+y) += val;
				*(dst++) += val;
				//cout<<endl<<endl;
			}
		}
	}
	//return NULL;
};

void ConvProcess_17cells(void *thread_arg,int f1,int* outdim)
{
	thread_data *args = (thread_data *)thread_arg;
	float *A = args->A;
	double *B = args->B;
	float *C = args->C+f1*outdim[1]*outdim[0];
	int *A_dims = args->A_dims;
	int *B_dims = args->B_dims;
	int *C_dims = args->C_dims;
	int num_features = A_dims[2];

	for (int f = 0; f < num_features; f++) 
	{
		float *dst = C;
		float *A_src = A + f*A_dims[0]*A_dims[1];  //image Feature 32層中的哪一層  (二維陣列)
		double *B_src = B + f*17; //filter(i) 32層中的哪一層   (二維陣列)

		for (int x = 0; x < C_dims[1]; x++) //Filter response size(w-filter.w+1)
		{
			for (int y = 0; y < C_dims[0]; y++)  //Filter response size(h-filter.h+1)
			{
				double val = 0;
				for (int xp = 0; xp < B_dims[1]; xp++) //filter size(filter.w)
				{
					float *A_off = A_src + (x+xp)*A_dims[0] + y; //將指標位置指到A_src第一個位置
					double *B_off = B_src;		 //將指標位置指到B_src第一個位置
					if (xp == 2)
					{
						switch(B_dims[0]) // 判斷filter 大小
						{
							case 20: val += A_off[19] * B_off[19];
							case 19: val += A_off[18] * B_off[18];
							case 18: val += A_off[17] * B_off[17];
							case 17: val += A_off[16] * B_off[16];
							case 16: val += A_off[15] * B_off[15];
							case 15: val += A_off[14] * B_off[14];
							case 14: val += A_off[13] * B_off[13];
							case 13: val += A_off[12] * B_off[12];
							case 12: val += A_off[11] * B_off[11];
							case 11: val += A_off[10] * B_off[10];
							case 10: val += A_off[9] * B_off[9];
							case 9: val += A_off[8] * B_off[8];
							case 8: val += A_off[7] * B_off[7];
							case 7: val += A_off[6] * B_off[6];
							case 6: val += A_off[5] * B_off[5];
							case 5: val += A_off[4] * B_off[10];
							case 4: val += A_off[3] * B_off[9];
							case 3: val += A_off[2] * B_off[8];
							case 2: val += A_off[1] * B_off[7];
							case 1: val += A_off[0] * B_off[6];
								//cout << A_off[0]<<" "<<A_off[1]<<" "<<A_off[2]<<" "<<A_off[3]<<" "<<A_off[4]<<endl;
								//cout << B_off[6]<<" "<<B_off[7]<<" "<<B_off[8]<<" "<<B_off[9]<<" "<<B_off[10]<<endl;
								break;
								for (int yp = 0; yp < B_dims[0]; yp++) 
								{
									val += *(A_off++) * *(B_off++);
								}
						}
					}

					else if (xp==0 || xp==4)
					{
						//cout<<A_off[0]<<" "<<A_off[2]<<" "<<A_off[4]<<endl;
						//cout<<B_off[0]<<" "<<B_off[1]<<" "<<B_off[2]<<endl;
						val += A_off[0] * B_off[0]+A_off[2] * B_off[1]+A_off[4] * B_off[2];
						if (xp==4)
						{
						//cout<<A_off[0]<<" "<<A_off[2]<<" "<<A_off[4]<<endl;
						//cout<<B_off[14]<<" "<<B_off[15]<<" "<<B_off[16]<<endl;
							val += A_off[0] * B_off[14]+A_off[2] * B_off[15]+A_off[4] * B_off[16];
						}
					}
					else if (xp==1 || xp==3)
					{
						//cout<<A_off[1]<<" "<<A_off[2]<<" "<<A_off[3]<<endl;
						//cout<<B_off[3]<<" "<<B_off[4]<<" "<<B_off[5]<<endl;
						val += A_off[1] * B_off[3]+A_off[2] * B_off[4]+A_off[3] * B_off[5];
						if (xp==3)
						{
							//cout<<A_off[1]<<" "<<A_off[2]<<" "<<A_off[3]<<endl;
							//cout<<B_off[11]<<" "<<B_off[12]<<" "<<B_off[13]<<endl;
							val += A_off[1] * B_off[11]+A_off[2] * B_off[12]+A_off[4] * B_off[13];
						}
					}

				}
				//*(dst+x*C_dims[0]+y) += val;
				*(dst++) += val;
				//cout<<endl<<endl;
			}
		}
	}
	//return NULL;
};

void ConvProcess_9cells(void *thread_arg,int f1,int* outdim)
{
	thread_data *args = (thread_data *)thread_arg;
	float *A = args->A;
	double *B = args->B;
	float *C = args->C+f1*outdim[1]*outdim[0];
	int *A_dims = args->A_dims;
	int *B_dims = args->B_dims;
	int *C_dims = args->C_dims;
	int num_features = A_dims[2];

	for (int f = 0; f < num_features; f++) 
	{
		float *dst = C;
		float *A_src = A + f*A_dims[0]*A_dims[1];  //image Feature 32層中的哪一層  (二維陣列)
		double *B_src = B + f*9;//B_dims[0]*B_dims[1]; //filter(i) 32層中的哪一層   (二維陣列)

		for (int x = 0; x < C_dims[1]; x++) //Filter response size(w-filter.w+1)
		{
			for (int y = 0; y < C_dims[0]; y++)  //Filter response size(h-filter.h+1)
			{
				double val = 0;
				for (int xp = 0; xp < B_dims[1]; xp++) //filter size(filter.w)
				{
					float *A_off = A_src + (x+xp)*A_dims[0] + y; //將指標位置指到A_src第一個位置
					double *B_off = B_src;	 //將指標位置指到B_src第一個位置
					if (xp == 2)
					{
						switch(B_dims[0]) // 判斷filter 大小
						{
						case 20: val += A_off[19] * B_off[19];
						case 19: val += A_off[18] * B_off[18];
						case 18: val += A_off[17] * B_off[17];
						case 17: val += A_off[16] * B_off[16];
						case 16: val += A_off[15] * B_off[15];
						case 15: val += A_off[14] * B_off[14];
						case 14: val += A_off[13] * B_off[13];
						case 13: val += A_off[12] * B_off[12];
						case 12: val += A_off[11] * B_off[11];
						case 11: val += A_off[10] * B_off[10];
						case 10: val += A_off[9] * B_off[9];
						case 9: val += A_off[8] * B_off[8];
						case 8: val += A_off[7] * B_off[7];
						case 7: val += A_off[6] * B_off[6];
						case 6: val += A_off[5] * B_off[5];
						case 5: val += A_off[4] * B_off[6];
						case 4: val += A_off[3] * B_off[5];
						case 3: val += A_off[2] * B_off[4];
						case 2: val += A_off[1] * B_off[3];
						case 1: val += A_off[0] * B_off[2];
							//cout << A_off[0]<<" "<<A_off[1]<<" "<<A_off[2]<<" "<<A_off[3]<<" "<<A_off[4]<<" ";
							//cout << B_off[2]<<" "<<B_off[3]<<" "<<B_off[4]<<" "<<B_off[5]<<" "<<B_off[6]<<" ";
							break;
						default:	    	      
							for (int yp = 0; yp < B_dims[0]; yp++) 
							{
								val += *(A_off++) * *(B_off++);
							}
						}
					}

					else if (xp==0 || xp==1)
					{
						//cout<<A_off[2]<<" ";
						//cout<<B_off[xp]<<" ";
						val += A_off[2] * B_off[xp];
					}
					else{
						//cout<<A_off[2]<<" ";
						//cout<<B_off[xp+4]<<" ";
						val += A_off[2] * B_off[xp+4];}
				}
				//*(dst+x*C_dims[0]+y) += val;
				*(dst++) += val;
				//cout<<endl<<endl;
			}
		}
	}
	//return NULL;
};

void LandmarkDetector::pca_recon(float**out,float*in,double* SCORE,int filter_num,int* indim)
{
	float *b = new float[filter_num*indim[0]*indim[1]];
	::memset(b,0,sizeof(float)*filter_num*indim[0]*indim[1]);
	//#pragma omp parallel for num_threads(10)
	for (int i=0;i<filter_num;i++)
	{
		int flag = 0;
		for (int j=0;j<indim[1];j++)
		{
			for (int k=0;k<indim[0];k++)
			{
				*( b+flag*filter_num+i ) = *( in+i*indim[1]*indim[0]+j*indim[0]+k );
				//cout << *( in+i*indim[1]*indim[0]+j*indim[0]+k )<< "  ";
				flag++;
			}
		}
		//cout << endl<<endl;
	}

	//float *a = new float[filter_num*model_->filters_.size()];
	//::memset(a,0,sizeof(float)*filter_num*model_->filters_.size());
	////#pragma omp parallel for num_threads(10)
	//for (int i=0;i<filter_num;i++)
	//{
	//	for (int j=0;j<model_->filters_.size();j++)
	//	{
	//		*( a+i*model_->filters_.size()+j ) = *( SCORE+i*model_->filters_.size()+j );
	//		//cout << a[i*model_->filters_.size()+j]<< "  ";
	//	}
	//	//cout << endl<<endl;
	//}

	float *c= new float[indim[0]*indim[1]*model_->filters_.size()]; // c = a*b;
	::memset(c,0,sizeof(float)*indim[0]*indim[1]*model_->filters_.size());
	float temp_c=0;  int flag = 0; int flag2 = 0;
	//#pragma omp parallel for num_threads(10)
	for (int i=0;i<model_->filters_.size();i++)
	{
		
		for (int j=0;j<indim[0]*indim[1];j++)
		{		
			float sum=0;
			for (int k=0;k<filter_num;k++)
			{
				temp_c = *( SCORE+k*model_->filters_.size() + i )**( b+j*filter_num+k );
				sum += temp_c;
				//cout <<*( a+k*model_->filters_.size() + i )<< "  ";
			}
			*( c+j*model_->filters_.size()+flag )  = sum;
			//cout <<*( c+i*model_->filters_.size()+j )<< "  ";
		}
		flag++;
		//cout << endl<<endl;
	}

	//delete [] a;
	delete [] b;

	float *out_temp= new float[model_->filters_.size()*indim[0]*indim[1]];
	::memset(out_temp,0,sizeof(float)*model_->filters_.size()*indim[0]*indim[1]);
	flag = -1;
	for (int i=0;i<model_->filters_.size();i++)
	{
		flag++; int flag2=0;
		for (int j=0;j<indim[1];j++)
		{		
			for (int k=0;k<indim[0];k++)
			{
				*( out_temp+i*indim[1]*indim[0] + j*indim[0] + k ) = *( c+flag2*model_->filters_.size() + flag );
				flag2++;
				//cout <<*( a+k*model_->filters_.size() + i )<< "  ";
			}
		}
		//cout << endl<<endl;
	}
	*out = out_temp;
	delete [] c;
}


void LandmarkDetector::dt1d(float *src, float *dst, int *ptr, int step, int len, double a, double b, int dshift, double dstep)
{
	int   *v = new int[len];
	float *z = new float[len+1];
	int k = 0;
	int q = 0;
	v[0] = 0;
	z[0] = -INF;
	z[1] = +INF;

	for (q = 1; q <= len-1; q++) 
	{
		float s = ((src[q*step] - src[v[k]*step]) - b*(q - v[k]) + a*(square(q) - square(v[k]))) / (2*a*(q-v[k]));
		while (s <= z[k]) {
			k--;
			s  = ((src[q*step] - src[v[k]*step]) - b*(q - v[k]) + a*(square(q) - square(v[k]))) / (2*a*(q-v[k]));
		}
		k++;
		v[k]   = q;
		z[k]   = s;
		z[k+1] = +INF;
	}

	k = 0;
	q = dshift;

	for (int i=0; i <= len-1; i++) {
		while (z[k+1] < q)
			k++;
		dst[i*step] = a*square(q-v[k]) + b*(q-v[k]) + src[v[k]*step];
		ptr[i*step] = v[k];
		q += dstep;
	}

	delete [] v;//v=NULL;
	delete [] z;//z=NULL;
}
// void LandmarkDetector::shiftdt(float** temp2,int** tempIx,int** tempIy,std::vector<Data_s> &parts,int* dims)
//{
//
//	
//	float *total_score;
//	*temp2 = total_score;
//	
//	delete [] tmpM;//tmpM=NULL;
//	delete [] tmpIy;//tmpIy=NULL;
//	delete [] Ix;
//	delete [] Iy;
//}

//inline void shiftdt(float** Mptr, int** Ixptr, int** Iyptr, float* vals, double* w,int offx, int offy, int* dims,int step)
//{
//	float* M = new float[dims[0]*dims[1]];
//	int* Ix = new int[dims[0]*dims[1]];
//	int* Iy = new int[dims[0]*dims[1]];
//	float*tmpM = new float[dims[0]*dims[1]];
//	//int *tmpIx = new int[dims[0]*dims[1]];
//	int *tmpIy = new int[dims[0]*dims[1]];
//	offy-=1;
//	offx-=1;
//
//
//	*Mptr = M;
//	*Ixptr = Ix;
//	*Iyptr = Iy;
//
////#pragma omp parallel for num_threads(10)
//	for (int x = 0; x < dims[1]; x++)
//	{
//		dt1d(vals+x*dims[0], tmpM+x*dims[0], tmpIy+x*dims[0], 1, dims[0], -w[2], -w[3], offy, step);
//	}
//
////#pragma omp parallel for num_threads(10)
//	for (int y = 0; y < dims[0]; y++)
//	{
//		dt1d(tmpM+y, M+y, Ix+y, dims[0], dims[1], -w[0], -w[1], offx, step);
//	}
//	cout <<endl;
//	for (int i=0;i<dims[0]*dims[1];i++)
//	{
//		cout <<  *(M+i) << " ";
//	}
//
//	//get argmins and adjust for matlab indexing from 1
//		for (int x = 0; x < dims[1]; x++) 
//		{
//			for (int y = 0; y < dims[0]; y++) 
//			{
//				int p = x*dims[0]+y;
//				Iy[p] = tmpIy[Ix[p]*dims[0]+y]+1;	
//				Ix[p] = Ix[p]+1;
//			}
//		}
//		delete [] tmpM;tmpM=NULL;
//		//delete [] tmpIx;tmpIx = NULL;
//		delete [] tmpIy;tmpIy=NULL;
//}

void features(float** feature ,int* out,float* im , int w,int h , int sbin)
{
	int dims[3];
	dims[0] = w;
	dims[1] = h;
	dims[2] = 3;

	int blocks[2];
	blocks[0] = (int)round((double)dims[0]/(double)sbin);
	blocks[1] = (int)round((double)dims[1]/(double)sbin);
	double *hist = new double[blocks[0]*blocks[1]*18];//(double *)mxCalloc(blocks[0]*blocks[1]*18, sizeof(double));
	::memset(hist, 0, sizeof(double)*blocks[0]*blocks[1]*18);
	double *norm = new double[blocks[0]*blocks[1]];//(double *)mxCalloc(blocks[0]*blocks[1], sizeof(double));
	::memset(norm, 0, sizeof(double)*blocks[0]*blocks[1]);

	// memory for HOG features
	//int out[3];
	out[0] = max(blocks[0]-2, 0);
	out[1] = max(blocks[1]-2, 0);
	out[2] = 27+4+1;
	float* feat = new float[out[0]*out[1]*out[2]];
	::memset(feat, 0, sizeof(float)*out[0]*out[1]*out[2]);	
	*feature = feat;

	int visible[2];
	visible[0] = blocks[0]*sbin;
	visible[1] = blocks[1]*sbin;

	for (int x = 1; x < visible[1]-1; x++) 
	{
		for (int y = 1; y < visible[0]-1; y++) 
		{
			// first color channel
			float *s = im + min(x, dims[1]-2)*dims[0] + min(y, dims[0]-2);
			double dy = *(s+1) - *(s-1);
			double dx = *(s+dims[0]) - *(s-dims[0]);
			double v = dx*dx + dy*dy;

			// second color channel
			s += dims[0]*dims[1];
			double dy2 = *(s+1) - *(s-1);
			double dx2 = *(s+dims[0]) - *(s-dims[0]);
			double v2 = dx2*dx2 + dy2*dy2;

			// third color channel
			s += dims[0]*dims[1];
			double dy3 = *(s+1) - *(s-1);
			double dx3 = *(s+dims[0]) - *(s-dims[0]);
			double v3 = dx3*dx3 + dy3*dy3;

			// pick channel with strongest gradient
			if (v2 > v) 
			{
				v = v2;
				dx = dx2;
				dy = dy2;
			} 
			if (v3 > v) 
			{
				v = v3;
				dx = dx3;
				dy = dy3;
			}

			// snap to one of 18 orientations
			double best_dot = 0;
			int best_o = 0;
			for (int o = 0; o < 9; o++) 
			{
				double dot = uu[o]*dx + vv[o]*dy;
				if (dot > best_dot) 
				{
					best_dot = dot;
					best_o = o;
				} 
				else if (-dot > best_dot) 
				{
					best_dot = -dot;
					best_o = o+9;
				}
			}

			// add to 4 histograms around pixel using linear interpolation
			double xp = ((double)x+0.5)/(double)sbin - 0.5;
			double yp = ((double)y+0.5)/(double)sbin - 0.5;
			int ixp = (int)floor(xp);
			int iyp = (int)floor(yp);
			double vx0 = xp-ixp;
			double vy0 = yp-iyp;
			double vx1 = 1.0-vx0;
			double vy1 = 1.0-vy0;
			v = sqrt(v);

			if (ixp >= 0 && iyp >= 0) {
				*(hist + ixp*blocks[0] + iyp + best_o*blocks[0]*blocks[1]) += 
					vx1*vy1*v;
			}

			if (ixp+1 < blocks[1] && iyp >= 0) {
				*(hist + (ixp+1)*blocks[0] + iyp + best_o*blocks[0]*blocks[1]) += 
					vx0*vy1*v;
			}

			if (ixp >= 0 && iyp+1 < blocks[0]) {
				*(hist + ixp*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1]) += 
					vx1*vy0*v;
			}

			if (ixp+1 < blocks[1] && iyp+1 < blocks[0]) {
				*(hist + (ixp+1)*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1]) += 
					vx0*vy0*v;
			}
		}
	}

	for (int o = 0; o < 9; o++) 
	{
		double *src1 = hist + o*blocks[0]*blocks[1];
		double *src2 = hist + (o+9)*blocks[0]*blocks[1];
		double *dst = norm;
		double *end = norm + blocks[1]*blocks[0];
		while (dst < end) 
		{
			*(dst++) += (*src1 + *src2) * (*src1 + *src2);
			src1++;
			src2++;
		}
	}

	// compute features
	for (int x = 0; x < out[1]; x++) 
	{
		for (int y = 0; y < out[0]; y++) 
		{
			float *dst = feat + x*out[0] + y;      
			double *src, *p, n1, n2, n3, n4;

			p = norm + (x+1)*blocks[0] + y+1;
			n1 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
			p = norm + (x+1)*blocks[0] + y;
			n2 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
			p = norm + x*blocks[0] + y+1;
			n3 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
			p = norm + x*blocks[0] + y;      
			n4 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);

			double t1 = 0;
			double t2 = 0;
			double t3 = 0;
			double t4 = 0;

			// contrast-sensitive features
			src = hist + (x+1)*blocks[0] + (y+1);
			for (int o = 0; o < 18; o++) 
			{
				double h1 = min(*src * n1, 0.2);
				double h2 = min(*src * n2, 0.2);
				double h3 = min(*src * n3, 0.2);
				double h4 = min(*src * n4, 0.2);
				*dst = 0.5 * (h1 + h2 + h3 + h4);
				t1 += h1;
				t2 += h2;
				t3 += h3;
				t4 += h4;
				dst += out[0]*out[1];
				src += blocks[0]*blocks[1];
			}

			// contrast-insensitive features
			src = hist + (x+1)*blocks[0] + (y+1);
			for (int o = 0; o < 9; o++) 
			{
				double sum = *src + *(src + 9*blocks[0]*blocks[1]);
				double h1 = min(sum * n1, 0.2);
				double h2 = min(sum * n2, 0.2);
				double h3 = min(sum * n3, 0.2);
				double h4 = min(sum * n4, 0.2);
				*dst = 0.5 * (h1 + h2 + h3 + h4);
				dst += out[0]*out[1];
				src += blocks[0]*blocks[1];
			}

			// texture features
			*dst = 0.2357 * t1;
			dst += out[0]*out[1];
			*dst = 0.2357 * t2;
			dst += out[0]*out[1];
			*dst = 0.2357 * t3;
			dst += out[0]*out[1];
			*dst = 0.2357 * t4;

			// truncation feature
			dst += out[0]*out[1];
			*dst = 0;
		}
	}
	delete [] hist;hist=NULL;
	delete [] norm;norm=NULL;
}

void LandmarkDetector::featpyramid(cv::Mat&img ,LandmarkModel *model)
{
	int out[3];
	int intervel = model->interval_;
	int sbin=model->sbin_;
	int padx = max(model->max_height_-1-1,0);
	int pady = max(model->max_width_-1-1,0);
	double sc=pow(2,((double)1/intervel));

	int width = img.cols;
	int height = img.rows / 3;
	int max_scale=1+floor(log((double) min(width,height)/(5*sbin))/log(sc));
	this->FP_level = max_scale+intervel;

	for (int i=0;i<intervel;i++)
	{
		int sw = round(width * 1/std::pow((double)sc, (double)(i)));
		int sh = round(height * 1/std::pow((double)sc, (double)(i)));
		cv::Mat scaled(sh*3,sw, CV_32FC1);
		resize(scaled,img);

		//for (int i=0;i<scaled.cols;i++)
		//{
		//	for (int j=470;j<scaled.rows;j++)
		//	{
		//		cout << scaled.at<float>(i,j)<<"  ";
		//	}
		//	cout << endl<<endl;
		//}

		// "first" 2x interval //
		features(&pyra.feat[i] , out ,(float*) scaled.data,sw,sh,sbin/2);
		pyra.scale[i]=2/pow(sc,i);
		pyra.dim_c[i]=out[0];pyra.dim_r[i]=out[1];

		// "second" 2x interval //
		features(&pyra.feat[i+intervel] , out ,(float*) scaled.data,sw,sh,sbin);
		pyra.scale[i+intervel]=1/pow(sc,i);
		pyra.dim_c[i+intervel]=out[0];pyra.dim_r[i+intervel]=out[1];


		cv::Mat* previous_scale = &scaled;

		//imshow("previous_scale",*previous_scale);
		//cvWaitKey(0);
		
		// remaining interals //
		for (int j=i+intervel;j<max_scale;j+=intervel)
		{
			int hsw = round(sw / 2);
			int hsh = round(sh / 2);
			cv::Mat half(hsh*3,hsw, CV_32FC1);
			cv::Mat* current_scale = &half;
			reduce(*current_scale, *previous_scale);
			//cv::Mat tt=*current_scale;
			//tt=tt.t();
			//imshow("current_scale",tt);
			//cvWaitKey(0);

			features(&pyra.feat[j+intervel] , out,(float*) current_scale->data,hsw,hsh,sbin);
			pyra.scale[j+intervel]=0.5*pyra.scale[j];
			pyra.dim_c[j+intervel]=out[0];pyra.dim_r[j+intervel]=out[1];
			*previous_scale=*current_scale;
			//std::swap(current_scale, previous_scale);
			sw = hsw; sh = hsh;
		}

		
	}

	int ori_dim[3],paddim[3],newdim[3];
	paddim[0]=padx+1; paddim[1]=pady+1; paddim[2]=0;
	for (int pl=0;pl<this->FP_level;pl++)
	{
		ori_dim[0]=pyra.dim_c[pl]; ori_dim[1]=pyra.dim_r[pl]; ori_dim[2]=32;
		pyra.feat[pl]=PadArray(pyra.feat[pl],ori_dim,&paddim[0], &newdim[0]);  //行列上下各增加4個pixel
		pyra.dim_c[pl]=newdim[0];pyra.dim_r[pl]=newdim[1];
	}


	for (int i=0;i<sizeof(pyra.scale)/sizeof(pyra.scale[0]);i++) //i<28
	{
		pyra.scale[i]=model->sbin_/pyra.scale[i];
	}
	pyra.intervel=intervel;
	pyra.imx=height;
	pyra.imy=width;
	pyra.padx=padx;
	pyra.pady=pady;
}

static float* PadArray( float* oldArray, int* dim, int* paddim, int* newdim)
{
	newdim[0] = dim[0] + 2*paddim[0]; //列
	newdim[1] = dim[1] + 2*paddim[1]; //行
	newdim[2] = dim[2] + 2*paddim[2]; //面
	float* newArray = new float[newdim[0]*newdim[1]*newdim[2]]; // 動態記憶體配置三層矩陣
	::memset(newArray, 0, sizeof(float)*newdim[0]*newdim[1]*newdim[2]);
	int ozoffset = dim[0]*dim[1];
	int nzoffset = newdim[0]*newdim[1];
	int nyoffset = paddim[2]*newdim[0]*newdim[1] + paddim[1]*newdim[0] + paddim[0];
	for(int z = 0; z < dim[2]; ++z)
	{
		float* ozPtr = oldArray + z*ozoffset;
		float* nzPtr = newArray + z*nzoffset + nyoffset;
		for(int y = 0; y < dim[1]; ++y)
		{
			::memcpy(nzPtr, ozPtr, sizeof(float)*dim[0]);
			nzPtr += newdim[0];
			ozPtr += dim[0];
		}
	}

	for (int i=0;i<newdim[2];i++){
		for(int j = 0; j != newdim[1]; j++) { //列
			for(int k = 0; k != newdim[0]; k++) //行
			{
				if (i==31)
				{
					if (j == 0 || j == 1 || j == 2 || j == 3 || j == newdim[1]-4 || j == newdim[1]-3 || j == newdim[1]-2 || j == newdim[1]-1 
						|| k == 0 || k == 1 || k == 2 || k == 3 || k == newdim[0]-4 || k == newdim[0]-3 || k == newdim[0]-2 || k == newdim[0]-1)
					{
						*(newArray+(i*newdim[1]*newdim[0])+ (j * newdim[0]) + k) = 1;
					}
				}
			}
		}
	}
	delete [] oldArray;oldArray=NULL;
	return newArray;
}

void resize1dtran_redue(float *src, int sheight, float *dst, int dheight, int width, int chan) 
{
	memset(dst,0,chan*width*dheight*sizeof(float));
	int y;
	float *s, *d;

	for (int c = 0; c < chan; c++) {
		for (int x = 0; x < width; x++) {
			s  = src + c*width*sheight + x*sheight;
			d  = dst + c*dheight*width + x;

			// First row
			*d = s[0]*.6875 + s[1]*.2500 + s[2]*.0625;      

			for (y = 1; y < dheight-2; y++) {	
				s += 2;
				//cout << s[-2]<<" "<< s[-1] << endl;
				//cout<<*s<<endl;
				//printf("%f\n",s); printf("%f\n",&s);
				d += width;
				*d = s[-2]*0.0625 + s[-1]*.25 + s[0]*.375 + s[1]*.25 + s[2]*.0625;
			}

			// Last two rows
			s += 2;
			d += width;
			if (dheight*2 <= sheight) {
				*d = s[-2]*0.0625 + s[-1]*.25 + s[0]*.375 + s[1]*.25 + s[2]*.0625;
			} else {
				*d = s[1]*.3125 + s[0]*.3750 + s[-1]*.2500 + s[-2]*.0625;
			}
			s += 2;
			d += width;
			*d = s[0]*.6875 + s[-1]*.2500 + s[-2]*.0625;
		}
	}
}

struct alphainfo
{
	int si, di;
	double alpha;
};

void alphacopy(float *src, float *dst, struct alphainfo *ofs, int n) 
{
	struct alphainfo *end = ofs + n;
	while (ofs != end) 
	{
		dst[ofs->di] += ofs->alpha * src[ofs->si];
		ofs++;
	}
}

void resize1dtran_resize(float *src, int sheight, float *dst, int dheight, int width, int chan) 
{
	float scale = (float)dheight/(float)sheight;
	float invscale = (float)sheight/(float)dheight;

	// we cache the interpolation values since they can be 
	// shared among different columns
	int len = (int)ceil(dheight*invscale) + 2*dheight;
	struct alphainfo *ofs = new struct alphainfo[len];
	int k = 0;
	for (int dy = 0; dy < dheight; dy++) 
	{
		float fsy1 = dy * invscale;
		float fsy2 = fsy1 + invscale;
		int sy1 = (int)std::ceil(fsy1);
		int sy2 = (int)std::floor(fsy2);       

		if (sy1 - fsy1 > 1e-3) 
		{
			//assert(k < len);
			//assert(sy-1 >= 0);
			ofs[k].di = dy*width;
			ofs[k].si = sy1-1;
			ofs[k++].alpha = (sy1 - fsy1) * scale;
		}

		for (int sy = sy1; sy < sy2; sy++) 
		{
			//assert(k < len);
			//assert(sy < sheight);
			ofs[k].di = dy*width;
			ofs[k].si = sy;
			ofs[k++].alpha = scale;
		}

		if (fsy2 - sy2 > 1e-3) 
		{
			//assert(k < len);
			//assert(sy2 < sheight);
			ofs[k].di = dy*width;
			ofs[k].si = sy2;
			ofs[k++].alpha = (fsy2 - sy2) * scale;
		}
	}

	// resize each column of each color channel
	memset(dst, 0, chan*width*dheight*sizeof(float));
	for (int c = 0; c < chan; c++) 
	{
		for (int x = 0; x < width; x++) {
			float *s = src + c*width*sheight + x*sheight;
			float *d = dst + c*width*dheight + x;
			alphacopy(s, d, ofs, k);
		}
	}
	delete [] ofs;ofs=NULL;
}

void reduce(cv::Mat& dest, cv::Mat& srcimg)
{
	float *src = (float*) srcimg.data;
	int sdims[3];
	sdims[0] = srcimg.cols;
	sdims[1] = srcimg.rows / 3;
	sdims[2] = 3;

	//double scale = mxGetScalar(mxscale);
	//if (scale > 1)
	//mexErrMsgTxt("Invalid scaling factor");   

	int ddims[3];
	ddims[0] = dest.cols;
	ddims[1] = dest.rows / 3;
	ddims[2] = 3;
	float *dst = (float*) dest.data;

	float *tmp = new float[ddims[0]*sdims[1]*sdims[2]];//(double *)mxCalloc(ddims[0]*sdims[1]*sdims[2], sizeof(double));
	resize1dtran_redue(src, sdims[0], tmp, ddims[0], sdims[1], sdims[2]);
	resize1dtran_redue(tmp, sdims[1], dst, ddims[1], ddims[0], sdims[2]);
	delete [] tmp;tmp=NULL;
}

void resize(cv::Mat& dest, cv::Mat& srcimg)
{
	float* src = (float*) srcimg.data;

	int sdims[3];
	sdims[0] = srcimg.cols;
	sdims[1] = srcimg.rows /3;
	sdims[2] = 3;

	//double scale = mxGetScalar(mxscale);
	//if (scale > 1)
	//mexErrMsgTxt("Invalid scaling factor");   

	int ddims[3];
	ddims[0] = dest.cols;
	ddims[1] = dest.rows /3;
	ddims[2] = 3;
	float *dst = (float*) dest.data;

	float *tmp = new float[ddims[0]*sdims[1]*sdims[2]];//(double *)mxCalloc(ddims[0]*sdims[1]*sdims[2], sizeof(double));
	resize1dtran_resize(src, sdims[0], tmp, ddims[0], sdims[1], sdims[2]);
	resize1dtran_resize(tmp, sdims[1], dst, ddims[1], ddims[0], sdims[2]);
	delete [] tmp;tmp=NULL;
}

void LandmarkDetector::backtrack(float box[],vector<int> X,vector<int> Y,std::vector<Data_s> parts,double pyrascale[],int padx,int pady, int* rootdim)
//void LandmarkDetector::backtrack(float box[],int X[],int Y[],std::vector<Data_s> parts,double pyrascale[],int padx,int pady, int* rootdim,int index)
{
	Data_s p;
	int numparts=parts.size();		
	//vector<int> x,y ;
	int x[500],y[500];
	int inds[500];
	int ptr[500][68][2];

	p=parts[0];
	for (int kk=0; kk<X.size(); kk++)
	{
		ptr[kk][0][0] = X[kk]; 
		ptr[kk][0][1] = Y[kk]; 
	}

	double scale = pyrascale[p.level-1];

	for (int kk=0; kk<X.size(); kk++)
	{
		box[kk*68*4+0*4+0] = ( X[kk]-1-padx )*scale + 1;
		box[kk*68*4+0*4+1] = ( Y[kk]-1-padx )*scale + 1;
		box[kk*68*4+0*4+2] = box[kk*68*4+0*4+0]+ p.sizx_*scale - 1;
		box[kk*68*4+0*4+3] = box[kk*68*4+0*4+1]+ p.sizx_*scale - 1;
	}

	for (int k=1 ; k<numparts ; k++)
	{
		//memset(inds,0,sizeof(int)*500);
		int num=0,temp=0;
		p = parts[k];
		int par = p.parent_-1;
		for (int kk=0; kk<X.size() ; kk++)
		{	
			x[kk]=ptr[kk][par][0];
			y[kk]=ptr[kk][par][1];
			/*x.push_back( ptr[kk][par][0] );
			y.push_back( ptr[kk][par][1] );*/
		}											 

		//for (int kk=0; kk <X.size() ; kk++)
		//{
		//	cout << x[kk] << "  " << y[kk] << endl;
		//}
		//cout<< endl;cout<< endl;


		for(int jj = 0; jj != rootdim[1]; ++jj) 
		{
			for(int kk = 0; kk != rootdim[0]; ++kk) 
			{
				if ( temp==X.size() )
				{
					break;
				}
				else
				{
					for (int temp1=0; temp1<X.size();temp1++) //檢查是否有重複座標
					{
						if ( jj==x[temp1]-1 )
						{
							if ( kk== y[temp1]-1 )
							{
								inds[temp1] = num+1;
								temp+=1;
							}
						}
					}

					num+=1;
				}
			}
		}

		//for (int kk=0; kk <X.size() ; kk++)
		//{
		//	cout << inds[kk] << endl;
		//}
		//cout<< endl;cout<< endl;



		temp=0,num=0;
		for(int b1 = 0; b1 != rootdim[1]; ++b1) 
		{
			for(int b2 = 0; b2 != rootdim[0]; ++b2) 
			{
				if ( num<X.size() )
				{
					temp+=1; 
					for (int temp1=0; temp1<X.size(); temp1++)
					{
						if ( temp == inds[temp1] )
						{
							ptr[temp1][k][0] = *( p.Ix + (b1*rootdim[0]) + b2 );
							ptr[temp1][k][1] = *( p.Iy + (b1*rootdim[0]) + b2 ); 
							num+=1;
						}
					}
				}
			}
		}

		//for (int kk=0; kk <X.size() ; kk++)
		//{
		//	cout << ptr[kk][k][0] << "  " << ptr[kk][k][1]   << endl;
		//}
		//cout<< endl;cout<< endl;

		scale = pyra.scale[p.level-1];

		for (int kk=0; kk<X.size(); kk++)
		{
			box[kk*68*4+k*4+0] = ( ptr[kk][k][0]-1 -padx )*scale + 1; 
			box[kk*68*4+k*4+1] = ( ptr[kk][k][1]-1 -pady )*scale + 1; 
			box[kk*68*4+k*4+2] = box[kk*68*4+k*4+0] + p.sizx_*scale - 1; 
			box[kk*68*4+k*4+3] = box[kk*68*4+k*4+1] + p.sizy_*scale - 1; 

			//cout << box[kk*68*4+k*4+0] << "  " << box[kk*68*4+k*4+1]<< "  " << box[kk*68*4+k*4+2] << "  " << box[kk*68*4+k*4+3]  << endl;

		}
		//vector<int>().swap(x);//x.clear();
		//vector<int>().swap(y);//y.clear();

	}
}
float sum(vector<float>time)
{
	float sum=0;
	for (int i=0; i<time.size(); i++)
	{
		sum+=time[i];
	}
	return sum;
}

void LandmarkDetector::clipboxes(int imy,int imx ,std::vector<Data_bs> boxes)
{
	//this->numparts;
	//int imy=img.rows;int imx=img.cols;
	float b[68][4];
	for (int i=0;i<boxes.size();i++)
	{
		memcpy(b,boxes[i].xy,sizeof(float)*68*4);
		for (int j=0;j<this->numparts;j++)
		{
			b[j][0]=max( b[j][0],1);
			b[j][1]=max( b[j][1],1);
			b[j][2]=min( b[j][2],imx);
			b[j][3]=min( b[j][3],imy);
		}
		memcpy(boxes[i].xy,b,sizeof(float)*68*4);
		//boxes[i].xy=b;
	}
}

void LandmarkDetector::nmsface(std::vector<Data_bs> boxes,float overlap,std::vector<Data_bs>  &top)
{
	int numparts = this->numparts;
	int N=boxes.size();
	std::vector<int> pick;
	std::vector<float> s1,s2;
	std::vector<int> I,I_tranform,suppress,j;
	std::vector<float>x1_temp,y1_temp,x2_temp,y2_temp;
	//float x1_temp[68],x2_temp[68],y1_temp[68],y2_temp[68];
	std::vector<float> x1,x2,y1,y2,area;
	//float x1[15000],x2[15000],y1[15000],y2[15000],area[15000];
	//float xx1[15000],xx2[15000],yy1[15000],yy2[15000],w[15000],h[15000],o1[15000],o2[15000];
	std::vector<float> xx1,xx2,yy1,yy2,w,h,inter,o1,o2;

	for (int nb=0; nb<N; nb++)
	{
		for (int nc=0; nc<numparts; nc++)
		{
			//x1_temp[nc]=boxes[nb].xy[nc][0];
			//y1_temp[nc]=boxes[nb].xy[nc][1];
			//x2_temp[nc]=boxes[nb].xy[nc][2];
			//y2_temp[nc]=boxes[nb].xy[nc][3];
			x1_temp.push_back( boxes[nb].xy[nc][0] );
			y1_temp.push_back( boxes[nb].xy[nc][1] );
			x2_temp.push_back( boxes[nb].xy[nc][2] );
			y2_temp.push_back( boxes[nb].xy[nc][3] );
		}

		//for (int i=0;i<numparts;i++)
		//{
		//	printf("%f ",*(x1_temp+i));
		//}

		//BubbleSort(x1_temp,numparts);BubbleSort(y1_temp,numparts);
		//BubbleSort(x2_temp,numparts);BubbleSort(y2_temp,numparts);
		std::sort(x1_temp.begin(),x1_temp.end());
		std::sort(y1_temp.begin(),y1_temp.end());
		std::sort(x2_temp.begin(),x2_temp.end());
		std::sort(y2_temp.begin(),y2_temp.end());

		//x1[nb]=x1_temp[0];
		//y1[nb]=y1_temp[0];
		//x2[nb]=x2_temp[numparts-1];
		//y2[nb]=y2_temp[numparts-1];
		x1.push_back( x1_temp[0] );
		y1.push_back( y1_temp[0] );
		x2.push_back( x2_temp[numparts-1] );
		y2.push_back( y2_temp[numparts-1] );

		area.push_back( (x2[nb]-x1[nb]+1)*(y2[nb]-y1[nb]+1) );
		//area[nb]=(x2[nb]-x1[nb]+1)*(y2[nb]-y1[nb]+1);

		//s1[nb]=boxes[nb].s;
		s1.push_back(boxes[nb].s);
		vector<float>().swap(x1_temp);vector<float>().swap(x2_temp);//x1_temp.clear();x2_temp.clear();y1_temp.clear();y2_temp.clear();
		vector<float>().swap(y1_temp);vector<float>().swap(y2_temp);
	}
	s2 = s1;

	std::sort(s2.begin(),s2.end());
	for (int i=0; i<s1.size(); i++) // find score index (   [vals, I] = sort(s);   )
	{
		vector<float>::iterator iter=std::find(s1.begin(),s1.end(),s2[i]);
		I.push_back( std::distance(s1.begin(), iter) );
	}

	while (I.empty() == false)
	{
		int last = I.size();
		int i = I[last-1];
		pick.push_back(i);
		suppress.push_back(last-1);
		j.insert(j.begin(),I.begin(),I.end()-1); //j = I(1:last-1);
		for (int nd=0;nd<j.size();nd++)
		{
			//xx1[nd]=( max( x1[i] , x1[ j[nd] ]) );
			//yy1[nd]=( max( y1[i] , y1[ j[nd] ]) );
			//xx2[nd]=( min( x2[i] , x2[ j[nd] ]) );
			//yy2[nd]=( min( y2[i] , y2[ j[nd] ]) );
			xx1.push_back( max( x1[i] , x1[ j[nd] ]) );
			yy1.push_back( max( y1[i] , y1[ j[nd] ]) );
			xx2.push_back( min( x2[i] , x2[ j[nd] ]) );
			yy2.push_back( min( y2[i] , y2[ j[nd] ]) );

			w.push_back( xx2[nd]-xx1[nd]+1 );
			//w[nd]=( xx2[nd]-xx1[nd]+1 );
			if ( w[nd]<0 )
			{
				w[nd] = 0;
			}
			h.push_back( yy2[nd]-yy1[nd]+1 );
			//h[nd]=( yy2[nd]-yy1[nd]+1 );
			if ( h[nd]<0 )
			{
				h[nd] = 0;
			}
			inter.push_back(w[nd]*h[nd]);

			//o1[nd]=( w[nd]*h[nd] / area[ j[nd] ] );
			//o2[nd]=( w[nd]*h[nd] / area[ i ] );

			o1.push_back( inter[nd] / area[ j[nd] ] );
			o2.push_back( inter[nd] / area[ i ] );

			if ( o1[nd]>overlap || o2[nd]>overlap)
			{
				suppress.push_back(nd);
			} 
		}

		for (int i=0;i<suppress.size();i++)
		{
			I[suppress[i]] = 0;
		}

		auto max = max_element(std::begin(I), std::end(I));

		if (*max == 0)
		{
			I.clear();
		}
		else{
			for (int i=0;i<I.size();i++)
			{
				if ( I[i]!=0 )
				{
					I_tranform.push_back(I[i]);
				}
			}
			I = I_tranform;
		}
		vector<int>().swap(suppress); vector<int>().swap(j); vector<int>().swap(I_tranform);
		vector<float>().swap(xx1); vector<float>().swap(xx2); vector<float>().swap(yy1);vector<float>().swap(yy2);
		vector<float>().swap(w);vector<float>().swap(h);
		vector<float>().swap(inter);vector<float>().swap(o1);vector<float>().swap(o2);
		vector<float>().swap(s1);vector<float>().swap(s2);
		//I.clear(),suppress.clear(),j.clear();
		//xx1.clear(),xx2.clear(),yy1.clear(),yy2.clear(),w.clear(),h.clear();
		//inter.clear(),o1.clear(),o2.clear();
	}
	vector<float>().swap(x1);vector<float>().swap(y1);vector<float>().swap(x2);vector<float>().swap(y2);vector<float>().swap(area);
	vector<int>().swap(I); 
	while ( 1 )
	{
		if (pick.empty() == true)
		{
			break;
		}
		else
			for (int i=0 ;i<pick.size();i++)
			{
				top.push_back( boxes[pick[i]] );
			}
		break;
	}
		
	
	vector<Data_bs>().swap(boxes);
	vector<int>().swap(pick);
	//pick.clear();
}

void MixScore(float* root, float* part, int* partdims)
{
	float* rPtr = root;
	float* pPtr = part;
	for(int h = 0; h < partdims[1]*partdims[0]; ++h, rPtr++ , pPtr++)
	{
		//cout << "before add - To_root" << *rPtr<<*pPtr<<endl;
		*rPtr = *rPtr + *pPtr;
		//cout << "after add - To_root" << *rPtr<<endl;
	}
}

vector<cv::Point2f> LKTrincingProcessLandMark(cv::Mat prevFrame, cv::Mat Frame,vector<cv::Point2f> LpsPre)
{
	bool Lps_flag=true;vector<uchar> status;
	cv::Size winSize(30,30);
	cv::TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 30, 0.03);
	vector<cv::Point2f> Lps;
	vector<float> err;

	cv::calcOpticalFlowPyrLK(prevFrame, Frame, LpsPre, Lps, status, err, winSize,3, termcrit, 0, 0.001);

	size_t i, k;

	for( i = k = 0; i < Lps.size(); i++ )
	{
		if( !status[i] || err[i] > 10)
			Lps_flag=true;
	}
	//cout<<endl;

	return Lps;
}

vector<cv::Point2f> GPUbrox(cv::Mat prevGray, cv::Mat gray,std::vector<cv::Point2f>LpsPre ,int point)
{
	vector<cv::Point2f> Lps;
	Lps=LpsPre;

	cv::gpu::GpuMat d_frame0f;
	cv::gpu::GpuMat d_frame1f;

	cv::gpu::GpuMat d_frame0(prevGray);
	cv::gpu::GpuMat d_frame1(gray);

	cv::gpu::GpuMat d_flowx(prevGray.size(), CV_32FC1);
	cv::gpu::GpuMat d_flowy(prevGray.size(), CV_32FC1);

	cv::Mat FlowX,FlowY;

	d_frame0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
	d_frame1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);

	cv::gpu::BroxOpticalFlow brox(0.197f, 50.0f, 0.8f, 10, 77, 10);
	brox(d_frame0f,d_frame1f,d_flowx,d_flowy);

	d_flowx.download(FlowX);
	d_flowy.download(FlowY);

	for (int i=0;i<point;i++)
	{
		Lps[i].x=LpsPre[i].x+FlowX.at<float>(LpsPre[i].y,LpsPre[i].x);
		Lps[i].y=LpsPre[i].y+FlowY.at<float>(LpsPre[i].y,LpsPre[i].x);
	}

	return Lps;

}

vector<cv::Point2f> GPUTVL1 (cv::Mat prevGray, cv::Mat gray,std::vector<cv::Point2f>LpsPre ,int point)
{
	vector<cv::Point2f> Lps;
	Lps=LpsPre;

	cv::gpu::GpuMat d_frame0f;
	cv::gpu::GpuMat d_frame1f;

	cv::gpu::GpuMat d_frame0(prevGray);
	cv::gpu::GpuMat d_frame1(gray);

	cv::gpu::GpuMat d_flowx(prevGray.size(), CV_32FC1);
	cv::gpu::GpuMat d_flowy(prevGray.size(), CV_32FC1);

	cv::Mat FlowX,FlowY;

	d_frame0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
	d_frame1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);

	cv::gpu::OpticalFlowDual_TVL1_GPU tvl1;
	tvl1(d_frame0, d_frame1, d_flowx, d_flowy);

	d_flowx.download(FlowX);
	d_flowy.download(FlowY);

	for (int i=0;i<point;i++)
	{
		Lps[i].x=LpsPre[i].x+FlowX.at<float>(LpsPre[i].y,LpsPre[i].x);
		Lps[i].y=LpsPre[i].y+FlowY.at<float>(LpsPre[i].y,LpsPre[i].x);
	}

	return Lps;

}

std::vector<cv::Rect> detectAndDisplay( cv::Mat frame,cv::CascadeClassifier& face_cascade_frontal, cv::CascadeClassifier& face_cascade_profile )
{
   std::vector<cv::Rect> faces,faces1,faces2;

   cv::Mat frame_gray;

   cvtColor( frame, frame_gray, cv::COLOR_BGR2GRAY );
   equalizeHist( frame_gray, frame_gray );

   //-- Detect faces
   
   face_cascade_frontal.detectMultiScale( frame_gray, faces, 1.1, 3, 0|CV_HAAR_SCALE_IMAGE, cv::Size(80, 80) );
  

  
   if(faces.empty())
   {
       face_cascade_profile.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(100,100) );
	   
   }
 
   
   
   
   for( size_t i = 0; i < faces.size(); i++ )
    {
      cv::Mat faceROI = frame_gray( faces[i] );
      std::vector<cv::Rect> eyes;

      //-- In each face, detect eyes
      //eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );
      //if( eyes.size() == 2)
      //{ 
         //-- Draw the face

	  /*  Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
	  ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar( 255, 0, 0 ), 2, 8, 0 );*/


		cv::Point center2,center3;
		 center2.x=cvRound(faces[i].x);
		 center3.x=cvRound(faces[i].x+faces[i].width);
		 center2.y=cvRound(faces[i].y);
		 center3.y=cvRound(faces[i].y+faces[i].height);
		 //rectangle(frame,center2,center3,cv::Scalar( 100, 0, 255),3,8,0);


         //for( size_t j = 0; j < eyes.size(); j++ )
         // { //-- Draw the eyes
         //   Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
         //   int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
         //   circle( frame, eye_center, radius, Scalar( 255, 0, 255 ), 3, 8, 0 );
         // }
       //}

    }
   //-- Show what you got
   //imshow( "L1_Result", frame );
   //cvWaitKey(0);
   return faces;
}

cv::Rect Face_boundary(std::vector<cv::Point2f> Lps,int point)
{
	cv::Rect FB;

	int x_min=500,x_max=0,y_min=500,y_max=0;
	for (int n=point-1; n>=0; n--)
	{
		x_min = x_min<Lps[n].x ? x_min:Lps[n].x;
		x_max = x_max>Lps[n].x ? x_max:Lps[n].x;
		y_min = y_min<Lps[n].y ? y_min:Lps[n].y;
		y_max = y_max>Lps[n].y ? y_max:Lps[n].y;
	}
	FB.x = x_min; FB.y = y_min;
	FB.width = x_max-x_min; FB.height = y_max-y_min;

	return FB;
}