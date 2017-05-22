
#include "opencv2/opencv.hpp"
#include "opencv2/legacy/compat.hpp"

CvMat Normalize8(CvMat *Mat1,CvMat *Mat4);
CvMat robust_postprocessor(CvMat *Mat6,CvMat *Mat13);
CvMat TT(CvMat *ResizeImage, CvSize ImageSize, CvMat *Result,int type);
//double cvMean(CvMat *Mat_m);
