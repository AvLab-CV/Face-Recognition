// STL Header
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>

// OpenCV Header
#include <opencv2/opencv.hpp>

// OpenNI Header
#include <OpenNI.h>

// namespace
using namespace std;
using namespace cv;
using namespace openni;

int main()
{
	float width_long=320;
	float height_long=240;
	
	// Initial OpenNI
	if( OpenNI::initialize() != STATUS_OK )
	{
		cerr << "OpenNI Initial Error: " << OpenNI::getExtendedError() << endl;
		return 0;
	}

	// Open Device
	Device	devDevice;
	if( devDevice.open( ANY_DEVICE ) != STATUS_OK )
	{
		cerr << "Can't Open Device: " << OpenNI::getExtendedError() << endl;
		return 0;
	}

	// Create depth stream
	VideoStream vsDepth;
	if( devDevice.hasSensor( SENSOR_DEPTH ) )
	{
		if( vsDepth.create( devDevice, SENSOR_DEPTH ) == STATUS_OK )
		{
			// set video mode
			VideoMode mMode;
			mMode.setResolution( width_long, height_long );
			mMode.setFps( 30 );
			mMode.setPixelFormat( PIXEL_FORMAT_DEPTH_1_MM );

			if( vsDepth.setVideoMode( mMode) != STATUS_OK )
				cout << "Can't apply VideoMode: " << OpenNI::getExtendedError() << endl;
		}
		else
		{
			cerr << "Can't create depth stream on device: " << OpenNI::getExtendedError() << endl;
			return 0;
		}
	}
	else
	{
		cerr << "ERROR: This device does not have depth sensor" << endl;
	}

	// Create color stream
	VideoStream vsColor;
	if( devDevice.hasSensor( SENSOR_COLOR ) )
	{
		if( vsColor.create( devDevice, SENSOR_COLOR ) == STATUS_OK )
		{
			// set video mode
			VideoMode mMode;
			mMode.setResolution( width_long, height_long );
			mMode.setFps( 15 );
			mMode.setPixelFormat( PIXEL_FORMAT_RGB888 );

			if( vsColor.setVideoMode( mMode) != STATUS_OK )
				cout << "Can't apply VideoMode: " << OpenNI::getExtendedError() << endl;

			// image registration
			if( devDevice.isImageRegistrationModeSupported( IMAGE_REGISTRATION_DEPTH_TO_COLOR ) )
				devDevice.setImageRegistrationMode( IMAGE_REGISTRATION_DEPTH_TO_COLOR );
			else
				cerr << "This device doesn't support image registration" << endl;
		}
		else
		{
			cerr <<  "Can't create color stream on device: " << OpenNI::getExtendedError() << endl;
			return 0;
		}
	}
	else
	{
		cerr << "This device doesn't support color sensor" << endl;
	}

	// create OpenCV Window and start
	if( vsDepth.isValid() )
	{
		cv::namedWindow( "Depth Image",  CV_WINDOW_AUTOSIZE );
		vsDepth.start();
	}

	if( vsColor.isValid() )
	{
		cv::namedWindow( "Color Image",  CV_WINDOW_AUTOSIZE );
		vsColor.start();
	}

	cv::Mat rgbpoint;
	int inputKey;
	bool s_flag=false;
	int count=0;
	while( true )
	{
		// get color frame
		VideoFrameRef vfColorFrame;
		vsColor.readFrame( &vfColorFrame );

		// convert data to OpenCV format
		const cv::Mat mImageRGB( vfColorFrame.getHeight(), vfColorFrame.getWidth(), CV_8UC3, const_cast<void*>( vfColorFrame.getData() ) );

		// convert form RGB to BGR
		cv::Mat mImageBGR;
		cv::cvtColor( mImageRGB, mImageBGR, CV_RGB2BGR );
		cv::imshow( "Color Image", mImageBGR );
		mImageBGR.copyTo(rgbpoint) ;
		vfColorFrame.release();

		VideoFrameRef vfDepthFrame;
		const DepthPixel* pDepthArray = NULL;
		vsDepth.readFrame( &vfDepthFrame );
		pDepthArray = (const DepthPixel*)vfDepthFrame.getData();
		const cv::Mat mImageDepth( vfDepthFrame.getHeight(), vfDepthFrame.getWidth(), CV_16UC1, const_cast<void*>( vfDepthFrame.getData() ) );
		cv::Mat_<float> Depth_show;
		//cv::Mat Depth_show1;
		// get max depth value
		int iMaxDepth = vsDepth.getMaxPixelValue();
		//mImageDepth.convertTo(Depth_show1,CV_8U,255.0/iMaxDepth);
		cv::normalize(mImageDepth,Depth_show,0,1,cv::NORM_MINMAX,-1);
		cv::imshow( "Depth Image", Depth_show );

		if (s_flag)
		{
			s_flag = !s_flag;
			static int count_num=0;
			//cv::Mat_<float> ply_x(480,640),ply_y(480,640),ply_z(480,640);
			//cv::Mat_<int>ply_r(480,640),ply_g(480,640),ply_b(480,640);
			cv::Mat_<float> ply_x(height_long,width_long),ply_y(height_long,width_long),ply_z(height_long,width_long);
			cv::Mat_<int>ply_r(height_long,width_long),ply_g(height_long,width_long),ply_b(height_long,width_long);
			uchar pr, pg, pb;
			FILE *fply_FRGC,*fply_FRGC_show,*fply_FRGC_crop,*fply_FRGC_show_crop;
			cv::flip(ply_x,ply_x,1);
			cv::flip(ply_y,ply_y,1);
			cv::flip(ply_z,ply_z,1);
			cv::flip(ply_r,ply_r,1);
			cv::flip(ply_g,ply_g,1);
			cv::flip(ply_b,ply_b,1);


			stringstream int2str;
			string num;
			int2str << count;
			int2str >> num;

			char depth_FRGC[50],depth_FRGC_show[50],depth_FRGC_crop[50],depth_FRGC_crop_show[50];
			//sprintf(depth_FRGC,"kinect_data\\current_grab.ply");
			//sprintf(depth_FRGC_show,"kinect_data\\ID_%s\\current_grab_show.ply",name);
			//sprintf(depth_FRGC_show,"test_1.ply");
			string depth_FRGC_show_s="Data/test_"+num+".ply";
			//sprintf(depth_FRGC_crop,"kinect_data\\ID_%s\\current_grab_show_crop.txt",name);
			//sprintf(depth_FRGC_crop_show,"kinect_data\\ID_%s\\current_grab_show_crop.ply",name);
			//fply_FRGC=fopen(depth_FRGC,"wt");
			fply_FRGC_show=fopen(depth_FRGC_show_s.c_str(),"wt");
			//fply_FRGC_crop=fopen(depth_FRGC_crop,"wt");
			//fply_FRGC_show_crop=fopen(depth_FRGC_crop_show,"wt");
			//int tol_num=0;
			//for( int y = 0; y < vfDepthFrame.getHeight(); ++ y )
			//{	
			//	for( int x = 0; x < vfDepthFrame.getWidth(); ++ x )
			//	{
			//		int idx = x + y * vfDepthFrame.getWidth();
			//		const DepthPixel&  rDepth = pDepthArray[idx];
			//		float fX, fY, fZ;
			//		CoordinateConverter::convertDepthToWorld( vsDepth,x, y, rDepth,&fX, &fY, &fZ );
			//		if (fZ>0 && fZ<1500)
			//		{
			//			tol_num++;
			//		}

			//	}
			//}

			fprintf(fply_FRGC_show,"ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n",76800);
			//fprintf(fply_FRGC_show_crop,"ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n",tol_num);
			//fprintf(fply_FRGC_crop,"%d\n",tol_num);
			//tol_num=0;
			for( int y = 0; y < vfDepthFrame.getHeight(); ++ y )
			{	
				uchar* rgb_ptr = rgbpoint.ptr<uchar>(y);
				for( int x = 0; x < vfDepthFrame.getWidth(); ++ x )
				{
					ply_b(y,x) = rgb_ptr[3*x];
					ply_g(y,x) = rgb_ptr[3*x+1];
					ply_r(y,x) = rgb_ptr[3*x+2];
					int idx = x + y * vfDepthFrame.getWidth();
					const DepthPixel&  rDepth = pDepthArray[idx];
					float fX, fY, fZ;
					CoordinateConverter::convertDepthToWorld( vsDepth,x, y, rDepth,&fX, &fY, &fZ );
					ply_x(y,x) = fX;
					ply_y(y,x) = fY;
					ply_z(y,x) = fZ;
					//fprintf(fply_FRGC,"%f %f %f %d %d %d\n",ply_x(y,x),ply_y(y,x),ply_z(y,x),ply_r(y,x),ply_g(y,x),ply_b(y,x));
					fprintf(fply_FRGC_show,"%f %f %f %d %d %d\n",ply_x(y,x),ply_y(y,x),ply_z(y,x),ply_r(y,x),ply_g(y,x),ply_b(y,x));

					//if (fZ>0 && fZ<1500)
					//{
					//	fprintf(fply_FRGC_show_crop,"%f %f %f %d %d %d\n",ply_x(y,x),ply_y(y,x),ply_z(y,x),ply_r(y,x),ply_g(y,x),ply_b(y,x));
					//	fprintf(fply_FRGC_crop,"%f %f %f %d %d %d 255\n",ply_x(y,x),ply_y(y,x),ply_z(y,x),ply_r(y,x),ply_g(y,x),ply_b(y,x));
					//}
				}
			}
			//fclose(fply_FRGC_crop);
			//fclose(fply_FRGC);
			fclose(fply_FRGC_show);
			//fclose(fply_FRGC_show_crop);
			//imwrite("kinect_data\\current_grab.jpg",rgbpoint);
			string depth_FRGC_show_s_rgb="Data/test_"+num+".jpg";
			flip(rgbpoint,rgbpoint,1);
			imwrite(depth_FRGC_show_s_rgb,rgbpoint);
			//char saveimg[50];
			//sprintf(saveimg,"kinect_data\\ID_%s\\current_grab.jpg",name);
			//imwrite(saveimg,rgbpoint);
			//imshow("save img",rgbpoint);cvWaitKey(100);
			printf("save Kinect data \n");
			//cvDestroyWindow("save img");
			count_num++;
			count++;
		}

		inputKey=waitKey(5);
		if(inputKey == VK_ESCAPE) {vsColor.destroy();vsDepth.destroy();break;} 
		else if(char(inputKey) == 's') {s_flag = !s_flag;} 
	}
	devDevice.close();
	OpenNI::shutdown();
	return 0;
}