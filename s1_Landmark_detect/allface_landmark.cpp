#include "dlib/all/source.cpp"
#include <dlib/opencv.h>
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/gui_widgets.h>

using namespace dlib;
using namespace std;

int main()
{
    try
    {
        cv::VideoCapture cap(0);
        //image_window win;

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector(); //內建人臉偵測modelㄝ, 目測 +-45
        shape_predictor pose_model,pose_model2;
        deserialize("../../using_data/dlib-model/shape_predictor_68_face_landmarks.dat") >> pose_model;
		deserialize("../../using_data/dlib-model/sp.dat") >> pose_model2;
		typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;
		object_detector<image_scanner_type> detector1;
		deserialize("../../using_data/dlib-model/face_detector.svm") >> detector1; //側臉左右臉45度(不含)以上

        // Grab and process frames until the main window is closed by the user.
        while(1)
        {
            // Grab a frame
            cv::Mat temp,temp2;
            cap >> temp;
			//cv::resize(temp,temp,cv::Size(temp.cols*2,temp.rows*2));
			temp.copyTo(temp2);
			
            cv_image<bgr_pixel> cimg(temp);
			int number_size=0;
			cv::Point2f points_plot; 
			std::vector<full_object_detection> shapes; // 存點位置
            // Detect faces 
            std::vector<rectangle> faces = detector(cimg);
			if (faces.size()!=0)
			{
            // Find the pose of each face.
				
				for (unsigned long i = 0; i < faces.size(); ++i)
				{
					full_object_detection shape = pose_model(cimg, faces[i]);
				    shapes.push_back(pose_model(cimg, faces[i]));
					number_size=shape.num_parts();
				}
				char bufCount[100];
				cv::vector<cv::Point2d> Landmarks;
				for (int i=0;i<number_size;i++)
				{
					points_plot.x=shapes[0].part(i).x();
					points_plot.y=shapes[0].part(i).y();
					Landmarks.push_back(points_plot);
					circle(temp2,points_plot,2,cv::Scalar(0,255,255),(-1));
					sprintf(bufCount,"%d",(i+1));
					putText( temp2, bufCount, Landmarks[Landmarks.size()-1], 0, 0.25, cv::Scalar(255,0,0), 1, CV_AA);
				}
			}
			/*else
			{
				faces = detector1(cimg);
				for (unsigned long i = 0; i < faces.size(); ++i)
				{

					full_object_detection shape = pose_model2(cimg, faces[i]);
					shapes.push_back(pose_model2(cimg, faces[i]));
					number_size=shape.num_parts();
				}
				char bufCount[100];
				cv::vector<cv::Point2d> Landmarks;
				for (int i=0;i<number_size;i++)
				{
					points_plot.x=shapes[0].part(i).x();
					points_plot.y=shapes[0].part(i).y();
					Landmarks.push_back(points_plot);
					circle(temp2,points_plot,2,cv::Scalar(0,255,255),(-1));
					sprintf(bufCount,"%d",(i+1));
					putText( temp2, bufCount, Landmarks[Landmarks.size()-1], 0, 0.25, cv::Scalar(255,0,0), 1, CV_AA);
				}
			}*/
				cv::imshow("temp2",temp2);cv::waitKey(1);//cv::imwrite("land.jpg",temp2);
			//cv::imshow("temp2",temp2);cv::waitKey(1);
            // Display it all on the screen
            //win.clear_overlay();
           // win.set_image(cimg);
           // win.add_overlay(render_face_detections(shapes));
        }
    }
    catch(serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
		//system("pause");
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
		//system("pause");
    }
}

