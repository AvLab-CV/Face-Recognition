#include "stdlib.h"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

//// Load file name //
void Load_insideFile_name(string input_path, vector<string>* output_path);//讀file內所有檔名//
//// Load ply x y z nx ny nz r g b //
void Read_PLY(string name_model, string name_index,string name_mask,Mat *Model_x,Mat *Model_y,Mat *Model_z,Mat *Model_r,Mat *Model_g,Mat *Model_b,Mat *Model_nx,Mat *Model_ny,Mat *Model_nz);