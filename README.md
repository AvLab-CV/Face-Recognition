# Face-Recognition
Robust Cross-Pose Face Recognition using Landmark Oriented Depth Warping 

Pre-requirement: Visual Studio 2010

And, following the step for completing the environment.

    step1: Download the <a href="https://drive.google.com/file/d/0BwJ2me84dFHIYURLRlZpZXcwMlE/view?usp=sharing">External_lib</a>, unzip into the External_lib/.

step2: Download the <a href="https://drive.google.com/file/d/0BwJ2me84dFHIT2syY0FUWWo3QmM/view?usp=sharing">workingdir64-r</a>, unzip into the workingdir64-r/.

Or you can build the library by youself: Opencv2.4.9, OpenGL, OpenNI2.

step3: Move the folder named "using_data" outside the solution dir, like "move using_data ../using_data".

step4: Download the model file <a href="https://drive.google.com/file/d/0BwJ2me84dFHISHFaaGhqeml0c1U/view?usp=sharing">using_data</a>, unzip into the using_data/.

=====Operation=====

1. 3D Face Reconstruction

    The main code: a2_3_MPIE_Reg_demo.cpp, you will see the generated 3D model by using using_data/MPIE_classification/F00_05_1/06/all/003_01_01_051_06.png, result to the folder using_data/Reg_model/06/all/003/

2. Face Alignment

    The main code: a3_3_pose_align_demo.cpp, when compile and run, the command win will stop and wait for type in 2 then "enter". Here 2 means the index of sample image at the folder using_data/MPIE_classification/L15_14_0/06/all/001_01_01_140_06.png. The alignment results are saved at using_data/Test_data/

3. SRC Recognition

    3a. The main code: a4_1_SRC_create_gallery_data_demo.cpp, extract the gabor feature of gallery image.

    3b. The main code: a4_2_SRC_create_test_data_demo.cpp, extract the gabor feature of test image.

    3c. The main code: a4_3_SRC_test_demo.cpp use above feature to output the result at using_data/Test_data_xml.
