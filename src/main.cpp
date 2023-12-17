#include "Panorama.h"
#include "cropper.h"

#include <string>
#include <iostream>

using namespace std;

int surf_threshold = 500;

int main(int argc, char *argv[])
{
	if(argc != 3){
		cout << "args error" << endl;
		return -1;
	}

	int imgs_num = atoi(argv[1]);
	surf_threshold = atoi(argv[2]);
	cout << "imgs_num:" << imgs_num << ", " << "surf_threshold:" << surf_threshold << endl;

	std::vector<cv::Mat> vecImgs;

	cv::Mat panoImg = panorama(vecImgs, imgs_num);
	if(panoImg.empty()){
		cout << "Panorama concatenation failure o(╥﹏╥)o" << endl;
		return -1;
	}

	cout << "Cropping image..." << endl;
	// 去掉图片边缘黑框部分
	Cropper *cropper = new Cropper();
	panoImg = cropper->crop(panoImg);

	cout << "Writting image..." << endl;
	cv::imwrite("./image/result.jpg", panoImg);

	cout << "Finish all" << endl;

	return 0;
}