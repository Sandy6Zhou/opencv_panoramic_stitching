#include "Panorama.h"
#include "cropper.h"

#include <string>
#include <iostream>
#include <cstdlib>

using namespace std;

int surf_threshold = 500;

int main(int argc, char *argv[])
{
	if(argc != 4){
		cout << "args error" << endl;
		return -1;
	}
	int64_t start = cv::getTickCount();// 记录开始时间

	int imgs_num = atoi(argv[1]);
	surf_threshold = atoi(argv[2]);
	int screenflip = atoi(argv[3]);
	cout << "imgs_num:" << imgs_num << ", " << "surf_threshold:" << surf_threshold << endl;

	std::vector<cv::Mat> vecImgs;

	cv::Mat panoImg = panorama(vecImgs, imgs_num, screenflip);
	if(panoImg.empty()){
		cout << "Panorama concatenation failure o(╥﹏╥)o" << endl;
		return -1;
	}

	cout << "Cropping image..." << endl;
	cv::Mat panoImg_result;

	// 去掉图片边缘黑框部分,方式一
	// Cropper *cropper = new Cropper();
	// panoImg_result = cropper->crop(panoImg);

	// 去掉图片边缘黑框部分,方式二
	Cropper *cropper = new Cropper();
	cropper->RemoveBlackBorder(panoImg, panoImg_result);

	cout << "Writting image..." << endl;
	cv::imwrite("./image/result.jpg", panoImg_result);

	int64_t end = cv::getTickCount();// 记录结束时间
    double elapsedSeconds = (end - start) / cv::getTickFrequency();// 计算执行时间（单位：秒）

	std::system("printf \"%s\" \"$(date)\": >> ./Panoramic_splicing_time.txt");
	char str[128]={0};
	sprintf(str, "echo opencv splicing time:%.2fs >> ./Panoramic_splicing_time.txt", elapsedSeconds);
	std::system(str);

	cout << "Finish all" << endl;

	return 0;
}