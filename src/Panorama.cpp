#include "Panorama.h"
#include <iostream>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

extern int surf_threshold;
static int dynamic_threshold;

#define Img_Width 	200
#define Img_Height 	180
#define Cols_Cut	60
/*******************************************************
 * Description: 将多张图片进行特征提取，变换，RANSAC 等处理
 * 				以将多张图片组成一张全景图
 *
 * @para sourceImages 进行拼接的所有图片
 * @para panoImg 拼接结果图片
 *******************************************************/
cv::Mat panorama(std::vector<cv::Mat> & sourceImages, unsigned nImgs, unsigned screen_flip)
{
	cv::Mat panoImg;

	if (nImgs < 2) {
		if (nImgs == 1)
			panoImg = sourceImages[0].clone();
		return panoImg;
	}

	std::vector<cv::Mat> Hs(nImgs);
	cv::Mat eyeMatrix = cv::Mat::eye(3, 3, CV_32FC1);
	if(screen_flip){
		eyeMatrix.at<float>(0,2) = (nImgs-1) * Img_Width;
	}
	
	Hs.front() = eyeMatrix;
	cv::Mat tempImg = cv::Mat(Img_Height, Img_Width*nImgs, CV_32FC4);

	cout << "Loading image...:1" << endl;
	
	cv::Mat front_img, back_img;
	cv::Mat current_img = cv::imread("./image/ptz_panorama_1.jpg").colRange(Cols_Cut, Cols_Cut+Img_Width).clone();
	if(current_img.empty()){
		cout << "image is no exist" << endl;
		return panoImg;
	}

	blendImage(current_img, Hs[0], tempImg);
	cv::imwrite("./image/test_" + std::to_string(0+1) + ".jpg", tempImg);
	cv::Mat last_img = current_img;
	front_img = current_img;
	
	for (unsigned i = 1; i < nImgs; i++)
	{
		cout << "Loading image...:" << i+1 << endl;
		current_img = cv::imread("./image/ptz_panorama_" + std::to_string(i+1) + ".jpg").colRange(Cols_Cut, Cols_Cut+Img_Width).clone();
		if(current_img.empty()){
			cout << "image is no exist" << endl;
			return panoImg;
		}
		dynamic_threshold = surf_threshold;
loop:
		alignImage(last_img, current_img, Hs[i]);
		cout << "Hs-"<< i << ":" << Hs[i] << endl;
		float diff_x = Hs[i].at<float>(0, 2);
		float diff_y = Hs[i].at<float>(1, 2);

		bool is_modifythreshold = false;
		if(screen_flip){
			if(!(diff_x < -50 && diff_x > (-Img_Width)) || !(diff_y > -10 && diff_y < 10)){
				is_modifythreshold = true;
			}
		}else{
			if(!(diff_x > 50 && diff_x < Img_Width) || !(diff_y > -10 && diff_y < 10)){
				is_modifythreshold = true;
			}
		}

		if(is_modifythreshold){
			dynamic_threshold -= 25;
			if(diff_x == 0 && diff_y == 0){
				dynamic_threshold = 0;
				cout << "Fight recklessly image:" << i+1 << endl;
			}else{
				cout << "matrix error, Adjustment parameter dynamic_threshold:" << dynamic_threshold << endl;
			}

			if(dynamic_threshold>=25){
				goto loop;
			}else{
				if(i==1){return panoImg;}
				Hs[i].at<float>(0,2) = Hs[i-1].at<float>(0,2) / (i-1);
				Hs[i].at<float>(1,2) = Hs[i-1].at<float>(1,2) / (i-1);
				cout << "Hs-"<< i << ":" << Hs[i] << endl;
				Hs[i] = Hs[i - 1] * Hs[i];
				blendImage(current_img, Hs[i], tempImg);
				last_img = current_img;
			}
		}else{
			Hs[i] = Hs[i - 1] * Hs[i];
			blendImage(current_img, Hs[i], tempImg);
			last_img = current_img;
		}
		cv::imwrite("./image/test_" + std::to_string(i+1) + ".jpg", tempImg);
	}

	back_img = current_img;

	float cols = Hs[nImgs-1].at<float>(0, 2)+Img_Width;
	cv::Mat temp_Img = tempImg.colRange(0, cols).clone() ;

	normalizeBlend(tempImg, panoImg);
	affineDeform(panoImg, front_img, Hs.front(), back_img, Hs.back());

	cout << "Adjusting image size..." << endl;
	// 处理图片大小
	adjustSize(panoImg);

	return panoImg;
}

/*******************************************************
 * Description: 使用 SIFT 算子进行特征提取
 *
 * @para image 进行特征提取的图片
 * @para keypoints 提取的特征点
 * @para descriptor 特征向量描述
 *******************************************************/
void detectFeature(const cv::Mat &image,
			std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptor)
{
	Ptr<SURF>orb = SURF::create(dynamic_threshold);

	// Initialize temporrary variable to be pushed insisde the input keypoints and descriptors vector
	Mat tmp_Mat;
	vector<KeyPoint> tmp_kp;

	// Compute keypoints and descriptor of i-th image
	orb->detectAndCompute(image, Mat(), keypoints, descriptor);
}

/*******************************************************
 * Description: 进行量图像的特征匹配
 *
 * @para image1, image2 进行特征提取的图片
 * @para keyPoint1, keyPoint2 图片的特征点
 * @para matches 匹配结果
 *******************************************************/
void matchFeatures(const cv::Mat & image1, const cv::Mat & image2,
	std::vector<cv::KeyPoint> & keyPoint1,
	std::vector<cv::KeyPoint> & keyPoint2,
	std::vector<cv::DMatch>   & matches)
{
	cv::Mat descriptors1, descriptors2;

	// 检测特征点，然后计算特征向量
	detectFeature(image1, keyPoint1, descriptors1);
	detectFeature(image2, keyPoint2, descriptors2);
	if(keyPoint2.size() == 0 || descriptors2.cols == 0){//图片(纯黑色图片)没有匹配点时，进行硬拼
		return;
	}

	// 特征点匹配
	cv::FlannBasedMatcher   matcher;
	std::vector<cv::DMatch> originMatches;
	matcher.match(descriptors1, descriptors2, originMatches);

	// 计算特征点之间的最小距离，然后取所有特征点之中一定距离内的特征点
	// 以减少个别特征点引起误差，也就是计算所谓的 “好” 特征点
	float minDist = FLT_MAX;
	const unsigned nOriMatches = originMatches.size();
	for (unsigned i = 0; i < nOriMatches; i++)
	{
		float dist = originMatches[i].distance;
		if (dist < minDist) minDist = dist;
	}

	for (unsigned i = 0; i < nOriMatches; i++)
	{
		if (originMatches[i].distance <= std::max(2 * minDist, 0.02f))
		{
			matches.push_back(originMatches[i]);
		}
	}

	cout << "matches.size():" << matches.size() << endl;
}

/*******************************************************
 * Description: 进行图像对齐
 *
 * @para image1, image2 进行对齐的图片
 * @para H 变换矩阵
 *******************************************************/
void alignImage(const cv::Mat & image1, const cv::Mat & image2, cv::Mat & H)
{
	// 找到图像中的特征点对
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	std::vector<cv::DMatch>   matches;
	matchFeatures(image1, image2, keypoints1, keypoints2, matches);
	if(matches.size() == 0){//图片(纯黑色图片)没有匹配点时，进行硬拼
		H = cv::Mat::eye(3, 3, CV_32FC1);
		return;
	}

	std::vector<cv::Point2f> pts1, pts2;
	const unsigned nMatches = matches.size();

	// 进行匹配的特征点是前面判断的“好”特征点之中获取
	for (unsigned i = 0; i < nMatches; i++)
	{
		pts1.push_back(keypoints1[matches[i].queryIdx].pt);
		pts2.push_back(keypoints2[matches[i].trainIdx].pt);
	}

	// 计算变换矩阵
	computeHomography(H, pts2, pts1);
}

/*******************************************************
 * Description: 图像融合
 *
 * @para sourceImages 进行图像融合的所有图像
 * @para H 变换矩阵
 * @para panoImg 融合后的图像
 *******************************************************/
void blendPanorama(const std::vector<cv::Mat> & sourceImages,
	std::vector<cv::Mat> & H, cv::Mat & panoImg)
{
	cv::Mat tempImg;
	
	// 计算融合后的图像大小
	computeSize(sourceImages, H, tempImg);

	// 进行图像的融合
	const unsigned nImgs = sourceImages.size();
	for (unsigned i = 0; i < nImgs; i++)
	{
		blendImage(sourceImages[i], H[i], tempImg);
	}

	normalizeBlend(tempImg, panoImg);
	affineDeform(panoImg, sourceImages.front(), H.front(), sourceImages.back(), H.back());
}

/*******************************************************
 * Description: 计算变换矩阵
 *
 * @para H 变换矩阵
 * @para pts2, pts1 变换匹配点
 *******************************************************/
void computeHomography(cv::Mat & H, 
	const std::vector<cv::Point2f> & pts2,
	const std::vector<cv::Point2f> & pts1)
{
	const float thresh = 4.0f;

	unsigned count = 0;	
	std::vector<unsigned> inliers;

	unsigned n = pts1.size();
	for (unsigned i = 0; i < n; i++)
	{
		float tu = pts1[i].x - pts2[i].x;
		float tv = pts1[i].y - pts2[i].y;

		std::vector<unsigned> tempInliers;
		for (unsigned j = 0; j < n; j++)
		{
			float tx = pts2[j].x + tu;
			float ty = pts2[j].y + tv;

			float d = dist(tx, ty, pts1[j].x, pts1[j].y);
			if (d <= thresh)
			{
				tempInliers.push_back(j);
			}
		}

		if (count < tempInliers.size())
		{
			count = tempInliers.size();
			inliers.swap(tempInliers);
		}
	}

	float u = 0.0f, v = 0.0f;
	for (const auto i : inliers)
	{
		u += pts1[i].x - pts2[i].x;
		v += pts1[i].y - pts2[i].y;
	}
	u /= inliers.size();
	v /= inliers.size();

	H = cv::Mat::eye(3, 3, CV_32FC1);
	H.at<float>(0, 2) = u;
	H.at<float>(1, 2) = v;
}

/*******************************************************
 * Description: 计算两点距离
 *
 * @para x1, y1 点 1 坐标
 * @para x2, y2 点 2 坐标
 *******************************************************/
float dist(const float &x1, const float &y1, const float &x2, const float &y2)
{
	float x = x1 - x2;
	float y = y1 - y2;

	return sqrtf(x * x + y * y);
}

/*******************************************************
 * Description: 计算目标图像大小
 *
 * @para sourceImages 原图像组
 * @para Hs 变换矩阵
 * @para panoImg 目标图像
 *******************************************************/
void computeSize(const std::vector<cv::Mat> &sourceImages,
	std::vector<cv::Mat> &Hs, cv::Mat &panoImg)
{
	const unsigned nImgs = sourceImages.size();
	for (unsigned i = 1; i < nImgs; i++)
	{		
		Hs[i] = Hs[i - 1] * Hs[i];
	}

	float minX = FLT_MAX, maxX = 0.0f;
	float minY = FLT_MAX, maxY = 0.0f;
	
	// 计算目标图像边界
	for (unsigned i = 0; i < nImgs; i++)
	{
		const int w = sourceImages[i].cols;
		const int h = sourceImages[i].rows;

		std::vector<cv::Point2f> corners;

		corners.push_back(cv::Point2f(0.0f, 0.0f));
		corners.push_back(cv::Point2f(0.0f, h - 1.0f));
		corners.push_back(cv::Point2f(w - 1.0f, 0.0f));
		corners.push_back(cv::Point2f(w - 1.0f, h - 1.0f));

		cv::perspectiveTransform(corners, corners, Hs[i]);

		for (const auto & corner : corners)
		{
			if (corner.x < minX) minX = corner.x;
			if (corner.x > maxX) maxX = corner.x;
			if (corner.y < minY) minY = corner.y;
			if (corner.y > maxY) maxY = corner.y;
		}
	}

	cv::Mat transform = cv::Mat::eye(3, 3, CV_32FC1);

	transform.at<float>(0, 2) = -minX;
	transform.at<float>(1, 2) = -minY;

	for (unsigned i = 0; i < nImgs; i++)
	{
		Hs[i] = transform * Hs[i];
	}

	int height = (int)(ceil(maxY) - floor(minY));
	int width  = (int)(ceil(maxX) - floor(minX));

	panoImg = cv::Mat(height, width, CV_32FC4);
}

/*******************************************************
 * Description: 将一张图像融合进目标图像
 *
 * @para img 进行融合的图像
 * @para Hs 变换矩阵
 * @para panoImg 目标图像
 *******************************************************/
void blendImage(const cv::Mat & img, cv::Mat & H, cv::Mat & panoImg)
{
	const float blendWidth = 100.0;

	int minX, minY, maxX, maxY;
	
	// 计算目标图像边界
	imageBoundry(img, H, minX, minY, maxX, maxY);
	cout << "H:" << H << "minX:" << minX << "minY:" << minY << "maxX:" << maxX << "maxY:" << maxY << endl;
	
	cv::Mat invH = H.inv();

	const int h = panoImg.rows;
	const int w = panoImg.cols;

	const float r = img.rows - 1.0f;
	const float c = img.cols - 1.0f;

	for (int i = minY; i <= maxY; i++)
	{
		for (int j = minX; j <= maxX; j++)
		{
			// 舍弃超出边界的点
			if (i < 0 || i >= h || j < 0 || j >= w)
			{
				continue;
			}

			std::vector<cv::Point2f> pts;
			pts.push_back(cv::Point2f((float)j, (float)i));
			cv::perspectiveTransform(pts, pts, invH);
			cv::Point2f & pt = pts.front();

			if (pt.x < 0.0 || pt.x >= c || pt.y < 0.0 || pt.y >= r)
			{
				continue;
			}

			cv::Vec3b color;

			// 使用线性方式计算该点的颜色值
			computeColor(img, pt.x, pt.y, color);

			float weight = std::min(j - minX, maxX - j) / blendWidth;
			if (weight > 1.0f) weight = 1.0f;

			cv::Vec4f & val = panoImg.at<cv::Vec4f>(i, j);

			val[0] += weight * color[0];
			val[1] += weight * color[1];
			val[2] += weight * color[2];
			val[3] += weight;
		}
	}
}

/*******************************************************
 * Description: 计算图像边界
 *
 * @para img 计算边界的图像
 * @para Hs 变换矩阵
 * @para minX, minY, maxX, maxY 边界值
 *******************************************************/
void imageBoundry(const cv::Mat & img, const cv::Mat & H,
	int & minX, int & minY, int & maxX, int & maxY)
{
	float minXf = FLT_MAX, maxXf = 0.0f;
	float minYf = FLT_MAX, maxYf = 0.0f;

	const int h = img.rows;
	const int w = img.cols;

	std::vector<cv::Point2f> corners;
	
	// 图像四个角点
	corners.push_back(cv::Point2f(0.0, 0.0));
	corners.push_back(cv::Point2f(0.0, h - 1.0));
	corners.push_back(cv::Point2f(w - 1.0, 0.0));
	corners.push_back(cv::Point2f(w - 1.0, h - 1.0));

	// 映射到图像的四个角
	cv::perspectiveTransform(corners, corners, H);

	for (const auto & corner : corners)
	{
		if (corner.x < minXf) minXf = corner.x;
		if (corner.x > maxXf) maxXf = corner.x;
		if (corner.y < minYf) minYf = corner.y;
		if (corner.y > maxYf) maxYf = corner.y;
	}

	minX = (int)floor(minXf);
	minY = (int)floor(minYf);
	maxX = (int)ceil (maxXf);
	maxY = (int)ceil (maxYf);
}

/*******************************************************
 * Description: 计算某点颜色
 *
 * @para img 需要进行计算的图像
 * @para x, y 图像点坐标
 * @para color 图像 x, y 像素点颜色值
 *******************************************************/
void computeColor(const cv::Mat & img,
	const float &x, const float &y, cv::Vec3b &color)
{
	int xf = (int)floor(x);
	int yf = (int)floor(y);
	int xc = xf + 1;
	int yc = yf + 1;
	
	std::vector<float> wts;
	wts.push_back(dist(x, y, xf, yf));
	wts.push_back(dist(x, y, xc, yf));
	wts.push_back(dist(x, y, xf, yc));
	wts.push_back(dist(x, y, xc, yc));

	// 计算距离
	float wtTot = std::accumulate(wts.begin(), wts.end(), 0.0f);
	for (float & wt : wts) wt = wtTot - wt;

	wtTot = std::accumulate(wts.begin(), wts.end(), 0.0f);

	for (float & wt : wts)
		wt /= wtTot;

	for (unsigned i = 0; i < 3; i++)
	{
		float colVal = 0.0f;

		colVal += wts[0] * img.at<cv::Vec3b>(yf, xf)[i];
		colVal += wts[1] * img.at<cv::Vec3b>(yf, xc)[i];
		colVal += wts[2] * img.at<cv::Vec3b>(yc, xf)[i];
		colVal += wts[3] * img.at<cv::Vec3b>(yc, xc)[i];

		color[i] = (uchar)colVal;
	}
}

/*******************************************************
 * Description: 根据多张图颜色权值计算图片颜色，进行融合
 *
 * @para tempImg 包含个图像权值的图片
 * @para panoImg 融合后的图片
 *******************************************************/
void normalizeBlend(cv::Mat & tempImg, cv::Mat & panoImg)
{
	const int h = tempImg.rows;
	const int w = tempImg.cols;

	panoImg = cv::Mat(h, w, CV_8UC3);

	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			cv::Vec4f & val   = tempImg.at<cv::Vec4f>(i, j);
			cv::Vec3b & color = panoImg.at<cv::Vec3b>(i, j);

			if (val[3])
			{
				color[0] = (uchar)(val[0] / val[3]);
				color[1] = (uchar)(val[1] / val[3]);
				color[2] = (uchar)(val[2] / val[3]);
			}
			else
			{
				color[0] = (uchar)val[0];
				color[1] = (uchar)val[1];
				color[2] = (uchar)val[2];
			}			
		}
	}
	tempImg.release();
}

/*******************************************************
 * Description: 对图像进行仿射变换，解决垂直方向的偏差问题
 *
 * @para img0 第一张图片
 * @para imgN 最后一张图片
 * @para H0, HN 变换矩阵
 *******************************************************/
void affineDeform(cv::Mat & panoImg,
	const cv::Mat & img0, const cv::Mat & H0,
	const cv::Mat & imgN, const cv::Mat & HN)
{
	std::vector<cv::Point2f> pts;
	pts.push_back(cv::Point2f(0.5f * img0.cols, 0.5f * img0.rows));
	cv::perspectiveTransform(pts, pts, H0);
	cv::Point2f pt_init = pts.front();
	
	pts.clear();
	pts.push_back(cv::Point2f(0.5f * imgN.cols, 0.5f * imgN.rows));
	cv::perspectiveTransform(pts, pts, HN);
	cv::Point2f pt_final = pts.front();
	
	cv::Mat A = cv::Mat::eye(3, 3, CV_32FC1);
	A.at<float>(1, 0) = -(pt_final.y - pt_init.y) / (pt_final.x - pt_init.x);
	A.at<float>(1, 2) = -pt_init.x * A.at<float>(1, 0);

	cv::Mat tempImg;
	cv::warpPerspective(panoImg, tempImg, A, panoImg.size());

	int diff = (int)ceil(abs(pt_final.y - pt_init.y));
	cv::Mat subImg(tempImg, cv::Rect(0, diff, tempImg.cols, tempImg.rows - diff));
	subImg.copyTo(panoImg);
}

/*******************************************************
 * Description: 处理图片大小，是图片最终是宽度为 2048
 *
 * @para panoImg 进行处理的图片
 *******************************************************/
void adjustSize(cv::Mat & panoImg)
{
	const int maxWidth = 2048;
	if (panoImg.cols > maxWidth)
	{
		int height = maxWidth * panoImg.rows / panoImg.cols;
		cv::resize(panoImg, panoImg, cv::Size(maxWidth, height));
	}
}
