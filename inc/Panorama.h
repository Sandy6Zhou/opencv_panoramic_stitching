#ifndef PANORAMA_H
#define PANORAMA_H

#include "opencv2/core/core.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
// #include "opencv2/calib3d/calib3d.hpp"
// #include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <numeric>
#include <vector>

cv::Mat panorama(std::vector<cv::Mat> & vecImgs, unsigned nImgs, unsigned screen_flip);

void detectFeature(const cv::Mat & image,
		std::vector<cv::KeyPoint> & keypoints,
		cv::Mat & descriptor);

void matchFeatures(const cv::Mat & img_1, const cv::Mat & img_2,
		std::vector<cv::KeyPoint> & keyPts_1,
		std::vector<cv::KeyPoint> & keyPts_2,
		std::vector<cv::DMatch> & matches);

void alignImage(const cv::Mat & img_1, const cv::Mat & img_2,
		cv::Mat & H);

void blendPanorama(const std::vector<cv::Mat> & vecImgs,
		std::vector<cv::Mat> & H, cv::Mat & panoImg);

void computeHomography(cv::Mat & H,
		const std::vector<cv::Point2f> & pts_2,
		const std::vector<cv::Point2f> & pts_1);

float dist(const float & x1, const float & y1,
		const float & x2, const float & y2);

void computeSize(const std::vector<cv::Mat> & vecImgs,
		std::vector<cv::Mat> & Hs, cv::Mat & panoImg);

void blendImage(const cv::Mat & img, cv::Mat & H, cv::Mat & panoImg);

void imageBoundry(const cv::Mat & img, const cv::Mat & H,
		int & min_x, int & min_y, int & max_x, int & max_y);

void computeColor(const cv::Mat & img,
		const float & x, const float & y, cv::Vec3b & color);

void normalizeBlend(cv::Mat & tempImg, cv::Mat & panoImg);

void affineDeform(cv::Mat & panoImg,
		const cv::Mat & img_0, const cv::Mat & H_0, 
		const cv::Mat & img_n, const cv::Mat & H_n);

void adjustSize(cv::Mat & panoImg);

#endif
