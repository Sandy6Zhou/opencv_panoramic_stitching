#ifndef __SlamView__cropper__
#define __SlamView__cropper__

#include <iostream>
#include "opencv2/opencv.hpp"
// #include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

class Cropper{
    public:
    int kstep = 10;
        
    public:
        Mat crop(const Mat& img);
        
        void crop(const Mat& src, Mat& dst, cv::Rect& dstBound);
        
        
        inline void setStep(int kp){
            kstep = kp;
        }
        
        void setROI(const cv::Rect& roi){
            finalROI = roi;
        }
        
        ~Cropper(){
            mask.release();
        }
        
    private:
        Mat mask;
        cv::Rect rect;
        
        cv::Rect finalROI;
        
    private:
        void getMask(const Mat& img);
        void expandROI(cv::Rect& r);
        void framing();
        
        bool lineHasBlankPixels(const Point2i& st, const Point2i& ed);
        
        bool rectHasBlankPixels(const cv::Rect& roi);
        
        bool lineHasContent(const Point2i& st, const Point2i& ed);
        
        void searchContentLine(Point2i& p1, Point2i& p2, Point2i& delta);
        
};

#endif

