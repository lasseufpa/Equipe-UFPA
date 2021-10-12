#ifndef CENTRALIZE_H
#define CENTRALIZE_H

#include <ros/ros.h>
#include <ros/assert.h> 

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <sensor_msgs/image_encodings.h>

class Centralize{
private:
    cv::Point2d last_mc_ = cv::Point2d(0x3f3f3f3f, 0x3f3f3f3f);

    /* Image */
    cv::Mat img_;
    int x_, y_;
    int center_offset_ = 27; // Constructor param
    cv::Point2i center_;

    /* Color thresholds */
    cv::Scalar low_yellow_ = cv::Scalar(26, 70, 245);
    cv::Scalar high_yellow_ = cv::Scalar(35, 127, 255);
    cv::Scalar low_blue_ = cv::Scalar(110, 50, 210);
    cv::Scalar high_blue_ = cv::Scalar(127, 101, 255);

    cv::Scalar contours_color_ = cv::Scalar(0, 0, 250);

    void noiseFilterMask(cv::Mat&, cv::Mat&);
    void contourMc(std::vector<std::vector<cv::Point>>&, std::vector<cv::Point2d>&, std::vector<double>&);
    bool isCentralized(cv::Point2d);
    cv::Point2d calculateOffset(cv::Point2d);

public:
    Centralize() { }
    Centralize(int center_offset) : center_offset_(center_offset) { }

    void update(cv::Mat& img) {
        img_ = img;
        y_ = img.size().height;
        x_ = img.size().width;
        center_ = {x_ / 2, y_ / 2};
    }

    cv::Point2d getOffset();
};

#endif