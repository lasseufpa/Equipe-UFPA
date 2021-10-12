#include <vector>

#include "phase0_base_centralize/centralize.h"

void Centralize::noiseFilterMask(cv::Mat& mask, cv::Mat& sure_bg) 
{
    cv::Mat opening;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)); //MORPH_RECT=0

    cv::morphologyEx(mask, opening, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);  // MORPH_OPEN=2
    cv::dilate(opening, sure_bg, kernel, cv::Point(-1, -1), 3);
}

void Centralize::contourMc(std::vector<std::vector<cv::Point>>& contours, std::vector<cv::Point2d>& mc, std::vector<double>& perimeter)
{
    mc = std::vector<cv::Point2d>(contours.size());
    perimeter = std::vector<double> (contours.size());

    for (int i = 0; i < (int)contours.size(); ++i) {
        auto mu = cv::moments(contours[i]);
        mc[i] = {mu.m10 / (mu.m00 + 1e-5), mu.m01 / (mu.m00 + 1e-5)};
        perimeter[i] = cv::arcLength(contours[i], true);
    }
}

bool Centralize::isCentralized(cv::Point2d mc) 
{
    if (center_.x - center_offset_ <= mc.x && center_.x + center_offset_ >= mc.x &&
        center_.y - center_offset_ <= mc.y && center_.y + center_offset_ >= mc.y) 
    {  
        return true;
    }
    return false;
}

cv::Point2d Centralize::calculateOffset(cv::Point2d mc) {
    cv::Point2d offset(0, 0);

    if (abs(mc.x - center_.x) > 49) 
    {
        offset.y = -0.4;
    } 
    else if (abs(mc.x - center_.x) > center_offset_) 
    {
        offset.y = -0.1;
    }

    if (abs(mc.y - center_.y) > 49) 
    {
        offset.x = -0.4;
    } 
    else if (abs(mc.y - center_.y) > center_offset_) 
    {
        offset.x = -0.1;
    }

    if (mc.x - center_.x < 0 && offset.y != 0) 
    {
        offset.y = -offset.y;
    } 
    if (mc.y - center_.y < 0 && offset.x != 0) 
    {
        offset.x = -offset.x;
    } 
    return offset;
}



cv::Point2d Centralize::getOffset()
{
    cv::Mat hsv;
    cv::cvtColor(img_, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask1, mask2;
    cv::inRange(hsv, low_yellow_, high_yellow_, mask1);
    cv::inRange(hsv, low_blue_, high_blue_, mask2);
    
    cv::Mat sure_blue, sure_yellow;
    noiseFilterMask(mask2, sure_blue);
    noiseFilterMask(mask1, sure_yellow);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(sure_blue, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) 
    {
        return cv::Point2d(-10, -10);
    }

    std::vector<cv::Point2d> mc;
    std::vector<double> perimeter;
    contourMc(contours, mc, perimeter);

    auto euclideanDist = [&] (cv::Point2d a, cv::Point2i b) {
        double dx = a.x - b.x;
        double dy = a.y - b.y;
        return sqrt(dx * dx + dy * dy);
    };

    int max_idx = std::max_element(perimeter.begin(), perimeter.end()) - perimeter.begin();
    double dist = euclideanDist(mc[max_idx], center_);

    for (int i = 0; i < (int)mc.size(); ++i) {
        double d = euclideanDist(mc[i], center_);
        if (d < dist) {
            max_idx = i;
            dist = d; 
        }
    }

    if (last_mc_ == mc[max_idx]) {
        return cv::Point2d(0, 0);
    }
    last_mc_ = mc[max_idx];

    if (isCentralized(mc[max_idx])) {
        return cv::Point2d(0, 0);
    }

    return calculateOffset(mc[max_idx]);
}
