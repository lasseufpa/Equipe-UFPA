#include <algorithm>
#include <mutex> 

#include "ros/ros.h"
#include <ros/assert.h> 

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <sensor_msgs/image_encodings.h>

#include <phase0_base_centralize/GetOffset.h>
#include <phase0_base_centralize/centralize.h>

std::mutex g_mtx;
cv::Mat g_image;

void imageCb(const sensor_msgs::ImageConstPtr& msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    {
        std::lock_guard<std::mutex> lck (g_mtx);
        g_image = cv_ptr->image;
    }
}

Centralize g_central;

bool base_centralize(phase0_base_centralize::GetOffset::Request& req, 
                    phase0_base_centralize::GetOffset::Response& res) 
{
    try 
    {
        cv::Mat img;
        {
            std::lock_guard<std::mutex> lck (g_mtx);
            img = g_image;
        }
        
        ROS_ASSERT(img.empty() == 0);

        // cv::imshow("window", img);
        // cv::waitKey(2000);
        // cv::destroyAllWindows();

        g_central.update(img);
        
        cv::Point2d offset = g_central.getOffset();
        res.any_base = res.sucess = true;
        res.offset[0] = offset.x;
        res.offset[1] = offset.y;
        
        if (offset.x == -10 && offset.y == -10) {
            res.any_base = false;
        }

        return true;
    }
    catch (...)
    {
        ROS_ERROR("Exception in base_centralize service");
        res.sucess = false;
        return true;
    }
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "base_centralize");
    
    ros::NodeHandle nh;

    image_transport::ImageTransport it(nh);
    image_transport::Subscriber image_sub;
    image_sub = it.subscribe("/uav1/bluefox_optflow/image_raw", 1, imageCb);

    ros::ServiceServer service = nh.advertiseService("base_centralize", base_centralize);
    ROS_INFO("/base_centralize service started");

    ros::spin();
    return 0;
}