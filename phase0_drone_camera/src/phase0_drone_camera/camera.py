#!/usr/bin/env python

from __future__ import print_function

import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


from phase0_control.collector import Collector

class GetImage(Collector):

    def __init__(self, image_topic='/uav1/bluefox_optflow/image_raw'):
        super(GetImage, self).__init__()
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(image_topic, Image, self.callback)
        rospy.sleep(1)

    def _process(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return None
    
        return cv_image

    def get(self):
        return self.data

def main():
    rospy.init_node('image_converter', anonymous=True)

    ic = GetImage()
    img = ic.get()
    cv2.imshow('PC', img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
