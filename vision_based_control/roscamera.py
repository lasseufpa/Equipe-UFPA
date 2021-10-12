#!/usr/bin/env python
from __future__ import print_function

import roslib
#roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time
import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description='Follow bases using a trace file'
    )

    parser.add_argument('--name',
                        help='image name',
                        type=str)

    return parser.parse_args()

class image_converter:

  def __init__(self, image_name):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/uav1/bluefox_optflow/image_raw",Image,self.callback)
    self.image_name = image_name

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
#Depis daqui pode usar a imagem pra qualquer coisa com opencv

    '''(rows,cols,channels) = cv_image.shape
    if cols > 60 and rows > 60 :
      cv2.circle(cv_image, (50,50), 10, 255)'''
    #cv2.imshow('PC', cv_image)
    #cv2.waitKey(3)
    cv2.imwrite(self.image_name,cv_image)
    cv2.waitKey(2)
    #rospy.sleep(1)
    #cv2.destroyAllWindows()
    #exit(-1)


def main(args):
  args = get_args()
  ic = image_converter(args.name)
  rospy.init_node('image_converter', anonymous=True)
  #rospy.spin()
  '''try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")'''
  #cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
