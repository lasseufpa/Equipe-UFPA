import roslib
import sys
import rospy
import cv2
import numpy as np
import random


def binarize_image(image):
    
    #cv2.imshow('window_name', image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows() 

    # pre-process the image by resizing it, converting it to
    # graycale, blurring it, and computing an edge map
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    low_white_display = (0, 0, 121)
    high_white_display = (0, 0, 255)

    # threshold the warped image, then apply a series of morphological
    # operations to cleanup the thresholded image
    sure_display = cv2.inRange(HSV,low_white_display, high_white_display) #sensor green

    #cv2.imwrite(str(random.randint(0, 99)) + str(random.randint(0, 99)) + str(random.randint(0, 99)) + '.png', sure_display)

    #cv2.imshow('window_name', sure_display)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()   
    
    return sure_display

