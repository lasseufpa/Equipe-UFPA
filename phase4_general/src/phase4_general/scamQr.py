import roslib
import sys
import rospy
import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar


def readQr(image):
    decodedObjects = pyzbar.decode(image)

    if len(decodedObjects) == 0:
        return [False, '0']
    else:
        return [True, decodedObjects]
