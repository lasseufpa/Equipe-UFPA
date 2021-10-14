#!/usr/bin/python3
# @Lucas Damasceno - 2021 - LASSE UFPA
# lucas.damasceno.silva@itec.ufpa.br

import argparse
import rospy
from std_srvs.srv import Trigger
from phase0_control.controller import Controller
from phase0_drone_camera.camera import GetImage
from panoramic import baseScam
import pyzbar.pyzbar as pyzbar
from std_srvs.srv import SetBool
import numpy as np
import cv2


def box_center_and_area(image):
    image_rgb = image
    # Center from the image
    center_image = np.shape(image)
    center_image = np.array(center_image[0:2])/2
    center_image = [center_image[1], center_image[0]]
    # define range of yellow color in HSV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #low = np.array([22, 60 , 0])#custom yellow
    # low = np.array([22, 93, 0])#yellos
    #up = np.array([45, 255, 255])#yellow
    low = np.array([0,0,230]) #white
    up = np.array([10,255,255]) #white

    # Create the mask
    mask_hsv = cv2.inRange(image, low, up)
    _, thresh = cv2.threshold(mask_hsv, thresh=114, maxval=151, type=cv2.THRESH_BINARY)
    im_thresh = cv2.bitwise_and(mask_hsv, thresh)

    # Filtragem
    kernel = np.ones((5,5),np.uint8)
    im_thresh = cv2.morphologyEx(im_thresh, cv2.MORPH_CLOSE, kernel, iterations=10)
    cv2.imwrite('./debug.png',mask_hsv)
    
    # Find contours
    _, thresh = cv2.threshold(im_thresh, thresh=114, maxval=151, type=cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    print(contours)
    if contours == []:
        print('Sem contorno')
        return False, False, False

    # Find Rectangle
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    # This is the image with the rectangle
    img = cv2.rectangle(image_rgb,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imwrite('./contour.png',img)

    area = w*h
    center = [x+w//2,y+h//2]
    print(area, center, center_image)
    return area, center, center_image

def calculate_offset(image):
    area, center, center_image = box_center_and_area(image)
    if area == False:
        return [False,0,0]
    # Calculate offset based on the environment coordinates
    x = center[1] - center_image[1]
    y = center[0] - center_image[0]
    #3660 magic number
    #z = np.sqrt(3660-area)
    z = area
    offset = [x,y,z]
    return offset

def is_centralized(offset, epsi=2):
    print(offset)
    for i in offset[0:2]:
        if np.abs(i) > epsi:
            return False
    print('centralizou x e y')
    return True


def centralize_base():

    control = Controller()
    image = GetImage()
    img = image.get()
    
    offset = calculate_offset(img)

    while offset[0] == False:
        control.change_reference_pos(is_abs=False, x=5, y=0, z=0)
        rospy.sleep(3)
        print('Sem offset')
        offset = calculate_offset(img)

    while(not is_centralized(offset)):
        x, y = 0, 0
        if np.abs(offset[0])>30:
            if offset[0]>0:
                x = -2.7
            else:
                x = 3
        elif np.abs(offset[0])>8:
            if offset[0]>0:
                x = -0.09
            else:
                x = 1
        if np.abs(offset[1])>30:
            if offset[0]>0:
                y = -1.7
            else:
                y = 2.2
        if np.abs(offset[1])>3:
            if offset[1]>0:
                y=-0.8
            else:
                y=0.05 
        #print('moving to ->',x,',',y)
        control.change_reference_pos(is_abs=False, x=x, y=y, z=0)
        rospy.sleep(7)
        img = image.get()
        offset = calculate_offset(img)

def readQr(image):
    decodedObjects = pyzbar.decode(image)

    if len(decodedObjects) == 0:
        return [False, '0']
    else:
        return [True, decodedObjects]

land_srv = '/uav1/uav_manager/land'
takeoff_srv = '/uav1/uav_manager/takeoff'
land_msg = '/uav1/control_manager/landoff_tracker/land'
garmin_src = '/uav1/odometry/toggle_garmin'


def main():
    rospy.init_node('ufpa_phase1')

    land = rospy.ServiceProxy(land_srv, Trigger, persistent=True)
    takeoff = rospy.ServiceProxy(takeoff_srv, Trigger, persistent=True)
    land_verify = rospy.ServiceProxy(land_msg, Trigger, persistent=True)
    control = Controller()
    camera = GetImage()
    garmin = rospy.ServiceProxy(garmin_src, SetBool, persistent=True)


    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--jump', action='store_true',
                        help='Run only the land in fixed bases')
    parser.add_argument('-f', '--first', action='store_true',
                        help='To Do')
    parser.add_argument('-k', '--krai', action='store_true',
                        help='To Do')
    parser.add_argument('-d', '--dama', action='store_true',
                        help='To Do')
    parser.add_argument('-a', '--airton', action='store_true',
                        help='To Do')
    parser.add_argument('-b', '--barometro', action='store_true',
                        help='To Do')
    args = parser.parse_args()

    if args.barometro:
        z_land = -15
    else:
        z_land = -11
    
    zones = []
    
    if args.first:
        zones.append('18, -4')
    if args.krai:
        zones.append('0, 0')
    if args.dama:
        zones.append('-13, 2')
    if args.airton:
        zones.append('46, -55')
    
    garmin.call(False)
    rospy.loginfo('Garmin False')
    #base 1 - at the highest point
    control.change_reference_pos(is_abs=True, x=0, y=0, z=7)
    while(control.utils_arrived.arrived([0, 0, 7]) == False):
        rospy.sleep(0.2)
    
    #base 1 - at the highest point
    control.change_reference_pos(is_abs=True, x=-19, y=-21, z=13.6)
    while(control.utils_arrived.arrived([-19, -21, 13.6]) == False):
        rospy.sleep(0.2)
    
    #garmin.call(True)
    #rospy.loginfo('Garmin True')
    
    #altitude adjustment
    control.change_reference_pos(is_abs=True, x=-19, y=-21, z=-0.1)
    while(control.utils_arrived.arrived([-19, -21, -0.1]) == False):
        rospy.sleep(0.2)
    
    #garmin.call(True)
    #rospy.loginfo('Garmin True')
    
    cap = camera.get()
    isQr, qrData = readQr(cap)
    
    if isQr == True:
        print(qrData[0])
        rospy.sleep(3)

    while(isQr==False):
        cap = camera.get()
        isQr, qrData = readQr(cap)
        if isQr == True:
            print(qrData[0])
            rospy.sleep(3)

    #garmin.call(False)
    #rospy.loginfo('Garmin False')

    #movement out of the collision area
    control.change_reference_pos(is_abs=True, x=-19, y=-10, z=2.6)
    while(control.utils_arrived.arrived([-19, -10, 2.6]) == False):
        rospy.sleep(0.2)
    
    #base 2 - near the pipe
    control.change_reference_pos(is_abs=True, x=-54, y=-35, z=9.85)
    while(control.utils_arrived.arrived([-54, -35, 9.85]) == False):
        rospy.sleep(0.2)
    

    #garmin.call(True)
    #rospy.loginfo('Garmin True')
    #altitude adjustment
    control.change_reference_pos(is_abs=True, x=-53.7, y=-35, z=-3.5)
    while(control.utils_arrived.arrived([-53.7, -35, -3.5]) == False):
        rospy.sleep(0.2)
    
    #garmin.call(True)
    #rospy.loginfo('Garmin True')
    
    cap = camera.get()
    isQr, qrData = readQr(cap)
    if isQr == True:
        print(qrData[0])
        rospy.sleep(3)

    while(isQr==False):
        cap = camera.get()
        isQr, qrData = readQr(cap)
        if isQr == True:
            print(qrData[0])
            rospy.sleep(3)

    #garmin.call(False)
    #rospy.loginfo('Garmin False')
    #movement out of the collision area
    control.change_reference_pos(is_abs=True, x=-50, y=-24, z=4)
    while(control.utils_arrived.arrived([-50, -24, 4]) == False):
        rospy.sleep(0.2)

    #data = [[-30, 30, 2], [60, 0, 2], [30, -55, 2]]
    if args.first:
        #movel base1
        control.change_reference_pos(is_abs=True, x=-30, y=30, z=1)
        while(control.utils_arrived.arrived([-30, 30, 1]) == False):
            rospy.sleep(0.2)
        
        if centralize_base():
            cap = camera.get()
            isQr, qrData = readQr(cap)
            if isQr == True:
                print(qrData[0])
                rospy.sleep(3)

            while(isQr==False):
                cap = camera.get()
                isQr, qrData = readQr(cap)
                if isQr == True:
                    print(qrData[0])
                    rospy.sleep(3)
        
        #movel safe height
        control.change_reference_pos(is_abs=True, x=-20, y=30, z=5)
        while(control.utils_arrived.arrived([-20, 30, 5]) == False):
            rospy.sleep(0.2)


    #coster base
    control.change_reference_pos(is_abs=True, x=10.5, y=90, z=1)
    while(control.utils_arrived.arrived([10.5, 90, 1]) == False):
        rospy.sleep(0.2)
	
    #altitude adjustment
    control.change_reference_pos(is_abs=True, x=10.5, y=90, z=0.5)
    while(control.utils_arrived.arrived([10.5, 90, 0.5]) == False):
        rospy.sleep(0.2)
    
    rospy.loginfo('LANDING...')
    
    land()


if __name__ == '__main__':
    main()