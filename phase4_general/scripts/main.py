#!/usr/bin/python3
# @Lucas Damasceno - 2021 - LASSE UFPA
# lucas.damasceno.silva@itec.ufpa.br
import rospy
import os
import time
import cv2
import numpy as np
from std_srvs.srv import Trigger
from phase0_control.controller import Controller
from phase0_base_centralize.srv import GetOffset
from phase0_drone_camera.camera import GetImage
from phase4_general.scamQr import readQr
from std_srvs.srv import Trigger
from std_srvs.srv import SetBool
from mrs_msgs.srv import String
from std_srvs.srv import Trigger
from controller_g_msgs.srv import GetStance

land_srv = '/uav1/uav_manager/land'
takeoff_srv = '/uav1/uav_manager/takeoff'
centralize_srv = '/base_centralize'
land_msg = '/uav1/control_manager/landoff_tracker/land'
takeoff_msg = '/uav1/control_manager/landoff_tracker/takeoff'
gripper_src = '/uav1/control_manager/controller_gripper'
garmin_src = '/uav1/odometry/toggle_garmin'
change_alt_estimator_srv = '/uav1/odometry/change_alt_estimator_type_string'
controller_srv = '/uav1/control_manager/switch_controller'

def recognizeQr(image, control, number_of_images=41):
    assert (number_of_images < 50)
    moves = 0
    deltas = [[0.05, 0], [-0.1, 0], [0.05, 0.05], [-0.1, 0], [0.01, 0.01]]
    listz = [-0.01, -0.02, -0.01, -0.02, -0.01]

    for i in range(1, number_of_images + 1):
        img = image.get()
        isQr, qrData = readQr(img)
        if isQr == True:
            return True, qrData
        rospy.sleep(0.1)

        if i % 10 == 0:
            control.change_reference_pos(is_abs=False, x=deltas[moves][0], y=deltas[moves][1], z=listz[moves])
            rospy.sleep(2)
            moves += 1

    return False, qrData

def readQRCode(image, control, delivered=[]):
    # Read Qrcode
    isQr, qrData = recognizeQr(image, control)

    # QR code detected

    if isQr == True:
        rospy.loginfo('Qrcode encontrado')

        rospy.loginfo(qrData)

        for qrValue in qrData:
            if qrValue[0] in delivered:
                #rospy.loginfo('Repeated')
                continue
            else:
                address = qrValue[0]
                rospy.loginfo('Qrcode value: %s', address)
    else:
        rospy.loginfo('Qrcode não encontrado')

def stop(garmin, change_alt_sensor, switch_controller):
    garmin.call(True)
    change_alt_sensor('HEIGHT')
    rospy.sleep(2)

    switch_controller('MpcController')
    rospy.sleep(2)

def motion(garmin, change_alt_sensor, switch_controller):
    garmin.call(False)
    change_alt_sensor('BARO')
    rospy.sleep(2)

    switch_controller('Se3Controller')
    rospy.sleep(2)

def main():
    rospy.init_node('ufpa_phase1')
    
    #bases = [[45.8, 10, 2.6], [-19.8, -21, 13.6], [-54, -35, 9.85]]

    land = rospy.ServiceProxy(land_srv, Trigger, persistent=True)
    takeoff = rospy.ServiceProxy(takeoff_srv, Trigger, persistent=True)
    centralize = rospy.ServiceProxy(centralize_srv, GetOffset, persistent=True)
    land_verify = rospy.ServiceProxy(land_msg, Trigger, persistent=True)
    takeoff_verify = rospy.ServiceProxy(takeoff_msg, Trigger, persistent=True)
    control = Controller()
    gripper = rospy.ServiceProxy(gripper_src, GetStance, persistent=True)
    garmin = rospy.ServiceProxy(garmin_src, SetBool, persistent=True)
    change_alt_sensor = rospy.ServiceProxy(change_alt_estimator_srv, String, persistent=True)
    switch_controller = rospy.ServiceProxy(controller_srv, String, persistent=True)
    image = GetImage()

    motion(garmin, change_alt_sensor, switch_controller)

    control.change_reference_pos(is_abs=True, x=0, y=0, z=7)
    while(control.utils_arrived.arrived([0, 0, 7]) == False):
        rospy.sleep(0.2)
    
    control.change_reference_pos(is_abs=True, x=0, y=0, z=-4)
    while(control.utils_arrived.arrived([0, 0, -4]) == False):
        rospy.sleep(0.2)
    
    camera = GetImage()
    cap = camera.get()
    cv2.imwrite('./bases/0--7.png', cap)
    
    """ control.change_reference_pos(is_abs=True, x=45.8, y=10, z=-10)
    while(control.utils_arrived.arrived([45.8, 10, -10]) == False):
        rospy.sleep(0.2)
    
    rospy.loginfo('LANDING')
    land()
    rospy.sleep(2)
    rospy.loginfo(land_verify().success)
    while(land_verify().success == True):
        pass
    rospy.sleep(2)

    rospy.loginfo('TAKING OFF')
    takeoff()
    rospy.sleep(2)
    while(takeoff_verify().success == True):
        rospy.sleep(0.2)
    rospy.sleep(2) """

    control.change_reference_pos(is_abs=True, x=-19, y=-21, z=13.6)
    while(control.utils_arrived.arrived([-19, -21, 13.6]) == False):
        rospy.sleep(0.2)
    
    control.change_reference_pos(is_abs=True, x=-19, y=-21, z=1.5)
    while(control.utils_arrived.arrived([-19, -21, 1.5]) == False):
        rospy.sleep(0.2)
    
    stop(garmin, change_alt_sensor, switch_controller)

    readQRCode(image, control, delivered=[])

    motion(garmin, change_alt_sensor, switch_controller)

    """ rospy.loginfo('LADING')
    rospy.wait_for_service(land_srv, timeout=60)
    land()
    rospy.sleep(12)

    rospy.wait_for_service(takeoff_srv, timeout=60)
    rospy.loginfo('TAKING OFF')
    takeoff()
    rospy.sleep(10) """

    control.change_reference_pos(is_abs=True, x=-19, y=-10, z=2.6)
    while(control.utils_arrived.arrived([-19, -10, 2.6]) == False):
        rospy.sleep(0.2)
    
    control.change_reference_pos(is_abs=True, x=-54, y=-35, z=9.85)
    while(control.utils_arrived.arrived([-54, -35, 9.85]) == False):
        rospy.sleep(0.2)
    
    control.change_reference_pos(is_abs=True, x=-53.7, y=-35, z=-1)
    while(control.utils_arrived.arrived([-53.7, -35, -1]) == False):
        rospy.sleep(0.2)

    stop(garmin, change_alt_sensor, switch_controller)

    readQRCode(image, control, delivered=[])

    motion(garmin, change_alt_sensor, switch_controller)
    
    """ rospy.loginfo('LADING')
    rospy.wait_for_service(land_srv, timeout=60)
    land()
    rospy.sleep(12)

    rospy.wait_for_service(takeoff_srv, timeout=60)
    rospy.loginfo('TAKING OFF')
    takeoff()
    rospy.sleep(10) """
    
    control.change_reference_pos(is_abs=True, x=-50, y=-24, z=4)
    while(control.utils_arrived.arrived([-50, -24, 4]) == False):
        rospy.sleep(0.2)

    control.change_reference_pos(is_abs=True, x=11, y=90, z=1)
    while(control.utils_arrived.arrived([11, 90, 1]) == False):
        rospy.sleep(0.2)
	
    control.change_reference_pos(is_abs=True, x=11, y=90, z=0.5)
    while(control.utils_arrived.arrived([11, 90, 0.5]) == False):
        rospy.sleep(0.2)
    
    rospy.sleep(1)
    
    land()

if __name__ == '__main__':
    main()
