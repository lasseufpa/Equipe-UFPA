#!/usr/bin/python3
# @Lucas Damasceno - 2021 - LASSE UFPA
# lucas.damasceno.silva@itec.ufpa.br

import rospy
from std_srvs.srv import Trigger
from phase0_control.controller import Controller

land_srv = '/uav1/uav_manager/land'
takeoff_srv = '/uav1/uav_manager/takeoff'
land_msg = '/uav1/control_manager/landoff_tracker/land'


def main():
    rospy.init_node('ufpa_phase1')

    land = rospy.ServiceProxy(land_srv, Trigger, persistent=True)
    takeoff = rospy.ServiceProxy(takeoff_srv, Trigger, persistent=True)
    land_verify = rospy.ServiceProxy(land_msg, Trigger, persistent=True)
    control = Controller()

    #pier base
    control.change_reference_pos(is_abs=True, x=45.8, y=10, z=2.6)
    while(control.utils_arrived.arrived([45.8, 10, 2.6]) == False):
        rospy.sleep(0.2)
    
    #altitude adjustment
    control.change_reference_pos(is_abs=True, x=45.8, y=10, z=-9)
    while(control.utils_arrived.arrived([45.8, 10, -9]) == False):
        rospy.sleep(0.2)
    
    #land and takeoff actions
    rospy.loginfo('LANDING...')
    
    land()
    rospy.sleep(2)
    
    while(land_verify().success == True):
        rospy.sleep(1)
    
    rospy.sleep(2)

    rospy.loginfo('TAKING OFF...')
    
    takeoff()
    rospy.sleep(10)

    #base 1 - at the highest point
    control.change_reference_pos(is_abs=True, x=-19, y=-21, z=13.6)
    while(control.utils_arrived.arrived([-19, -21, 13.6]) == False):
        rospy.sleep(0.2)
    
    #altitude adjustment
    control.change_reference_pos(is_abs=True, x=-19, y=-21, z=0.5)
    while(control.utils_arrived.arrived([-19, -21, 0.5]) == False):
        rospy.sleep(0.2)
    
    #land and takeoff actions
    rospy.loginfo('LANDING...')
    
    land()
    rospy.sleep(2)
    
    while(land_verify().success == True):
        rospy.sleep(1)
    
    rospy.sleep(2)

    rospy.loginfo('TAKING OFF...')
    
    takeoff()
    rospy.sleep(10)

    #movement out of the collision area
    control.change_reference_pos(is_abs=True, x=-19, y=-10, z=2.6)
    while(control.utils_arrived.arrived([-19, -10, 2.6]) == False):
        rospy.sleep(0.2)
    
    #base 2 - near the pipe
    control.change_reference_pos(is_abs=True, x=-54, y=-35, z=9.85)
    while(control.utils_arrived.arrived([-54, -35, 9.85]) == False):
        rospy.sleep(0.2)
    
    #altitude adjustment
    control.change_reference_pos(is_abs=True, x=-53.7, y=-35, z=0)
    while(control.utils_arrived.arrived([-53.7, -35, 0]) == False):
        rospy.sleep(0.2)
    
    #land and takeoff actions
    rospy.loginfo('LANDING...')
    
    land()
    rospy.sleep(2)
    
    while(land_verify().success == True):
        rospy.sleep(1)
    
    rospy.sleep(2)

    rospy.loginfo('TAKING OFF...')
    
    takeoff()
    rospy.sleep(10)
    
    #movement out of the collision area
    control.change_reference_pos(is_abs=True, x=-50, y=-24, z=4)
    while(control.utils_arrived.arrived([-50, -24, 4]) == False):
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