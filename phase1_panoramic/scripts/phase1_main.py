#!/usr/bin/python3
# @Lucas Damasceno - 2021 - LASSE UFPA
# lucas.damasceno.silva@itec.ufpa.br

import argparse
import rospy
from std_srvs.srv import Trigger
from phase0_control.controller import Controller
from phase0_drone_camera.camera import GetImage
from panoramic import baseScam

land_srv = '/uav1/uav_manager/land'
takeoff_srv = '/uav1/uav_manager/takeoff'
land_msg = '/uav1/control_manager/landoff_tracker/land'


def main():
    rospy.init_node('ufpa_phase1')

    land = rospy.ServiceProxy(land_srv, Trigger, persistent=True)
    takeoff = rospy.ServiceProxy(takeoff_srv, Trigger, persistent=True)
    land_verify = rospy.ServiceProxy(land_msg, Trigger, persistent=True)
    control = Controller()
    camera = GetImage()

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
    
    if not(args.jump):
        control.change_reference_pos(is_abs=True, x=56.1, y=9, z=7)
        while(control.utils_arrived.arrived([56.1, 9, 7]) == False):
            rospy.sleep(0.2)
        
        for base in zones:
            x_target = float(base.split(",")[0])
            y_target = float(base.split(",")[1])

            control.change_reference_pos(is_abs=True, x=x_target, y=y_target, z=7)
            while(control.utils_arrived.arrived([x_target, y_target, 7]) == False):
                rospy.sleep(0.2)
            
            cap = camera.get()
            pan = baseScam(cap)
            x_offset, y_offset = pan.scan_area()

            x_target += x_offset
            y_target += y_offset
            
            control.change_reference_pos(is_abs=True, x=x_target, y=y_target, z=-4)
            while(control.utils_arrived.arrived([x_target, y_target, -4]) == False):
                rospy.sleep(0.2)
            
            cap = camera.get()
            pan.set_img(cap)
            x_coord, y_coord = pan.centralize()
            print("X:", x_coord)
            print("Y:", y_coord)

            control.change_reference_pos(is_abs=True, x=x_target, y=y_target, z=z_land)
            while(control.utils_arrived.arrived([x_target, y_target, z_land]) == False):
                rospy.sleep(0.2)

            control.change_reference_pos(is_abs=False, x=x_coord+0.75, y=y_coord, z=-1)
            rospy.sleep(2.5)

            cap = camera.get()
            pan.set_img(cap)
            mc = pan.get_mc()
            print(mc[0][0])
            
            if mc[0][0] < 320:
                rospy.loginfo('320')
                control.change_reference_pos(is_abs=False, x=0, y=+1, z=0)
                rospy.sleep(1)
            elif mc[0][0] < 366:
                rospy.loginfo('366')
                control.change_reference_pos(is_abs=False, x=0, y=+0.5, z=0)
                rospy.sleep(1)
            elif mc[0][0] > 381:
                rospy.loginfo('381')
                control.change_reference_pos(is_abs=False, x=0, y=+0.5, z=0)
                rospy.sleep(1)
            elif mc[0][0] > 379:
                rospy.loginfo('379')
                control.change_reference_pos(is_abs=False, x=0, y=+0.05, z=0)
                rospy.sleep(1)
            
            rospy.loginfo('LANDING...')
            land()
            rospy.sleep(2)
            
            while(land_verify().success == True):
                rospy.sleep(2)
            
            rospy.sleep(2)

            rospy.loginfo('TAKING OFF...')
            
            takeoff()
            rospy.sleep(10)
    
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
    rospy.sleep(12)

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
    rospy.sleep(12)

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
    rospy.sleep(15)
    
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