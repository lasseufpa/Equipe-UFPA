#!/usr/bin/env python

import rospy
from std_srvs.srv import Trigger
from phase0_control.controller import Controller 
from phase0_base_centralize.srv import GetOffset
from phase0_drone_camera.camera import GetImage
from phase1_panoramic.panoramic import baseScam
from salesman_trajectory import travellingSalesmanProblem

land_srv = '/uav1/uav_manager/land'
takeoff_srv = '/uav1/uav_manager/takeoff'
centralize_srv = '/base_centralize'

def center_base(centralize, control):
    while True:
        data = centralize()
        offset = data.offset
        any_base = data.any_base

        if any_base == False:
            return False
        
        if offset[0] == 0 and offset[1] == 0:
            break

        control.change_reference_pos(is_abs=False, x=offset[0], y=offset[1], z=-0.01)

        rospy.sleep(4)

    return True

def main():

    rospy.init_node('ufpa_phase1')

    land = rospy.ServiceProxy(land_srv, Trigger, persistent=True)
    takeoff = rospy.ServiceProxy(takeoff_srv, Trigger, persistent=True)
    centralize = rospy.ServiceProxy(centralize_srv, GetOffset, persistent=True)
    control = Controller()
    image = GetImage()

    control.change_reference_pos(is_abs=True, x=10, y=90, z=7)
    while(control.utils_arrived.arrived([10,90,7]) == False):
        rospy.sleep(0.2)

    rospy.sleep(10)

    bases=[[25,10,6], [-19.5,-21,13.612], [30.4897,-22.328,-0.192949], [36.67,-14.36,-0.0425], [38,-22.52,-0.3350], [-54,-35,9.85]]
    #falta colocar a segunda base da plataforma, a que fica atrás.

    """
    
    [-19.5,-21,13.612] base petroleira
 
    [30.4897,-22.328,-0.192949] barquinho1

    [36.67,-14.36,-0.0425] barquinho2

    [38,-22.52,-0.3350] barquinho3
    """

    for base in bases:

        control.change_reference_pos(is_abs=True, x=base[0], y=base[1], z=base[2])

        while(control.utils_arrived.arrived(base) == False):
            rospy.sleep(0.2)
        
        rospy.sleep(2)

        control.change_reference_pos(is_abs=False, x=0, y=0, z=-0.8)

        rospy.sleep(4)

        if (center_base(centralize, control) == True):

            rospy.wait_for_service(land_srv, timeout=60)
            land()

            rospy.sleep(12)

            rospy.wait_for_service(takeoff_srv, timeout=60) 
            takeoff()

            rospy.sleep(8)

    control.change_reference_pos(is_abs=True, x=10, y=90, z=2.5)
    while(control.utils_arrived.arrived([1,-1,2.5]) == False):
        rospy.sleep(0.2)

    control.change_reference_pos(is_abs=True, x=10, y=90, z=1.5)
    while(control.utils_arrived.arrived([1,-1,1.5]) == False):
        rospy.sleep(0.2)

    rospy.sleep(4)

    if (center_base(centralize, control) == True):
    
        land()

if __name__ == '__main__':
    main()
