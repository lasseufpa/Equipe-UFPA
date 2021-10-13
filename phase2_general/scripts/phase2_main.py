#!/usr/bin/python3
# @Lucas Damasceno - 2021 - LASSE UFPA
# lucas.damasceno.silva@itec.ufpa.br

import cv2
import rospy
import numpy as np
from std_srvs.srv import Trigger
from mrs_msgs.srv import String
from phase0_control.controller import Controller
from phase0_base_centralize.srv import GetOffset
from phase2_general.phase2_vision import Fase_2

land_srv = '/uav1/uav_manager/land'
change_alt_estimator_srv = '/uav1/odometry/change_alt_estimator_type_string'


def main():
	rospy.init_node('ufpa_phase2')
	
	bases = [[0,0,0], [0,-1.3,0], [0,-1.3,0], [0,-1.3,0], [0,-1.3,0],
			 [0,-1.3,0], [0,-1.3,0], [0,-1.3,0], [0,-1.3,0], [0,-1.3,0],
			 [0,-1.3,0], [0,-1.3,0], [0,-1.3,0], [0,-1.3,0], [0,-1.3,0],
			 [0,-1.3,0]]

	land = rospy.ServiceProxy(land_srv, Trigger, persistent=True)
	change_alt_sensor = rospy.ServiceProxy(change_alt_estimator_srv, String, persistent=True)
	control = Controller()
	vision = Fase_2()

	img_green = np.zeros((300,300,3),np.uint8)
	img_red = np.zeros((300,300,3),np.uint8)

	img_green[:,:,1] = 255
	img_red[:,:,2] = 255

	control.change_reference_pos(is_abs=True, x=-49.6, y=-24, z=1.5)
	while(control.utils_arrived.arrived([-49.6, -24, 1.5]) == False):
	    rospy.sleep(0.2)
	
	rospy.sleep(1)
	
	control.change_reference_pos(is_abs=True, x=-49.6, y=-25, z=-1)
	while(control.utils_arrived.arrived([-49.6, -25, -1]) == False):
	    rospy.sleep(0.2)
	
	rospy.sleep(1)
	
	control.change_reference_pos(is_abs=True, x=-49.6, y=-25, z=-2.5)
	while(control.utils_arrived.arrived([-49.6, -25, -2.5]) == False):
	    rospy.sleep(0.2)
	
	rospy.sleep(2)

	change_alt_sensor('HEIGHT')

	rospy.sleep(2)

	control.change_reference_pos(is_abs=True, x=-49.6, y=-24.5, z=0.8)
	while(control.utils_arrived.arrived([-49.6, -24.5, 0.8]) == False):
	    rospy.sleep(0.2)

	i = 0
	for base in bases:

		control.change_reference_pos(is_abs=False, x=base[0], y=base[1], z=base[2])

		rospy.sleep(1.5)

		isRed, mcRed, isGreen, mcGrenn, flag = vision.scan_pipe()

		if flag == 0:
			if isRed > 0:
				for x in range(isRed):
					if mcRed[x][0] > 330:
						rospy.loginfo('Sensor Vermelho - Com Defeito - Centro de Massa {}'.format(mcRed[x][0]))
						
						control.change_reference_pos(is_abs=False, x=0, y=0, z=0.5)
						rospy.sleep(1)
						control.change_reference_pos(is_abs=False, x=0, y=0, z=-0.5)
						rospy.sleep(1)
		
		elif flag == 1:
			if isGreen > 0:
				for x in range(isGreen):
					if mcGrenn[x][0] > 330:
						rospy.loginfo('Sensor Verde - Sem Defeito - Centro de Massa {}'.format(mcGrenn[x][0]))
						rospy.sleep(0.5)
		
		elif flag == 2:
			if (isRed > 0) and (isGreen > 0):
				for x in range(isRed):
					if mcRed[x][0] > 330:
						rospy.loginfo('Sensor Vermelho - Com Defeito - Centro de Massa {}'.format(mcRed[x][0]))
						
						control.change_reference_pos(is_abs=False, x=0, y=0, z=0.5)
						rospy.sleep(1)
						control.change_reference_pos(is_abs=False, x=0, y=0, z=-0.5)
						rospy.sleep(1)
			
				for x in range(isGreen):
					if mcGrenn[x][0] > 330:
						rospy.loginfo('Sensor Verde - Sem Defeito - Centro de Massa {}'.format(mcGrenn[x][0]))
						rospy.sleep(0.5)

		elif flag == 3:
			rospy.loginfo('Nenhum Sensor Detectado')
			rospy.sleep(0.5)

	cv2.destroyAllWindows()
	
	change_alt_sensor('BARO')

	rospy.sleep(2)
	
	control.change_reference_pos(is_abs=True, x=-50, y=-24, z=4)
	while(control.utils_arrived.arrived([-50, -24, 4]) == False):
		rospy.sleep(0.2)

	control.change_reference_pos(is_abs=True, x=10.5, y=90, z=1)
	while(control.utils_arrived.arrived([10.5,90,1]) == False):
		rospy.sleep(0.2)
	
	control.change_reference_pos(is_abs=True, x=10.5, y=90, z=0.5)
	while(control.utils_arrived.arrived([10.5,90,0.5]) == False):
		rospy.sleep(0.2)

	rospy.sleep(2)
	
	rospy.loginfo('LANDING...')
	land()


if __name__ == '__main__':
    main()
