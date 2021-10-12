#!/usr/bin/python3
# @Lucas Damasceno - 2021 - LASSE UFPA
# lucas.damasceno.silva@itec.ufpa.br
import rospy
import os
import time
import cv2
import numpy as np
from std_srvs.srv import Trigger
from mrs_msgs.srv import String
from phase0_control.controller import Controller
from phase0_base_centralize.srv import GetOffset
from phase2_general.phase2_vision import Fase_2

land_srv = '/uav1/uav_manager/land'
takeoff_srv = '/uav1/uav_manager/takeoff'
centralize_srv = '/base_centralize'
change_alt_estimator_srv = '/uav1/odometry/change_alt_estimator_type_string'

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
	rospy.init_node('ufpa_phase2')

	'''bases = [[-49.6,-24.5,-2.8], [-49.6,-25.5,-2.8], [-49.6,-26.5,-2.8], [-49.6,-27.5,-2.8], [-49.6,-28.5,-2.8],
			 [-49.6,-29.5,-2.8], [-49.6,-30.5,-2.8], [-49.6,-31.5,-2.8], [-49.6,-32.5,-2.8], [-49.6,-33.5,-2.8],
			 [-49.6,-34.5,-2.8], [-49.6,-35.5,-2.8], [-49.6,-36.5,-2.8], [-49.6,-37.5,-2.8], [-49.6,-38.5,-2.8],
			 [-49.6,-39.5,-2.8], [-49.6,-40.5,-2.8], [-49.6,-41.5,-2.8], [-49.6,-42.5,-2.8], [-49.6,-43.5,-2.8]]
			 [0,-1.3,0], [0,-1.3,0], [0,-1.3,0], [0,-1.3,0], [0,-1.3,0]'''
	
	bases = [[0,0,0], [0,-1.3,0], [0,-1.3,0], [0,-1.3,0], [0,-1.3,0],
			 [0,-1.3,0], [0,-1.3,0], [0,-1.3,0], [0,-1.3,0], [0,-1.3,0],
			 [0,-1.3,0], [0,-1.3,0], [0,-1.3,0], [0,-1.3,0], [0,-1.3,0]]

	land = rospy.ServiceProxy(land_srv, Trigger, persistent=True)
	takeoff = rospy.ServiceProxy(takeoff_srv, Trigger, persistent=True)
	centralize = rospy.ServiceProxy(centralize_srv, GetOffset, persistent=True)
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
	
	rospy.sleep(1)
	
	control.change_reference_pos(is_abs=True, x=-49.6, y=-24.5, z=-3)
	while(control.utils_arrived.arrived([-49.6, -24.5, -3]) == False):
	    rospy.sleep(0.2)

	rospy.sleep(2)

	#labels = vision.scan_panoramic()

	change_alt_sensor('HEIGHT')

	rospy.sleep(2)

	control.change_reference_pos(is_abs=True, x=-49.6, y=-24.5, z=0.8)
	while(control.utils_arrived.arrived([-49.6, -24.5, 0.8]) == False):
	    rospy.sleep(0.2)

	i = 0
	for base in bases:

		control.change_reference_pos(is_abs=False, x=base[0], y=base[1], z=base[2])

		#while(control.utils_arrived.arrived(base) == False):
		#    rospy.sleep(0.2)

		rospy.sleep(1.5)

		isRed, mcRed, isGreen, mcGrenn, flag = vision.scan_pipe()
		#path = './img_cano_{}.png'.format(i)
		#img = vision.scan_pipe(path)
		#i += 1
		#continue

		#rospy.sleep(2)

		if flag == 0:
			if isRed > 0:
				for x in range(isRed):
					if mcRed[x][0] > 330:
						rospy.loginfo('Sensor Vermelho - Com Defeito')
						#rospy.sleep(1)
						
						control.change_reference_pos(is_abs=False, x=0, y=0, z=0.5)
						rospy.sleep(1.5)
						control.change_reference_pos(is_abs=False, x=0, y=0, z=-0.5)
						rospy.sleep(1.5)
		
		elif flag == 1:
			if isGreen > 0:
				for x in range(isGreen):
					if mcGrenn[x][0] > 330:
						rospy.loginfo('Sensor Verde - Sem Defeito')
						rospy.sleep(0.5)
		
		elif flag == 2:
			if (isRed > 0) and (isGreen > 0):
				for x in range(isRed):
					if mcRed[x][0] > 330:
						rospy.loginfo('Sensor Vermelho - Com Defeito')
						#rospy.sleep(1)
						
						control.change_reference_pos(is_abs=False, x=0, y=0, z=0.5)
						rospy.sleep(1.5)
						control.change_reference_pos(is_abs=False, x=0, y=0, z=-0.5)
						rospy.sleep(1.5)
			
				for x in range(isGreen):
					if mcGrenn[x][0] > 330:
						rospy.loginfo('Sensor Verde - Sem Defeito')
						rospy.sleep(0.5)

		elif flag == 3:
			rospy.loginfo('Nenhum Sensor Detectado')
			rospy.sleep(0.5)
		
		'''if isRed > 0:
			for x in range(isRed):
				if mcRed[x][0] > 330:
					rospy.loginfo('Sensor Vermelho - Com Defeito')
					rospy.sleep(3)
		
		elif isGreen > 0:
			for x in range(isGreen):
				if mcGreen[x][0] > 330:
					rospy.loginfo('Sensor Verde - Sem Defeito')
					rospy.sleep(3)
		
		else:
			rospy.loginfo('Nenhum Sensor Detectado')
			rospy.sleep(3)'''

	#rospy.loginfo('Sensores localizados no cano:')
	#rospy.loginfo(labels)

	'''for i in range(len(labels)):
		if labels[i] == 'Green':
			cv2.imshow("Sensor_" +str(i)+ "", img_green)
			cv2.waitKey(1000)
			cv2.destroyAllWindows()

		if labels[i] == 'Red':
			cv2.imshow("Sensor_" +str(i)+ "", img_red)
			cv2.waitKey(1000)
			cv2.destroyAllWindows()'''

	change_alt_sensor('BARO')

	rospy.sleep(2)
	
	control.change_reference_pos(is_abs=True, x=-50, y=-24, z=4)
	while(control.utils_arrived.arrived([-50, -24, 4]) == False):
		rospy.sleep(0.2)

	control.change_reference_pos(is_abs=True, x=10, y=90, z=1)
	while(control.utils_arrived.arrived([10,90,1]) == False):
		rospy.sleep(0.2)
	
	control.change_reference_pos(is_abs=True, x=10, y=93, z=0.5)
	while(control.utils_arrived.arrived([10,90,0.5]) == False):
		rospy.sleep(0.2)

	rospy.sleep(2)

	land()

	#if (center_base(centralize, control) == True):
	#	land()


if __name__ == '__main__':
    main()
