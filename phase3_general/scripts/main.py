#!/usr/bin/env python

import rospy
import cv2
from std_srvs.srv import Trigger
from mrs_msgs.srv import String
from phase0_control.controller import Controller
from phase0_base_centralize.srv import GetOffset
from phase0_drone_camera.camera import GetImage
from salesman_trajectory import travellingSalesmanProblem

from phase3_general.preprocessing import binarize_image
import phase_3_display_classification.aux_functions_for_classifier as display
from phase_3_display_classification.rotate_display.util_interpret_digits import is_there_a_digit_display_in_this_image

land_srv = '/uav1/uav_manager/land'
takeoff_srv = '/uav1/uav_manager/takeoff'
centralize_srv = '/base_centralize'

def recognizeDigits(image, control):
    number_of_images = 33
    images = []; moves = 0
    deltas = [[0.05, 0], [-0.1, 0], [0.05, 0.05]]

    for i in range(1, number_of_images + 1):
        img = binarize_image(image.get())
        images.append(img)
        rospy.sleep(0.1)

        if i % 10 == 0:
            control.change_reference_pos(is_abs=False, x=deltas[moves][0], y=deltas[moves][1], z=0)
            rospy.sleep(2)
            moves += 1

    upper_number, down_number = display.mnist_classifier_by_voting(images)

    rospy.loginfo('percentual de gas (numero de cima): %d', upper_number)
    rospy.loginfo('ajuste de ZERO (numero de baixo): %d', down_number)

    return upper_number, down_number

def isConforme(upper_number, down_number):
    upper_ans = 'Nao conforme'
    if 45 <= upper_number <= 55:
        upper_ans = 'Conforme'

    down_ans = 'Nao conforme'
    if -5 <= down_number <= 5:
        down_ans = 'Conforme'

    return [upper_ans, down_ans]


def main():

    rospy.init_node('ufpa_phase3_main')

    #data = [[3.26, -0.06, 2], [0.15, -5.95, 2], [0.2221916014762265, -4.046130015533255, 2], [5.2802585562667596, -3.049278823968442, 2], [6.4236754655651335, -1.0209832588243547, 2]]
    data = [[-19.5, -21, 2], [30.4897, -22.328, 2], [36.67, -14.36, 2], [38, -22.52, 2], [45, 10, 2]]
    #data = [[-53.5,-35.5,9.85],[-53.5,-23,5],[-19.0, -21.2, 3], [26, -15, 0.5], [36.67, -14.36, 0.5], [38, -22.52, 0.5], [45.6, 10, 0.5]]
    #path = travellingSalesmanProblem(data)
    rospy.set_param('/bases', data)

    bases = rospy.get_param('/bases')

    land = rospy.ServiceProxy(land_srv, Trigger, persistent=True)
    takeoff = rospy.ServiceProxy(takeoff_srv, Trigger, persistent=True)
    centralize = rospy.ServiceProxy(centralize_srv, GetOffset, persistent=True)
    control = Controller()
    image = GetImage()

    #desired_height = [2, 2, 1.3, 0.9, 0.9]
    desired_height = [5, 5, 5, 5, 5, 5]
    number_of_displays = 0
    for _i, base in enumerate(bases):

        if number_of_displays == 3:
            break

        # Fly at high Z
        control.change_reference_pos(is_abs=True, z=desired_height[_i], arrive=True)

        # Go to base
        control.change_reference_pos(is_abs=True, x=base[0], y=base[1], z=desired_height[_i], arrive=True)

        # Go to low Z to centralize
        control.change_reference_pos(is_abs=True, z=1.5, arrive=True)

        #if not control.center_at_base(centralize, descend_factor=-0.01):
            #continue
        #    pass

        # Lower Z to find display
        #control.change_reference_pos(is_abs=True, z=1.2, arrive=True)
        ##### DIGITS

        img = binarize_image(image.get())

        any_display = is_there_a_digit_display_in_this_image(img)
        print(any_display)
        if any_display:
            rospy.loginfo('DISPLAY DETECTADO')

            number_of_displays += 1
            rospy.sleep(0.1)
            #window_name = 'image'
            #cv2.imshow(window_name,image.get())
            upper_number, down_number = recognizeDigits(image, control)

            resposta = isConforme(upper_number, down_number)

            for _ in range(10):
                rospy.loginfo('percentual de gas: %s', resposta[0])
                rospy.sleep(1)

            rospy.sleep(30)

            for _ in range(10):
                rospy.loginfo('ajuste de ZERO: %s', resposta[1])
                rospy.sleep(1)


        else:
            rospy.loginfo('DISPLAY NAO DETECTADO')

    ## GO BACK HOME

    # Fly at high Z
    control.change_reference_pos(is_abs=True, z=4, arrive=True)

    rospy.sleep(1)

    #control.change_reference_pos(is_abs=True, x=1, y=-1, z=2)
    control.change_reference_pos(is_abs=True, x=10.6, y=90, z=7)
    while(control.utils_arrived.arrived([10.6,90,7]) == False):
        rospy.sleep(0.2)

    control.change_reference_pos(is_abs=True, x=10.6, y=90, z=2)
    while(control.utils_arrived.arrived([10.6,90,2]) == False):
        rospy.sleep(0.2)

    rospy.sleep(4)

    control.center_at_base(centralize)

    land()

    rospy.sleep(10)



if __name__ == '__main__':
    main()
