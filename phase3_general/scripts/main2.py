#!/usr/bin/python3

import rospy
import cv2
from std_srvs.srv import Trigger 
from std_srvs.srv import SetBool
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
garmin_srv = '/uav1/odometry/toggle_garmin'
controller_srv = '/uav1/control_manager/switch_controller'
change_alt_estimator_srv = '/uav1/odometry/change_alt_estimator_type_string'
change_speed_srv = '/uav1/constraint_manager/set_constraints'

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

    rospy.init_node('ufpa_phase3_main2')

    #data = [[-19.5, -21, 2], [30.4897, -22.328, 2], [36.67, -14.36, 2], [38, -22.52, 2], [45, 10, 2]]
    data = [[-30, 30, 2], [60, 0, 2], [30, -55, 2]]

    rospy.set_param('/bases', data)

    bases = rospy.get_param('/bases')

    land = rospy.ServiceProxy(land_srv, Trigger, persistent=True)
    takeoff = rospy.ServiceProxy(takeoff_srv, Trigger, persistent=True)
    centralize = rospy.ServiceProxy(centralize_srv, GetOffset, persistent=True)
    change_alt_sensor = rospy.ServiceProxy(change_alt_estimator_srv, String,persistent=True)
    garmin = rospy.ServiceProxy(garmin_srv, SetBool, persistent=True)
    switch_controller = rospy.ServiceProxy(controller_srv, String, persistent=True)
    change_speed = rospy.ServiceProxy(change_speed_srv, String, persistent=True)
    control = Controller()
    image = GetImage()

    
    #change_alt_sensor('BARO')
    

    #desired_height = [2, 2, 1.3, 0.9, 0.9]
    desired_height = [6, 6, 6, 6, 6, 6]
    number_of_displays = 0
    '''for _i, base in enumerate(bases):

        if number_of_displays == 1:
            break
        garmin.call(False)
        change_speed('fast')
        rospy.sleep(1)
        # Fly at high Z
        control.change_reference_pos(is_abs=True, z=desired_height[_i], arrive=True)

        # Go to base
        control.change_reference_pos(is_abs=True, x=base[0], y=base[1], z=desired_height[_i], arrive=True)

        # Go to low Z to centralize
        change_speed('medium')
        rospy.sleep(1)
        switch_controller('MpcController')
        rospy.sleep(1)
        control.change_reference_pos(is_abs=True, z=1.2, arrive=True)

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
        
            upper_number, down_number = recognizeDigits(image, control)

            resposta = isConforme(upper_number, down_number)

            for _ in range(10):
                rospy.loginfo('Percentual de gas: %s', resposta[0])
                rospy.loginfo('Ajuste de ZERO: %s', resposta[1])
                rospy.sleep(1)

        else:
            rospy.loginfo('DISPLAY NAO DETECTADO')
        switch_controller('Se3Controller')
        rospy.sleep(1)'''

    garmin.call(False)
    #change_speed('fast')
    rospy.sleep(1)
    
    #offshore 1
    control.change_reference_pos(is_abs=True, x=-19, y=-21, z=13.6)
    while(control.utils_arrived.arrived([-19, -21, 13.6]) == False):
        rospy.sleep(0.2)
    
    
    #change_speed('medium')
    #garmin.call(True)
    #change_alt_sensor('HEIGHT')
    #switch_controller('MpcController')
    #rospy.sleep(2)
    
    #altitude adjustment
    control.change_reference_pos(is_abs=True, x=-19, y=-21, z=0.45)
    while(control.utils_arrived.arrived([-19, -21, 0.45]) == False):
        rospy.sleep(0.2)
    
    img = binarize_image(image.get())

    any_display = is_there_a_digit_display_in_this_image(img)
    print(any_display)
    if any_display:
        rospy.loginfo('DISPLAY DETECTADO')

    else:
        rospy.loginfo('DISPLAY NAO DETECTADO')
    #switch_controller('Se3Controller')
    rospy.sleep(2)

    #garmin.call(False)
    #rospy.sleep(5)
    #change_speed('fast')
    #change_alt_sensor('BARO')
    #rospy.sleep(2)

    # Fora area colisao
    control.change_reference_pos(is_abs=True, x=-19, y=-10, z=2.6)
    while(control.utils_arrived.arrived([-19, -10, 2.6]) == False):
        rospy.sleep(0.2)

    # Fora area colisao
    #control.change_reference_pos(is_abs=True, x=-9, y=-26, z=4)
    #while(control.utils_arrived.arrived([-9, -26, 4]) == False):
    #    rospy.sleep(0.2)

    #base cano
    control.change_reference_pos(is_abs=True, x=-54, y=-35, z=9.85)
    while(control.utils_arrived.arrived([-54, -35, 9.85]) == False):
        rospy.sleep(0.2)

    #garmin.call(True)
    
    #change_speed('medium')
    #rospy.sleep(1)
    #change_alt_sensor('HEIGHT')
    #switch_controller('MpcController')
    #rospy.sleep(2)

    #altitude adjustment
    control.change_reference_pos(is_abs=True, x=-54, y=-35, z=-1)
    while(control.utils_arrived.arrived([-54, -35, -1]) == False):
        rospy.sleep(0.2)

    img = binarize_image(image.get())

    any_display = is_there_a_digit_display_in_this_image(img)
    print(any_display)
    if any_display:
        rospy.loginfo('DISPLAY DETECTADO')
    else:
        rospy.loginfo('DISPLAY NAO DETECTADO')
    #switch_controller('Se3Controller')
    rospy.sleep(2)

    #garmin.call(False)
    #rospy.sleep(5)
    #change_alt_sensor('BARO')
    #change_speed('fast')
    #rospy.sleep(2)

    ## GO BACK HOME

    # Fly at high Z
    #coster base
    control.change_reference_pos(is_abs=True, x=-50, y=-24, z=4)
    while(control.utils_arrived.arrived([-50, -24, 4]) == False):
        rospy.sleep(0.2)

    #garmin.call(True)
    
    #change_speed('medium')
    #rospy.sleep(1)
    #switch_controller('MpcController')
    #rospy.sleep(1)

    #altitude adjustment
    control.change_reference_pos(is_abs=True, x=10.5, y=90, z=1)
    while(control.utils_arrived.arrived([10.5, 90, 1]) == False):
        rospy.sleep(0.2)
    
    control.change_reference_pos(is_abs=True, x=10.5, y=90, z=0.5)
    while(control.utils_arrived.arrived([10.5, 90, 0.5]) == False):
        rospy.sleep(0.2)

    rospy.loginfo('LANDING...')

    land() 



if __name__ == '__main__':
    main()