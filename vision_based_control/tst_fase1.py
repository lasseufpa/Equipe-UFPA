import os
import numpy as np
import time
import rospy
from random import randint

bases = np.array([[3.3,0,0.5], [0.2,-5.5,0.5], [1.8,-4,-0.5], [5.2,-1.2,-0.5], [5,-4.5,-0.5]])

for i in range(5):

    if(i == 0):
        json_out = ('[[{},{},{}], [0,0,-1.0]]').format(bases[i][0],bases[i][1],bases[i][2])
        
    else:
        json_out = ('[[{},{},{}]]').format(bases[i][0],bases[i][1],bases[i][2])
    
    correct = True

    if not os.path.exists('./tmp'):
                    os.mkdir('./tmp')
    with open('test_trace.json', 'w') as f:
        f.write(json_out)
    step = 'step.txt'
    while(correct):
        image_name = ('./tmp/base_{}.jpg').format(randint(0,200))
        os.system('rosrun phase0_control_gps control.py --trace ./test_trace.json')
        while(not(os.path.exists(image_name))):
            os.system('python roscamera.py --name '+image_name)
        os.system('python3 visao_fase1.py --name '+image_name)
        os.system('rm ' + image_name)
        with open(step, 'r') as reader:
            status = reader.read()
        if status == 'land':
            #os.system('rosservice call /uav1/uav_manager/land')
            correct = False

    #os.system('rosrun phase0_control_gps control.py --trace ./test_trace.json')
    os.system('rosservice call /uav1/uav_manager/land')
    rospy.sleep(16)
    os.system('rosservice call /uav1/uav_manager/takeoff')
    rospy.sleep(10)
    os.system('rosservice call /uav1/control_manager/goto [0.1,-0.1,1.5,0]')
    rospy.sleep(10)


