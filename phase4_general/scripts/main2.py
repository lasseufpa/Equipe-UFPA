#!/usr/bin/env python

import rospy
from std_srvs.srv import Trigger
from phase0_control.controller import Controller
from phase0_base_centralize.srv import GetOffset
from phase0_drone_camera.camera import GetImage
from phase4_general.scamQr import readQr
from phase4_general.centralize import centralize
from gazebo_ros_link_attacher.srv import Attach, AttachRequest, AttachResponse
from math import sqrt

land_srv = '/uav1/uav_manager/land'
takeoff_srv = '/uav1/uav_manager/takeoff'
centralize_srv = '/base_centralize'
attach_srv = '/link_attacher_node/attach'
detach_srv = '/link_attacher_node/detach'

# 752 x 480

def getCenter(image, address, control):
    decodedObjects = recognizeQr(image, control, number_of_images=21)

    if decodedObjects[0] == True:

        for qrValue in decodedObjects[1]:
            if qrValue[0] == address:
                center_width = qrValue[2][0] + qrValue[2][2]/2
                center_height = qrValue[2][1] + qrValue[2][3]/2
                return center_width, center_height

    return -1, -1

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

def calculateOffset(p, center, center_offset):
    offset = [0, 0]

    if abs(p[0] - center[0])  > 49:
        offset[1] = -0.09
    elif abs(p[0] - center[0]) > center_offset:
        offset[1] = -0.04

    if abs(p[1] - center[1]) > 49:
        offset[0] = -0.1
    elif abs(p[1] - center[1]) > center_offset:
        offset[0] = -0.05

    if p[0] - center[0] < 0 and offset[1] != 0:
        offset[1] = -offset[1]
    if p[1] - center[1] < 0 and offset[0] != 0:
        offset[0] = -offset[0]

    return offset


def centralize_qrcode(address, control, image):

    shape = image.get().shape
    center = [shape[1] // 2, shape[0] // 2]
    center_offset = 20

    #oy, ox = getCenter(image.get(), address)

    while True:
        oy, ox = getCenter(image, address, control)

        if (oy == -1 and ox == -1):
            #rospy.loginfo("qr nao encontrado")
            return False

        offset = calculateOffset([oy, ox], center, center_offset)

        rospy.loginfo('QrOffset: ' + str(offset))

        if offset[0] == 0 and offset[1] == 0:
            break

        control.change_reference_pos(is_abs=False, x=offset[0], y=offset[1], z=0)
        rospy.sleep(5)

    return True


def delivery(address, bases, centralize, control, attach, detach, image):
    mapper = {
        'A' : bases[4],
        'B' : bases[3],
        'C' : bases[2],
        'D' : bases[1],
        'E' : bases[0],
    }

    #rospy.sleep(1)
    #control.change_reference_pos(is_abs=False, x=-0, y = 0, z = -0.1, arrive=True)
    rospy.loginfo("Centralizando Qrcode")
    if centralize_qrcode(address, control, image) == False:
        pass
    rospy.loginfo("Qrcode centralizado")
    #decentralize(address, control, image, centralize)
    control.change_reference_pos(is_abs=True, z=0.35, arrive=True)
    control.change_reference_pos(is_abs=False, x=-0.04, y=0.01, z =0, arrive=True)

    req = AttachRequest()
    req.model_name_1 = "uav1"
    req.link_name_1 = "base_link"
    req.model_name_2 = "equipment" + str(address)
    req.link_name_2 = "link_" + str(address)

    rospy.loginfo("Coletando caixa")
    attach.call(req)

    rospy.sleep(2)
    rospy.loginfo("Caixa coletada")

    control.change_reference_pos(is_abs=True, z=1.8, arrive=True)

    rospy.loginfo("Saiu para entrega")

    # Base to deliver package
    coordenates = mapper[str(address)]

    # Go to destiny base
    control.change_reference_pos(is_abs=True, x=coordenates[0], y=coordenates[1], z=2, arrive=True)

    rospy.sleep(1)

    #if control.center_at_base(centralize):

    control.change_reference_pos(is_abs=False, x=0.1, y=0.1, z=-1, arrive=True)
    #rospy.sleep(2)
    rospy.loginfo("Entregue")

    detach.call(req)

    rospy.sleep(2)

    return address



def main():

    rospy.init_node('ufpa_phase4')

    data = [[3.26, -0.06, 2], [0.15, -5.95, 2], [0.22, -4.04, 2], [6.6, -1.02, 2], [5.28, -3.04, 2]]
    #data = [[0.22, -4.04, 2], [6.55, -1.02, 2], [5.28, -3.04, 2]]

    rospy.set_param('/bases', data)

    bases = rospy.get_param('/bases')
    desired_height = [2, 2, 1.3, 0.9, 0.9]

    delivered = []

    land = rospy.ServiceProxy(land_srv, Trigger, persistent=True)
    takeoff = rospy.ServiceProxy(takeoff_srv, Trigger, persistent=True)
    centralize = rospy.ServiceProxy(centralize_srv, GetOffset, persistent=True)
    attach = rospy.ServiceProxy(attach_srv, Attach, persistent=True)
    detach = rospy.ServiceProxy(detach_srv, Attach, persistent=True)
    control = Controller()
    image = GetImage()

    for _i, base in enumerate(bases):

        # Fly at high Z
        control.change_reference_pos(is_abs=True, z=desired_height[_i], arrive=True)

        # Go to base
        control.change_reference_pos(is_abs=True, x=base[0], y=base[1], z=desired_height[_i], arrive=True)

        #Go to low Z to centralize
        control.change_reference_pos(is_abs=True, z=1, arrive=True)

        if not control.center_at_base(centralize, descend_factor=-0.01):
            continue

        # Lower Z to find qrcode
        control.change_reference_pos(is_abs=True, z=0.55, arrive=True)
        control.change_reference_pos(is_abs=False, x=-0.01, y=0, z=0, arrive=True)
        rospy.sleep(1)
        # Read Qrcode

        isQr, qrData = recognizeQr(image, control)

        # QR code detected

        if isQr == True:
            rospy.loginfo('Qrcode encontrado')

            #rospy.loginfo(qrData)

            for qrValue in qrData:
                if qrValue[0] in delivered:
                    #rospy.loginfo('Repeated')
                    continue
                else:
                    address = qrValue[0]
                    rospy.loginfo('Qrcode value: %s', address)

            packDelivered = delivery(address, bases, centralize, control, attach, detach, image)

            delivered.append(packDelivered)


    ## GO BACK HOME

    # Fly at high Z
    control.change_reference_pos(is_abs=True, z=2.5, arrive=True)

    rospy.sleep(1)

    control.change_reference_pos(is_abs=True, x=1, y=-1, z=2)
    while(control.utils_arrived.arrived([1,-1,2]) == False):
        rospy.sleep(0.2)

    control.change_reference_pos(is_abs=True, x=1, y=-1, z=1.5)
    while(control.utils_arrived.arrived([1,-1,1.5]) == False):
        rospy.sleep(0.2)

    rospy.sleep(4)

    control.center_at_base(centralize)

    land()

    rospy.sleep(10)


if __name__ == '__main__':
    main()
