'''
Utilities for digits phase

Code from https://blog.paperspace.com/data-augmentation-for-bounding-boxes/
and others

Oct. 31 - 2020 - AK
'''

import cv2
import numpy as np
import os
import random

#from https://stackoverflow.com/questions/3964681/find-all-files-in-a-directory-with-extension-txt-in-python
def get_files(path, extension):
    #path = os.getcwd()
    all_files_list = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(extension)]
    return all_files_list

def binarize_image(image):
    # pre-process the image by resizing it, converting it to
    # graycale, blurring it, and computing an edge map
    #image = imutils.resize(image, height=500)
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low_white_display = (0, 0, 131)
    high_white_display = (0, 0, 255)
    # threshold the warped image, then apply a series of morphological
    # operations to cleanup the thresholded image
    sure_display = cv2.inRange(HSV,low_white_display, high_white_display) #sensor green
    return sure_display

def write_all_object_name():
    '''
    Objects:
    display  - class 0
    percent  - class 1
    sinal_apenas (-) - class 2 ou 10
    sinal_e_um (-1) - class 3 ou 11
    sinal_zero (0, tudo apagado) - class 4 ou 00
    sinal_mais (1) - class 5 ou 01
    sup_esq_0 - class 6 + digit
    sup_esq_1
    sup_esq_9
    sup_dir_0 - class 16 + digit
    sup_dir_9
    inf_dir_0 - class 26 + digit
    inf_dir_9
    '''
    print('display')
    print('percent')
    print('sinal_apenas')
    print('sinal_e_um')
    print('sinal_zero')
    print('sinal_mais')
    for i in range(10):
        print('sup_esq_' + str(i))
    for i in range(10):
        print('sup_dir_' + str(i))
    for i in range(10):
        print('inf_dir_' + str(i))

#to generate the file, use > to redirect stdout
#write_all_object_name()
#exit(1)

'''
Format: https://github.com/AlexeyAB/Yolo_mark/issues/60
<object-class> - integer number of object from 0 to (classes-1)
<x> <y> <width> <height> - float values relative to width and height of image, it can be equal from (0.0 to 1.0]
for example: <x> = <absolute_x> / <image_width> or <height> = <absolute_height> / <image_height>
atention: <x> <y> - are center of rectangle (are not top-left corner)
'''
def convert_to_yolo_bounding_box_format(image_shape, bounding_boxes):
    if True:
        dw = 1. / image_shape[1]
        dh = 1. / image_shape[0]
    else: #AK: I think this is the correct convention for Yolo
        dw = 1. / image_shape[0]
        dh = 1. / image_shape[1]

    num_boxes, num_entries = bounding_boxes.shape
    if num_entries != 5:
        print('error. must be 5')
        exit(-1)
    output_information = np.zeros(bounding_boxes.shape)
    for ii in range(num_boxes):
        x = (bounding_boxes[ii,0] + bounding_boxes[ii,2]) / 2.0
        y = (bounding_boxes[ii,1] + bounding_boxes[ii,3]) / 2.0
        w = bounding_boxes[ii,2] - bounding_boxes[ii,0]
        h = bounding_boxes[ii,3] - bounding_boxes[ii,1]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        this_class = bounding_boxes[ii,4]
        if True:
            output_information[ii,] = np.array([this_class,x,y,w,h])
        else: #AK: I think this is the correct convention for Yolo
            output_information[ii,] = np.array([this_class,y,x,h,w])
    return output_information

def write_boxes_into_file(file_name, bounding_boxes):
    '''This is not using Yolo's syntax'''
    np.savetxt(file_name, bounding_boxes, fmt='%d')

def write_boxes_into_file_as_yolo(file_name, bounding_boxes):
    '''This is not using Yolo's syntax'''
    #from https://stackoverflow.com/questions/40030481/numpy-savetxt-save-one-column-as-int-and-the-rest-as-floats
    N=4 #4 float numbers
    np.savetxt(file_name, bounding_boxes, fmt=' '.join(['%i'] + ['%1.6f']*N))

def change_extension(file_name, new_extension):
    dir_name = os.path.dirname(file_name)
    file_name = os.path.basename(file_name)
    file_name = os.path.splitext(file_name)[0] #discard extension
    new_name = os.path.join(dir_name, file_name + '.' + new_extension)
    return new_name

def extract_classes_from_file_name(file_name):
    file_name = os.path.basename(file_name)
    file_name = os.path.splitext(file_name)[0] #discard extension
    num_classes = 5 #each "class" is a digit in the file name, e.g. 00107 in dig0_00107.png
    classes = np.zeros((num_classes,))
    for i in range(num_classes-1):
        classes[i] = int (file_name[-num_classes+i:-num_classes+i+1])
    classes[-1] = int (file_name[-1])
    return classes

#print (extract_class_from_file_name('ddk/dkdk/dig0_00107.png'))
#exit(-1)

# from https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv
def overlay_image(img, img_overlay, pos):
    """Overlay img_overlay on top of img_digit at the position specified by
    pos.
    """

    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    img[y1:y2, x1:x2, :] = img_overlay[y1o:y2o, x1o:x2o, :]

# from https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv
#the other version is for color images, this is for gray levels
def overlay_1_channel_image(img, img_overlay, pos):
    """Overlay img_overlay on top of img_digit at the position specified by
    pos.
    """

    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    img[y1:y2, x1:x2] = img_overlay[y1o:y2o, x1o:x2o]
