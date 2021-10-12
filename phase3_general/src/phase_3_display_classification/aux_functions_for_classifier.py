import torch
import cv2
import glob, os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
np.random.seed(130)

#Import from AK
from .data_aug.data_aug import *
from .data_aug.bbox_util import *
import cv2
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy
import random
from .util_digits import *
from .rotate_display.util_interpret_digits import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(24500 , 50) 
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 24500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def digit_classifier(image):
    net = Net()
    # READ IMAGES
    #filename_a = "./digit_img/digit_a.png"
    #img_a = cv2.imread(filename_a,0)

    #filename_b = "./digit_img/digit_b.png"
    #img_b = cv2.imread(filename_b,0)

    #filename_sign = "./digit_img/digit_sign.png"
    #img_sign = cv2.imread(filename_sign,0)

    #filename_d = "./digit_img/digit_d.png"
    #img_d = cv2.imread(filename_d,0)
    #Run AK code to detect the display and cut the digit into four regiions    

    # READ CLASSIFIERS
    classifier_digit_a = Net()
    classifier_digit_a.load_state_dict(torch.load('./classifiers/classifier_digits_a.pth'))
    classifier_digit_a.eval()

    classifier_digit_b = Net()
    classifier_digit_b.load_state_dict(torch.load('./phase_3_display_classification/classifiers/classifier_digits_b.pth'))
    classifier_digit_b.eval()


    classifier_digit_sign = Net()
    classifier_digit_sign.load_state_dict(torch.load('./classifiers/classifier_digits_sign.pth'))
    classifier_digit_sign.eval()

    classifier_digit_d = Net()
    classifier_digit_d.load_state_dict(torch.load('./classifiers/classifier_digits_d.pth'))
    classifier_digit_d.eval()


    images_tensors = torch.tensor(image[1])
    images_exp = images_tensors.unsqueeze_(0)
    images_exp = images_tensors.unsqueeze_(0)

    output_a =  classifier_digit_a(images_exp.float())
    pred_a = output_a.data.max(1, keepdim=True)[1]

    #print('digit_superior_esquerdo: ', pred_a.item())


    images_tensors = torch.tensor(image[2])
    images_exp = images_tensors.unsqueeze_(0)
    images_exp = images_tensors.unsqueeze_(0)

    output_b =  classifier_digit_b(images_exp.float())
    pred_b = output_b.data.max(1, keepdim=True)[1]
    #print('digit_superior_direito: ', pred_b.item())

    digit_upper = pred_a.item()*10 + pred_b.item()
    


    images_tensors = torch.tensor(image[3])
    images_exp = images_tensors.unsqueeze_(0)
    images_exp = images_tensors.unsqueeze_(0)

    output_sign =  classifier_digit_sign(images_exp.float())
    pred_sign = output_sign.data.max(1, keepdim=True)[1]
    #print('digito_inferior_esquerdo: ', pred_sign.item())


    images_tensors = torch.tensor(image[4])
    images_exp = images_tensors.unsqueeze_(0)
    images_exp = images_tensors.unsqueeze_(0)

    output_d =  classifier_digit_d(images_exp.float())
    pred_d = output_d.data.max(1, keepdim=True)[1]


    if pred_sign == 0:
        digit_down  = 0*10 + pred_d.item()
    elif pred_sign == 1:
        digit_down  = pred_sign.item()*10 + pred_d.item()
    elif pred_sign == 2:
        digit_down = (1*10 + pred_d.item())*(-1)
    elif pred_sign == 3:
        digit_down = -pred_d.item()


    #print('digito_inferior_direito: ', pred_d.item())
    #print("digit upper: %d"% digit_upper)
    #print("digit down: %d"% digit_down)
    #print("=========================")
    return digit_upper, digit_down





debug_flavio_part = False # From flavio script. If true, it will generate several plots written by flavio
def mnist_classifier(final_image):


    #From AK Script
    #template_file = './rotate_display/template_1.png' #template_1.png is the largest
    template_file = './rotate_display/template_1.png' #template_1.png is the largest
    percent_template = cv2.imread(template_file,0) #keep it in memory to speedup
    #These are the important digits for training the MNIST DNNs
    num_pixels_output_image = 290 #this image will not be used, has all display
    num_pixels_each_digit = 154 #this image is the input image to the NNs



    #Final image will be drone image
    cropped_binary_image, cropped_image = step_all_from_camera_to_rotated_display(final_image, percent_template)

    all_images = extract_digits_from_display(cropped_binary_image,num_pixels_output_image,num_pixels_each_digit)

    if debug_flavio_part:
        cv2.imshow('all_images_1',all_images[1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        cv2.imshow('all_images_2',all_images[2])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow('all_images_3',all_images[3])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        cv2.imshow('all_images_4',all_images[4])
        cv2.waitKey(0)
        cv2.destroyAllWindows()



        cv2.imshow('cropped_binary_image',cropped_binary_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow('cropped_image',cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow('final_image',final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    upper_digit, down_digit =  digit_classifier(all_images)   
    return upper_digit, down_digit


def mnist_classifier_by_voting(drone_images):
    upper_digits = [] #To store all the upper digits predictions
    down_digits = [] #To store all the down digits predictions
    for image in drone_images:
       upper_digit, down_digit = mnist_classifier(image) # Do the prediction in the image
       upper_digits.append(upper_digit) #store the upper digits predictions
       down_digits.append(down_digit) #store the down predictions
    #After all the prediction, we just need to select the one who repeats more
    #Convert to numpy because it is easier to work on

    upper_digits_numpy = np.array(upper_digits) #Convert the list from numpy because it's easier to work on
    down_digits_numpy =  np.array(down_digits) #Convert the list from numpy because it's easier to work on


    #Get all the unique values (if we have a list with [52,52,32,32], the output will be only [52,32])
    upper_digits_unique_numpy = np.unique(np.array(upper_digits)) 
    down_digits_unique_numpy = np.unique(np.array(down_digits))
    
    upper_digits_votes = []
    down_digits_votes = []
    #Select the most voted digits!
    for votes in upper_digits_unique_numpy:
        upper_digits_votes.append(upper_digits_numpy[upper_digits_numpy == votes].shape[0])
        
    for votes in down_digits_unique_numpy:
        down_digits_votes.append(down_digits_numpy[down_digits_numpy == votes].shape[0])
    
    #select the most voted
    indice_of_the_most_voted_upper_digit = np.argmax(upper_digits_votes)
    indice_of_the_most_voted_down_digit = np.argmax(down_digits_votes)
    upper_digit = upper_digits_unique_numpy[indice_of_the_most_voted_upper_digit]
    down_digit =  down_digits_unique_numpy[indice_of_the_most_voted_down_digit]
    return upper_digit, down_digit
