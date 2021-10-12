'''
This code generates digits to enable networks like MNIST to be used.
AK.
'''

from data_aug.data_aug import *
from data_aug.bbox_util import *
import cv2
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy
import random
from util_digits import *

#Define key parameters
num_epochs = 1 #number of runs over all digits. There are 3900 digits (combinations of digits), so if you choose 2, you end up with 2*3900 output images
should_show_images = True #use True to plot of False to run faster
should_add_noise = False #use to add noise
should_disturb_box_center = False

#define resolutions for:
#
# num_pixels_digits = 209 #a) images of digits, e.g. 209 for 209 x 209 images - DO NOT CHANGE IT
num_pixels_digits_y = 84
num_pixels_digits_x = 50
#num_pixels_base = 1900 #b) images of the base, e.g. 1900 for 1900 x 1900 images
#c) final (output) images, e.g. 680 x 480 images:
#num_pixels_final_x = 680 #width
num_pixels_final_x = 120 #width
num_pixels_final_y = 120 #height

random_center_shift = 8 #number of pixels the center of the image will be shifted, from - to +

#2) file with all possible display (digit) combinations
path = "./mnist_digits"
all_digit_files_list = get_files(path, "png")
#print(all_digit_files_list)

#3) output folder that will contain the digits on top of the base, with data augmentation
output_folder = './digitsE_train/' #'./new_digits_yolo/' #end it with /
os.makedirs(output_folder, exist_ok=True)

total_number_of_noise_patches = 2
max_number_of_pixels_in_noise_patches = 40 # 10 for 10 x 10 patches
number_of_pixels_away_from_display = 50

#initialize seed to facilitate reproducing bugs 
random.seed(3696)

'''
Other boxes assuming 209 x 209 digit display images
topleft x, y, rightbottom x, y. I got the numbers below manually
a - 25 15 76 99
b - 80 15 130 99
c - 20 107 76 187
d - 80 107 130 187
e - 131 1 208 208
'''

N = len(all_digit_files_list)
num_of_written_images = 0
num_of_skipped_images = 0
for epoch in range(num_epochs): #a complete sweep over all digits
    for n in range(N): #go over all files
        file_name = all_digit_files_list[n]
        img_digit = cv2.imread(file_name)[:, :, ::-1] #OpenCV uses BGR channels

        print('#', num_of_written_images, '/', (num_of_skipped_images+num_of_written_images), file_name)

        if len(img_digit.shape) < 3:
            raise Exception('I am assuming a color image with 3 channels')

        five_classes = extract_classes_from_file_name(file_name) #each "class" is a digit in the file name, e.g. 00107 in dig0_00107.png
        a_class = 6 + five_classes[0]
        b_class = 16 + five_classes[1]
        d_class = 26 + five_classes[4]
        if five_classes[2]==1 and five_classes[3]==0:
            c_class = 2 #sinal_apenas (-) - class 2 ou 10
        elif five_classes[2]==1 and five_classes[3]==1:
            c_class = 3 #sinal_e_um (-1) - class 3 ou 11
        elif five_classes[2]==0 and five_classes[3]==0:
            c_class = 4 #sinal_zero (0, tudo apagado) - class 4 ou 00
        elif five_classes[2]==0 and five_classes[3]==1:
            c_class = 5 #sinal_mais (1) - class 5 ou 01
        else:
            raise Exception('Error in parsing or file name:', file_name, '=>', five_classes)

        '''
        Other boxes assuming 209 x 209 digit display images
        topleft x, y, rightbottom x, y. I got the numbers below manually
        a - 25 15 76 99
        b - 80 15 130 99
        c - 20 107 76 187
        d - 80 107 130 187
        e - 131 1 208 208
        '''
        #img_single_digit = np.zeros( img_digit.shape, dtype=int)
        digit_indices = np.array([15 , 25,  99, 75])
        img_single_digit = img_digit[digit_indices[0]:digit_indices[2], digit_indices[1]:digit_indices[3],:]
        #print(digit_indices)
        
        #print('aaa',np.unique(img_single_digit))
        #img_single_digit = img_digit[digit_indices[0]:digit_indices[2], digit_indices[1]:digit_indices[3]]
        if False: #should_show_images:
            plt.imshow(img_single_digit) #img_single_digit)
            #plt.imshow(img_digit) #)
            plt.show()

        #print(img_single_digit.shape)
        binary_img_single_digit = binarize_image(img_single_digit)
        #print(binary_img_single_digit.shape)
        #print('aaa',np.unique(final_image))        
        #print('bbb',np.unique(img_single_digit))        

        if False: #should_show_images:
            #plt.imshow(draw_rect(final_image, target_bounding_boxes))
            plt.imshow(binary_img_single_digit)
            plt.show()

        #print('Bounding boxes before transformations=', target_bounding_boxes)
        final_image = np.zeros ( (num_pixels_final_y, num_pixels_final_x), dtype=int)

        center_x = int(num_pixels_final_x/2)
        center_y = int(num_pixels_final_y/2)

        if should_disturb_box_center:
            center_x += random.randint(-random_center_shift, random_center_shift)
            center_y += random.randint(-random_center_shift, random_center_shift)

        #print(center_x, center_y, 'centers')

        top_left_final_x = center_x- int( num_pixels_digits_x/2 )
        top_left_final_y = center_y- int( num_pixels_digits_y/2 )
        bottom_right_final_x = top_left_final_x + num_pixels_digits_x
        bottom_right_final_y = top_left_final_y + num_pixels_digits_y

        final_image[top_left_final_y:bottom_right_final_y,top_left_final_x:bottom_right_final_x] = binary_img_single_digit

        #final_image = binarize_image(final_image)

        #add noise
        if should_add_noise:
            center_x = int (final_image.shape[0]/2)
            center_y = int (final_image.shape[1]/2)
            for ii in range(total_number_of_noise_patches):
                noise_probability = np.random.uniform(low=0.1, high=1.0, size=1)
                patch_size = np.random.randint(2, max_number_of_pixels_in_noise_patches+1)
                mask = np.random.uniform(low=0.0, high=1.0, size=(patch_size,patch_size)) < noise_probability
                mask = 255 * mask #AK-TODO the images should be 0 or 1, instead of 0 or 255
                patch = mask.astype(np.uint8)
                if np.random.uniform(low=0.0, high=1.0, size=1) > 0.5:
                    pos_noise_x = center_x + random.randint(number_of_pixels_away_from_display, final_image.shape[0] - center_x)
                else:
                    pos_noise_x = random.randint(0, center_x - number_of_pixels_away_from_display)
                if np.random.uniform(low=0.0, high=1.0, size=1) > 0.5:
                    pos_noise_y = center_y + random.randint(number_of_pixels_away_from_display, final_image.shape[1] - center_y)
                else:
                    pos_noise_y = random.randint(0, center_y - number_of_pixels_away_from_display)
                #print((pos_noise_x, pos_noise_y))
                #print(patch)
                #print(np.max(final_image))
                overlay_1_channel_image(final_image, patch, (pos_noise_x, pos_noise_y))

        #save image with boxes
        if False:
            final_image = draw_rect(final_image, final_bounding_boxes)

        final_image = final_image.astype(np.uint8)

        output_file_name = output_folder + str(int(five_classes[0])) + '_' + str(num_of_written_images) + '.png'
        if final_image.shape[1] == num_pixels_final_x and final_image.shape[0] == num_pixels_final_y:
            #cv2.imwrite(output_file_name, final_image[:, :, ::-1]) #[:, :, ::-1] #OpenCV uses BGR channels
            #make uint8 (could not use binary) image
            #https://stackoverflow.com/questions/44587613/how-to-save-a-binary-imagewith-dtype-bool-using-cv2
            #final_image = final_image / 255
            #print(np.unique(final_image))
            #cv2.imwrite(output_file_name, final_image,'uint8')
            cv2.imwrite(output_file_name, final_image)
            print('Wrote', output_file_name)
            num_of_written_images += 1
        else:
            print('Invalid', output_file_name)
            num_of_skipped_images += 1
            #exit(-1)

        if should_show_images:
            #cv2.imshow("Final image", draw_rect(final_image, final_bounding_boxes))
            #print(np.unique(final_image))
            #print('ss',final_image.shape)
            cv2.imshow("Final image", final_image)
            cv2.waitKey(100)
            #plt.imshow(draw_rect(final_image, final_bounding_boxes))
            #plt.draw()
            #plt.pause(0.05)



print("Finished processing.", num_of_written_images, "files were generated and ", num_of_skipped_images, 'were skipped')
if should_show_images:
    plt.show()
