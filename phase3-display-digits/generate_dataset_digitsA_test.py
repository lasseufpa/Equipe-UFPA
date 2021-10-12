'''
This code is a version of dataset_task3_digits_black_white_noisy.py
used to generate a specific dataset.

Oct. 31 - 2020 - AK: I will check how to use this script again
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
num_epochs = 3 #number of runs over all digits. There are 3900 digits (combinations of digits), so if you choose 2, you end up with 2*3900 output images
should_show_images = False #use True to plot of False to run faster
should_add_noise = True #use to add noise

#define resolutions for:
num_pixels_digits = 209 #a) images of digits, e.g. 209 for 209 x 209 images - DO NOT CHANGE IT
num_pixels_base = 1900 #b) images of the base, e.g. 1900 for 1900 x 1900 images
#c) final (output) images, e.g. 680 x 480 images:
#num_pixels_final_x = 680 #width
num_pixels_final_x = 752 #width
num_pixels_final_y = 480 #height

random_center_shift = 80 #number of pixels the center of the image will be shifted, from - to +

#1) file with image of the base (image with a + in center). It needs to match num_pixels_base defined before
img_base_original = cv2.imread("./source_base_images/bases_1900x1900.png")[:, :, ::-1] #OpenCV uses BGR channels
#img_overlay = img_base_original
#img_shape = img_overlay.shape

#2) file with all possible display (digit) combinations
#path = os.getcwd()
#path = "D:/ak/Projects/Petrobras_Desafio/labels"
#path = "/mnt/d/ak/Projects/Petrobras_Desafio/labels"
path = "./source_digits_images/small_images"
all_digit_files_list = get_files(path, "png")
#print(all_digit_files_list)

#3) output folder that will contain the digits on top of the base, with data augmentation
output_folder = './digitsA_test/' #'./new_digits_yolo/' #end it with /
os.makedirs(output_folder, exist_ok=True)

total_number_of_noise_patches = 2
max_number_of_pixels_in_noise_patches = 40 # 10 for 10 x 10 patches
number_of_pixels_away_from_display = 50

#initialize seed to facilitate reproducing bugs 
random.seed(52) #change it when considering train and test


#from defined resolutions
top_left_x = int((num_pixels_base/2) - (num_pixels_digits/2))
top_left_y = top_left_x #symmetry

# See for the modeling of objects
# https://docs.google.com/document/d/1b1R3ncgDhGfIEpoF7hteZg4nThMeSrUQM2pqORqP4Ww/edit?ts=5daa0d1e#
num_objects = 36
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
…
sup_esq_9
sup_dir_0 - class 16 + digit
...
sup_dir_9
inf_dir_0 - class 26 + digit
…
inf_dir_9
'''
# there are num_objects but only 6 boxes. Two are always classes 0 and 1 while there are other 4 classes that change
num_bounding_boxes = 6
display_bounding_boxes = np.zeros(( num_bounding_boxes, 5), )

object_class = 0 #there is only one class here

#follow convention of augmentation API
#box for display object
display_bounding_boxes[0, 0] = top_left_x
display_bounding_boxes[0, 1] = top_left_y
display_bounding_boxes[0, 2] = top_left_x + num_pixels_digits
display_bounding_boxes[0, 3] = top_left_y + num_pixels_digits
display_bounding_boxes[0, 4] = 0 #class always 0

'''
Other boxes assuming 209 x 209 digit display images
topleft x, y, rightbottom x, y. I got the numbers below manually
a - 25 15 76 99
b - 80 15 130 99
c - 20 107 76 187
d - 80 107 130 187
e - 131 1 208 208
'''
if num_pixels_digits != 209:
    raise Exception("Sorry. num_pixels_digits is currently limited to be equals to 209")
display_bounding_boxes[1] = np.array([25+top_left_x, 15+top_left_y, 76+top_left_x, 99+top_left_y, -1]) #non-initialized class -1
display_bounding_boxes[2] = np.array([80+top_left_x, 15+top_left_y, 130+top_left_x, 99+top_left_y, -1]) #non-initialized class -1
display_bounding_boxes[3] = np.array([20+top_left_x, 107+top_left_y, 76+top_left_x, 187+top_left_y, -1]) #non-initialized class -1
display_bounding_boxes[4] = np.array([80+top_left_x, 107+top_left_y, 130+top_left_x, 187+top_left_y, -1]) #non-initialized class -1
display_bounding_boxes[5] = np.array([131+top_left_x, 1+top_left_y, 208+top_left_x, 208+top_left_y, 1]) #class always 1

#print(display_bounding_boxes)
#print(display_bounding_boxes.shape)

#AK: keep the translation as the last one
#transforms = Sequence([RandomScale((0.8,1.2), diff = True), RandomRotate(90), RandomHSV(hue = None, saturation = None, brightness = 1), RandomTranslate(0.1)])
first_transforms_set = Sequence([RandomScale((-0.7,-0.6)),RandomRotate(360)])

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

        #update boxes
        display_bounding_boxes[1][4] = a_class
        display_bounding_boxes[2][4] = b_class
        display_bounding_boxes[3][4] = c_class
        display_bounding_boxes[4][4] = d_class

        #need deep copy because augmentation API changes the arrays inside
        target_bounding_boxes = deepcopy(display_bounding_boxes)
        #will create large image and position the original target in its center
        target_expanded = deepcopy(img_base_original) #not sure I need deep copy

        pos = (top_left_x, top_left_y)
        #superimpose:
        overlay_image(target_expanded, img_digit, pos)

        if False: #should_show_images:
            plt.imshow(draw_rect(target_expanded, display_bounding_boxes))
            plt.show()

        #print('Bounding boxes before transformations=', target_bounding_boxes)

        #implement several transformations here
        final_image, final_bounding_boxes = first_transforms_set(target_expanded, target_bounding_boxes)
        if len(final_bounding_boxes) == 0:
            print('Skipping file', file_name, '. Invalid augmentation!')
            num_of_skipped_images += 1
            continue  #skip, invalid augmentation
        #print('Bounding boxes after transformations=', final_bounding_boxes)

        #extract final image by centering bounding box 
        center_x = int((final_bounding_boxes[0][0] + final_bounding_boxes[0][2])/2)
        center_y = int((final_bounding_boxes[0][1] + final_bounding_boxes[0][3])/2)            

        should_disturb_box_center = True

        if should_disturb_box_center:
            center_x += random.randint(-random_center_shift, random_center_shift)
            center_y += random.randint(-random_center_shift, random_center_shift)

        #print(center_x, center_y, 'centers')

        top_left_final_x = center_x- int( (num_pixels_final_x)/2 )
        top_left_final_y = center_y- int( (num_pixels_final_y)/2 )
        bottom_right_final_x = top_left_final_x + num_pixels_final_x
        bottom_right_final_y = top_left_final_y + num_pixels_final_y

        if False: #should_show_images:
            #cv2.imshow("just here", draw_rect(final_image, final_bounding_boxes))
            #cv2.waitKey(100)
            plt.imshow(draw_rect(final_image, final_bounding_boxes))
            plt.show()

        #black_background = np.zeros(final_image.shape).astype(np.uint8)

        final_image = final_image[top_left_final_y:bottom_right_final_y,top_left_final_x:bottom_right_final_x,:]

        #need deep copy because augmentation API changes the arrays inside
        #target_bounding_boxes = deepcopy(display_bounding_boxes)
        #will create large image and position the original target in its center
        
        #pos = (top_left_x, top_left_y)
        #superimpose:
        #overlay_image(black_background, final_image, pos)
        #if True: #should_show_images:
        #    plt.imshow(black_background)
        #    #plt.imshow(draw_rect(final_image, final_bounding_boxes))
        #    plt.show()




        if np.prod(final_image.shape) != (num_pixels_final_x*num_pixels_final_y*3):
        #if np.prod(final_image.shape) != (num_pixels_final_x*num_pixels_final_y*1): #use 1 because image is B&W now
            #the transformation generated an image smaller than expected
            print('Skipping file', file_name, '. Invalid augmentation!')
            num_of_skipped_images += 1
            continue  #skip, invalid augmentation

        final_image = binarize_image(final_image)

        #adjust bounding boxes
        for ii in range(num_bounding_boxes):
            final_bounding_boxes[ii][0] -= top_left_final_x
            final_bounding_boxes[ii][2] -= top_left_final_x
            final_bounding_boxes[ii][1] -= top_left_final_y
            final_bounding_boxes[ii][3] -= top_left_final_y

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


        output_file_name = output_folder + 'dig' + str(epoch) + '_' + os.path.basename(file_name)
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

        use_yolo_label_format = True

        if use_yolo_label_format: #enable to write pixel values in Yolo format or not
            boxes_file_name = change_extension(output_file_name, 'txt')
            final_bounding_boxes_as_yolo = convert_to_yolo_bounding_box_format(final_image.shape, final_bounding_boxes)
            if np.max(final_bounding_boxes_as_yolo[:,1:]) > 1:
                #cv2.waitKey(0)
                raise Exception('Yolo bounding box cannot be larger than 1!')
                #print(np.max(final_bounding_boxes_as_yolo[:,1:]))
            write_boxes_into_file_as_yolo(boxes_file_name, final_bounding_boxes_as_yolo)
        else:
            boxes_file_name = change_extension(output_file_name, 'txt')
            write_boxes_into_file(boxes_file_name, final_bounding_boxes)

        if should_show_images:
            #cv2.imshow("Final image", draw_rect(final_image, final_bounding_boxes))
            cv2.imshow("Final image", final_image)
            cv2.waitKey(100)
            #plt.imshow(draw_rect(final_image, final_bounding_boxes))
            #plt.draw()
            #plt.pause(0.05)



print("Finished processing.", num_of_written_images, "files were generated and ", num_of_skipped_images, 'were skipped')
if should_show_images:
    plt.show()
