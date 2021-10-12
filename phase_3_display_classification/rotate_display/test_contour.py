'''
Test Phase 3.
'''
from util_interpret_digits import *

debug_on = True
plots_on = False

#file_name = '../digitsA_train/dig0_00005.png'
#file_name = 'dig0_00005.png'
test_digit = 2
file_name = './images_from_simulator/' + str(test_digit) + '.png'
output_file_name = 'only_digit_' + str(test_digit) + '.png'

original_image = cv2.imread(file_name)
#image = process_image(original_image)

image = step_1_find_largest_display_candidate(original_image)
if plots_on:
    cv2.imshow('3', image) 
    cv2.waitKey(0) 

binary_image = binarize_image(image)
if plots_on:
    cv2.imshow('6', binary_image) 
    cv2.waitKey(0) 

rectangle = step_2_find_display_bounding_box(binary_image, image)
#print('rectangle', rectangle)
angle = rectangle[2]
#print('angle', rectangle[2])
#print('Extrema:', extLeft, extRight, extTop, extBot)
if False:
    image2 = deepcopy(image)
    box = cv2.boxPoints(rectangle)
    box = np.int0(box)
    cv2.drawContours(image2,[box],0,(0,0,255),2)
    cv2.imshow('display found via contours', image2)
    cv2.waitKey(0)

cropped_binary_image, cropped_image = step_3_rotate_display(binary_image, rectangle, image)
if True:
    cv2.imshow('rotated_binary_image', cropped_binary_image) 
    cv2.waitKey(0) 

cv2.imwrite(output_file_name,cropped_image)

print('End of processing')