# This script will take as input the drone image and outputs the digit predictions. The digit predictions is divided into two parts: digit_upper and digit_down


from aux_functions_for_classifier import * # to import the Neural Network functions

#Imports from AK script


final_image_name = './rotate_display/images_from_simulator/4.png'
drone_image = cv2.imread(final_image_name)



#If you want to use voting system (which will be better)
drone_images = [] #This list will store all the images

final_image_name = './rotate_display/images_from_simulator/4.png'
drone_image_1 = cv2.imread(final_image_name)
drone_images.append(drone_image_1)


final_image_name = './rotate_display/images_from_simulator/4.png'
drone_image_2 = cv2.imread(final_image_name)
drone_images.append(drone_image_2)



final_image_name = './rotate_display/images_from_simulator/4.png'
drone_image_3 = cv2.imread(final_image_name)
drone_images.append(drone_image_3)



final_image_name = './rotate_display/images_from_simulator/4.png'
drone_image_4 = cv2.imread(final_image_name)
drone_images.append(drone_image_4)



final_image_name = './rotate_display/images_from_simulator/4.png'
drone_image_5 = cv2.imread(final_image_name)
drone_images.append(drone_image_5)


upper_digit, down_digit = mnist_classifier(drone_image)
upper_digit_by_voting, down_digit_by_voting = mnist_classifier_by_voting(drone_images)
print("upper digit by voting: %d" %upper_digit_by_voting)
print("down digit by voting: %d" %down_digit_by_voting)