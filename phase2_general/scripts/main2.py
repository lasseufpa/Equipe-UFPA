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

imgg = cv2.imread('image5.png')

img = binarize_image(imgg)

cv2.imshow("window_name", img)

cv2.waitKey(0) 
  
#closing all open windows 
cv2.destroyAllWindows()


any_display = is_there_a_digit_display_in_this_image(img)
print(any_display)

upper_number, down_number = display.mnist_classifier_by_voting(img)

print(upper_number)