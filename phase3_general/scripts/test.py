import cv2
import phase_3_display_classification.aux_functions_for_classifier as display
from phase_3_display_classification.rotate_display.util_interpret_digits import is_there_a_digit_display_in_this_image
from phase3_general.preprocessing import binarize_image

image = cv2.imread('../data/22.png')
img = binarize_image(image)
any_display = is_there_a_digit_display_in_this_image(img)

print (any_display)