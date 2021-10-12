'''
Test Phase 3.
'''
from util_interpret_digits import *

debug_on = True
plots_on = False

test_digit = 4 #use 1, 2 or 4
file_name = './images_from_simulator/' + str(test_digit) + '.png'
output_file_name = 'only_digit_' + str(test_digit) + '.png'

original_image = cv2.imread(file_name)
#image = process_image(original_image)

if False: #enable create empty image with some noise that should test False
    original_image = np.zeros( original_image.shape, dtype='uint8' )
    original_image[0:100] = 255

found = is_there_a_digit_display_in_this_image(original_image)
print('is_there_a_digit_display_in_this_image?', found)

print('End of processing')