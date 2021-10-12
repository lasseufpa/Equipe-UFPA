'''
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html

Seems good: cv2.TM_CCOEFF
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt
from util_interpret_digits import *

template_file = 'template_1.png' #template_1.png is the largest
percent_template = cv2.imread(template_file,0) #keep it in memory to speedup

#file_name = 'D:/aksimul/petrobras2020/digits_yolo/dig0_00003.png'
#template_file = 'D:/aksimul/petrobras2020/preprocess_digits/template_2.png'
#file_name = './images_from_simulator/1.png'
file_name = 'only_digit_2.png'
template_file = 'template_1.png' #template_1.png is the largest

img = cv2.imread(file_name,0)
img2 = img.copy()
template = cv2.imread(template_file,0)
w, h = template.shape[::-1]
#print('a',np.unique(img), np.unique(template))

template = resize_template(img, template)

#cv2.imshow('ss',template)
#cv2.waitKey(0)

img = binarize_image(img)
template = binarize_image(template)

angles = np.array([0,90,-90,180])
angle_max = get_best_rotation_angle(img, template, angles)

if angle_max != 0:
    rotated_image = rotate_bound(img, angle_max)
else:
    rotated_image = img

cv2.imshow('final',rotated_image)
cv2.waitKey(0)

print(img.shape, template.shape)
print('b',np.unique(img), np.unique(template))

# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img,top_left, bottom_right, 255, 2)

    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()