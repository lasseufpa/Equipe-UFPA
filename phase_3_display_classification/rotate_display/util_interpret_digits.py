'''
Some utility functions to detect digits in Phase 3 Petrobras

Refs:
https://docs.opencv.org/3.4/d9/d8b/tutorial_py_contours_hierarchy.html
https://github.com/gabrielburnworth/SSD-Reader/blob/master/SSD_Reader.py
https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0
https://answers.opencv.org/question/215531/python-cv2drawcontours-function-is-not-drawing-contours-i-dont-have-any-error-in-the-program-i-need-help/

https://stackoverflow.com/questions/50432349/combine-contours-vertically-and-get-convex-hull-opencv-python
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy

is_debug_enabled = False
show_plots_to_debug = False

#Key parameter for extracting white pixels in HSV color space
#This was setup by Ailton:
low_white_display = (0, 0, 131)
high_white_display = (0, 0, 255)

#Parameters for morphology
blur_block_size = 25 # odd
threshold_block_size = 31 #31 # odd
threshold_constant = 3
threshold = 110 # region average pixel value segment detection limit
morph_block_size = 8

def binarize_image(image):
    if len(image.shape) == 2:
        #in case of 2D array (gray image), convert to BGR
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    # pre-process the image by resizing it, converting it to
    # graycale, blurring it, and computing an edge map
    #image = imutils.resize(image, height=500)
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # threshold the warped image, then apply a series of morphological
    # operations to cleanup the thresholded image
    sure_display = cv2.inRange(HSV,low_white_display, high_white_display) #sensor green
    return sure_display

#Ailton's routine
def noise_filter(mask_x,kernel):
    #kernel = np.ones((2,2),np.uint8)
    opening = cv2.morphologyEx(mask_x,cv2.MORPH_OPEN,kernel, iterations = 1)
    # sure background area
    dilate = cv2.dilate(opening,kernel,iterations = 3)
    sure_bg = cv2.erode(dilate,kernel,iterations = 1)
    return(sure_bg)
    #return(opening)    

def plot_digit_images(all_images):
    #first image is the full one
    for i in range(4):
        plt.subplot(3,2,i+1),plt.imshow(all_images[i+1],cmap = 'gray')
        plt.xticks([]), plt.yticks([])

    plt.subplot(3,2,5),plt.imshow(all_images[0],cmap = 'gray')
    plt.title('Original image'), plt.xticks([]), plt.yticks([])
    plt.show()

def expand_image_to_given_size(image, num_output_pixels):
    y = image.shape[0]
    x = image.shape[1]
    if y > num_output_pixels or x > num_output_pixels:
        raise Exception('increase num_output_pixels to ' + str(x) + ' or ' + str(y))
    new_image = np.zeros( (num_output_pixels, num_output_pixels), dtype='uint8')
    new_image[0:y,0:x] = image
    return new_image

'''
# (x,y) top-left  (x,y) bottom-right
'''
def crop_image(image, indices):
    #print(len(indices))    
    tolerance = 8  #importante, extra pixels we add to capture the digit
    y = image.shape[0]
    x = image.shape[1]
    top_left_x = indices[0]
    top_left_y = indices[1]
    bottom_right_x = indices[2]
    bottom_right_y = indices[3]
    min_y = np.maximum(0,top_left_y-tolerance)
    max_y = np.minimum(y-1,bottom_right_y+tolerance)
    min_x = np.maximum(0,top_left_x-tolerance)
    max_x = np.minimum(x-1,bottom_right_x+tolerance)
    new_image = image[min_y:max_y,min_x:max_x]
    return deepcopy(new_image)

'''
#returns 5 images in a list: resized one with num_pixels_output_image pixels
# and 4 images with "digits" each with num_pixels_each_digit pixels
'''
def extract_digits_from_display(image, num_pixels_output_image, num_pixels_each_digit):
    #returns 5 images in a list: resized one and digits
    # Sequence is resized one, a, b, c, d
    all_images = list()
    image_h, image_w = image.shape
    num_pixels = num_pixels_output_image
    half_num_pixels = int (num_pixels/2)
    #I took below from observation
    first_digits_end_x = int ((105.0/290.0)*num_pixels)
    second_digits_end_x = int ((195.0/290.0)*num_pixels)

    if image_h != image_w or num_pixels != image_w:
        resized_image = cv2.resize(image, (num_pixels,num_pixels), interpolation = cv2.INTER_CUBIC)
    else:
        resized_image =image
    all_images.append(resized_image)

    #cv2.imwrite('sss.png', resized_image)

    # (x,y) top-left  (x,y) bottom-right
    #a = np.array([25, 15, 76, 99])
    
    #all_indices = np.array( (4,4), dtype='int')
    #a = np.array([0, 0, half_num_pixels, first_digits_end_x])
    #b = np.array([first_digits_end_x, second_digits_end_x, half_num_pixels])
    #c = np.array([0, half_num_pixels, first_digits_end_x, num_pixels-1])
    #d = np.array([first_digits_end_x, half_num_pixels, second_digits_end_x, num_pixels-1])
    #e = np.array([131, 1, 208, 208])

    # (x,y) top-left  (x,y) bottom-right
    # Sequence is a, b, c, d
    all_indices = np.zeros( (4,4), dtype=int)
    all_indices[0] = [0, 0, first_digits_end_x, half_num_pixels]
    all_indices[1] = [first_digits_end_x, 0, second_digits_end_x, half_num_pixels]
    all_indices[2] = [0, half_num_pixels, first_digits_end_x, num_pixels-1]
    all_indices[3] = [first_digits_end_x, half_num_pixels, second_digits_end_x, num_pixels-1]

    for i in range(4):
        cropped_image = crop_image(resized_image, all_indices[i])
        #print('cropped_image', cropped_image.shape)
        digit_image = expand_image_to_given_size(cropped_image, num_pixels_each_digit)
        all_images.append(digit_image)

    return all_images

#this uses only OpenCV.
def get_best_rotation_angle(image, template, angles):
    #show_plots_to_debug = True #debug
    tolerance = 0 #threshold for telling the percent is at the right
    value_tolerance = 0.1 #threshold, based on distortion
    top_left_tolerance = 2 #number of pixels
    N = len(angles)
    w, h = template.shape[::-1]
    values = np.zeros( (N,) )
    top_lefts_x = np.zeros( (N,) )
    #is_at_right = np.zeros( (N,) )
    for i in range(N):
        angle = angles[i]
        #cv2.imshow('ss',image)
        #cv2.waitKey(0)
        rotated_image = rotate_bound(image, angle)

        # Apply template Matching
        res = cv2.matchTemplate(rotated_image,template,cv2.TM_CCOEFF_NORMED)

        # avoid false negatives by imposing heavy distortions at the left side
        half_results_x = int(res.shape[1]/2) - tolerance
        res[:,0:half_results_x] = -1e30

        #res = cv2.matchTemplate(rotated_image,template,cv2.TM_SQDIFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        values[i] = max_val
        top_lefts_x[i] = top_left[0]
        #print('angle', angle, 'top_left', top_left, 'max_val', max_val)        
        #if top_left[0] > (image.shape[1] / 2):
        #    is_at_right[i] = 1
        if show_plots_to_debug:
            #print('angle',angle)
            #print('is_at_right[i]',is_at_right[i])
            #print('template.shape=',template.shape)
            new_rotated_image = deepcopy(rotated_image)
            cv2.rectangle(new_rotated_image,top_left, bottom_right, 255, 2)
            #cv2.imshow('ss',new_rotated_image)
            #cv2.waitKey(0)
            plt.subplot(131),plt.imshow(res,cmap = 'gray')
            plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            plt.subplot(132),plt.imshow(new_rotated_image,cmap = 'gray')
            plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
            plt.subplot(133),plt.imshow(template,cmap = 'gray')
            plt.title('Template'), plt.xticks([]), plt.yticks([])
            plt.show()

    max_value = np.max(values)
    #valid_indices = is_at_right > 0
    valid_indices = values >= (max_value - value_tolerance)
    valid_values = values[valid_indices]
    valid_angles = angles[valid_indices]
    valid_top_lefts_x = top_lefts_x[valid_indices]
    if valid_indices.any():
        if False: #strategy based on maximum value
            index_best = np.argmax(valid_values)
        else: #strategy based on region most to the right
            index_best = np.argmax(valid_top_lefts_x)
            index_max = np.argmax(valid_values)
            if index_max != index_best: #check if they are very close, and then choose max value
                if np.abs(valid_top_lefts_x[index_max] - valid_top_lefts_x[index_best]) <= top_left_tolerance:
                   index_best = index_max
        angle_best = valid_angles[index_best]
    else:
        angle_best = 0
        print('WARNING: template matching could not find percent %')
    if is_debug_enabled:
        print(values)
        print('angle_best', angle_best)
        print('top_lefts_x',top_lefts_x)
        print('valid_top_lefts_x',valid_top_lefts_x)
    return angle_best

def resize_template(image, template):
    decrease_in_size = 10 # num of pixels the template should be smaller than image
    h=image.shape[0]
    w=image.shape[1]
    target_h = h - decrease_in_size

    h_template = template.shape[0]
    w_template = template.shape[1]
    if h_template != h:
        resizing_factor = target_h / h_template
        target_w = int( w_template* resizing_factor + 0.5 )
        new_template = cv2.resize(template, (target_w,target_h), interpolation = cv2.INTER_AREA)
        return new_template
    else:
        return deepcopy(template)

#from https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
def rotate_bound(image, angle):
    #supports negative and positive angles, no need for:
    #if angle < 0:
    #    angle += 360
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

# determine the most extreme points along the contour
# extLeft, extRight, extTop, extBot = get_extrema_of_contour(c):
def get_extrema_of_contour(c):
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    return extLeft, extRight, extTop, extBot

# TODO Ailton diz para usar morphologia OPEN no inicio
def process_image_with_morphology(image):
    #thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
    #      np.ones((morph_block_size, morph_block_size), np.uint8))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_block_size, blur_block_size), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 
          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
          threshold_block_size, threshold_constant)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE,
    #thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE,
          np.ones((morph_block_size, morph_block_size), np.uint8))
    #invert
    thresh = 255 - thresh
    return thresh

def extract_region_given_contour(image, cnt):
    x,y,w,h = cv2.boundingRect(cnt)
    if show_plots_to_debug:
        image2 = deepcopy(image)
        cv2.rectangle(image2,(x,y),(x+w,y+h),(0,255,0),2)    
        cv2.imshow('2', image2) 
        cv2.waitKey(0)
    return image[y:y+h,x:x+w,:]

def find_contour_of_maximum_area(contours):
    max_area = -1
    max_area_index = -1
    for index, c in enumerate(contours):
        area = cv2.contourArea(c)
        if is_debug_enabled:
            print(index, 'area=', area)
        if area > max_area:
            max_area = area
            max_area_index = index
        index += 1
    c = contours[max_area_index]
    return c

def create_contour(extLeft, extRight, extTop, extBot):
    c = np.zeros( (4,1,2), dtype='int32')
    c[0,0] = extLeft
    c[1,0] = extTop
    c[2,0] = extRight
    c[3,0] = extBot
    return c

def get_display_contour(image, extLeft, extRight, extTop, extBot):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    if is_debug_enabled:
        print('# contours = ', len(contours))
    c = find_contour_of_maximum_area(contours)
    #print(c)
    print(c.shape)
    print(get_extrema_of_contour(c))
    print(extLeft, extRight, extTop, extBot)

def write_line(image, extLeft, extRight, extTop, extBot):
    line_thickness = 2
    cv2.line(image, extLeft, extTop, (0, 255, 0), thickness=line_thickness)
    cv2.line(image, extTop, extRight, (0, 255, 0), thickness=line_thickness)
    cv2.line(image, extRight, extBot, (0, 255, 0), thickness=line_thickness)
    cv2.line(image, extBot, extLeft, (0, 255, 0), thickness=line_thickness)

def step_1_find_largest_display_candidate(original_image):
    image = process_image_with_morphology(original_image)
    if show_plots_to_debug:
        cv2.imshow('1', image) 
        cv2.waitKey(0) 
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    if is_debug_enabled:
        print('# contours = ', len(contours))
    c = find_contour_of_maximum_area(contours)
    region = extract_region_given_contour(original_image, c)
    return region

def step_2_find_display_bounding_box(binary_image, image):
    # Finding Contours 
    # Use a copy of the image e.g. edged.copy() 
    # since findContours alters the image 
    #https://docs.opencv.org/3.4/d9/d8b/tutorial_py_contours_hierarchy.html
    #contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) 
    #contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) 

    #show_plots_to_debug = True #to debug
    if show_plots_to_debug:
        image2 = deepcopy(image)
        cv2.drawContours(image2, contours, -1, (255,0,0), 2)
        cv2.imshow('step_2_find_display_bounding_box', image2)
        cv2.waitKey(0)

    if is_debug_enabled:
        print('# contours = ', len(contours))
    #find total number of points in all contours
    num_points = 0
    for c in contours:
        num_points += c.shape[0]
        if c.shape[1] != 1:
            raise Exception('sifu1')
        if c.shape[2] != 2:
            raise Exception('sifu2')
    all_points = np.zeros( (num_points, 1, 2), dtype=int) #stores all points, from all contours
    index = 0
    for c in contours:
        for ii in range(c.shape[0]):
            all_points[index,:,:] = c[ii,:,:]
            index += 1 
    #print(all_points)

    if show_plots_to_debug:
        image2 = deepcopy(image)
        cv2.drawContours(image2, all_points, -1, (255,0,0), 2)
        cv2.imshow('drawContours', image2)
        cv2.waitKey(0)

    #from Section 7.b at https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
    #Here, bounding rectangle is drawn with minimum area, so it considers the rotation also. 
    #The function used is cv.minAreaRect(). It returns a Box2D structure which contains following
    #details - ( center (x,y), (width, height), angle of rotation ). But to draw this rectangle,
    #we need 4 corners of the rectangle. It is obtained by the function cv.boxPoints()
    rect = cv2.minAreaRect(all_points)
    return rect

def step_3_rotate_display(binary_image, rectangle, image, percent_template):
    #from Section 7.b at https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
    #Here, bounding rectangle is drawn with minimum area, so it considers the rotation also. 
    #The function used is cv.minAreaRect(). It returns a Box2D structure which contains following
    #details - ( center (x,y), (width, height), angle of rotation ). But to draw this rectangle,
    #we need 4 corners of the rectangle. It is obtained by the function cv.boxPoints()
    angle = rectangle[2]
    #the rectangle is rotated by angle, so we rotate the images by -angle
    rotated_binary_image = rotate_bound(binary_image, -angle)
    rotated_image = rotate_bound(image, -angle)
    if show_plots_to_debug:
        cv2.imshow('after 1st rotation', rotated_image)
        cv2.waitKey()
        cv2.imshow('after 1st rotation binary', rotated_binary_image)
        cv2.waitKey()

    new_rectangle = step_2_find_display_bounding_box(rotated_binary_image, rotated_image)
    angle = new_rectangle[2]
    if angle != 0 and angle != -90 and angle != 90:
        print("WARNING: angle should be 0, -90 or 90 degrees, but it is ", angle)
    #convert to a single box:
    box = cv2.boxPoints(new_rectangle)
    box = np.int0(box)
    y_min = np.min(box[:,0])
    x_min = np.min(box[:,1])
    y_max = np.max(box[:,0])
    x_max = np.max(box[:,1])

    #make sure image is square: width = height
    diff_y = y_max - y_min
    diff_x = x_max - x_min
    if diff_x != diff_y:
        both_diff = np.maximum(diff_x, diff_y)
        if diff_y > diff_x:
            x_max = x_min + both_diff
        else:
            y_max = y_min + both_diff

    #crop images
    cropped_binary_image = rotated_binary_image[y_min:y_max, x_min:x_max]
    cropped_image = rotated_image[y_min:y_max, x_min:x_max]

    #rotate if necessary, but assuming the square has been aligned with x and y axes
    #and consider only the angles: 0,90,-90,180
    angles = np.array([0,90,-90,180])
    resized_percent_template = resize_template(cropped_binary_image, percent_template)
    angle_max = get_best_rotation_angle(cropped_binary_image, resized_percent_template, angles)

    if angle_max != 0:
        cropped_binary_image = rotate_bound(cropped_binary_image, angle_max)
        cropped_image = rotate_bound(cropped_image, angle_max)

    return cropped_binary_image, cropped_image

#implements all steps
def step_all_from_camera_to_rotated_display(original_image, percent_template):
    image = step_1_find_largest_display_candidate(original_image)
    if show_plots_to_debug:
        cv2.imshow('step_all_1', image) 
        cv2.waitKey(0) 
    binary_image = binarize_image(image)
    if show_plots_to_debug:
        cv2.imshow('step_all_2', binary_image) 
        cv2.waitKey(0) 
    rectangle = step_2_find_display_bounding_box(binary_image, image)
    if show_plots_to_debug:
        image2 = deepcopy(image)
        box = cv2.boxPoints(rectangle)
        box = np.int0(box)
        cv2.drawContours(image2,[box],0,(0,0,255),2)
        cv2.imshow('step_all_display found via contours', image2)
        cv2.waitKey(0)
    cropped_binary_image, cropped_image = step_3_rotate_display(binary_image, rectangle, image, percent_template)
    if show_plots_to_debug:
        cv2.imshow('step_all_rotated_binary_image', cropped_binary_image) 
        cv2.waitKey(0) 
    return cropped_binary_image, cropped_image

def is_there_a_digit_display_in_this_image(original_image):
    #from step_1_find_largest_display_candidate(original_image):
    threshold_max_area = 450 #30 x 30 should be 900
    threshold_distance = np.minimum(original_image.shape[0],original_image.shape[1])/3

    image = process_image_with_morphology(original_image)
    if show_plots_to_debug:
        cv2.imshow('1', image) 
        cv2.waitKey(0) 
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    if is_debug_enabled:
        print('# contours = ', len(contours))
    if len(contours) == 0:
        #nothing found
        return False
    cnt = find_contour_of_maximum_area(contours)
    max_area = cv2.contourArea(cnt)
    #from region = extract_region_given_contour(original_image, c)
    x,y,w,h = cv2.boundingRect(cnt) #I could use center of mass too
    center_y, center_x = original_image.shape[0:2]
    center_y /= 2
    center_x /= 2
    distance = np.sqrt( (center_x-x)**2 + (center_y-y)**2 )
    #print(threshold_distance, distance)
    if show_plots_to_debug:
        image2 = deepcopy(image)
        cv2.rectangle(image2,(x,y),(x+w,y+h),(0,255,0),2)    
        cv2.imshow('2', image2) 
        cv2.waitKey(0)

    display_was_found = False
    if max_area > threshold_max_area and distance < threshold_distance:
        display_was_found = True
    return display_was_found
    