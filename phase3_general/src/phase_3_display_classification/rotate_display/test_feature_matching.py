import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
#from util_digits import *

def binarize_color_image(image):
    # pre-process the image by resizing it, converting it to
    # graycale, blurring it, and computing an edge map
    #image = imutils.resize(image, height=500)
    HSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    low_white_display = (0, 0, 131)
    high_white_display = (0, 0, 255)
    # threshold the warped image, then apply a series of morphological
    # operations to cleanup the thresholded image
    sure_display = cv.inRange(HSV,low_white_display, high_white_display) #sensor green
    return sure_display


file_name = '../digitsA_train/dig0_00005.png'
template_file = 'template_2.png'

#img1 = cv.imread(file_name,cv.IMREAD_GRAYSCALE)          # queryImage
img1 = cv.imread(file_name)          # queryImage

img1 = binarize_color_image(img1)

img2 = cv.imread(template_file,cv.IMREAD_GRAYSCALE) # trainImage
# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

print(len(matches))

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    #matchesMask[i]=[1,0]
    #if m.distance < 0.7*n.distance:
    if m.distance < 0.9*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
plt.imshow(img3,),plt.show()