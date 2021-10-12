import cv2
import numpy as np
from matplotlib import pyplot as plt
from util_digits import *

file_name = 'D:/aksimul/petrobras2020/digits_yolo/dig0_00003.png'
template_file = 'D:/aksimul/petrobras2020/preprocess_digits/template_2.png'



image = cv2.imread(file_name)

image = binarize_image(image)

cv2.imshow('binary', image) 

cv2.waitKey(0) 
  
# Grayscale 
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
  
# Find Canny edges 
edged = cv2.Canny(image, 30, 200) 
cv2.waitKey(0) 
  
# Finding Contours 
# Use a copy of the image e.g. edged.copy() 
# since findContours alters the image 
contours, hierarchy = cv2.findContours(edged,  
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
  
cv2.imshow('Canny Edges After Contouring', edged) 
cv2.waitKey(0) 
  
print("Number of Contours found = " + str(len(contours))) 
print(contours)
  
# Draw all contours 
# -1 signifies drawing all contours 
cv2.drawContours(edged, contours, -1, (0, 255, 0), 3) 
  
cv2.imshow('Contours', edged) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 

exit(1)

####### FIM 

img2 = img.copy()
template = cv2.imread(template_file)
#w, h = template.shape[::-1]

imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

plt.imshow(imgray)
cv2.drawContours(imgray, contours, -1, (0,255,0), 3)

ret, binary = cv2.threshold(imgray,40,255,cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(imgray, contours, -1, (255, 0, 0), 1)
plt.show()

for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(imgray,(x,y),(x+w,y+h),(155,155,0),1)
    cv2.imshow('nier2',imgray)

plt.show()