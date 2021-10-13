# @Ailton Oliveira - 2020 - LASSE UFPA 
# Ailton.pinto@itec.ufpa.br

import cv2
import numpy as np

class baseScam:
    def __init__(self, image):
        self.image = image
        

    def boat_noise_filter(self,mask_x,kernel):
        #Barco
        dilate = cv2.dilate(mask_x,kernel,iterations = 6)
        return(dilate)

    def base_noise_filter(self,mask_x,kernel):
		#Base
        kernel2 = np.ones((2,2),np.uint8)
        sure_bg = cv2.erode(mask_x,kernel2,iterations = 1)
        dilate = cv2.dilate(sure_bg,kernel,iterations = 4)
        return(dilate)

    def mc(self,contours):
        mu = [None]*len(contours)
        for i in range(len(contours)):
            mu[i] = cv2.moments(contours[i])
        
        # Get the mass centers
        mc = [None]*len(contours)
        area = [None]*len(contours)
        for i in range(len(contours)):
            # add 1e-5 to avoid division by zero (Cx,Cy)
            mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))
            area[i] = cv2.contourArea(contours[i])

        return mc, area
    
    def check_range(self, valuex, valuey, array, limit):
        for i in array:
            if ( valuex-limit <= i[0] <= valuex+limit) and (valuey-limit <= i[1] <= valuey+limit):
                return False
            else:
                continue
        return True
    
    def _map(self, x, in_min, in_max, out_min, out_max):
        return float((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)
    
    def centralize(self):
        #path = (self.image)
        cap = self.image
        match = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)
        low_base = (20, 130, 160)
        high_base = (120, 180, 255)
        kernel = np.ones((6,6),np.uint8)

        mask2 = cv2.inRange(match,low_base, high_base)
        out = self.base_noise_filter(mask2,kernel)

        #Extrair contornos
        contours, hierarchy = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mc, area = self.mc(contours) #centro de massa, e area dos contornos
        #demo = cv2.drawContours(image=cap, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        #Interpolação z=7
        y_img = np.array([67.801,173.517,224.788,232.32,267.952,334.112, 385.225])
        x_gaz = np.array([6,4,2,0,-2,-4,-6])
        #x_interp = scipy.interpolate.interp1d(y_img, x_gaz)
        x_img = np.array([78.885,192.831,252.358,275.255,378.233,472.890,507.74,593.269,639.9312])
        y_gaz = np.array([12.5,9.5,6.5,4.5,-0.5,-5.5,-7.5,-10.5,-13.5])
        #y_gaz = np.array([13,10,7,5,0,-5,-7,-10,-13])
        #y_interp = scipy.interpolate.interp1d(x_img, y_gaz)
        #print('Interpolando!')
        x_rel = np.interp(mc[0][1], y_img, x_gaz)
        y_rel = np.interp(mc[0][0], x_img, y_gaz)
        #x_rel = x_interp(mc[0][1])
        #y_rel = y_interp(mc[0][0])
        
        return x_rel,y_rel


    def scan_area(self):
        #z = 7
        #path = (self.image)
        cap = self.image
        # Change when not use path
        match = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)
        low_blue = (0, 0, 185)
        high_blue = (117, 131, 200)
        kernel = np.ones((6,6),np.uint8)

        mask2 = cv2.inRange(match,low_blue, high_blue)
        out = self.boat_noise_filter(mask2,kernel)
        

        #Extrair contornos
        contours, hierarchy = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mc, area = self.mc(contours) #centro de massa, e area dos contornos
        #demo = cv2.drawContours(image=cap, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        #TODO
        #If mc.shape[0] > 1: cv2.dilate
        
        #Work with a panoramic fly at z=7
        #Interpolation data -> z=7
        y_img = np.array([78.458, 60.1,27.681, 178.82, 168.33, 128.1, 141.974, 440.038, 430.083, 427.3689])
        x_gaz = np.array([16,15,14,7,5,4,6,-21,-19,-20])
        #x_interp = scipy.interpolate.interp1d(y_img, x_gaz)
        x_img = np.array([66.647,375.583,673.343,50.417, 223.415, 547.704,686.79,80.720,265.410, 518.971, 670.1])
        y_gaz = np.array([30,0,-30,31.5,17,-15,-31,30.5,11,-14,-29])
        #y_interp = scipy.interpolate.interp1d(x_img, y_gaz)

        x_offset = np.interp(mc[0][1], y_img, x_gaz)
        y_offset = np.interp(mc[0][0], x_img, y_gaz)
        #x_rel = x_interp(mc[0][1])
        #y_rel = y_interp(mc[0][0])
        #print(mc)
        return x_offset, y_offset
    
    def set_img(self, img):
        self.image = img
    
    def get_mc(self):
        #path = (self.image)
        cap = self.image
        match = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)
        low_base = (20, 130, 160)
        high_base = (120, 180, 255)
        kernel = np.ones((6,6),np.uint8)

        mask2 = cv2.inRange(match,low_base, high_base)
        out = self.base_noise_filter(mask2,kernel)

        #Extrair contornos
        contours, hierarchy = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mc, area = self.mc(contours) #centro de massa, e area dos contornos
        return mc
            
        
def main():
    #fase_1 = baseScam('./base_cent1.png')
    #bases = fase_1.scan_area()
    print('tst')
    

if __name__ == "__main__":
    main()