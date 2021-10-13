#!/usr/bin/python3
# @Ailton Oliveira - 2021 - LASSE UFPA 
# ailton.pinto@itec.ufpa.br
# @Lucas Damasceno - 2021 - LASSE UFPA
# lucas.damasceno.silva@itec.ufpa.br

import cv2
import numpy as np
from phase0_drone_camera.camera import GetImage

class Fase_2:

    def noise_filter(self,mask_x,kernel):
        opening = cv2.morphologyEx(mask_x,cv2.MORPH_OPEN,kernel, iterations = 1)
        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations = 1)
        return(sure_bg)

    def area_and_mc(self,contours):
        mu = [None]*len(contours)
        for i in range(len(contours)):
            mu[i] = cv2.moments(contours[i])

        # Get the mass centers
        mc = [None]*len(contours)
        for i in range(len(contours)):
            # add 1e-5 to avoid division by zero
            mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))

        return mc

    def scan_pipe(self):
        camera = GetImage()
        cap = camera.get()
        match = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)

        #///treshold///
        low_green = (43, 64, 55)
        high_green = (69, 255, 255)
        low_red = (0, 200, 205)
        high_red = (7, 255, 255)

        mask1 = cv2.inRange(match,low_green, high_green) #sensor green
        mask2 = cv2.inRange(match,low_red, high_red) #sensor red

        kernel = np.ones((3,3),np.uint8)
        sure_red = self.noise_filter(mask2, kernel=kernel)
        sure_green = self.noise_filter(mask1, kernel=kernel)
        contours_red, hierarchy = cv2.findContours(sure_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        contours_green, hierarchy = cv2.findContours(sure_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        mc_r = self.area_and_mc(contours_red)
        mc_g = self.area_and_mc(contours_green)

        demo = cv2.drawContours(image=cap, contours=contours_red, contourIdx=-1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        demo = cv2.drawContours(image=demo, contours=contours_green, contourIdx=-1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

        cv2.imshow('Camera', demo)
        k = cv2.waitKey(32)

        if (len(contours_red)>0) and (len(contours_green)>0):
            return len(contours_red), mc_r, len(contours_green), mc_g, 2
        elif len(contours_red)>0:
            return len(contours_red), mc_r, 0, 0, 0
        elif len(contours_green)>0:
            return 0, 0, len(contours_green), mc_g, 1
        else:
            return 0, mc_r, 0, mc_g, 3


def main():
    fase_2 = Fase_2('image')
    status = fase_2.scan_pipe()


if __name__ == "__main__":
    main()
