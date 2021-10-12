#!/usr/bin/python3
# @Ailton Oliveira - 2020 - LASSE UFPA 
# ailton.pinto@itec.ufpa.br
# @Lucas Damasceno - 2020 - LASSE UFPA
# lucas.damasceno.silva@itec.ufpa.br
import cv2
import os
import sys
import numpy as np
from operator import itemgetter
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
        #perimetro = [None]*len(contours)
        for i in range(len(contours)):
            # add 1e-5 to avoid division by zero
            mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))
            #perimetro[i] = cv2.arcLength(contours[i], True)

        return mc

    def scan_pipe(self):
        camera = GetImage()
        cap = camera.get()

        #cv2.imwrite(img_name, cap)
        #return 0
        
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

        if (len(contours_red)>0) and (len(contours_green)>0):
            return len(contours_red), mc_r, len(contours_green), mc_g, 2
        elif len(contours_red)>0:
            return len(contours_red), mc_r, 0, 0, 0
        elif len(contours_green)>0:
            return 0, 0, len(contours_green), mc_g, 1
        else:
            return 0, mc_r, 0, mc_g, 3

    def scan_panoramic(self):
        camera = GetImage()
        cap = camera.get()
        match = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)

        #///treshold///
        low_green = (43, 64, 55)
        high_green = (69, 255, 255)
        low_red = (0, 200, 205)
        high_red = (7, 255, 255)
        #low_orange = (1, 140, 160)
        #high_orange = (15, 255, 255)

        img_green = np.zeros((300,300,3),np.uint8)
        img_red = np.zeros((300,300,3),np.uint8)

        img_green[:,:,1] = 255
        img_red[:,:,2] = 255

        mask1 = cv2.inRange(match,low_green, high_green) #sensor green
        mask2 = cv2.inRange(match,low_red, high_red) #sensor red
        #mask3 = cv2.inRange(match,low_orange, high_orange) #pipe

        kernel = np.ones((3,3),np.uint8)
        sure_red = self.noise_filter(mask2, kernel=kernel)
        sure_green = self.noise_filter(mask1, kernel=kernel)
        contours_red, hierarchy = cv2.findContours(sure_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        contours_green, hierarchy = cv2.findContours(sure_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        '''contours_color = (250,250,250)

        sure_bg = sure_red | sure_green
        demonstration = cv2.bitwise_and(cap, cap, mask = sure_bg)

        cv2.drawContours(demonstration, contours_red, 0, contours_color, 1, cv2.LINE_8, hierarchy, 0)
        cv2.drawContours(demonstration, contours_green, 0, contours_color, 1, cv2.LINE_8, hierarchy, 0)'''

        #Adjusting of mass center
        coords_g = np.zeros((len(contours_green), 4))
        coords_r = np.zeros((len(contours_red), 4))

        for i in range(len(contours_green)):
            (coords_g[i,0], coords_g[i,1], coords_g[i,2], coords_g[i,3]) = cv2.boundingRect(contours_green[i])

        for i in range(len(contours_red)):
            (coords_r[i,0], coords_r[i,1], coords_r[i,2], coords_r[i,3]) = cv2.boundingRect(contours_red[i])

        coord_x_g = []
        coord_y_g = []
        coord_x_r = []
        coord_y_r = []

        for i in range(len(coords_g)):
            coord_x_g.append((coords_g[i,0]+coords_g[i,0]+coords_g[i,2])/2)
            coord_y_g.append((coords_g[i,1]+coords_g[i,1]+coords_g[i,3])/2)

        for i in range(len(coords_r)):
            coord_x_r.append((coords_r[i,0]+coords_r[i,0]+coords_r[i,2])/2)
            coord_y_r.append((coords_r[i,1]+coords_r[i,1]+coords_r[i,3])/2)

        coord_central_g = np.zeros((len(contours_green), 2))
        coord_central_r = np.zeros((len(contours_red), 2))

        for i in range(len(coords_g)):
            coord_central_g[i,0] = coord_x_g[i]
            coord_central_g[i,1] = coord_y_g[i]

        for i in range(len(coords_r)):
            coord_central_r[i,0] = coord_x_r[i]
            coord_central_r[i,1] = coord_y_r[i]

        coords = {}

        for i in range(len(coords_g)):
            coords.update({coord_central_g[i,0]: 'Green'})

        for i in range(len(coords_r)):
            coords.update({coord_central_r[i,0]: 'Red'})

        coords_ord = sorted(coords.items(), key=itemgetter(0))

        labels = []

        for i in range(len(coords_ord)):
            labels.append(coords_ord[i][1])

        '''print ('Sensores localizados no cano:')
        print(labels)

        for i in range(len(labels)):
            if labels[i] == 'Green':
                cv2.imshow("Sensor_" +str(i)+ "", img_green)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()

            if labels[i] == 'Red':
                cv2.imshow("Sensor_" +str(i)+ "", img_red)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()'''

        return labels




def main():
    fase_2 = Fase_2('image')
    status = fase_2.scan_pipe()

if __name__ == "__main__":
    main()
