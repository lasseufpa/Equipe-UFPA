# @Ailton Oliveira - 2020 - LASSE UFPA 
# Ailton.pinto@itec.ufpa.br
import cv2
import os
import sys
import numpy as np
import argparse
from phase0_drone_camera.camera import GetImage

camera = GetImage()

class baseScam:
    def __init__(self, image):
        self.image = image
        

    def noise_filter(self,mask_x,kernel):
        #kernel = np.ones((2,2),np.uint8)
        opening = cv2.morphologyEx(mask_x,cv2.MORPH_OPEN,kernel, iterations = 1)
        # sure background area
        dilate = cv2.dilate(opening,kernel,iterations = 3)
        sure_bg = cv2.erode(dilate,kernel,iterations = 1)
        return(sure_bg)
        #return(opening)

    def mc(self,contours):
        mu = [None]*len(contours)
        for i in range(len(contours)):
            mu[i] = cv2.moments(contours[i])
        
        # Get the mass centers
        mc = [None]*len(contours)
        #perimetro = [None]*len(contours)
        area = [None]*len(contours)
        for i in range(len(contours)):
            # add 1e-5 to avoid division by zero (Cx,Cy)
            mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))
            #perimetro[i] = cv2.arcLength(contours[i], True)
            area[i] = cv2.contourArea(contours[i])

        '''for i in range(len(contours)):
            print(' * Contour[%d] - Area (M_00) = %.2f - Area OpenCV: %.2f - Length: %.2f' % (i, mu[i]['m00'], cv2.contourArea(contours[i]), cv2.arcLength(contours[i], True)))'''
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

    def scan_pipe(self):
        scan = True
        while(scan):
            cap = camera.get()
            match = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)
            
            low_blue = (110, 50, 210)
            high_blue = (127, 101, 255)
            mask2 = cv2.inRange(match,low_blue, high_blue) 
            
            kernel = np.ones((3,3),np.uint8)
            sure_blue = self.noise_filter(mask2, kernel=kernel)
            
            #Extrair contornos
            _, contours_blue, hierarchy = cv2.findContours(sure_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            mc, area = self.mc(contours_blue) #centro de massa, e area dos contornos
            
            #Amazenar centros unicos
            centros = np.zeros((6,2)) 
            centros[0][0] = mc[0][0]
            centros[0][1] = mc[0][1]
            j = 0
            draw_list =[0] #Debug - Lista pra visualizacao
            erro_de_indice = False
            for i in range(1,len(area)):
                Cx = mc[i][0] #X da imagem e Y do drone
                Cy = mc[i][1] #Y da imagem e X do drone
                #funcao para verificar se ja existem centros naquela regiao
                #limit(4) e para passar o valor aceitavel de margem de erro entre centros repetidos
                if self.check_range(Cx,Cy,centros,10): 
                    j+=1
                    draw_list.append(i)
                    try:
                        centros[j] = (Cx, Cy)
                    except(IndexError):
                        erro_de_indice = True
            if erro_de_indice:
                #print('scanear de novo')
                continue
            scan = False
            hector_x = []
            hector_y = []
            #print(centros)
            for base in centros:
                if 215 <= base[1] <= 226 and 73 <= base[0] <= 101:
                    #print('suspensa1')
                    continue
                if 448 <= base[1] <= 456 and 602 <= base[0] <= 612:
                    #print('suspensa2')
                    continue
                if 427 <= base[1] <= 434 and 175 <= base[0] <= 200:
                    #print('costeira')
                    continue
                hector_x.append(self._map(base[1],439,12,0,6.8))
                hector_y.append(self._map(base[0],175,591,0,-6.6))
                #print(base)
                #print(self._map(base[1],431,12,0,6.8))
                #print(self._map(base[0],180,591,0,-6.6))
            #print("Coordenadas filtradas")
            Output = [[3.3,0,2], [0.2,-5.8,2], [hector_x[0], hector_y[0],2], [hector_x[1], hector_y[1],2], [hector_x[2], hector_y[2],2]]
        return Output
            
        
def main():
    fase_1 = Fase_1('image')
    bases = fase_1.scan_pipe()
    #print(bases)
    

if __name__ == "__main__":
    main()
