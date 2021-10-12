# @Ailton Oliveira - 2020 - LASSE UFPA 
# Ailton.pinto@itec.ufpa.br
import cv2
import os
import sys
import numpy as np
import argparse

#np.set_printoptions(threshold=sys.maxsize)
def get_args():
    parser = argparse.ArgumentParser(
        description='Follow bases using a trace file'
    )

    parser.add_argument('--name',
                        help='image name',
                        type=str)

    return parser.parse_args()
class Fase_2:
    def __init__(self, image_name):
        self.image_name = image_name

    def noise_filter(self,mask_x,kernel):
        #kernel = np.ones((2,2),np.uint8)
        opening = cv2.morphologyEx(mask_x,cv2.MORPH_OPEN,kernel, iterations = 1)
        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations = 4)
        return(sure_bg)
        #return(opening)

    def mc(self,contours):
        mu = [None]*len(contours)
        for i in range(len(contours)):
            mu[i] = cv2.moments(contours[i])
        
        # Get the mass centers
        mc = [None]*len(contours)
        perimetro = [None]*len(contours)
        area = [None]*len(contours)
        for i in range(len(contours)):
            # add 1e-5 to avoid division by zero (Cx,Cy)
            mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))
            perimetro[i] = cv2.arcLength(contours[i], True)
            area[i] = cv2.contourArea(contours[i])

        '''for i in range(len(contours)):
            print(' * Contour[%d] - Area (M_00) = %.2f - Area OpenCV: %.2f - Length: %.2f' % (i, mu[i]['m00'], cv2.contourArea(contours[i]), cv2.arcLength(contours[i], True)))'''
        return perimetro, mc, area
    
    def check_range(self, valuex, valuey, array, limit):
        #Limit é para passar o valor aceitavel de margem de erro entre centros repetidos
        for i in array:
            if ( valuex-limit <= i[0] <= valuex+limit) and (valuey-limit <= i[1] <= valuey+limit):
                return False
            else:
                continue
        return True

    def scan_pipe(self):
        #path = ('../Dataset/{}.png').format(i)
        path = (self.image_name) #path da imagem (arg)
        print(('reading {}').format(path))
        cap = cv2.imread(path)
        #find the center of the camera/drone
        resolution = cap.shape
        y,x = resolution[0],resolution[1] #y e x da imagem (Opencv entrega invertido)
        
        match = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)
        #///treshold/// mascaras do amarelo e azul cenário do desafio Petrobras
        low_yellow = (26, 70, 245)
        high_yellow = (35, 127, 255)
        low_blue = (110, 50, 210)
        high_blue = (127, 101, 255)
        mask1 = cv2.inRange(match,low_yellow, high_yellow) 
        mask2 = cv2.inRange(match,low_blue, high_blue) 
        
        #Filtragem de ruído
        kernel = np.ones((3,3),np.uint8)
        sure_blue = self.noise_filter(mask2, kernel=kernel)
        sure_yellow = self.noise_filter(mask1, kernel=kernel)
        
        #Extrair contornos
        contours_blue, hierarchy = cv2.findContours(sure_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, hierarchy = cv2.findContours(sure_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_color = (0,250,0) #cor para visualização
        perimetro, mc, area = self.mc(contours_blue) #perimetro, centro de massa, e area dos contornos
        
        #Armazenar os centros únicos
        centros = np.zeros((6,2)) 
        centros[0][0] = mc[0][0]
        centros[0][1] = mc[0][1]
        j = 0
        draw_list =[0] #Debug - lista apenas para visualização dos contornos
        for i in range(1,len(area)):
            print(mc[i])
            Cx = mc[i][0] #X da imagem e Y do drone
            Cy = mc[i][1] #Y da imagem e X do drone
            #função para verificar se já existem centros naquela região
            #limit(4) é para passar o valor aceitavel de margem de erro entre centros repetidos
            if self.check_range(Cx,Cy,centros,4): 
                j+=1
                draw_list.append(i)
                centros[j] = (Cx, Cy)
        sure_bg = sure_blue | sure_yellow #Debug - junção das mask apenas para visualização
        demonstration = cv2.bitwise_and(cap, cap, mask = sure_bg) #Debug - junção das mask apenas para visualização
        print(centros) #saída final

        #Debug - Desenho dos contornos para visualização
        for j in draw_list:
            cv2.drawContours(demonstration, contours_blue, j, contours_color, 1, cv2.LINE_8, hierarchy, 0)
            cv2.imshow('vsss',demonstration)
            cv2.waitKey(0)#debug tools
            cv2.destroyAllWindows()
        
def main():
    args = get_args()
    fase_2 = Fase_2(args.name)
    fase_2.scan_pipe()
    

if __name__ == "__main__":
    main()
