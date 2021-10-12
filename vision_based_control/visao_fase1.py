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
        sure_bg = cv2.dilate(opening,kernel,iterations = 3)
        return(sure_bg)
        #return(opening)

    def mc(self,contours):
        mu = [None]*len(contours)
        for i in range(len(contours)):
            mu[i] = cv2.moments(contours[i])
        
        # Get the mass centers
        mc = [None]*len(contours)
        perimetro = [None]*len(contours)
        for i in range(len(contours)):
            # add 1e-5 to avoid division by zero (Cx,Cy)
            mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))
            perimetro[i] = cv2.arcLength(contours[i], True)

        '''for i in range(len(contours)):
            print(' * Contour[%d] - Area (M_00) = %.2f - Area OpenCV: %.2f - Length: %.2f' % (i, mu[i]['m00'], cv2.contourArea(contours[i]), cv2.arcLength(contours[i], True)))'''
        return perimetro, mc
    
    def scan_pipe(self):
        #path = ('../Dataset/{}.png').format(i)
        path = (self.image_name)
        print(('reading {}').format(path))
        trace = './test_trace.json'
        step = 'step.txt'
        bug = 'bug.txt'
        if not os.path.exists(trace):
            os.mknod(trace)
        if not os.path.exists(step):
            os.mknod(step)
        if not os.path.exists(bug):
            os.mknod(bug)
        cap = cv2.imread(path)
        #find the center of the camera/drone
        resolution = cap.shape
        y,x = resolution[0],resolution[1]
        Center_offset = 27 #Deve mudar de acordo com o Z que definirmos como altitude de busca
        if (x % 2) == 0:
            Cx = x/2
        else:
            Cx = (x-1)/2
        if (y % 2) == 0:
            Cy = y/2
        else:
            Cy = (y-1)/2
        #cap = np.uint8([[[145,232,254]]])
        match = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)
        #///treshold///
        low_yellow = (26, 70, 245)
        high_yellow = (35, 127, 255)
        low_blue = (110, 50, 210)
        high_blue = (127, 101, 255)
        mask1 = cv2.inRange(match,low_yellow, high_yellow) #sensor green
        mask2 = cv2.inRange(match,low_blue, high_blue) #sensor red
        #mask3 = cv2.inRange(match,low_orange, high_orange) #pipe
        
        kernel = np.ones((3,3),np.uint8)
        sure_blue = self.noise_filter(mask2, kernel=kernel)
        sure_yellow = self.noise_filter(mask1, kernel=kernel)
        contours_blue, hierarchy = cv2.findContours(sure_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, hierarchy = cv2.findContours(sure_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours_blue) == 0:
            y_offset = 0.1
            x_offset = 0.1
            Output = 'find nothing'
            json_out = ('[[{},{},0]]'.format(x_offset,y_offset))
            with open(trace, 'w') as f:
                f.write(json_out)
            with open(step, 'w') as f:
                f.write(Output)
            return
        #tst = np.hstack((sure_bg,mask3)) #debug
        #contours_color = (0,0,250)
        #contours_color2 = (0,250,0)
            
        #sure_bg = sure_blue | sure_yellow
        #demonstration = cv2.bitwise_and(cap, cap, mask = sure_bg)
        # Get the moments
        perimetro, mc = self.mc(contours_blue)
        max_ele, max_id=perimetro[0],0
        for i in range(0,len(perimetro)): 
            if perimetro[i]>max_ele: 
                max_ele=perimetro[i]
                max_id = i
        x_offset = 0
        y_offset = 0
        if  (Cx-Center_offset) <= mc[max_id][0] <= (Cx+Center_offset) and (Cy-Center_offset) <= mc[max_id][1] <= (Cy+Center_offset):
            Output = 'land'
            print('use /uav1/uav_manager/land')
        else:
            Output = 'correct coordinates'
            print('rosrun')
            if (abs(mc[max_id][0] - Cx)) > 49:
                y_offset = -0.5
            elif ((abs(mc[max_id][0] - Cx)) > Center_offset):
                y_offset = -0.1
            else:
                y_offset = 0
            if (abs(mc[max_id][1] - Cy)) > 49:
                x_offset = -0.5
            elif ((abs(mc[max_id][1] - Cy)) > Center_offset):
                x_offset = -0.1
            else:
                x_offset = 0
            if (mc[max_id][0] - Cx) < 0:
                y_offset = y_offset*-1
            if (mc[max_id][1] - Cy) < 0:
                x_offset = x_offset*-1
        with open(bug, 'r') as reader:
            status = reader.read()
        centros = str(mc[max_id][0])+','+str(mc[max_id][1])
        if status == centros:
            y_offset = 0
            x_offset = 0
            Output = 'correct coordinates'
        with open(bug, 'w') as f:
            f.write(centros)
        print(mc[max_id])
        print(Cx, Cy)
        print(y_offset, x_offset)
        #cv2.drawContours(demonstration, contours_blue, max_id, contours_color, 1, cv2.LINE_8, hierarchy, 0)
        #cv2.imshow('vsss',demonstration)
        #cv2.waitKey(0)#debug tools
        #cv2.destroyAllWindows()
        json_out = ('[[{},{},0]]'.format(x_offset,y_offset))
        with open(trace, 'w') as f:
            f.write(json_out)
        with open(step, 'w') as f:
            f.write(Output)
            #cv2.drawContours(demonstration, contours_yellow, max_id, contours_color, 1, cv2.LINE_8, hierarchy, 0)
            #cv2.imshow('vsss',demonstration)
            #cv2.waitKey(0)#debug tools
            

        #cv2.destroyAllWindows()

def main():
    args = get_args()
    fase_2 = Fase_2(args.name)
    fase_2.scan_pipe()
    #os.system('rm ./tmp/fase2.png')

if __name__ == "__main__":
    main()
