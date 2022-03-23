import cv2
import numpy as np
from skimage import morphology
from numpy import *

def line1(textline,img_name):
    img = np.zeros((512,512))
    rgb =(255,255,255)
    width = 1
    points,points_ = [],[]
    for line in textline:
        for i in range(len(line)):
            theta1,theta2 = [],[]
            for j in range(1,4):
                theta1.append((line[j][1]-line[0][1])/(line[j][0]-line[0][0]))
                theta2.append((line[-1-j][1]-line[-1][1])/(line[-1-j][0]-line[-1][0]))
        theta1,theta2 = np.mean(theta1),np.mean(theta2)
        points.append([line[0][0], line[0][1],theta1])
        points_.append([line[-1][0], line[-1][1],theta2])

    M = 15
    threshold = 0.45 #0.5
    line1_l,line1_r = [],[]
    for point in points:
        dis_min1,dis_min2= M*M*2,M*M*2
        j1 = 0
        for point1 in points:
            if point1[0] >point[0]-M and point1[0] <point[0]+M and point1[1] >point[1] and point1[1] <point[1]+M:
                dis = (point1[0]-point[0])**2+(point1[1]-point[1])**2
                z = (point1[0]-point[0])/(point1[1]-point[1])
                z1 = abs((z+point[2])/(1-z*point[2]))

                if z1 < threshold and dis_min1 > dis:
                    zz = point1
                    dis_min1 = dis
                    j1 = 1
        if j1:
            cv2.line(img,(point[0],point[1]),(zz[0],zz[1]),rgb,width)
            line1_l.append([point[0],point[1],zz[0],zz[1]])


    for point in points_:
        dis_min1,dis_min2= M*M*2,M*M*2
        j1 = 0
        for point1 in points_:
            if point1[0] >point[0]-M and point1[0] <point[0]+M and point1[1] >point[1] and point1[1] <point[1]+M:
                dis = (point1[0]-point[0])**2+(point1[1]-point[1])**2
                z = abs((point1[0]-point[0])/(point1[1]-point[1]))
                z = (point1[0]-point[0])/(point1[1]-point[1])
                z1 = abs((z+point[2])/(1-z*point[2]))

                if z1 < threshold and dis_min1 > dis:
                    zz = point1
                    dis_min1 = dis
                    j1 = 1
        if j1:
            cv2.line(img,(point[0],point[1]),(zz[0],zz[1]),rgb,width)
            line1_r.append([point[0],point[1],zz[0],zz[1]])


    img1 = np.zeros((512,512))
    rgb =(255,255,255)
    line1_l,line1_r = [],[]
    for point in points:
        dis_min1,dis_min2= M*M*2,M*M*2
        j1 = 0
        for point1 in points:
            if point1[0] >point[0]-M and point1[0] <point[0]+M and point1[1] <point[1] and point1[1] >point[1]-M:
                dis = (point1[0]-point[0])**2+(point1[1]-point[1])**2
                z = (point1[0]-point[0])/(point1[1]-point[1])
                z1 = abs((z+point[2])/(1-z*point[2]))

                if z1 < threshold and dis_min1 > dis:
                    zz = point1
                    dis_min1 = dis
                    j1 = 1
        if j1:
            cv2.line(img1,(point[0],point[1]),(zz[0],zz[1]),rgb,width)
            line1_l.append([point[0],point[1],zz[0],zz[1]])


    for point in points_:
        dis_min1,dis_min2= M*M*2,M*M*2
        j1 = 0
        for point1 in points_:
            if point1[0] >point[0]-M and point1[0] <point[0]+M and point1[1] <point[1] and point1[1] >point[1]-M:
                dis = (point1[0]-point[0])**2+(point1[1]-point[1])**2
                z = abs((point1[0]-point[0])/(point1[1]-point[1]))
                z = (point1[0]-point[0])/(point1[1]-point[1])
                z1 = abs((z+point[2])/(1-z*point[2]))

                if z1 < threshold and dis_min1 > dis:
                    zz = point1
                    dis_min1 = dis
                    j1 = 1
        if j1:
            cv2.line(img1,(point[0],point[1]),(zz[0],zz[1]),rgb,width)
            line1_r.append([point[0],point[1],zz[0],zz[1]])
    
    gray_img = img*img1
    cv2.imwrite("result/vertical_line/"+img_name,gray_img)
    
    _, th = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY)

    th = morphology.skeletonize(th/255).astype(np.uint8)*255
    _, labels, stats,_ = cv2.connectedComponentsWithStats(th)

    i = 1
    gap = 10
    points = []
    for stat in stats[1:]:
        point = []
        if  stat[3]>30: 
            line = np.where(labels == i)
            for j in range(int(len(line[0])/gap)):
                point.append([line[1][gap*j], line[0][gap*j]])
            point.append([line[1][-1], line[0][-1]])  
            if point:
                points.append(np.array(point))
        i+=1
    return points

