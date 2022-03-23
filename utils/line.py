import cv2
import numpy as np
from skimage import morphology
from numpy import *

def line(img):
    img = img.astype(np.uint8)
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    th = morphology.skeletonize(th/255).astype(np.uint8)*255

    _, labels, stats,_ = cv2.connectedComponentsWithStats(th)
    i = 1
    points = []
    for stat in stats[1:]:
        point = []
        if  stat[2]>60:
            inter =16
            line = np.where(labels.T == i)
            for j in range(int(len(line[0])/inter)):
                point.append([line[0][inter*j], line[1][inter*j]])
            point.append([line[0][-1], line[1][-1]])  
            if point:
                points.append(np.array(point))
        i+=1
    return points