import cv2
import numpy as np
import math
from utils import *
from KLT import KLT
from Sfm import Sfm
from VideoSfm import VideoSfm

################  HW2  #####################
# Nathana Facion                 RA:191079
# Rafael Mariottini Tomazela     RA:192803
############################################

def run_kmeans(img):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness,labels,centers = cv2.kmeans(z,2,None,criteria,10,flags)
    return labels



def main():
    img = cv2.imread('input/p4-1-0.png')
    kmeans_labels = run_kmeans(img)
    print(labels)
    #labels_to_im = 
    #debug('img1', )


    

if __name__ == '__main__':
   main()


