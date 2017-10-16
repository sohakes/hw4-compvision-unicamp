import cv2
import numpy as np
import math
from utils import *
from ImageFeatures import ImageFeatures
import glob

################  HW2  #####################
# Nathana Facion                 RA:191079
# Rafael Mariottini Tomazela     RA:192803
############################################




def main():
  
    #imgs = [ImageFeatures(cv2.imread(path)) for path in glob.glob('input/*')
    boat1 = ImageFeatures(cv2.imread('input/p4-images/boat_2.jpg'))
    boat2 = ImageFeatures(cv2.imread('input/p4-images/boat_3.jpg'))
    beach1 = ImageFeatures(cv2.imread('input/p4-images/beach_1.jpg'))
    cherry1 = ImageFeatures(cv2.imread('input/p4-images/cherry_1.jpg'))
    pond1 = ImageFeatures(cv2.imread('input/p4-images/pond_1.jpg'))
    sunset = ImageFeatures(cv2.imread('input/p4-images/sunset1_5.jpg'))
    print('boat1, boat2', boat1.calc_dissimilarity(boat2))
    print('boat1, beach1', boat1.calc_dissimilarity(beach1))
    print('boat1, cherry1', boat1.calc_dissimilarity(cherry1))
    print('boat1, pond1', boat1.calc_dissimilarity(pond1))
    print('boat1, sunset', boat1.calc_dissimilarity(sunset))
    

    #print(img)
    #kmeans_labels = run_kmeans(img)
    #kmeans_labels_img = kmeans_labels.reshape((img.shape[0], img.shape[1]))
    #step_color = 255/5
    #kmeans_label_img_color = kmeans_labels_img * step_color
    #print(kmeans_labels_img)
    #print(kmeans_label_img_color)
    #debug('kmeans', kmeans_label_img_color.astype('uint8'))
    #labels_to_im = 
    #debug('img1', )


    

if __name__ == '__main__':
   main()


