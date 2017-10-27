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

def calc_all_imgs():
    imgs_feat = [ImageFeatures(cv2.imread(path), path) for path in glob.glob('input/*.jpg')]

    dists_for_each_image = []
    for i in range(len(imgs_feat)):
        curr_dists = []
        for j in range(len(imgs_feat)):
            #if i == j:
            #    continue
            dis = imgs_feat[i].calc_dissimilarity(imgs_feat[j])
            
            curr_dists.append((j, dis))
        curr_dists.sort(key=lambda x: x[1])
        dists_for_each_image.append(curr_dists)

        debug('img ' + imgs_feat[i].path, imgs_feat[i].simg)
        for j in range(1, 4):
            print('dis:', imgs_feat[i].path, imgs_feat[curr_dists[j][0]].path, curr_dists[j][1])
            debug('img ' + str(j), imgs_feat[curr_dists[j][0]].simg)
        for j in range(len(imgs_feat)):
            print('dis:', imgs_feat[i].path, imgs_feat[curr_dists[j][0]].path, curr_dists[j][1])

def calc_input_imgs(k, dist_id):
    imgs_feat = [ImageFeatures(cv2.imread(path), path, k, dist_id) for path in glob.glob('input/*.jpg')]

    file_imgs = open("input/input_files.txt", "r") 
    fout = open("output/out_k_"+str(k)+"_dist_id_"+str(dist_id)+".txt","w")
    filenames = []
    for line in file_imgs:
        filenames.append('input/'+line.rstrip('\n')+'.jpg')

    #print(filenames)

    dists_for_each_image = []
    for i in range(len(imgs_feat)):
        curr_dists = []
        if imgs_feat[i].path not in filenames:
            #print('n achou', imgs_feat[i].path)
            continue
        #print('achou')
        for j in range(len(imgs_feat)):
            #if i == j:
            #    continue
            dis = imgs_feat[i].calc_dissimilarity(imgs_feat[j])
            
            curr_dists.append((j, dis))
        curr_dists.sort(key=lambda x: x[1])
        dists_for_each_image.append(curr_dists)

        debug('img ' + imgs_feat[i].path, imgs_feat[i].simg)
        fout.write(imgs_feat[i].path+': ')
        for j in range(1, 4):
            #print('dis:', imgs_feat[i].path, imgs_feat[curr_dists[j][0]].path, curr_dists[j][1])
            debug('img ' + str(j), imgs_feat[curr_dists[j][0]].simg)
            fout.write(imgs_feat[curr_dists[j][0]].path+' ')
        #for j in range(len(imgs_feat)):
            #print('dis:', imgs_feat[i].path, imgs_feat[curr_dists[j][0]].path, curr_dists[j][1])
        fout.write('\n')

def calc_pair(n1, n2):
    im1 = ImageFeatures(cv2.imread('input/'+n1+'.jpg'))
    im2 = ImageFeatures(cv2.imread('input/'+n2+'.jpg'))
    print(n1 + ' ' + n2, im1.calc_dissimilarity(im2))



def main():
    #calc_pair('beach_3', 'beach_4')
    calc_input_imgs(10, 0)    
    calc_input_imgs(5, 0)
    calc_input_imgs(10, 1)
    calc_input_imgs(5, 1)
    calc_input_imgs(10, 2)
    calc_input_imgs(5, 2)    
    calc_input_imgs(10, 3)
    calc_input_imgs(5, 3) 
    calc_input_imgs(10, 4)
    calc_input_imgs(5, 4) 
    calc_input_imgs(10, 5)
    calc_input_imgs(5, 5) 

    """
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
    """
    

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


