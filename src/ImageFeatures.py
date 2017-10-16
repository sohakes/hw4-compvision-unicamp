import cv2
import numpy as np 
from utils import *
from ClusterFeatures import ClusterFeatures
from copy import copy

class ImageFeatures:
    def _run_kmeans(self, img, k):
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        img = np.float32(img)
        #img = img.astype('float')
        #print(img.shape)
        #print(img)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness,labels,centers = cv2.kmeans(img,k,None,criteria,10,flags)
        return labels, centers


    def _describe_image(self, img, k):
        labels, mean_colors = self._run_kmeans(img, k)
        kmeans_labels_img = labels.reshape((img.shape[0], img.shape[1]))
        step_color = 255/k
        kmeans_label_img_color = kmeans_labels_img * step_color
        print(mean_colors)
        debug('colors' ,kmeans_label_img_color.astype('uint8'))
        size_clusters = [np.sum(j == labels) for j in range(k)]
        print(size_clusters)

        

        return [ClusterFeatures(size_clusters[i], mean_colors[i]) for i in range(k)]

    def calc_dissimilarity(self, other):
        my_clusters = copy(self.image_segments)
        other_clusters = copy(other.image_segments)
        
        sum_dis = 0

        best_matches = []
        for c in my_clusters:
            best_comb = 0
            best_comb_dis = c.calc_dissimilarity(other_clusters[0])     
            for i in range(1, len(other_clusters)):
                dis = c.calc_dissimilarity(other_clusters[i])
                print(dis, best_comb_dis, 'hey')
                if dis < best_comb_dis:
                    dis = best_comb_dis
                    best_comb = i
            best_matches.append((c, other_clusters[best_comb], best_comb_dis)) 
            sum_dis += best_comb_dis
            del other_clusters[best_comb]
        
        return sum_dis
        

    def __init__(self, img, k=2):
        self.image_segments = self._describe_image(img, k)
