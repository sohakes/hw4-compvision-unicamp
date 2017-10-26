import cv2
import numpy as np 
from utils import *

class ClusterFeatures:

    def calc_dissimilarity(self, other):
        dif_size = abs(other.size_cluster - self.size_cluster)
        #print('size_cluster', other.size_cluster)
        dist_color = np.sqrt(np.sum(other.mean_color - self.mean_color) ** 2)
        #print('histogram', self.histogram, other.histogram, self.histogram.shape, self.histogram-other.histogram, ((self.histogram - other.histogram) ** 2))
        dist_pos = np.sqrt(np.sum(other.mean_pos - self.mean_pos) ** 2)/800
        return np.sum((self.histogram - other.histogram) ** 2)/5 + dist_pos + dif_size/307200 + dist_color/360
        #print('dissi:', np.sum((self.histogram - other.histogram) ** 2), dist_pos)
        #return 0 * (1 + dif_size/307200) * dist_color/360 + 20 * cv2.compareHist(self.histogram, other.histogram, cv2.HISTCMP_BHATTACHARYYA)

    def calc_distance(self, other):
        return self.calc_dissimilarity(other)
        #return np.sqrt(np.sum((other.mean_pos - self.mean_pos) ** 2))

    def print_features(self):
        print("features:")
        print(self.size_cluster)
        print(self.mean_color)
        print(self.histogram)
        print(self.mean_pos)
        print('\n')

    def __init__(self, size_cluster, mean_color, histogram, mean_pos):
        self.size_cluster = size_cluster
        self.mean_color = mean_color
        self.histogram = histogram
        self.mean_pos = mean_pos

        #print('mean pos', mean_pos)
