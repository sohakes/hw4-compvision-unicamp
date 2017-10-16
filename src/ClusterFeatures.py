import cv2
import numpy as np 
from utils import *

class ClusterFeatures:

    def calc_dissimilarity(self, other):
        dif_size = abs(other.size_cluster - self.size_cluster)
        #print('size_cluster', other.size_cluster)
        dist_color = np.sqrt(np.sum(other.mean_color - self.mean_color) ** 2)
        return dif_size * dist_color

    

    def __init__(self, size_cluster, mean_color):
        self.size_cluster = size_cluster
        self.mean_color = mean_color

