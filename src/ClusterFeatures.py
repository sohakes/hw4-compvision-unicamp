import cv2
import numpy as np 
from utils import *

class ClusterFeatures:

    def calc_dissimilarity(self, other, dist_id):
        dif_size = abs(other.size_cluster - self.size_cluster)
        #print('size_cluster', other.size_cluster)
        dist_color = np.sqrt(np.sum(other.mean_color - self.mean_color) ** 2)
        dist_texture = np.sqrt(np.sum(other.texture_feats - self.texture_feats) ** 2)
        #print('textures', dist_texture, self.texture_feats, other.texture_feats)
        #print('histogram', self.histogram, other.histogram, self.histogram.shape, self.histogram-other.histogram, ((self.histogram - other.histogram) ** 2))
        dist_pos = np.sqrt(np.sum(other.mean_pos - self.mean_pos) ** 2)/800
        if dist_id == 0:
            return np.sum((self.histogram - other.histogram) ** 2)/4 + dist_pos + dif_size/307200 + dist_color/360 + dist_texture*2
        elif dist_id == 1:
            return np.sum((self.histogram - other.histogram) ** 2)/5 + dist_pos*4 + dif_size/307200 + dist_color/360 + dist_texture*2
        elif dist_id == 2:
            return np.sum((self.histogram - other.histogram) ** 2)/5 + dist_pos*2 + dif_size/307200 + dist_color/360 + dist_texture*4
        elif dist_id == 3:
            return dist_pos*2 + dif_size/307200 + dist_color/360 + dist_texture*4
        elif dist_id == 4:
            return np.sum((self.histogram - other.histogram) ** 2)/5 + dist_pos*3 + dif_size/307200 + dist_color/360 + dist_texture*6
        elif dist_id == 5:
            return np.sum((self.histogram - other.histogram) ** 2)/2 + dist_pos*3 + 2*dif_size/307200 + 2*dist_color/360 + dist_texture*2
        #print('dissi:', np.sum((self.histogram - other.histogram) ** 2), dist_pos)
        #return 0 * (1 + dif_size/307200) * dist_color/360 + 20 * cv2.compareHist(self.histogram, other.histogram, cv2.HISTCMP_BHATTACHARYYA)

    def calc_distance(self, other, dist_id):
        return self.calc_dissimilarity(other, dist_id)
        #return np.sqrt(np.sum((other.mean_pos - self.mean_pos) ** 2))

    def print_features(self):
        print("features:")
        print(self.size_cluster)
        print(self.mean_color)
        print(self.histogram)
        print(self.mean_pos)
        print('\n')

    def __init__(self, size_cluster, mean_color, histogram, mean_pos, texture_feats):
        self.size_cluster = size_cluster
        self.mean_color = mean_color
        self.histogram = histogram
        self.mean_pos = mean_pos
        self.texture_feats = texture_feats

        #print('mean pos', mean_pos)
