import cv2
import numpy as np 
from utils import *
from ClusterFeatures import ClusterFeatures
from copy import copy
import random
from scipy.optimize import linear_sum_assignment

class ImageFeatures:
    def _run_kmeans(self, img, k):
        #img = img.copy()
        indices = np.moveaxis(np.indices((img.shape[0], img.shape[1])), 0, 2)
        print(img.shape, indices.shape)
        img = np.append(img, indices, axis=2)
        img = img.reshape((img.shape[0] * img.shape[1], 5))
        img = np.float32(img)
        #img = img.astype('float')
        #print(img.shape)
        #print(img)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness,labels,centers = cv2.kmeans(img,k,None,criteria,10,flags)
        return labels, centers


    def _describe_image(self, img, k):
        labels, means = self._run_kmeans(img, k)
        print('means', means)
        mean_colors = means[:,:3]
        mean_pos = means[:,3:]
        print('means2', mean_colors, mean_pos)
        kmeans_labels_img = labels.reshape((img.shape[0], img.shape[1]))
        step_color = 255/k
        kmeans_label_img_color = kmeans_labels_img * step_color
        print(mean_colors)
        #debug('colors' ,kmeans_label_img_color.astype('uint8'))
        size_clusters = [np.sum(j == labels) for j in range(k)]
        print(size_clusters)
        histograms = []
        for j in range(k):
            l = labels.copy()
            l[l == j] = 255
            l[l != j] = 0
            mask = l
            #h = cv2.calcHist([img], [0, 1, 2], None, [8, 3, 3], [0, 180,0, 256, 0, 256])
            h1 = cv2.calcHist([img], [0], None, [8], [0, 180])
            h2 = cv2.calcHist([img], [1], None, [3], [0, 256])
            h3 = cv2.calcHist([img], [2], None, [3], [0, 256])
            h1 = cv2.normalize(h1, h1)
            h2 = cv2.normalize(h2, h2)
            h3 = cv2.normalize(h3, h3)
            h1 = h1.flatten()
            h2 = h2.flatten()
            h3 = h3.flatten()
            print('hs',h1, h2, h3)
            h = np.concatenate((h1, h2, h3))
            print('ho',h)
            #h = cv2.calcHist([img], [0], None, [8], [0, 180])
            #h = cv2.normalize(h, h)
            #h = h.flatten()
            histograms.append(h)
            print('loop1\n\n\n')

        

        return [ClusterFeatures(size_clusters[i], mean_colors[i], histograms[i], mean_pos[i]) for i in range(k)]

    def _calc_dissimilarity_rec_dist(self, idx, useds, v1, v2, dists, disses, dp):
        if len(v2) == len(useds) or idx == len(v1):
            return (0, 0, [(-1, -1)])
        if (idx, str(useds)) in dp:
            #print('dp', dp[(idx, str(useds))])
            return dp[(idx, str(useds))]
        bestdiss = -1
        bestdist = -1
        bestcombs = None
        for i in range(len(v2)):
            if i in useds:
                continue
            ret = self._calc_dissimilarity_rec_dist(idx+1, useds + [i], v1, v2, dists, disses, dp)
            #print('the ret',ret, idx+1, useds + [i])
            diss, dist, combs = ret
            sdist = dist + dists[idx][i]
            if bestdist == -1 or sdist < bestdist:
                #print('changed')
                bestdist = sdist
                bestdiss = disses[idx][i] + diss
                combs = combs.copy()
                combs.append((idx, i))
                bestcombs = combs

        dp[(idx, str(useds))] = (bestdiss, bestdist, bestcombs)
        return (bestdiss, bestdist, bestcombs)
        


    def calc_dissimilarity(self, other):
        size = len(self.image_segments)
        my_clusters = copy(self.image_segments)
        other_clusters = copy(other.image_segments)
        dists = []
        diss = []
        for i in range(size):
            cdist = []
            cdiss = []
            for j in range(size):
                cdist.append(self.image_segments[i].calc_distance(other.image_segments[j]))
                cdiss.append(self.image_segments[i].calc_dissimilarity(other.image_segments[j]))
            dists.append(cdist)
            diss.append(cdiss)
        #ret = self._calc_dissimilarity_rec_dist(0, [], my_clusters, other_clusters, dists, diss, {})
        #print(dists)
        nret = linear_sum_assignment(dists)
        nret = list(zip(nret[0], nret[1]))
        
        #print('new', nret)
        step = 255/len(nret)
        a = 0
        nimg = self.simg.copy()
        thesum = 0
        for i, j in nret:
            if i == -1:
                continue
            #print('i, j:', i, j)
            #self.image_segments[i].print_features()
            #other.image_segments[j].print_features()
            x,y = self.image_segments[i].mean_pos[0], self.image_segments[i].mean_pos[1]
            cv2.circle(nimg,(x, y), 5, (100,step*a,step*a), 5)
            x,y = other.image_segments[j].mean_pos[0], other.image_segments[j].mean_pos[1]
            cv2.circle(nimg,(x, y), 5, (100,step*a,step*a), 5)
            a+=1
            thesum += diss[i][j]
        #debug('circled img', nimg)
        #return np.sum(diss)
        return thesum

    
    def _calc_dissimilarity_rec(self, idx, useds, v1, v2):
        if len(v2) == len(useds) or idx == len(v1):
            return 0
        best = -1
        for i in range(len(v2)):
            if i in useds:
                continue
            s = v1[idx].calc_dissimilarity(v2[i]) + self._calc_dissimilarity_rec(idx+1, useds + [i], v1, v2)
            if best == -1 or s < best:
                best = s

        return best
        


    def calc_dissimilarity_no_dist(self, other):
        size = len(self.image_segments)
        my_clusters = copy(self.image_segments)
        other_clusters = copy(other.image_segments)
        
        return self._calc_dissimilarity_rec(0, [], my_clusters, other_clusters)

    """
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
    """

    def __init__(self, img, path='', k=10):
        print(path)
        #path=''
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)   
        self.simg = img
        self.path = path
        print('path', path)
        if path == '':
            self.image_segments = self._describe_image(img, k)
        else:
            try:
                feats1 = np.load('featsave/' + path + '1.npy')
                feats2 = np.load('featsave/' + path + '2.npy')
                feats3 = np.load('featsave/' + path + '3.npy')
                feats4 = np.load('featsave/' + path + '4.npy')
                ab = list(range(k))
                random.shuffle(ab)
                self.image_segments = [ClusterFeatures(feats1[i], feats2[i], feats3[i], feats4[i]) for i in ab]
                #print('loaded')
            except IOError as e:
                #print('ioerror', e)
                self.image_segments = self._describe_image(img, k)
                feats1 = [x.size_cluster for x in self.image_segments]
                feats2 = [x.mean_color for x in self.image_segments]
                feats3 = [x.histogram for x in self.image_segments]
                feats4 = [x.mean_pos for x in self.image_segments]
                #print('feats1\n\n\n\n', feats1)
                #print('feats2\n\n\n\n', feats2)
                np.save('featsave/' + path + '1', feats1)
                np.save('featsave/' + path + '2', feats2)
                np.save('featsave/' + path + '3', feats3)
                np.save('featsave/' + path + '4', feats4)
