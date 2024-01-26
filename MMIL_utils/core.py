import torchmetrics
import torch.nn as nn
import torch
from .utils import *
import numpy as np
from sklearn.cluster import KMeans


class grouping:

    def __init__(self,groups_num,max_size=1e10):
        self.groups_num = groups_num
        self.max_size = int(max_size) # Max lenth 4300 for 24G RTX3090
        
    
    def indicer(self, labels):
        indices = []
        groups_num = len(set(labels))
        for i in range(groups_num):
            temp = np.argwhere(labels==i).squeeze()
            indices.append(temp)
        return indices
    
    def make_subbags(self, idx, features):
        index = idx
        features_group = []
        for i in range(len(index)):
            member_size = (index[i].size)
            if member_size > self.max_size:
                index[i] = np.random.choice(index[i],size=self.max_size,replace=False)
            temp = features[index[i]]
            temp = temp.unsqueeze(dim=0)
            features_group.append(temp)
            
        return features_group
        
    def coords_nomlize(self, coords):
        coords = coords.squeeze()
        means = torch.mean(coords,0)
        xmean,ymean = means[0],means[1]
        stds = torch.std(coords,0)
        xstd,ystd = stds[0],stds[1]
        xcoords = (coords[:,0] - xmean)/xstd
        ycoords = (coords[:,1] - ymean)/ystd
        xcoords,ycoords = xcoords.view(xcoords.shape[0],1),ycoords.view(ycoords.shape[0],1)
        coords = torch.cat((xcoords,ycoords),dim=1)
        
        return coords
    
    
    def coords_grouping(self,coords,features,c_norm=False):
        features = features.squeeze()
        coords = coords.squeeze()
        if c_norm:
            coords = self.coords_nomlize(coords.float())
        features = features.squeeze()
        k = KMeans(n_clusters=self.groups_num, random_state=0,n_init='auto').fit(coords.cpu().numpy())
        indices = self.indicer(k.labels_)
        features_group = self.make_subbags(indices,features)
        
        return features_group
    
    def embedding_grouping(self,features):
        features = features.squeeze()
        k = KMeans(n_clusters=self.groups_num, random_state=0,n_init='auto').fit(features.cpu().detach().numpy())
        indices = self.indicer(k.labels_)
        features_group = self.make_subbags(indices,features)

        return features_group
    
    def random_grouping(self, features):
        B, N, C = features.shape
        features = features.squeeze()
        indices = split_array(np.array(range(int(N))),self.groups_num)
        features_group = self.make_subbags(indices,features)
        
        return features_group
        
    def seqential_grouping(self, features):
        B, N, C = features.shape
        features = features.squeeze()
        indices = np.array_split(range(N),self.groups_num)
        features_group = self.make_subbags(indices,features)
        
        return features_group

    def idx_grouping(self,idx,features):
        idx = idx.cpu().numpy()
        idx = idx.reshape(-1)
        B, N, C = features.shape
        features = features.squeeze()
        indices = self.indicer(idx)
        features_group = self.make_subbags(indices,features)

        return features_group
