# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 20:18:34 2018

@author: kuzmin
"""

from numpy import linalg as LA
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

class DBSCAN():
    
    def __init__(self,data,epsilon,min_points):   # __init__= ze constructor
        self.cluster_num=1         #running cluster number
        self.epsilon=epsilon       #neighborhood radius
        self.min_points=min_points #minimum neighbors for cluster point
        self.points_dist=np.zeros((len(data),len(data))) 
        self.classification=np.zeros(len(data))
        self.data=data
        
        #building length matrics
        for i in range(len(self.data)):
            for j in range(len(self.data)):
                self.points_dist[i][j]=LA.norm(self.data[i]-self.data[j])  # norma shel efresh

    def get_neighbors(self,point):  # bodeket ma nikudot she nimzaot be sviva
        neighbors=[]                # np.nonzero(self.points_dist[i][point]<=self.epsilon) mahzir index indexim 0,1
        for i in range(len(self.points_dist)):
            if ((self.points_dist[i][point]<=self.epsilon) and (i!=point)):  # nekuda she le be alahson
                neighbors.append(i)
        return neighbors

    def expand_cluster(self,seed):        # self mazbia al objeict she anu nimzaim bo
        cluster=self.get_neighbors(seed)  # list of unclustered neighbors, mazanu kol haverim shel seed mesaviv
        self.classification[seed]=self.cluster_num # natanu cluster le seed
        while len(cluster):  # al kol ehad minekudot be sviva naavor al lehem
            next_gen=[]
            for i in cluster:
                sub_cluster=self.get_neighbors(i)   # kol nekudot be sviva
                self.classification[i]=self.cluster_num #gave a classification to all neighbors # notnim lahem cluster
                
                #if a neighbor is a cluster point, add all his neighbors to the cluster
                if len(sub_cluster)>=self.min_points: # nekudot hadashot, im kamut nekudot be sviva> minPts
                    for j in sub_cluster:   # bodkim nekuda, nekuda
                        if self.classification[j]==0: #if not clustered yet
                            next_gen.append(j)             
            cluster=next_gen[:] #proceed to next generation

    def show_clusters(self,x,y): #input - axis column x and y ,x=col, y=col
            X=self.data[:,x]
            Y=self.data[:,y]
            fig, ax = plt.subplots()           # mahzir 2 mishtanim figure, axis
            for i in range(self.cluster_num):  # ovrim al mispar clusters 0,1,2,3,  o=Noise
                if i==0:
                    plt.scatter(X[self.classification==i],Y[self.classification==i],label="Noise", alpha=0.3, edgecolors='none')
                else:
                    plt.scatter(X[self.classification==i],Y[self.classification==i],label=("cluster# "+str(i)))
            ax.legend()
            plt.show()

    def clusterize(self): #main function running on all data points        
        for i in range(len(self.data)):  # ovrim al kol data ehad, ehad, lehapes nekudot be sviva, ona core point i latet cluster
            if (len(self.get_neighbors(i))>=self.min_points) and (self.classification[i]==0): #new cluster point
                self.expand_cluster(i)
                self.cluster_num+=1 #increase number for next cluster

if __name__=='__main__':
    data = load_iris()
    epsilon=0.5
    min_points=3
    
    test=DBSCAN(data.data, epsilon, min_points)
    test.clusterize()
    test.show_clusters(1,3)#input - axis column x and y
