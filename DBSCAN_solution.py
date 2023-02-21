from numpy import linalg as LA
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

class DBSCAN():
    
    def __init__(self,data,epsilon,min_points):   
        self.cluster_num=1         
        self.epsilon=epsilon       
        self.min_points=min_points 
        self.points_dist=np.zeros((len(data),len(data))) 
        self.classification=np.zeros(len(data))
        self.data=data
        
        #building length matrics
        for i in range(len(self.data)):
            for j in range(len(self.data)):
                self.points_dist[i][j]=LA.norm(self.data[i]-self.data[j]) 

    def get_neighbors(self,point):  
        neighbors=[]                
        for i in range(len(self.points_dist)):
            if ((self.points_dist[i][point]<=self.epsilon) and (i!=point)):  
                neighbors.append(i)
        return neighbors

    def expand_cluster(self,seed):        
        cluster=self.get_neighbors(seed)  
        self.classification[seed]=self.cluster_num 
        while len(cluster):  
            next_gen=[]
            for i in cluster:
                sub_cluster=self.get_neighbors(i)  
                self.classification[i]=self.cluster_num 
                
                #if a neighbor is a cluster point, add all his neighbors to the cluster
                if len(sub_cluster)>=self.min_points: #
                    for j in sub_cluster:  
                        if self.classification[j]==0: 
                            next_gen.append(j)             
            cluster=next_gen[:] 

    def show_clusters(self,x,y): 
            X=self.data[:,x]
            Y=self.data[:,y]
            fig, ax = plt.subplots()          
            for i in range(self.cluster_num):  
                if i==0:
                    plt.scatter(X[self.classification==i],Y[self.classification==i],label="Noise", alpha=0.3, edgecolors='none')
                else:
                    plt.scatter(X[self.classification==i],Y[self.classification==i],label=("cluster# "+str(i)))
            ax.legend()
            plt.show()

    def clusterize(self): #main function running on all data points        
        for i in range(len(self.data)):  
            if (len(self.get_neighbors(i))>=self.min_points) and (self.classification[i]==0): 
                self.expand_cluster(i)
                self.cluster_num+=1 

if __name__=='__main__':
    data = load_iris()
    epsilon=0.5
    min_points=3
    
    test=DBSCAN(data.data, epsilon, min_points)
    test.clusterize()
    test.show_clusters(1,3)
