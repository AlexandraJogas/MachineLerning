# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 16:16:50 2019

@author: reutr
"""

import numpy as np    #arrays
from matplotlib import pyplot as plt
from sklearn    import datasets
from scipy.spatial import distance


all_data=datasets.load_iris()    
X = all_data.data
data=X[:,:2]   # lokhim 3 clusterim
#y = all_data.target 

K=3

#choosing K random points to be centroids for start: 
centroids_i=np.random.choice( len(data), K)  # zarih livhor nekodot randomaliet metoh data, k= nekodot
centroids=data[centroids_i]


#clustering points acording to given centroids:
def clustering(points, means):
    clusters=[]
    for point in points:
        i=-1
        cluster=-1          # lisport cluster 0,1,2
        min_dis=1000000
        for mean in means:  # esh K means=3
            i=i+1           #-1+1=0, 0+1=1, 1+1=2 clusters
            dis=distance.euclidean(point, mean) # kol nekuda bodkim 3 merhakim, 3 means 
            if dis<min_dis:
                min_dis=dis
                cluster=i
        clusters.append(cluster)    
    return np.array(clusters)


#changing centroids acording to given clustered points:
def finding_centroids(points, clusters, means):
    new_means=[]
    for i in range(len(means)):     # means= K=3: 0,1,2
        point_of_cluster=points[clusters==i]   # lakahat kol nekodot be clustet i=0,1,2
        new_mean=np.mean(point_of_cluster, axis=0)  # lehashev mean be kol mirhavim features be cluster i
        new_means.append(new_mean)
    return np.array(new_means)

def show_data(data,clusters,means):
    x=data[:,0]
    y=data[:,1]
    plt.scatter(x,y, c=clusters)    # c=color from cluster 0 or 1 or 2
    #means_l=list(means)
    xx=means[:,0]                   # memuzaim leazig be oto graf, bezeva adom
    yy=means[:,1]
    plt.scatter(xx,yy, c='r')
    plt.show()


def is_done(old_means, new_means):   # hishuv distans means bein old and new
    diss=[]
    for i in range(K):    # K= 0,1,2
        diss.append(distance.euclidean(old_means[i],new_means[i]))  # lehashev merhakim, matara she itkansu ihiu oto davar
    avg_diss = np.mean(np.array(diss))
    print(avg_diss)
    if avg_diss<0.0000001:
        return True
    else:
        return False
    

def k_means(data, centroids):
    done=False
    new_means=centroids
    while done==False:  # tnai itkansut
        old_means=new_means
        clusters  = clustering(data, old_means)                  # array[00000,111,222222222]
        new_means = finding_centroids(data, clusters, old_means) # array[[2,2,3,5],[2,6,9,4],[2,6,9,8]]
        show_data(data, clusters, new_means)
        done      = is_done(old_means, new_means)  # new_means kol pam mehashvim me: finding_centroids
    return new_means, clusters

 
final_means, final_clusters=k_means(data, centroids)

