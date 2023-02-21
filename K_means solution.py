import numpy as np    
from matplotlib import pyplot as plt
from sklearn    import datasets
from scipy.spatial import distance


all_data=datasets.load_iris()    
X = all_data.data
data=X[:,:2]   
#y = all_data.target 

K=3

#choosing K random points to be centroids for start: 
centroids_i=np.random.choice( len(data), K)  
centroids=data[centroids_i]


#clustering points acording to given centroids:
def clustering(points, means):
    clusters=[]
    for point in points:
        i=-1
        cluster=-1          
        min_dis=1000000
        for mean in means:  
            i=i+1          
            dis=distance.euclidean(point, mean) s 
            if dis<min_dis:
                min_dis=dis
                cluster=i
        clusters.append(cluster)    
    return np.array(clusters)


#changing centroids acording to given clustered points:
def finding_centroids(points, clusters, means):
    new_means=[]
    for i in range(len(means)):    
        point_of_cluster=points[clusters==i]   
        new_mean=np.mean(point_of_cluster, axis=0)  
        new_means.append(new_mean)
    return np.array(new_means)

def show_data(data,clusters,means):
    x=data[:,0]
    y=data[:,1]
    plt.scatter(x,y, c=clusters)    
    #means_l=list(means)
    xx=means[:,0]                   
    yy=means[:,1]
    plt.scatter(xx,yy, c='r')
    plt.show()


def is_done(old_means, new_means):   
    diss=[]
    for i in range(K):    # K= 0,1,2
        diss.append(distance.euclidean(old_means[i],new_means[i]))  
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
        clusters  = clustering(data, old_means)                 
        new_means = finding_centroids(data, clusters, old_means)
        show_data(data, clusters, new_means)
        done      = is_done(old_means, new_means) 
    return new_means, clusters

 
final_means, final_clusters=k_means(data, centroids)

