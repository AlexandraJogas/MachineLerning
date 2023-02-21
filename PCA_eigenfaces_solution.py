import numpy as np
import os
import imageio as io
from   skimage import color
import matplotlib.pyplot as plt
from   sklearn.model_selection import train_test_split
from   sklearn.decomposition   import PCA

path = r'C:\Users\Alexandra\Downloads\faces94\faces94'
h = 200
w = 180

def read_images(path):
    images = {}

    X = []
    y = []
    
    index = 0
    for root, dirs, files in os.walk(path,topdown=True):   
        for name in files:                                 
            full_file = os.path.join(root, name)
            if full_file.endswith(".jpg"):                
                #images[name] = color.rgb2gray(io.imread(full_file)).reshape([1,w*h]) 
                X.append(color.rgb2gray(io.imread(full_file)).reshape([1,w*h]))      
                y.append(name)   
                index +=1
    X = np.array(X)
    X = X.squeeze()   
    
    return X


# establish the pca

def run_sklearn_pca(X_data):
    n_components = 100
     
    pca = PCA(n_components=n_components).fit(X_data)
#    pca = PCA(n_components=n_components,whiten=True,svd_solver='randomized').fit(X_train)

    eigenfaces = pca.components_.reshape([n_components,h,w])  
    
    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]  

#    first_vector_projections = pca.transform(X_data[0].reshape(1,X_data.shape[1]))
    first_vector_projections = np.dot((X_data[0]-pca.mean_).reshape(1,X_data.shape[1]),pca.components_.T)  
    reconstructed_img = np.expand_dims(pca.mean_,axis=0) + np.dot(first_vector_projections, eigenfaces.reshape(n_components,w*h)) 
    plt.imshow(reconstructed_img.reshape( (h,w)),cmap=plt.cm.gray)                 
    plt.show()
    plt.imshow(X_data[0].reshape( (h,w)),cmap=plt.cm.gray)
    plt.show()
    
    plot_gallery(eigenfaces,eigenface_titles,h,w)

def plot_gallery(images,titles,h,w,n_row=2,n_col=4):
    plt.figure(figsize=(1.8*n_col,2.4*n_row))                              
    plt.subplots_adjust(bottom=0,left=0.01,right=0.99,top=0.9,hspace=0.35) 
    for i in range(n_row*n_col):
        plt.subplot(n_row,n_col,i+1)
        plt.imshow(images[i].reshape((h,w)),cmap=plt.cm.gray)              
        plt.title(titles[i],size=12)
        plt.xticks(())    
        plt.yticks(())
    plt.show()            
    
##################################################################################################    

def run_pca_linalg(XX):
    
    mean_image = np.mean(XX,axis=0)
    XX -= mean_image                          
        
#    Cov = np.dot(XX,XX.transpose())/XX.shape[1]
#    Cov = np.dot(XX,XX.transpose())/XX.shape[0]
    Cov = np.dot(XX,XX.transpose())
    
    eigen_val, eigen_vec = np.linalg.eig(Cov)
    eigen_val = eigen_val.real
    eigen_vec = eigen_vec.real
    
    # sort the eigenvec
    
    indices = np.argsort(eigen_val)[::-1]
    eigen_val = eigen_val[indices]
    eigen_vec = eigen_vec[indices]
    
    u = np.sum(eigen_val[:])
    epsilon = 0.9
    eigen_trshehold = 0
    k = 0
    
    for eigen in eigen_val:
        k += 1
        eigen_trshehold += eigen / u
        if eigen_trshehold >= epsilon:
            break
    print(k)
    eigen_val = eigen_val[:k]   
    eigen_vec = eigen_vec[:,:k]
        
    eigen_faces = np.dot(eigen_vec.T,XX)  

    norm = np.expand_dims(np.linalg.norm(eigen_faces,axis=1),axis=1)  
    eigen_faces /= norm                                               

    eigenface_titles = ["eigenface %d" % i for i in range(eigen_faces.shape[0])]
         
    first_vector_projections = np.dot(eigen_faces,np.expand_dims(XX[0],axis=1))
    reconstructed_img = mean_image + np.dot(first_vector_projections.T,eigen_faces.reshape(k,w*h))
    plt.imshow(reconstructed_img.reshape( (h,w)),cmap=plt.cm.gray)
    plt.show()
    plt.imshow(X_data[0].reshape( (h,w)),cmap=plt.cm.gray)
    plt.show()

    
    plot_gallery(eigen_faces,eigenface_titles,h,w)
    
if __name__ == '__main__':   # include
    X_data = read_images(path)
 
    run_sklearn_pca(X_data)
    
    run_pca_linalg(X_data)
    
    
    
    
