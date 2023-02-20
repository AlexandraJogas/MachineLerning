import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

#np.random.seed(5)

n_points_in_cluster = 99
n_clusters = 2
std = 0.3
mean_x1 = 0
mean_y1 = 0
mean_x2 = 1
mean_y2 = 1


pts_clust1 = np.random.multivariate_normal([mean_x1,mean_y1],np.diag([std,std]),n_points_in_cluster)
pts_clust2 = np.random.multivariate_normal([mean_x2,mean_y2],np.diag([std,std]),n_points_in_cluster)

plt.scatter(pts_clust1[:,0],pts_clust1[:,1])
plt.scatter(pts_clust2[:,0],pts_clust2[:,1])
plt.show()

data_points = np.r_[pts_clust1,pts_clust2]

#data_points=data.reshape(-1,2)

colors = cm.rainbow(np.linspace(0, 1, n_clusters))

nn=n_points_in_cluster

#for i in range(n_clusters):
#    plt.scatter(data_points[nn*i:nn*(i+1),0],data_points[nn*i:nn*(i+1),1],color=colors[i])

y=np.repeat(np.array(range(n_clusters)),n_points_in_cluster)
X = np.c_[data_points,np.ones(data_points.shape[0])]

def split_train_test(X,y,percentage_test):
    per_index=int(len(y)*(1-percentage_test))
    return X[:per_index,:],X[per_index:,:],y[:per_index],y[per_index:]

percentage_test = 0.2
indices= np.array(range(n_points_in_cluster*n_clusters))
np.random.shuffle(indices)

X_train, X_test, y_train, y_test = split_train_test(X[indices,:],y[indices],percentage_test)

def logit(z):
    return 1/(1+ np.exp(-z))

def predict_logit(x, teta):
    return logit(np.dot(x,teta))>0.5
            
EPS = 1e-7

def minus_log_likelihood(p,y):
    return -(y*np.log(p+EPS)+(1-y)*np.log(1-p+EPS)).mean()

def grad_minus_log_likelihood(p,y,X):
    return np.dot((p-y),X)/len(X)

def run_gradient_descent_logit(X,y,start,rate,epochs):
    t=start.copy()  
    for epoch in range(epochs):
        p = logit(np.dot(X,t))
        loss = minus_log_likelihood(p,y)
        grad = grad_minus_log_likelihood(p,y,X)
        t=t-rate*grad
        print('epoch {}, loss {}, new t {}'.format(epoch,loss,t))
    return t
        
start =  np.append(np.random.normal(0,0.1,(2)),0)
teta = run_gradient_descent_logit(X_train,y_train,start,0.1,100)
train_precision=(predict_logit(X_train, teta)==y_train).sum()/len(y_train)
test_precision=(predict_logit(X_test, teta)==y_test).sum()/len(y_test)
print('Train precision: {} Test precision: {}'.format(train_precision, test_precision))
print(teta)