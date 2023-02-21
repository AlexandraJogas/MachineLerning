import numpy as np
from sklearn import datasets
iris_path=''

class DataSet():
    def __init__(self, X,Y):
        assert len(X)==len(Y) # sanity check
        self.X=X
        self.Y=Y
        
    def train_test_random_split(self,p_train):
        """ returns X_train, Y_train, X_test, Y_test according to percentaget
        defined in p_train after randomly permuting the data """
        assert 0<p_train<1
        data_len=len(self.X)
        train_len=round(p_train*data_len)
        perm= np.random.permutation(data_len)
        train_pos=perm[:train_len]
        test_pos=perm[train_len:]
        return self.X[train_pos],self.Y[train_pos],self.X[test_pos],self.Y[test_pos]
    
    # property - enables calling function as as an attribute (this is a get property, set property also exist but has a different synthax)
    @property
    def std(self):
        return self.X.std(axis=0),self.Y.std(axis=0)

    @property
    def mean(self):
        return self.X.mean(axis=0),self.Y.mean(axis=0)
        

def distance(x,y):
    return np.linalg.norm(x-y)


    
class KNN():
    def __init__(self,X_train,Y_train, k):
        self.X_train=X_train
        self.Y_train=Y_train
        self.k=k

    @staticmethod  #static method - a method which does not depend on an 
    # instance and acts exactly like any standalone method outside of a classe
    # hence self is not passed as an argumetn
    def majority_vote(y_labels):
        votes={}
        for label in y_labels:
            if label in votes:
                votes[label]+=1
            else:
                votes[label]=1
        max_votes=0        
        result=None
        for label,votes in votes.items():
            if votes>max_votes:
                max_votes=votes
                result=label
        return result
        
    def _k_neighbours(self,point):
        distances=[]
        for i in range(len(self.X_train)):
            distances.append((self.Y_train[i],distance(point,self.X_train[i])))
        distances.sort(key=lambda tup: tup[1])
        return distances[:self.k]
    
    def _classify_point(self,point):
        k_neighbours=self._k_neighbours(point)
        labels=[tup[0] for tup in k_neighbours]
        # we call the static method using the class name (though self also works)
        return KNN.majority_vote(labels) 
        
    def train(self):
        # no training in this model, we will se later why this was done
        pass
    
    def test(self,X_test):
        results=[]
        for i in range(len(X_test)):
            results.append(self._classify_point(X_test[i]))
        return(np.array(results))

def success_percentage(Y_test,Y_classified):
    assert len(Y_test) == len(Y_classified)
    return (Y_test==Y_classified).sum()/len(Y_test)

iris=datasets.load_iris()
data_set=DataSet(iris.data,iris.target)

# This is not part of KNN, just class demo
print('Data set mean: {}, and standart deviation: {}'.format(data_set.mean, data_set.std))

X_train,Y_train,X_test,Y_test=data_set.train_test_random_split(0.8)
knn=KNN(X_train,Y_train,3)
Y_classified=knn.test(X_test)
print('Success percentage {:.2%}'.format(success_percentage(Y_test,Y_classified)))
