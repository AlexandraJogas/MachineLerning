# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 11:37:43 2018

@author: Lea
"""

import matplotlib.pyplot as plt
from sklearn import datasets,svm,model_selection
import numpy as np
#import seaborn as sns

# question 4
iris = datasets.load_iris()

#question 5
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

#question 6
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y)
mySvm = svm.SVC(gamma='auto')   # default  gamma=1/n_features
mySvm.fit(X_train,y_train)
score_reg1 = mySvm.score(X_test,y_test)

#question 7
artificial_data = np.array([[5,3.1],[4,2.7],[4.6,3.9],[5.8,3]])
labels_artificial = mySvm.predict(artificial_data)

#question 8
plt.scatter(artificial_data[:,0], artificial_data[:,1], c=labels_artificial)  # zir x, zir y, c=vector beoreh data, she ekabel zeva

#question 9
support_vectors = mySvm.support_vectors_
plt.scatter(support_vectors[:,0],support_vectors[:,1],c='r')
plt.show()

#question 10
X_10 = iris.data  # we only take the first two features.
y = iris.target
X_train_10, X_test_10, y_train, y_test = model_selection.train_test_split(X_10,y)

#question 11
#a. linear
my_svm_linear = svm.SVC(kernel='linear')
my_svm_linear.fit(X_train_10,y_train)
support_vectors_linear = my_svm_linear.support_vectors_
plt.scatter(support_vectors_linear[:,0],support_vectors_linear[:,1],c='r')
score_linear1 = my_svm_linear.score(X_test_10,y_test)

#b. plynomial kernel level 3
my_svm_poly = svm.SVC(kernel = 'poly',degree=3)
my_svm_poly.fit(X_train_10,y_train)
suppor_vectors_poly = my_svm_poly.support_vectors_
plt.scatter(suppor_vectors_poly[:,0],suppor_vectors_poly[:,1],c='r')
score_poly1 = my_svm_poly.score(X_test_10,y_test)

#c radial kernel (rbg)
my_svm_raidal = svm.SVC(kernel = 'rbf')
my_svm_raidal.fit(X_train_10,y_train)
support_vectors_raidal = my_svm_raidal.support_vectors_
plt.scatter(support_vectors_raidal[:,0],support_vectors_raidal[:,1],c='r')
score_radial1 = my_svm_raidal.score(X_test_10,y_test)

#question 12
digits = datasets.load_digits()
x_digits = digits.data
y_digits = digits.target

#question 13
#for i in range(10):
#    plt.imshow(digits.images[i])
    #plt.show() is in order to make sure that the next 
    #figure is not above the currnt one but a new one
#    plt.show()  
   
#question 14
num_rows = x_digits.shape[0]
num_rows_for_train = round(num_rows/2)
x_train_digits = x_digits[0:num_rows_for_train,:]   
y_train_digits = y_digits[0:num_rows_for_train] 
my_svm = svm.SVC(kernel='poly',gamma='auto')
my_svm.fit(x_train_digits,y_train_digits)

#question 15
x_test_digits = x_digits[num_rows_for_train:,:]
y_test_digits = y_digits[num_rows_for_train:]
score = my_svm.score(x_test_digits,y_test_digits)

#question 16
#a. c=0.0000001
my_svm_1 = svm.SVC(C=0.0000001,kernel='poly')
my_svm_1.fit(x_train_digits,y_train_digits)
score_1 = my_svm_1.score(x_test_digits,y_test_digits)

#b c = 4
my_svm_2 = svm.SVC(C=4,kernel='poly')
my_svm_2.fit(x_train_digits,y_train_digits)
score_2 = my_svm_2.score(x_test_digits,y_test_digits)

#c c = 0.000000001
my_svm_3 = svm.SVC(C=0.000000001,kernel='poly')
my_svm_3.fit(x_train_digits,y_train_digits)
score_3 = my_svm_3.score(x_test_digits,y_test_digits)

fig = plt.figure()
for i in range(1, 11):
    ax = fig.add_subplot(2, 5, i)
    ax.matshow(digits.images[i])
    plt.gray()
    plt.axis('off')
    plt.subplots_adjust(wspace=None, hspace=0.05)
    if i == 10:
        ax.set_title(str(0))
    else:
        ax.set_title(str(i))
plt.show()
plt.savefig('digits_data.png')  # lishmor bekovez tmunot

import pickle
file_name = 'my_file_pickle'
fid = open(file_name,'wb')
pickle.dump(my_svm_1,fid)   # save be C:\Users\Dom
fid_r = open(file_name,'rb')
my_svm_resoted = pickle.load(fid_r)

import joblib
file_name_jlib = 'my_file_jlib'
joblib.dump(my_svm_1,file_name_jlib)   # save be C:\Users\Dom
my_svm_joblib = joblib.load(file_name_jlib)
my_svm_joblib.score(x_test_digits,y_test_digits)
