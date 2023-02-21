# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 18:14:12 2018

@author: Lea
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#reading data and pre-processing
file_name = r'wdbc.data.txt'
data = pd.read_csv(file_name)

y = data['M']  # bohrim amuda M ze label= M, B
x = data.drop('M',axis=1) # drop amuda M, axis=1, ki rozim laasot tree al X

y_bool = y.apply(lambda x: 1 if x =='M' else 0)  # apply=lasim function(), lambda kshe rozim lehishtamesh ba paam ehad
x_train,x_test,y_train,y_test = train_test_split(x,y_bool,train_size=0.8)  #random shuffle, kol paam nekabel toza aheret
data = np.column_stack( (x_train, y_train) )   # mehaberet amudot
data_test = np.column_stack( (x_test, y_test))
#1
#a
def split_one_node(dataset,index,value):  # lefazel le 2 data
    left = []             # left=dataset[dataset[index]< value]   iten maarah true false
    right = []            # left=dataset[dataset[index]>=value]   iten maarah true false
    for row in dataset:   # ovrim shura-shura, bodkim katan, gadol, ve sholhim left, right
        if row[index]<value:
            left.append(row)
        else:
            right.append(row)
    return np.array(left), np.array(right)
#b
#first lets calcualte Gini coefficient
def calcualte_gini(left,right):  # mehashvim gini_left(0,1)+gini_right(0,1)
    gini = 0
    total_size = len(left) + len(right)  # oreh shurot
    for branch in [left,right]:   #over kodem al right, ve az left
        branch_score = 0          # count kdei lehashev sum(p^2)
        branch_size = len(branch) # kama shurot data be left, right, for hishuv istabrut
        if branch_size==0:
            continue
        for label in [0,1]:
            num_instances = [row[-1] for row in branch].count(label) # meyazer list labels: func: kama lebels 0,1 be left or be right
            probability_in_branch = num_instances/branch_size  # hishuv istabrut
            branch_score += probability_in_branch**2           # sum(istabrut^2)
        gini += (1-branch_score)*branch_size/total_size        # 1-sum(p^2)* mishkal === in right+ in left
    return gini
    
def get_best_split(dataset):    # overt al kol options xi attributes, ve lokahat ahi tova
    num_rows    = dataset.shape[0]
    num_columns = dataset.shape[1]
    best_gini = 999                  # shomrim gini min ahi tov  gini[0...1]
    for feature_index in range(num_columns-1):
        for row_num in range(num_rows):
            left, right = split_one_node( dataset, feature_index, dataset[row_num,feature_index] )
            gini = calcualte_gini( left, right)   # kibalnu left data, right data, lehashev gini
            if gini < best_gini:   # nishmor kol arahim the best if mitkaem minimum, kan oved al node ehad
                best_gini = gini
                best_feature = feature_index
                best_value = dataset[row_num,feature_index]
                best_right = right
                best_left = left
    return best_feature,best_value,best_right,best_left

#c
max_depth = 6    
class Node(object):
    
    def __init__(self,dataset,depth):
        self.right = None    # meathelim left, right, akol be athala
        self.left = None
        self.feature_index = None
        self.value = None
        self.depth_in_tree = depth
        self.dataset = dataset
        self.label = None    # ein label beathala
        
    def split_node_recursively(self):   # lokahat node, mefazelet left, right, ve az right mefazlim, ve az left mefazlim
        purity = self.calc_node_purity()  # mehashevet purity nodes
        if self.depth_in_tree > max_depth or purity == 1:     # esh klal azira shel rekursia, esh datot she purity =0.9 le tamid =1
            self.label = self.calc_majority_label()           # iganu le ale, leazig tozaa, label, rov kovea majority be ale
            return
        else:
            best_feature,best_value,right_dataset,left_dataset = get_best_split(self.dataset)  # mi feature xi mefazel bezura ahi tova, raz al kol opziet, 
            self.feature_index = best_feature  # ma feature ahi tov
            self.value = best_value            # ma value ahi tov
            self.right = Node(right_dataset,self.depth_in_tree+1)  # lefi ze nivne node right, nekadem omek
            self.right.split_node_recursively()                    # funkzia koret od paam et azma, yordim ahi amok be ez, ve az olhim le left
            self.left = Node(left_dataset,self.depth_in_tree+1)    # lefi ze nivne node left, nekadem omek
            self.left.split_node_recursively()                     # funkzia koret od paam et azma, olhim amok ad sof be left
            
    def calc_majority_label(self):
        labeled_zero = [row[-1] for row in self.dataset].count(0)  #sofrim be ale mi esh yoter label 0 or 1
        labeled_one =  [row[-1] for row in self.dataset].count(1)
        return 0 if labeled_zero > labeled_one else 1
            
    def calc_node_purity(self):       # bishvil hishuv tnai azira, le tamid purity tihie=1, esh data she ihie 0.9
        labeled_zero = [row[-1] for row in self.dataset].count(0)
        labeled_one  = [row[-1] for row in self.dataset].count(1)
        majority_label = self.calc_majority_label()
        if majority_label == 0:                                 # im 0 yoter az purity mehashvim bishvil 0
            purity = labeled_zero/(labeled_zero+labeled_one)
        else:
            purity = labeled_one/(labeled_zero+labeled_one)    # im 1 yoter az purity mehashvim bishvil 1
        return purity
    
    def predict(self,row):       # bodkim test
       if self.label != None:    # im be ale tahzir label
           return self.label
       else:                     # aheret tileh right or left, lefi value <, >=
           if row[self.feature_index] < self.value:
               return self.left.predict(row) # osim function predict al left, predict tamid mahzir label, ve bodkim oto shuv ad she le None
           else:
               return self.right.predict(row)
           
root_node = Node(data,1)  # yazarnu root node, omek 1
root_node.split_node_recursively()  # bonim tree mathilim me root, al train

test_labels = []      #bodkim ad kama asinu tov, bodkim al test
for row_data in data_test:
    row_label = root_node.predict(row_data)   # mafilim predict al test, shomrim label from tree
    test_labels.append(row_label)

accuracy = 1-sum(abs(test_labels - y_test))/len(y_test)    # mashvim test and prediction be tree, haim nahon, kama nahon sivagnu




