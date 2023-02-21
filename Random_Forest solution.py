import decision_tree_solution as dt  
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math
#reading data and pre-processing
file_name = r'wdbc.data.txt'
data = pd.read_csv(file_name)

y = data['M']  # lakahat amuda M
x = data.drop('M',axis=1).values # lizrok amuda M

y_bool = y.apply(lambda x: 1 if x=='M' else 0).values
x_train1,x_test1,y_train1,y_test1 = train_test_split(x,y_bool,train_size=0.8)
data = np.column_stack( (x_train1,y_train1) )
data_test = np.column_stack( (x_test1,y_test1))

#1
def subsample_create(dataset):
    num_rows = dataset.shape[0]
    num_rows_to_choose = math.sqrt(num_rows)
    rows_to_choose = range(num_rows)
    rows_indexing = np.random.choice(rows_to_choose,num_rows_to_choose) 
    mini_dataset = dataset[rows_indexing,:]
    return mini_dataset
#2
def get_best_split(dataset):
    num_rows = dataset.shape[0]
    num_columns = dataset.shape[1]
    best_gini = 999
    features_available = range(num_columns-1)
    num_features_choosing = math.sqrt(num_columns)
    features_to_search = np.random.choice(features_available,num_features_choosing)  
    for feature_index in features_to_search:
        for row_num in range(num_rows):
            left,right = dt.split_one_node(dataset,feature_index,dataset[row_num,feature_index])
            gini = dt.calcualte_gini(left,right)
            if gini < best_gini:
                best_gini = gini
                best_feature = feature_index
                best_value = dataset[row_num,feature_index]
                best_right = right
                best_left = left
    return best_feature,best_value,best_right,best_left

#3
def random_forests(dataset,n=5):
    trees_list = []
    for i in range(n):                 
        root = dt.Node(dataset,0)      
        root.split_node_recursively()
        trees_list.append(root)
    return trees_list

#4
def bagging_predict(trees_list,data_row):
    all_labels = []
    for tree in trees_list:               
        label = tree.predict(data_row)
        all_labels.append(label)
    num_zeros = all_labels.count(0)      
    num_ones = all_labels.count(1)
    if num_zeros>=num_ones:
        return 0
    else:
        return 1
      
#5
from sklearn.model_selection import KFold
kf = KFold(n_splits=4)   
accuracies = []     
for train_index,test_index in kf.split(x):
    X_train, X_test = x[train_index,:], x[test_index,:]
    y_train, y_test = y_bool[train_index], y_bool[test_index]
    data_train = np.column_stack( (X_train,y_train))    
    trees_list = random_forests(data_train)
    x_test_num_rows = X_test.shape[0]
    labels = []
    for data_row_index in range(x_test_num_rows):
        data_row = X_test[data_row_index]
        label = bagging_predict(trees_list,data_row)    
        labels.append(label)
    accuracy = 1-sum(abs(labels - y_test))/len(y_test)
    accuracies.append(accuracy)
        
#6
#regular random forest    
from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier()
random_forest_model.fit(x_train1,y_train1)
score_normal = random_forest_model.score(x_test1,y_test1)

#changing the number of trees 
random_forest_model = RandomForestClassifier(n_estimators = 5)
random_forest_model.fit(x_train1,y_train1)
score_five_trees = random_forest_model.score(x_test1,y_test1)

random_forest_model = RandomForestClassifier(n_estimators = 30)
random_forest_model.fit(x_train1,y_train1)
score_30_trees = random_forest_model.score(x_test1,y_test1)

#changing the number of features
random_forest_model = RandomForestClassifier(max_features='log2')
random_forest_model.fit(x_train1,y_train1)
score_log2 = random_forest_model.score(x_test1,y_test1)
