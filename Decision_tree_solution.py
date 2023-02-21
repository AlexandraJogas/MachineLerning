
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#reading data and pre-processing
file_name = r'wdbc.data.txt'
data = pd.read_csv(file_name)

y = data['M']  # bohrim amuda M ze label= M, B
x = data.drop('M',axis=1) 

y_bool = y.apply(lambda x: 1 if x =='M' else 0)  
x_train,x_test,y_train,y_test = train_test_split(x,y_bool,train_size=0.8)  
data = np.column_stack( (x_train, y_train) )   
data_test = np.column_stack( (x_test, y_test))
#1
#a
def split_one_node(dataset,index,value):  # lefazel le 2 data
    left = []             
    right = []            
    for row in dataset:   
        if row[index]<value:
            left.append(row)
        else:
            right.append(row)
    return np.array(left), np.array(right)
#b
#first lets calcualte Gini coefficient
def calcualte_gini(left,right):  
    gini = 0
    total_size = len(left) + len(right)  # oreh shurot
    for branch in [left,right]:   
        branch_score = 0          
        branch_size = len(branch) 
        if branch_size==0:
            continue
        for label in [0,1]:
            num_instances = [row[-1] for row in branch].count(label) 
            probability_in_branch = num_instances/branch_size  
            branch_score += probability_in_branch**2           
        gini += (1-branch_score)*branch_size/total_size       
    return gini
    
def get_best_split(dataset):    
    num_rows    = dataset.shape[0]
    num_columns = dataset.shape[1]
    best_gini = 999                  
    for feature_index in range(num_columns-1):
        for row_num in range(num_rows):
            left, right = split_one_node( dataset, feature_index, dataset[row_num,feature_index] )
            gini = calcualte_gini( left, right)   
            if gini < best_gini:  
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
        self.right = None    
        self.left = None
        self.feature_index = None
        self.value = None
        self.depth_in_tree = depth
        self.dataset = dataset
        self.label = None   
        
    def split_node_recursively(self):  
        purity = self.calc_node_purity() 
        if self.depth_in_tree > max_depth or purity == 1:    
            self.label = self.calc_majority_label()           
            return
        else:
            best_feature,best_value,right_dataset,left_dataset = get_best_split(self.dataset)  
            self.feature_index = best_feature  
            self.value = best_value           
            self.right = Node(right_dataset,self.depth_in_tree+1)  
            self.right.split_node_recursively()                   
            self.left = Node(left_dataset,self.depth_in_tree+1)    
            self.left.split_node_recursively()                    
            
    def calc_majority_label(self):
        labeled_zero = [row[-1] for row in self.dataset].count(0)  
        labeled_one =  [row[-1] for row in self.dataset].count(1)
        return 0 if labeled_zero > labeled_one else 1
            
    def calc_node_purity(self):      
        labeled_zero = [row[-1] for row in self.dataset].count(0)
        labeled_one  = [row[-1] for row in self.dataset].count(1)
        majority_label = self.calc_majority_label()
        if majority_label == 0:                                 
            purity = labeled_zero/(labeled_zero+labeled_one)
        else:
            purity = labeled_one/(labeled_zero+labeled_one)   
        return purity
    
    def predict(self,row):       
       if self.label != None:   
           return self.label
       else:                     
           if row[self.feature_index] < self.value:
               return self.left.predict(row) 
           else:
               return self.right.predict(row)
           
root_node = Node(data,1) 
root_node.split_node_recursively()  

test_labels = []      
for row_data in data_test:
    row_label = root_node.predict(row_data)   
    test_labels.append(row_label)

accuracy = 1-sum(abs(test_labels - y_test))/len(y_test)   




