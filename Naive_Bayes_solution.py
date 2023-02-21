import pandas as  pd
import numpy as np
from sklearn.model_selection import train_test_split
import math
import os   #operated sistem, sifria notenet efsharut laasot peulot be maarehet afala

path= 'C:\Alexandra\deep'
os.chdir(path)  #change dir

#1
file_name = 'diabetes.csv'
data = pd.read_csv(file_name).values
x = data[:,:-1]
y = data[:,-1]
x_train,x_test,y_train,y_test = train_test_split(x,y)

#2
mean_x = np.mean(x,axis=0)
std_x = np.std(x,axis=0)

def mean_and_std_per_label(x,y,wanted_label):
    x_with_specific_class = x[y==wanted_label]  # lokhim x bemakom she y=0 or y=1 ze x[index]
    mean = np.mean(x_with_specific_class,axis=0)
    std  = np.std(x_with_specific_class,axis=0)
    return mean, std
    
def calc_mean_std(x,y):
    mean_label_0,std_label_0 = mean_and_std_per_label(x,y,0)
    mean_label_1,std_label_1 = mean_and_std_per_label(x,y,1)
    return mean_label_0,std_label_0,mean_label_1,std_label_1    

#3
#taken from equation of probability of Gaussian
def predict_gussian(x,mean,std, godel_data):
    prob_array = (1/(np.sqrt(2*math.pi)*std))*np.exp(-(x-mean)**2/(2*std**2))
    prob_array = np.reshape(prob_array, (prob_array.shape[0],1))  # asinu reshape, biglyal she prod le avad bez nee
    prob = np.prod(prob_array,axis=0) *godel_data  # *p(A) zabula # lifnei bulo (8,) had memadi asinu dumemadi shape (8,1)
    return     # np.expand_dim- lehosif meimad  np.squeeze-lehorid meimad

#4
   
def predict_on_dataset(x_values,mean_label_0,std_label_0,mean_label_1,std_label_1):    
    prob0 = []
    prob1 = []
    for row in x_values:
        prob0_row = predict_gussian(row,mean_label_0,std_label_0, 0.75) # im o ma sikui ledata sheli
        prob1_row = predict_gussian(row,mean_label_1,std_label_1, 0.25) # im o1ma sikui ledata sheli
        prob0.append(prob0_row)
        prob1.append(prob1_row)
    probabilities = np.column_stack( (prob0,prob1))  # lhaber amudot
    y_calculated = np.argmax(probabilities,axis=1)   # learot index be amudot ma ereh maximali
    return y_calculated

#5
def evaluate_predictions(x_test,y_test,mean0,std0,mean1,std1):
    y_calculated = predict_on_dataset(x_test,mean0,std0,mean1,std1)    
    diff = y_calculated - y_test
    wrong_points = np.sum(np.sqrt(diff**2) )
    length = y_calculated.shape[0]
    accuracy = 1-wrong_points/length
    return accuracy   
  
    
#6    
class NaiveBayesClassifier():

    def __init__(self):
        self.alg_name = 'NaiveBayes'
        self.mean0 = ''
        self.std0 = ''
        self.mean1 = ''
        self.std1 = ''
            
    def fit(self,x_train,y_train):
        mean0,std0,mean1,std1 = calc_mean_std(x_train,y_train)
        self.mean0 = mean0
        self.mean1 = mean1
        self.std0 = std0
        self.std1 = std1
        
    def predict(self,x_test):
        prob0 = []
        prob1 = []
        for row in x_test:
            prob0_row = predict_gussian(row,self.mean0,self.std0)
            prob1_row = predict_gussian(row,self.mean1,self.std1)
            prob0.append(prob0_row)
            prob1.append(prob1_row)
        probabilities = np.column_stack( (prob0,prob1))
        y_calculated = np.argmax(probabilities)
        return y_calculated
    
    def evaluate(self,x_test,y_test):
        accuracy = evaluate_predictions(x_test,y_test,self.mean0,self.std0,self.mean1,self.std1)
        return accuracy
    
if __name__ == '__main__':
    my_classifier = NaiveBayesClassifier()  #meyazrit obect my_classifier
    my_classifier.fit(x_train,y_train)
    accuracy = my_classifier.evaluate(x_test,y_test)    
