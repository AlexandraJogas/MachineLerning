# Classifier flower Iris for 4 features: Sepal_length, Sepal_width, Petal_length, Petal_width     x= np.array([ 7.9, 3.1, 7.5, 1.8 ])  
# Y_Class=['Setosa', 'Versicolor','Virginica']
# X_Column=[Sepal_length, Sepal_width, Petal_length, Petal_width]
# h=F(xW+b) ,  x=[x1 x2], W=[ w11 w12 w13, w21 w22 w23 ], b=[b1 b2 b3]
# h1=F(x1w11+x2w21+b1), h2=F(x1w12+x2w22+b2), h3=F(x1w13+x2w23+b3)
# h=F(xW+b) --> t=xW+b, h=F(t)
# input layer:  4 neurons  ( X_Colum: Petal_length, Petal_width, Sepal_length, Sepal_widthn)
# 1 layer:      5 neurons  ( we choose ourselves)
# output layer: 3 neurons  ( Y_Class, last layer equal to number predict classes: 'Setosa', 'Versicolor','Virginica')
# t1=xW1+b1                (layer1, x,b-vectors,W1-matrix,  vector x--> W1+b1 -->t1 neuron)
# h1=F(t1)                 (layer1, h-vectors)
# t2=h1W2+b2               (layer2, x,b-vectors,W2-matrix,  neuron h1-->w2+b1 -->t2 neuron, not need activation function h2=F(t2) because is last layer)
# z=Softmax(t2)=S(t2)      (vector probabilities:  S(t)=e^ti/sum(e^ti) )
# F(t)=ReLu(t)=max(o,t)    (non-linear activation function)

import numpy as np

input_dim = 4      #(number of input  x)       x= np.array([ 7.9, 3.1, 7.5, 1.8 ]) 
out_dim   = 3      #(number of output y)       Y_Class=['Setosa', 'Versicolor','Virginica']
h_dim     = 10     #(number neurons in layer1)


# Weights and x temporarily will in random vector from normal distribution, after learning neural network it will learning weights

x = np.random.randn(input_dim)          # random vector from normal distribution    (1x4)
W1= np.random.randn(input_dimn, h_dimn) # weight matrix for layer1                  (4x10)
b1= np.random.randn(input_dimn, h_dimn) # vector bias= number of neurons in layaer1 (1x10)
W2= np.random.randn(h_dimn, out_dim)    # weight matrix for layer2                  (10x3)
b2= np.random.randn(out_dim)            # vector bias= number of neurons in layaer2 (1x3)

def_relu(t):
  return np.maximum(t, 0)

def softmax(t):
  out=np.exp(t)               # vector exponent
  return  out/np.sum(out)     # vector probabilities:  S(t)=e^ti/sum(e^ti) 
           
  
# Forward Propagation or Inference:
def predict(x):
    t1= x @ W1 + b1    # (@=dot, vector x multiply by matrix W1 + vector bias b1  is linear function, pass from x to t1)
    h1= relu(t1)       # (non-linear activation function for t1: give vector t1 --> return vector h1 is neuron in layer1)
    t2= h1 @ W2 +b2    # (is linear function, pass from h1 to t2 layer2)
    z=softmax(t2)      # (vector probabilities in function softmax)
    
probs=predict(x)                               # (return vector with 3 probabilities   out_dim = 3  number of output y)
pred_index_class = np.argmax(probs)            # (return index of class vector with maxmimum probability: 0 or 1 or 2)
Y_Class=['Setosa', 'Versicolor','Virginica'] 
print('Predicted class' ,  Y_Class[pred_index_class])


x= np.array([7.9, 3.1, 7.5, 1.8])   # give features of Iris in x vector
