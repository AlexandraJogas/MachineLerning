# -*- coding: utf-8 -*-
"""
@author: Dom
"""

###########   For version TF2.0  #############
import tensorflow as tf  
import numpy as np

print(tf.__version__)

node1= tf.constant([1,2,3,4,5])   # <tf.Tensor: id=0, shape=(5,), dtype=int32, numpy=array([1, 2, 3, 4, 5])>
node2= tf.constant([1,1,2,3,5])   # <tf.Tensor: id=1, shape=(5,), dtype=int32, numpy=array([1, 1, 2, 3, 5])>
node3= tf.multiply(node1, node2)  # tf.Tensor([ 1  2  6 12 25], shape=(5,), dtype=int32)
print(node3)    

print(node3.numpy())              # [ 1,  2,  6, 12, 25]

node4=node1+node2                 # <tf.Tensor: id=3, shape=(5,), dtype=int32, numpy=array([ 2,  3,  5,  7, 10])>
print(node3)    


###########   For version TF1.0  #############

import tensorflow as tf  
import tensorflow.compat.v1 as tf1

tf1.disable_eager_execution()
#tf1.disable_v2_behavior() 
node1= tf1.constant([1,2,3,4,5])    # <tf.Tensor 'Const:0' shape=(5,) dtype=int32>
node2= tf1.constant([1,1,2,3,5])    # <tf.Tensor 'Const_1:0' shape=(5,) dtype=int32>
node3= tf1.multiply(node1, node2)   # <tf.Tensor 'Mul:0' shape=(5,) dtype=int32>

sess= tf1.Session()
print(sess.run(node3))          # [ 1,  2,  6, 12, 25]

node4=node1+node2               # <tf.Tensor 'add:0' shape=(5,) dtype=int32>
print(sess.run(node4))          # [ 2  3  5  7 10]



x=tf1.placeholder(tf1.int32)  # <tf.Tensor 'Placeholder:0'   shape=<unknown> dtype=int32>
y=tf1.placeholder(tf1.int32)  # <tf.Tensor 'Placeholder_1:0' shape=<unknown> dtype=int32>
z=x+y                         # <tf.Tensor 'add_2:0'         shape=<unknown> dtype=int32>
sess= tf1.Session() 
print(sess.run(z, feed_dict= {x:1.0, y:5.0}))  # (z=output, feed_dict=input) = 6


w=tf1.get_variable("w1", initializer=tf1.constant(7)) # <tf.Variable 'w1:0'  shape=() dtype=int32>
b=tf1.get_variable("b1", initializer=tf1.constant(8)) # <tf.Variable 'b1:0'  shape=() dtype=int32>
suum=w+b                                              # <tf.Tensor 'add_3:0' shape=() dtype=int32>
sess= tf1.Session() 
sess.run(tf1.global_variables_initializer())
print(sess.run(suum))  # 15



##########   Linear Regression    #########

TRUE_W = 3.0
TRUE_b = 2.0
x='data_for_linear_regression_tf.csv'
x= tf.random.normal(shape=[1000])
noise= tf.random.normal(shape=[1000])
y= x* TRUE_W + TRUE_b + noise


class MyModel:                # create class Linear Regression 
    def __init__(self):       # method init for memorize self parameters
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
       outpu=self.W * x + self.b 
       return outpu 
         
 
model= MyModel()    
model.__call__(x)     

def loss(predicted_y, target_y):
        return tf.reduce_mean(tf.square(predicted_y - target_y))  # (1/N)(predicted_y-target_y)^2 
                                                                  # compute the mean of elements accross demension
    

#######################################################################
import matplotlib.pyplot as plt
plt.scatter(x, y, c='b')          # plot data
plt.scatter(x, model(x), c='r')   # plot prediction model
plt.show()
print('Current loss: %1.6f' % loss(model(x), y).numpy())
#######################################################################



def train(model, x, y, learning_rate):
  with tf.GradientTape() as t:                    # GradientTape=for automatic differentiation
       current_loss = loss(model(x), y)
       dW, db = t.gradient(current_loss, [model.W, model.b])
       model.W.assign_sub(learning_rate * dW)     # assign_sub=for decrementing a value
       model.b.assign_sub(learning_rate * db)   
    
  

Ws, bs = [], []  # Collect the history of W-values and b-values to plot later
epochs = range(1000)
for epoch in epochs:
  Ws.append(model.W.numpy())
  bs.append(model.b.numpy())
  current_loss = loss(model(x), y)

  train(model, x, y, learning_rate=0.1)
  print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' %
        (epoch, Ws[-1], bs[-1], current_loss))

# Let's plot it all
plt.plot(epochs, Ws, 'r',
         epochs, bs, 'b')
plt.plot([TRUE_W] * len(epochs), 'r--',
         [TRUE_b] * len(epochs), 'b--')
plt.legend(['W', 'b', 'True W', 'True b'])
plt.show()
 
    
    
    




