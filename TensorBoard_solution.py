# -*- coding: utf-8 -*-
"""
@author: Dom
"""
# C:\ProgramData\Anaconda3\pythonw.exe 
# https://www.youtube.com/watch?v=BqgTU7_cBnk&t=2s

import numpy  as np
import pandas as pd


diabetes1= pd.read_csv('C:/Users/Dom/Desktop/machine_learn_corse/targil31/diabetes.csv', header=None).values                    
X= diabetes[:,:-1]
y= diabetes[:,-1]


#########################################################################################################################
# Neural network with keras 
from keras.models    import Sequential
from keras.layers    import Dense
from keras.callbacks import Tensorboard

# define the keras model
model = Sequential()   # reshet adain reika, shhavot posledovatelnue
model.add(Dense(12, input_dim=8, activation='relu'))  # input_dim=8 input variables, 12= hidden layer num_neurons 
model.add(Dense(8, activation='relu'))                # 8= hidden layer num_neurons 
model.add(Dense(1, activation='sigmoid'))             # 1=output layer we have one neuron
# compile the keras model= optimizer, the loss function =for a binary classification, to use to evaluate a set of weights
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # compile= eize optimaizer lokhim
# fit the keras model on the dataset, to train our model
# Be Anaconda Promt :  TensorBoard --logdir=C:/Users/Dom/Downloads  (leyazer makom or shem tikia eifo lishmor log)
tensorboard= TensorBoard(log_dir="logs/{}".format(time()))  # create TensorBoardand point it to log directory where data should be collected
model.fit(X, y, epochs=150, batch_size=10, verbose=1, callbacks=[tensorboard])
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
#_, accuracy = model.evaluate(X, y, verbose=0)
print('Accuracy: %.2f' % (accuracy*100))  # Accuracy: 76.56


#########################################################################################################################
# Neural network with TensorFlow version 1 

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tf.summary.scalar(loss)
tf.summary.scalar(binary_crossentropy)
merged= tf.summary.merge_all()

file_writer= tf.create_file_writer()
with tf.session()  as sess:
        l,m= sess.run([loss,merged])
        file_writer.add_sammary(m)
        

#########################################################################################################################
# Neural network with TensorFlow version 2

import tensorflow as tf

writer= tf.summary.create_file_writer("/tmp/mylogs")
with writer.as_default():
         tf.summary.scalar(loss)
         tf.summary.scalar(binary_crossentropy)
         writer.flush





        
        
        
        
        







