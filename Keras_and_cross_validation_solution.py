import numpy  as np
import pandas as pd


diabetes= pd.read_csv('C:/Users/Dom/Desktop/machine_learn_corse/targil29/diabetes.csv', header=None).values
X= diabetes[:,:-1]
y= diabetes[:,-1]


#########################################################################################################################
# first neural network with keras tutorial
from keras.models import Sequential
from keras.layers import Dense

# define the keras model
model = Sequential()   # reshet adain reika, shhavot posledovatelnue
model.add(Dense(12, input_dim=8, activation='relu'))  # input_dim=8 input variables, 12= hidden layer num_neurons 
model.add(Dense(8, activation='relu'))                # 8= hidden layer num_neurons 
model.add(Dense(1, activation='sigmoid'))             # 1=output layer we have one neuron
# compile the keras model= optimizer, the loss function =for a binary classification, to use to evaluate a set of weights
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # compile= eize optimaizer lokhim
# fit the keras model on the dataset, to train our model
model.fit(X, y, epochs=150, batch_size=10)
#model.fit(X, y, epochs=150, batch_size=10, verbose=0)
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
#_, accuracy = model.evaluate(X, y, verbose=0)
print('Accuracy: %.2f' % (accuracy*100))  # Accuracy: 76.56

# make probability predictions with the model
# sigmoid activation function on the output layer, so the predictions will be a probability in the
# range between 0 and 1. We can easily convert them into a crisp binary prediction 
# for this classification task by rounding them.
predictions = model.predict(X)               # 0.3482 , 0.5891, 0.2687
rounded = [round(x[0]) for x in predictions] # round predictions 
for i in range(5):                           # summarize the first 5 cases
	print('%s => %d (expected %d)' % (X[i].tolist(), rounded[i], y[i]))


# make class predictions with the model
predictions = model.predict_classes(X)  # 0 or 1
for i in range(5):                      # summarize the first 5 cases
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))


#########################################################################################################################


# Use scikit-learn to grid search the batch size and epochs
import numpy
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection     import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection     import GridSearchCV

# Define a function returning a keras model 
def create_model(optimizer):
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation="relu"))
    model.add(Dense(8,  activation="relu"))
    model.add(Dense(1,  activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

# fix random seed for reproducibility
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=23)
# create model
model = KerasClassifier(build_fn=create_model)

                        
# define the grid search parameters to optimize
#batch_size = [10,20,30,40,50,60,70,80,90,100]
batch_size = [50,70,100]
epochs     = [10, 50, 100]
optimizer  = ['adam', 'sgd', 'adadelta', 'adagrad', 'adamax', 'nadam', 'rmsprop']
param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer)   # latet be dictionary parametrim

grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X_train,Y_train)   # loss: 0.5736 - accuracy: 0.6938
best_params = grid_result.best_params_
best_score  = grid_result.best_score_ 
 

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# Best: 0.682410 using {'batch_size': 70, 'epochs': 100, 'optimizer': 'nadam'}
means  = grid_result.cv_results_['mean_test_score']
stds   = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



