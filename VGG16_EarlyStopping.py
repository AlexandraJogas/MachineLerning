import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


trdata    = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory='C:\\Users\\Dom\\Desktop\\machine_learn_corse\\targil32\\train\\',      target_size=(224,224))
tsdata    = ImageDataGenerator()
testdata  = tsdata.flow_from_directory(directory='C:\\Users\\Dom\\Desktop\\machine_learn_corse\\targil32\\validation\\', target_size=(224,224))



model = Sequential()
model.add(Conv2D(input_shape=(224,224,3), filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=64,  kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(units=4096, activation="relu"))
model.add(Dense(units=4096, activation="relu"))
model.add(Dense(units=2,    activation="softmax"))


from keras.optimizers import Adam
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
model.summary()


from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import time

checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True,
                              save_weights_only=False, mode='auto', period=1)

earlystop = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')

# https://www.youtube.com/watch?v=BqgTU7_cBnk&t=2s
name="Cats-vs-Dogs-cnn-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='C:\\Users\\Dom\\Desktop\\machine_learn_corse\\targil32\\logs\\{}'.format(name))
#"logs/tf2{}"
#"logs/{}"

# AnacondaPromp:  C:\Users\Dom>tensorboard --logdir=C:\Users\Dom\Desktop\machine_learn_corse\targil32\logs
# http://localhost:6006/

hist = model.fit_generator(steps_per_epoch= 3,
                           generator      = traindata, 
                           validation_data= testdata, 
                           validation_steps=3,
                           epochs=5,
                           callbacks=[checkpoint, earlystop, tensorboard])



#  colab.research.google.com   -   leariz yoter maher reshet



















