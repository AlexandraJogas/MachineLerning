import keras
from keras.models import Model
from keras.layers import Dense
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image


trdata    = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory='C:\\Users\\Dom\\Desktop\\machine_learn_corse\\targil32\\train\\',      target_size=(224,224))
tsdata    = ImageDataGenerator()
testdata  = tsdata.flow_from_directory(directory='C:\\Users\\Dom\\Desktop\\machine_learn_corse\\targil32\\validation\\', target_size=(224,224))


from keras.applications.vgg16 import VGG16
vggmodel = VGG16(weights='imagenet', include_top=True)   # lokhim VGG16 reshet meumenet kvar
vggmodel.summary()


for layers in (vggmodel.layers)[:19]:   # lehabot kol shhavot ad shura 19 = list shhavot kan le ihie edkun mishkolot
    print(layers)
    layers.trainable = False
    
    
X = vggmodel.layers[-2].output                     # lakahat 2 shhavot ahronot
predictions = Dense(2, activation="softmax")(X)    # laasot classification, poelet al output beshihva ahat -2, osim keasher ein maspik data, reshet kvar meumenet, lokhim mishkolot muhanot
                                                   # Dense(2)= two classes so the last dense layer of model should be a 2 unit softmax dense layer
model_final = Model(input = vggmodel.input, output = predictions)    # esh secution, or model


from keras.optimizers import SGD
opt = SGD(lr=0.0001, momentum=0.9)
model_final.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
model_final.summary()


from keras.callbacks import ModelCheckpoint, EarlyStopping,  TensorBoard
import time

checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, 
                             save_weights_only=False, mode='auto', period=1)  # shomeret natunim data aharei kol epoch
                                                                              # shomer tozaa ahi tova mishkolot, loss
earlystop = EarlyStopping(monitor='val_acc', min_delta=0, patience=40, verbose=1, mode='auto')  # im kovaat parametr ve hu le mishtane le ever iter az hu mafsik leariz

name="Cats-vs-Dogs-cnn-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='C:\\Users\\Dom\\Desktop\\machine_learn_corse\\targil32\\logs\\{}'.format(name))



model_final.fit_generator(steps_per_epoch= 2,
                           generator      = traindata, 
                           validation_data= testdata, 
                           validation_steps=1,
                           epochs=5,
                           callbacks=[checkpoint, earlystop, tensorboard])  # tihie tuple zugot (label, pred))

model_final.save_weights("vgg16_1.h5")




