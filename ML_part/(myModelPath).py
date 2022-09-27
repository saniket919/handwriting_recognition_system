# coding: utf-8
model_S=keras.models.load_model("C:\Users\dixit\AndroidStudioProjects\digit_recognition\ML_part\bestmodel.h5")
model_S=keras.models.load_model("D://programming//python for ml//New folder//bestmodel.h5")
model_S=keras.models.load_model("D://programming//python for ml//New folder//bestmodel.h5")
print("hello to check")
print("hello to check")
import numpy as np
import matplotlib.pyplot as plt
#import pygame

import keras
from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
score = model_S.evaluate(X_test,y_test)
print(f",Model accuracy is {score[1]}")
score = model_S.evaluate(X_test,y_test)
print(f",Model accuracy is {score[1]}")
score = model_S.evaluate(X_test,y_test)
print(f",Model accuracy is {score[1]}")
score = model_S.evaluate(X_test,y_test)
print(f",Model accuracy is {score[1]}")
score = model_S.evaluate(X_test,y_test)
print(f",Model accuracy is {score[1]}")
score = model_S.evaluate(X_test,y_test)
print(f",Model accuracy is {score[1]}")
import numpy as np
import matplotlib.pyplot as plt
#import pygame

import keras
from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
get_ipython().run_line_magic('pinfo', 'mnist.load_data')
(X_train, y_train), (X_test, y_test)=mnist.load_data()
X_train.shape , y_train.shape , X_test.shape ,y_test.shape
def plot_input_img(i):
    plt.imshow(X_train[i] , cmap='binary')
    plt.title(y_train[i])
    plt.axis('off')
    plt.show()
for i in range(100):
    plot_input_img(i)
#pre process data
X_train = X_train.astype(np.float32)/255
X_test = X_test.astype(np.float32)/255
#reshape or expand the dimentions of images to (28,28)
X_train = np.expand_dims(X_train,-1)
X_test = np.expand_dims(X_test,-1)
#convert classes to one hot vector
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
model = Sequential()

model.add(Conv2D(32,(3,3),input_shape = (28,28,1) , activation='relu'))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(64,(3,3) , activation='relu'))
model.add(MaxPool2D((2,2)))

model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(10,activation="softmax"))
model.summary()
model.compile(optimizer='adam',loss=keras.losses.binary_crossentropy,metrics=['accuracy'])
#callbacks

from keras.callbacks import EarlyStopping, ModelCheckpoint
#earlystopping
es= EarlyStopping(monitor='accuracy', min_delta=0.01,patience=4, verbose=1 )
#model check point
mc=ModelCheckpoint("./bestmodel.h5",monitor="accuracy",verbose=1,save_best_only=True)

cb = [es,mc]
#model train
his=model.fit(X_train, y_train , epochs=5, validation_split=0.3, callbacks=cb)
model_S=keras.models.load_model("C:\Users\dixit\AndroidStudioProjects\digit_recognition\ML_part\bestmodel.h5")
model_S=keras.models.load_model("DC://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//bestmodel.h5")
model_S=keras.models.load_model("C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//bestmodel.h5")
model_S=keras.models.load_model("C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//bestmodel.h5")
score = model_S.evaluate(X_test,y_test)
print(f",Model accuracy is {score[1]}")
get_ipython().run_line_magic('pip', 'install tenserflow')
pip3 install tenserflow
get_ipython().run_line_magic('pip', 'install tensorflow')
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model_file('C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//bestmodel.h5')
converter = tf.lite.TFLiteConverter.from_saved_model('C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//bestmodel.h5')
tf.saved_model.save(model, "C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//bestmodel.h5")
converter = tf.lite.TFLiteConverter.from_saved_model('C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//bestmodel.h5')
model.save('C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//bestmodel.h5')
converter = tf.lite.TFLiteConverter.from_keras_model_file('C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//bestmodel.h5')
model.save('C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//bestmodel.h5')
model.save('C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//bestmodel.h5')
model.save('C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//bestmodel.h5')
model.save('C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//bestmodel.h5')
model.save('C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//bestmodel.h5')
model.save('C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//bestmodel.h5')
model.save('C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//bestmodel.h5')
converter = tf.lite.TFLiteConverterv2.from_keras_model_file('C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//bestmodel.h5')
converter = tf.lite.TFLiteConverter.from_keras_model_file('C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//bestmodel.h5')
# tensorflow 2.x
converter = tf.lite.TFLiteConverter.from_saved_model('C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//bestmodel.h5')
model.save('C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//bestmodel.h5')
model.save('C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5')
tf.saved_model.save('C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5')
import tensorflow as tf
tf.saved_model.save('C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5')
tf.saved_model.save("C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5")
model.save("C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5")
model.save("C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5")
model.save("C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5")
model.save("C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5")
model.save("C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5")
model.save("C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5")
model.save("C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5")
model.save("C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5")
model.save("C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5")
model.save("C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5")
model.save("C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5")
model.save("C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5")
model.save("C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5")
model.save("C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5")
model.save("C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5")
model.save("C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5")
model.save("C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5")
model.save("C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5")
model.save("C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5")
converter = tf.lite.TFLiteConverter.from_keras_model_file('C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//bestmodel.h5')
saved_keras_model="C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5"
model.save("C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5")
saved_keras_model="C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5"
converter = tf.lite.TFLiteConverter.from_keras_model_file(saved_keras_model)
converter = tf.lite.TFLiteConverter.from_keras_model_file(saved_keras_model)
converter = tf.lite.TFLiteConverter.from_keras_model_file(saved_keras_model)
converter = tf.lite.TFLiteConverter.from_keras_model_file(saved_keras_model)
converter = tf.tflite.TFLiteConverter.from_keras_model_file(saved_keras_model)
converter = tf.tflite.TFLiteConverter.from_keras_model_file(saved_keras_model)
converter = tflite.TFLiteConverter.from_keras_model_file(saved_keras_model)
converter = TFLiteConverter.from_keras_model_file(saved_keras_model)
converter = tf.lite.TFLiteConverter.from_keras_model_file(saved_keras_model)
from tensorflow.contrib import lite
print(tf.version)
print(tf.__version__)
converter = tf.lite.TFLiteConverter.from_saved_model(saved_keras_model)
model.save("C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5")
model.save("C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5")
model.save("C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5")
model.save("C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5")
saved_keras_model="C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5"
saved_keras_model="C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5"
saved_keras_model="C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5"
saved_keras_model="C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5"
saved_keras_model="C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5"
#model.save("C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5")
converter = tf.lite.TFLiteConverter.from_saved_model("C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5")
get_ipython().run_line_magic('pip', 'uninstall h5py')
get_ipython().run_line_magic('conda', 'install h5py')
import tensorflow as tf
print(tf.__version__)
print(tf.__version__)
print(tf.__version__)
model.save("C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5")
model.save(myModelPath)
myModelPath="C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5"
myModelPath="C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5"
myModelPath="C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5"
model.save(myModelPath)
model.save(myModelPath)
converter = tf.lite.TFLiteConverter.from_keras_model_file(myModelPath)
converter = tf.lite.TFLiteConverter.from_saved_model(myModelPath)
  tflite_model = converter.convert()
converter = tf.lite.TFLiteConverter.from_saved_model(myModelPath)
tflite_model = converter.convert()
converter = tf.lite.TFLiteConverter.from_keras_model(myModelPath)
tflite_model = converter.convert()
converter = tf.lite.TFLiteConverter.from_keras_model(myModelPath)
tflite_model = converter.convert()
converter = tf.lite.TFLiteConverter.from_keras_model(myModelPath)
#tflite_model = converter.convert()
converter = tf.lite.TFLiteConverter.from_keras_model(myModelPath)
#tflite_model = converter.convert()
converter = tf.lite.TFLiteConverter.from_keras_model(myModelPath)
#tflite_model = converter.convert()
converter = tf.lite.TFLiteConverter.from_keras_model(myModelPath)
#tflite_model = converter.convert()
converter = tf.lite.TFLiteConverter.from_keras_model(myModelPath)
#tflite_model = converter.convert()
tflite_model = converter.convert()
tf.lite.TFLiteConverter(
    funcs, trackable_obj=None
)
tf.lite.TFLiteConverter(funcs, trackable_obj=None)
tf.lite.TFLiteConverter(funcs, trackable_obj=None)
converter = tf.lite.TFLiteConverter.from_keras_model(myModelPath)
converter = tf.lite.TFLiteConverter.from_keras_model(myModelPath)
converter = tf.lite.TFLiteConverter.from_keras_model(myModelPath)
converter = tf.lite.TFLiteConverter.from_keras_model(myModelPath)
tflite_model = converter.convert()
score = model_S.evaluate(X_test,y_test)
print(f",Model accuracy is {score[1]}")
score = model_S.evaluate(X_test,y_test)
print(f",Model accuracy is {score[1]}")
score = model_S.evaluate(X_test,y_test)
print(f",Model accuracy is {score[1]}")
score = model_S.evaluate(X_test,y_test)
print(f",Model accuracy is {score[1]}")
import tensorflow as tf
print(tf.__version__)
myModelPath="C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5"
model.save(myModelPath)
converter = tf.lite.TFLiteConverter.from_keras_model(myModelPath)
tf.saved_model.save(myModelPath)
tf.saved_model.save(model,myModelPath)
tf.saved_model.save(model,"C://Users//dixit//AndroidStudioProjects//digit_recognition//ML_part//myBestModel.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(myModelPath)
converter = tf.lite.TFLiteConverter.from_keras_model(myModelPath)
tflite_model = converter.convert()
tf.saved_model.save(model)
get_ipython().run_line_magic('save', '(myModelPath)')
