import csv
import cv2
import numpy as np

#loadData
images=[]
measurements=[]
first = 0

#Read the data from csv file
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if first==0:
            first = 1
            continue

        #Use Centre, left, right caera images
        for i in range(3):
            path = 'data/' + line[i]
            image = cv2.imread(path)
            image1=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            images.append(image1)
            measurement = float(line[3])
            if i==1:
                measurement += 0.2
            if i==2:
                measurement -= 0.2
            measurements.append(measurement)

            #Flip the image to augment the dataSet
            images.append(cv2.flip(image,1))
            measurements.append(measurement*-1.0)

X_train = np.array(images)
y_train = np.array(measurements)
print(len(X_train), len(y_train))

import tensorflow as tf
import h5py
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
#USe Nvidia Architecture
model=Sequential()
#model.add(AveragePooling2D(pool_size=(4,4), strides=(4,4), border_mode='valid'))
model.add(Lambda(lambda x: (x/255.0) -0.5, input_shape=(160, 320,3)))
model.add(Cropping2D(cropping=((71,25),(0,0))))
model.add(Convolution2D(24, 5, 5, activation='relu', subsample=(2,2)))
model.add(Convolution2D(36, 5, 5, activation='relu', subsample=(2,2)))
model.add(Convolution2D(48, 5, 5, activation='relu', subsample=(2,2)))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#Train the model
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5, verbose=2)
model.save('model.h5')
