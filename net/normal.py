# -*- coding:utf-8
"""
@project:cifar_10
@author:xiezheng
@file:normal.py
"""
import os
import tensorflow as tf
from tensorflow import keras
from keras import Sequential,datasets,layers,optimizers,losses
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout

class Normal_net():
    def __init__(self):
        super(Normal_net,self).__init__()

    def process(self):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3),
                         input_shape=(32, 32, 3), activation='relu'
                         , padding='same'))
        model.add(Dropout(rate=0.25))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=64, kernel_size=(3, 3),
                         activation='relu', padding='same'
                         ))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))

        model.add(Flatten())
        model.add(Dropout(0.25))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(10, activation='softmax'))
        print(model.summary())
        return model