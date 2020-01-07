# -*- coding:utf-8
"""
@project:cifar_10
@author:xiezheng
@file:LeNet.py
"""
import tensorflow as tf
# from tensorflow import keras
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
from keras import optimizers
from keras.models import Sequential

def build_model():
    from keras.regularizers import l2
    weight_decay=0.0001
    model = Sequential()
    model.add(Conv2D(filters=6,kernel_size=(5,5),padding='valid',activation='relu',input_shape=(32,32,3),kernel_regularizer=l2(weight_decay),kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=16,kernel_size=(5,5),activation='relu',strides=(1,1),padding='valid',kernel_regularizer=l2(weight_decay),kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(120,activation='relu',kernel_initializer='he_normal',kernel_regularizer=l2(weight_decay)))
    model.add(Dense(84,activation='relu',kernel_initializer='he_normal',kernel_regularizer=l2(weight_decay)))
    model.add(Dense(10,activation='softmax',kernel_initializer='he_normal',kernel_regularizer=l2(weight_decay)))

    return model

def scheduler(epoch):
    if epoch < 50:
        return 0.001
    if epoch < 150:
        return 0.005
    return 0.0001
