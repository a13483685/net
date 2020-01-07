# -*- coding:utf-8
"""
@project:cifar_10
@author:xiezheng
@file:Xnception.py
"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import keras
from keras.datasets import cifar10
from keras import backend as K
from keras.layers import Input, Conv2D, Dense, BatchNormalization, Activation
from keras.layers import GlobalAveragePooling2D, MaxPooling2D, add
from keras.models import Model
from keras.layers import SeparableConv2D

from keras import optimizers,regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import he_normal
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint

num_classes        = 10
batch_size         = 64         # 64 or 32 or other
epochs             = 300
iterations         = 782
weight_decay=1e-4

def entryflow(x,params,top=False):
    if not top:
        x = Activation('relu')(x)
    residual = Conv2D(params[0],kernel_size=(1,1),strides=(2,2),padding='same')(x)
    residual = BatchNormalization(momentum=0.9,epsilon=1e-5)(residual)
    x = SeparableConv2D(params[1],kernel_size=(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = SeparableConv2D(params[2],kernel_size=(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    x = add([residual,x])
    return x

def middleflow(x,params):
    residual = x
    x = Activation('relu')(x)
    x = SeparableConv2D(filters=params[0],kernel_size=(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(filters=params[1], kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(filters=params[0], kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    x = add([x,residual])
    return x

def exitflow(x,params):
    residual = x
    residual = Conv2D(params[0],kernel_size=(1,1),strides=(2,2),padding='same')(residual)
    residual = BatchNormalization()(residual)

    x = Activation('relu')(x)
    x = SeparableConv2D(filters=params[1],kernel_size=(3,3),padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D(filters=params[2],kernel_size=(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    x = add([residual,x])

    x = SeparableConv2D(filters=params[3],kernel_size=(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(filters=params[4],kernel_size=(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes,activation='softmax')(x)
    return x

def xception(img_input,shallow=False, classes=10):

    x = Conv2D(filters=32,kernel_size=(3,3),strides=(2,2),padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=64,kernel_size=(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = entryflow(x=x,params=[128,128,128],top=True)
    x = entryflow(x=x,params=[256,256,256])
    x = entryflow(x=x,params=[728,728,728])
    x = middleflow(x,params=[728,728,728])
    x = exitflow(x,params=[1024,728,1024,1526,2048])
    return x

input = Input(shape=(32,32,3))
out = xception(img_input=input,classes=10)
model = Model(input,out)
print(model.summary())
