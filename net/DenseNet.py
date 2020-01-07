# -*- coding:utf-8
"""
@project:cifar_10
@author:xiezheng
@file:DenseNet.py
"""
weight_decay = 1e-5
growth_rate = 12
compression = 0.5
depth = 100
import tensorflow as tf
from keras.layers import Conv2D,Input,Dense,Dropout,BatchNormalization,Activation,AveragePooling2D,concatenate,GlobalAveragePooling2D
from keras.regularizers import l2
from keras import optimizers
def conv(x,out_filter,k_size):
    return Conv2D(filters=out_filter,kernel_size=k_size,
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(weight_decay),
                  padding='same',use_bias=False,
                  strides=(1,1))(x)

def dense_layer(x):
    return Dense(10,activation='softmax',kernel_initializer='he_normal',
                 kernel_regularizer=l2(weight_decay))(x)

def bn_relu(x):
    bn = BatchNormalization(momentum=0.9,epsilon=1e-5)(x)
    return Activation('relu')(x)

def bottleneck(x):
    channels = growth_rate * 4
    x= bn_relu(x)
    x = conv(x,channels,(1,1))
    x = bn_relu(x)
    x = conv(x,growth_rate,(3,3))
    return x

def trainsition(x,inchannels):
    outchannels = int(inchannels*compression)
    x = bn_relu(x)(x)
    x = conv(x,outchannels,(1,1))(x)
    x = AveragePooling2D((2,2),strides=(2,2))(x)
    return x,outchannels

def dense_block(x,blocks,nchannels):
    concat = x
    for i in range(blocks):
        x = bottleneck(concat)
        concat = concatenate([x,concat], axis=-1)
        nchannels += growth_rate
    return concat, nchannels

def densenet(img_input,class_num):
    nblocks = (depth - 4) //6
    nchannels = growth_rate * 2
    x = conv(img_input,nchannels,(3,3))
    x,nchannels = dense_block(x,nblocks,nchannels)
    x,nchannels = trainsition(x,nchannels)

    x, nchannels = dense_block(x, nblocks, nchannels)
    x, nchannels = trainsition(x, nchannels)

    x, nchannels = dense_block(x,nblocks,nchannels)
    x = bn_relu(x)
    x = GlobalAveragePooling2D()(x)
    x = dense_layer(x)
    return x



