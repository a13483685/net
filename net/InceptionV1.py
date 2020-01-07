# -*- coding:utf-8
"""
@project:cifar_10
@author:xiezheng
@file:InceptionV1.py
"""
import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBEL_DEVICES'] = "0"
import keras
import numpy as np
import math

from keras.layers import Conv2D,MaxPooling2D,AveragePooling2D,ZeroPadding2D,GlobalAveragePooling2D,Flatten,Dense,Dropout,BatchNormalization
from keras.models import Model
from keras.layers import Input,concatenate
from keras import regularizers

num_classes        = 10
batch_size         = 64         # 64 or 32 or other
epochs             = 300
iterations         = 782
USE_BN=True
LRN2D_NORM = True
DROPOUT=0.4
CONCAT_AXIS=3
WEIGHT_DECAY=1e-4
DATA_FORMAT='channels_last' # Theano:'channels_first' Tensorflow:'channels_last'

def conv2D_lrn2d(x,filters,kernel_size,strides=(1,1),padding='same',dataformat = DATA_FORMAT,
                 dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='he_normal',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, lrn2d_norm=LRN2D_NORM, weight_decay=WEIGHT_DECAY):
    if weight_decay:
        kernel_regularizer = regularizers.l2(weight_decay)
        bias_regularizer = regularizers.l2(weight_decay)
    else:
        kernel_regularizer = None
        bias_regularizer = None

    x = Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,
               activation=activation,kernel_initializer=kernel_initializer,kernel_regularizer = kernel_regularizer)(x)

    if lrn2d_norm:
        x = BatchNormalization()(x)

    return x

def inception_module(x,params,concat_axis,padding='same',data_format=DATA_FORMAT,dilation_rate=(1,1),activation='relu',
                     use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer=None,
                     bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,lrn2d_norm=LRN2D_NORM,weight_decay=None):
    (branch1,branch2,branch3,branch4)=params

    pathway1 = Conv2D(filters=branch1[0], kernel_size=(1, 1), strides=1, padding=padding, data_format=data_format,
                      dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(x)

    # 1x1->3x3
    pathway2 = Conv2D(filters=branch2[0], kernel_size=(1, 1), strides=1, padding=padding, data_format=data_format,
                      dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(x)
    pathway2 = Conv2D(filters=branch2[1], kernel_size=(3, 3), strides=1, padding=padding, data_format=data_format,
                      dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(pathway2)

    # 1x1->5x5
    pathway3 = Conv2D(filters=branch3[0], kernel_size=(1, 1), strides=1, padding=padding, data_format=data_format,
                      dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(x)
    pathway3 = Conv2D(filters=branch3[1], kernel_size=(5, 5), strides=1, padding=padding, data_format=data_format,
                      dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(pathway3)

    # 3x3->1x1
    pathway4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding=padding, data_format=DATA_FORMAT)(x)
    pathway4 = Conv2D(filters=branch4[0], kernel_size=(1, 1), strides=1, padding=padding, data_format=data_format,
                      dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(pathway4)

    return concatenate([pathway1, pathway2, pathway3, pathway4], axis=concat_axis)



def create_model(img_input):
    x=conv2D_lrn2d(img_input,64,(7,7),2,padding='same',lrn2d_norm=False)
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same',data_format=DATA_FORMAT)(x)
    x=BatchNormalization()(x)

    x=conv2D_lrn2d(x,64,(1,1),1,padding='same',lrn2d_norm=False)

    x=conv2D_lrn2d(x,192,(3,3),1,padding='same',lrn2d_norm=True)
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same',data_format=DATA_FORMAT)(x)

    x=inception_module(x,params=[(64,),(96,128),(16,32),(32,)],concat_axis=CONCAT_AXIS) #3a
    x=inception_module(x,params=[(128,),(128,192),(32,96),(64,)],concat_axis=CONCAT_AXIS) #3b
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same',data_format=DATA_FORMAT)(x)

    x=inception_module(x,params=[(192,),(96,208),(16,48),(64,)],concat_axis=CONCAT_AXIS) #4a
    x=inception_module(x,params=[(160,),(112,224),(24,64),(64,)],concat_axis=CONCAT_AXIS) #4b
    x=inception_module(x,params=[(128,),(128,256),(24,64),(64,)],concat_axis=CONCAT_AXIS) #4c
    x=inception_module(x,params=[(112,),(144,288),(32,64),(64,)],concat_axis=CONCAT_AXIS) #4d
    x=inception_module(x,params=[(256,),(160,320),(32,128),(128,)],concat_axis=CONCAT_AXIS) #4e
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same',data_format=DATA_FORMAT)(x)

    x=inception_module(x,params=[(256,),(160,320),(32,128),(128,)],concat_axis=CONCAT_AXIS) #5a
    x=inception_module(x,params=[(384,),(192,384),(48,128),(128,)],concat_axis=CONCAT_AXIS) #5b
    x=AveragePooling2D(pool_size=(1,1),strides=1,padding='valid',data_format=DATA_FORMAT)(x)

    x=Flatten()(x)
    x=Dropout(DROPOUT)(x)
    x=Dense(output_dim=10,activation='linear')(x)
    x=Dense(output_dim=10,activation='softmax')(x)
    return x
# img_input=Input(shape=(32,32,3))
# output = create_model(img_input)
# model=Model(img_input,output)
# model.summary()
# from keras.utils import vis_utils
# vis_utils.plot_model(model,to_file='inception_v1.jpg',show_shapes=True)