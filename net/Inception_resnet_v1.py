# -*- coding:utf-8
"""
@project:cifar_10
@author:xiezheng
@file:Inception_resnet_v1.py
"""
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

# import keras
# import numpy as np
# import math

# from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers import Flatten, Dense, Dropout,BatchNormalization,Activation, Convolution2D, add
from keras.models import Model
from keras.layers import Input, concatenate
from keras import optimizers, regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import he_normal
from keras.layers import Lambda
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

num_classes        = 10
batch_size         = 64         # 64 or 32 or other
epochs             = 300
iterations         = 782
USE_BN=True
DROPOUT=0.2 # keep 80%
CONCAT_AXIS=3
weight_decay=1e-4
DATA_FORMAT='channels_last' # Theano:'channels_first' Tensorflow:'channels_last'


def conv_block(x, nb_filter, nb_row, nb_col, border_mode='same', subsample=(1,1), bias=False):
    x = Convolution2D(nb_filter, nb_row, nb_col, subsample=subsample, border_mode=border_mode, bias=bias,
                     init="he_normal",dim_ordering='tf',W_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    return x

def create_stem(img_input):
    x = Conv2D(filters=32,kernel_size=(3,3),strides=1,activation='relu',kernel_initializer='he_normal',
           kernel_regularizer=regularizers.l2(weight_decay))(img_input)
    x = BatchNormalization(momentum=0.9,epsilon=1e-5)(x)
    x = Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same',kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = MaxPooling2D(pool_size=(3,3),padding='valid',strides=(1,1))(x)
    x = Conv2D(filters=80,kernel_size=(1,1),padding='same',kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(weight_decay),activation='relu')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Conv2D(filters=192,kernel_size=(3,3),kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    return x

def inception_A(x,params,concat_axis,padding='same',data_format=DATA_FORMAT,scale_residual=False,use_bias=True,kernel_initializer="he_normal",bias_initializer='zeros',kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,weight_decay=weight_decay):
    (branch1, branch2, branch3, branch4) = params
    if weight_decay:
        kernel_regularizer = regularizers.l2(weight_decay)
        bias_regularizer = regularizers.l2(weight_decay)
    else:
        kernel_regularizer = None
        bias_regularizer = None
    pathway1 = Conv2D(filters=branch1[0],kernel_size=(1,1),kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(weight_decay),activation='relu')(x)
    pathway1 = BatchNormalization(momentum=0.9,epsilon=1e-5)(pathway1)

    pathway2 = Conv2D(filters=branch2[0],kernel_size=(1,1),padding='same',kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(weight_decay),activation='relu')(x)
    pathway2 = BatchNormalization(momentum=0.9,epsilon=1e-5)(pathway2)

    pathway2 = Conv2D(filters=branch2[1], kernel_size=(3, 3),padding=padding, kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(pathway2)
    pathway2 = BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway2)

    pathway3 = Conv2D(filters=branch3[0], kernel_size=(1, 1), padding=padding,kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(x)
    pathway3 = BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway3)

    pathway3 = Conv2D(filters=branch3[1],kernel_size=(3,3),padding=padding,kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(pathway3)
    pathway3 = BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway3)
    pathway3 = Conv2D(filters=branch3[2], kernel_size=(3, 3), padding=padding,kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(pathway3)
    pathway123 = concatenate([pathway1,pathway2,pathway3],axis=concat_axis)
    pathway123 = Conv2D(filters=branch4[0],kernel_size=(1,1),padding='same',activation='linear',
                        kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(weight_decay))(pathway123)
    pathway123 = BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway123)
    if scale_residual:
        x =Lambda(lambda p:p*0.1)(x)
    return add([x,pathway123])

def reduce_A(x,params,concat_axis):
    (branch1, branch2) = params
    if weight_decay:
        kernel_regularizer = regularizers.l2(weight_decay)
        bias_regularizer = regularizers.l2(weight_decay)
    else:
        kernel_regularizer = None
        bias_regularizer = None
    pathway3 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid')(x)

    pathway1 = Conv2D(filters=branch1[0],strides=(2,2),kernel_size=(3,3), kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(x)
    pathway1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway1)

    pathway2 = Conv2D(filters=branch2[0],kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(x)
    pathway2 = BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway2)
    pathway2 = Conv2D(filters=branch2[1],kernel_size=(3,3),padding='same',kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(pathway2)
    pathway2 = BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway2)
    pathway2 = Conv2D(filters=branch2[2], kernel_size=(3, 3), strides=(2,2),kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(pathway2)
    pathway2 = BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway2)
    x = concatenate([pathway1,pathway2,pathway3],axis=concat_axis)
    return x

def inception_B(x,params,concat_axis =CONCAT_AXIS,padding='same',data_format=DATA_FORMAT,scale_residual=False,use_bias=True,kernel_initializer="he_normal",bias_initializer='zeros',kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,weight_decay=weight_decay):
    (branch1,branch2,branch12) = params
    if weight_decay:
        kernel_regularizer = regularizers.l2(weight_decay)
        bias_initializer = regularizers.l2(weight_decay)
    else:
        kernel_regularizer = None
        bias_initializer = None

    pathway1 = Conv2D(filters=branch1[0],kernel_size=(1,1),kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer,
                      padding=padding,bias_regularizer=bias_regularizer,activation='relu')(x)
    pathway1 = BatchNormalization(momentum=0.9,epsilon=1e-5)(pathway1)


    pathway2 = Conv2D(filters=branch2[0],kernel_size=(1,1),strides=(1,1),padding=padding,kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer,
                      bias_regularizer=bias_regularizer,activation='relu')(x)
    pathway2 = BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway2)

    pathway2 = Conv2D(filters=branch2[1], kernel_size=(1, 7), strides=(1, 1), padding=padding,
                      kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
                      bias_regularizer=bias_regularizer,activation='relu')(pathway2)
    pathway2 = BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway2)
    pathway2 = Conv2D(filters=branch2[2], kernel_size=(7, 1), strides=(1, 1), padding=padding,
                      kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
                      bias_regularizer=bias_regularizer,activation='relu')(pathway2)
    pathway2 = BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway2)

    pathway12 = concatenate([pathway1,pathway2],axis=concat_axis)
    # print('------------------------')
    # print('branch12[0] is {}'.format(branch12[0]))
    # print('x shape is {0},pathway12 shape is {1}'.format(x.shape, pathway12.shape))
    pathway12 = Conv2D(filters=branch12[0],kernel_size=(1,1),padding=padding,kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer,
                activation='linear')(pathway12)
    pathway12 = BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway12)

    x = add([x,pathway12])
    return x

def reduce_B(x,params,concat_axis,padding='same',data_format=DATA_FORMAT,use_bias=True,kernel_initializer="he_normal",bias_initializer='zeros',kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,weight_decay=weight_decay):
    [branch1,branch2,branch3] = params
    print('------------------------')
    print('x shape is {}'.format(x.shape))
    if weight_decay:
        kernel_regularizer = regularizers.l2(weight_decay)
        bias_initializer = regularizers.l2(weight_decay)
    else:
        kernel_regularizer = None
        bias_initializer = None
    pathway1 = Conv2D(filters=branch1[0],kernel_size=(1,1),activation='relu')(x)
    pathway1 = BatchNormalization(momentum=0.9,epsilon=1e-5)(pathway1)
    pathway1 = Conv2D(filters=branch1[1],kernel_size=(3,3),padding='valid',strides=(2,2),kernel_initializer=kernel_initializer,activation='relu')(pathway1)
    pathway1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway1)

    pathway2 = Conv2D(filters=branch2[0],kernel_size=(1,1),kernel_initializer=kernel_initializer,activation='relu')(x)
    pathway2 = BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway2)
    pathway2 = Conv2D(filters=branch2[1],kernel_size=(3,3),padding='valid',strides=(2,2),kernel_initializer=kernel_initializer,activation='relu')(pathway2)
    pathway2 = BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway2)

    pathway3 = Conv2D(filters=branch3[0],kernel_size=(1,1),strides=(1,1),kernel_initializer=kernel_initializer,activation='relu')(x)
    pathway3 =BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway3)

    pathway3 = Conv2D(filters=branch3[1], kernel_size=(3, 3), padding='same',strides=(1,1), kernel_initializer=kernel_initializer,
                      activation='relu')(pathway3)
    pathway3 = BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway3)

    pathway3 = Conv2D(filters=branch3[2], kernel_size=(3, 3), padding='valid',strides=(2,2), kernel_initializer=kernel_initializer,
                      activation='relu')(pathway3)
    pathway3 = BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway3)

    pathway4 =  MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid')(x)
    pathway4 = BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway4)
    x = concatenate([pathway1,pathway2,pathway3,pathway4])
    return x

def inception_C(x,params,concat_axis,padding='same',data_format=DATA_FORMAT,scale_residual=False,use_bias=True,kernel_initializer="he_normal",bias_initializer='zeros',kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,weight_decay=weight_decay):
    [branch1, branch2,branch12] = params
    if weight_decay:
        kernel_regularizer = regularizers.l2(weight_decay)
        bias_initializer = regularizers.l2(weight_decay)
    else:
        kernel_regularizer = None
        bias_initializer = None

    pathway1 = Conv2D(filters=branch1[0],kernel_size=(1,1),padding=padding,kernel_initializer=kernel_initializer,activation='relu')(x)

    pathway2 = Conv2D(filters=branch2[0],kernel_size=(1,1),padding=padding,kernel_initializer=kernel_initializer,activation='relu')(x)

    pathway2 = Conv2D(filters=branch2[1], kernel_size=(1, 3), padding=padding, kernel_initializer=kernel_initializer,activation='relu')(pathway2)

    pathway2 = Conv2D(filters=branch2[2], kernel_size=(3, 1), padding=padding, kernel_initializer=kernel_initializer,activation='relu')(pathway2)

    pathway12 = concatenate([pathway1,pathway2],axis=concat_axis)

    pathway12 = Conv2D(filters=branch12[0],kernel_size=(1,1),strides=(1,1),padding=padding, kernel_initializer=kernel_initializer,activation='linear')(pathway12)

    print('------------------------')
    # print('branch12[0] is {}'.format(branch12[0]))
    # print('x shape is {0},pathway12 shape is {1}'.format(x.shape, pathway12.shape))
    if scale_residual:
        x = Lambda(lambda p: p * 0.1)(x)
    x = add([x,pathway12])
    return x

def create_model(img_input):
    x = create_stem(img_input)
    for _ in range(5):
        x = inception_A(x,params=[(32,),(32,32),(32,32,32),(256,)],concat_axis=CONCAT_AXIS)
        # reduce A
    # print(x.shape)
    x = reduce_A(x, params=[(384,), (192, 224, 256)], concat_axis=CONCAT_AXIS)  # 768
    # 10 x inception_B
    for _ in range(10):
        x = inception_B(x, params=[(128,), (128, 128, 128), (896,)], concat_axis=CONCAT_AXIS)
    x = reduce_B(x, params=[(256, 384), (256, 256), (256, 256, 256)], concat_axis=CONCAT_AXIS)  # 1280
    for _ in range(5):
        x = inception_C(x, params=[(192,), (192, 192, 192), (1792,)], concat_axis=CONCAT_AXIS)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(DROPOUT)(x)
    x = Dense(num_classes, activation='softmax', kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)
    return x
# input = Input(shape=(32,32,3))
# out = create_model(img_input=input)
# model = Model(inputs=input,outputs=out)
# from keras.utils import vis_utils
# print(model.summary())
# vis_utils.plot_model(model,to_file='inception_resnet_v1.jpg',show_shapes=True)
