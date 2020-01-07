# -*- coding:utf-8
"""
@project:cifar_10
@author:xiezheng
@file:resnet.py
"""
import keras
import numpy as np
from keras.layers import Conv2D,Dense,Input,Activation,GlobalAveragePooling2D,add,AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler,TensorBoard
from keras.models import Model
from keras import optimizers,regularizers
from keras import backend as K

def resnet_32(img_input,weight_decay):
    x = Conv2D(16,strides=(1,1),kernel_size=(3,3),padding='same',kernel_regularizer=keras.regularizers.l2(weight_decay),kernel_initializer='he_normal')(img_input)
    #第1到5层
    for _ in range(0,5):
        b0 = BatchNormalization(momentum=0.9,epsilon=1e-5)(x)
        a0 = Activation('relu')(b0)
        conv_1 = Conv2D(16,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(weight_decay))(a0)
        b1 = BatchNormalization(momentum=0.9,epsilon=1e-5)(conv_1)
        a1 = Activation('relu')(b1)
        conv_2 = Conv2D(16,strides=(1,1),kernel_size=(3,3),padding='same',kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(weight_decay))(a1)
        x = add([x,conv_2])
    #第6层
    b0 = BatchNormalization(momentum=0.9,epsilon=1e-5)(x)
    a0 = Activation('relu')(b0)
    conv_1 = Conv2D(filters=32,kernel_size=(3,3),strides=(2,2),padding='same',
                    kernel_regularizer=regularizers.l2(weight_decay),
                    kernel_initializer='he_normal')(a0)
    b1 = BatchNormalization(momentum=0.9,epsilon=1e-5)(conv_1)
    a1 = Activation('relu')(b1)
    projection = Conv2D(filters=32,kernel_size=(2,2),strides=(2,2),padding='same',kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(weight_decay))(a0)
    conv_2 = Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='same',kernel_regularizer=
                    regularizers.l2(weight_decay),kernel_initializer='he_normal')(a1)
    x = add([conv_2,projection])#(32,32,32,3)
    #第7到10层
    for _ in range(1,5):
        b0 = BatchNormalization(momentum=0.9,epsilon=1e-5)(x)
        a0 = Activation('relu')(b0)
        conv_1 = Conv2D(filters=32,strides=(1,1),kernel_size=(3,3),padding='same',kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(weight_decay))(a0)
        b1 = BatchNormalization(momentum=0.9,epsilon=1e-5)(conv_1)
        a1 = Activation('relu')(b1)
        conv_2 = Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='same',
                        kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(weight_decay))(a1)
        x = add([conv_2,x])#(32,32,32,3)
    #第11层
    b0 = BatchNormalization(momentum=0.9,epsilon=1e-5)(x)
    a0 = Activation('relu')(b0)

    conv_1 = Conv2D(filters=64,kernel_size=(3,3),strides=(2,2),padding='same',kernel_regularizer=
                    regularizers.l2(weight_decay))(a0)
    b1 = BatchNormalization(momentum=0.9,epsilon=1e-5)(conv_1)
    a1 = Activation('relu')(b1)
    conv_2 = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(weight_decay))(a1)
    projection = Conv2D(filters=64,kernel_size=(2,2),strides=(2,2),padding='same',kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(weight_decay))(a0)

    x = add([projection,conv_2])#(64,32,32,3)
    #第12到15层
    for _ in range(1,5):
        b0 = BatchNormalization(momentum=0.9,epsilon=1e-5)(x)
        a0 = Activation('relu')(b0)
        conv_1 = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(weight_decay))(a0)
        b1 = BatchNormalization(momentum=0.9,epsilon=1e-5)(conv_1)
        a1 = Activation('relu')(b1)
        conv_2 = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(weight_decay))(a1)
        x = add([x,conv_2])
    x = BatchNormalization(momentum=0.9,epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(10,activation='softmax',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(weight_decay))(x)
    return x



# img_input = Input(shape=(32,32,3))
# output = resnet_32(img_input,weight_decay=0.00001)
# resnet = Model(img_input,output)
# from keras.utils.vis_utils import plot_model
# plot_model(resnet,to_file='resnet_model.jpg',show_shapes=True,show_layer_names=False)
# print(resnet.summary())
