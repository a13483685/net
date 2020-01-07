# -*- coding:utf-8
"""
@project:cifar_10
@author:xiezheng
@file:test.py
"""
import os
import tensorflow as tf
from net.normal import Normal_net
from tensorflow import keras
from keras import Sequential,datasets,layers,optimizers,losses
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from keras.models import Model

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

batchsz = 128
def process(x,y):
    x = tf.cast(x,dtype=tf.float32)/255.
    y = tf.cast(x,dtype=tf.int32)
    return x,y

from keras.datasets import cifar10
import numpy as np
np.random.seed(10)
# 载入数据集
(x_img_train,y_label_train),(x_img_test,y_label_test)=cifar10.load_data()
#数据增强：
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(horizontal_flip=True,width_shift_range=0.125,
                             height_shift_range=0.125,fill_mode='constant',
                             cval=0.)
datagen.fit(x_img_train)

print("train data:",'images:',x_img_train.shape,
      " labels:",y_label_train.shape)
print("test  data:",'images:',x_img_test.shape ,
      " labels:",y_label_test.shape)
# 归一化
x_img_train_normalize = x_img_train.astype('float32') / 255.0
x_img_test_normalize = x_img_test.astype('float32') / 255.0
# One-Hot Encoding
from keras.utils import np_utils
y_label_train_OneHot = np_utils.to_categorical(y_label_train)
y_label_test_OneHot = np_utils.to_categorical(y_label_test)
print(y_label_train_OneHot.shape)

from utils.utils import z_score
# mean : [126.02464140625, 123.70850419921875, 114.85431865234375]
# std  : [62.89639134921991, 61.93752718231365, 66.70605639561605]
mean,std = z_score(x_img_train_normalize)
print("mean list is {0},std list is {1}".format(mean,std))
for i in range(0,3):
    x_img_train_normalize[:,:,:,i] = (x_img_train_normalize[:,:,:,i] - mean[i])/std[i] #每一个元素都减去均值除以方差
    x_img_test_normalize[:,:,:,i] = (x_img_test_normalize[:,:,:,i] - mean[i])/std[i]
#--------------------------------------
# model = Normal_net()
# model = model.process()
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam', metrics=['accuracy'])
# print(model.metrics)
#
# #normal cnn
# train_history=model.fit(x_img_train_normalize, y_label_train_OneHot,
#                         validation_split=0.2,
#                         epochs=4, batch_size=128, verbose=1)

#--------------------------------------

#---------------------------------
    # LeNet
from net import resnet,LeNet
from keras.callbacks import TensorBoard,LearningRateScheduler
# model = LeNet.build_model()
from keras.layers import Input
img_input = Input(shape=(32,32,3))

def train(type):
    if type == 0:
        pass
    elif type ==1:
        pass
    elif type ==2:
        pass
    elif type ==3:#inception v1
        from net.InceptionV1 import create_model
        output = create_model(img_input)
        log_dir = 'logs/inception/inception_v1_he_normal'
        model_name = 'inception_v1_model'
    # output = resnet.resnet_32(img_input = img_input,weight_decay=0.00001)
    elif type ==4:
        from net.Inception_resnet_v1 import create_model
        output = create_model(img_input)
        log_dir = 'logs/Inception_resnet_v1'
        model_name = 'Inception_resnet_v1_model'

    elif type ==5:
        from net.Xnception import xception
        output = xception(img_input=img_input)
        log_dir = 'logs/Xnception'
        model_name = 'Xnception_model'


    model = Model(img_input,output)

    tb_cb = TensorBoard(log_dir=log_dir,histogram_freq=1)
    change_lr = LearningRateScheduler(schedule=LeNet.scheduler)
    cbks = [change_lr,tb_cb]
    # print(model.summary())
    sgd = optimizers.SGD(lr=.1,momentum=0.9,nesterov=True)
    #---------------------------
    #单GPU训练
    # model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])
    #
    # print(x_img_train.shape)
    # print(y_label_train_OneHot.shape)
    # model.fit(x=x_img_train_normalize, y=y_label_train_OneHot,batch_size=128, epochs=200,
    #           callbacks=cbks,validation_split=0.2,validation_data=(x_img_test_normalize,y_label_test_OneHot),
    #          verbose=1)
    from keras.utils import multi_gpu_model
    model_parallel = multi_gpu_model(model,gpus=4)
    model_parallel.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])
    model_parallel.fit_generator(datagen.flow(x_img_train_normalize,y_label_train_OneHot,batch_size=batchsz),
                                     epochs=200,callbacks=cbks,
                        validation_data=(x_img_test_normalize,y_label_test_OneHot))


    # score = model.evaluate(x_img_test_normalize,y_label_test_OneHot,verbose=0)
    # print(score)

    #---------------------------------
    from utils.utils import show_train_history,plot_images_labels_prediction,crosstab,save_model
    # show_train_history(train_history,'acc','val_acc')
    prediction=model.predict_classes(x_img_test_normalize)
    # prediction[:10]
    # #显示图像、预测与label的结果
    # plot_images_labels_prediction(x_img_test,y_label_test,prediction,0,100)
    # ——————————————————————————————————————

    #混淆矩阵
    print('prediction shape: is {0} ,label shape is {1}'.format(prediction.shape,y_label_test.shape))
    crosstab(y_label_test,prediction)
    #-------------------------
    save_model(model,model_name)
    model = model.to_json()
    #将json文件保存
    with open('models/{}.json'.format(model_name),'w') as json_file:
        json_file.write(model)

if __name__ == '__main__':
    # type =3 :incetion_v1
    # type =4 :Inception_resnet_v1
    train(type=5)