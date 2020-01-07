# -*- coding:utf-8
"""
@project:cifar_10
@author:xiezheng
@file:utils.py
"""
import matplotlib.pyplot as plt

def show_train_history(train_history,train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('train history')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train','test'],loc='upper left')
    plt.savefig('1.png')
    plt.show()

def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    label_dict = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
                  5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}

    fig = plt.gcf()
    fig.set_size_inches(12, 14)  # 控制图片大小
    if num > 25: num = 25  # 最多显示25张
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(images[idx], cmap='binary')
        title = str(i) + ',' + label_dict[labels[i][0]]  # i-th张图片对应的类别
        if len(prediction) > 0:
            title += '=>' + label_dict[prediction[i]]
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]);
        ax.set_yticks([])
        idx += 1
    plt.savefig('1.png')
    plt.show()

#混淆矩阵
def crosstab(test_label,prediction):
    import pandas as pd
    print(pd.crosstab(test_label.reshape(-1),prediction,rownames=['label'],colnames=['pridiction']))

def save_model(model,model_name):
    import os
    if not os.path.exists('models'):
        os.mkdir('models')
    model.save_weights('{}/{}.h5'.format('models',model_name))
    print('saved model to disk')

def z_score(train_data):
    import numpy as np
    mean_list = []
    std_list = []
    #对三个通道的数据计算均值和方差
    for i in range(0,3):
        mean = np.mean(train_data[:,:,:,i]).tolist()
        mean_list.append(mean)
        std = np.std(train_data[:,:,:,i]).tolist()
        std_list.append(std)
    # print("mean list is {0},std list is {1}".format(mean_list,std_list))
    return  mean_list,std_list