#-*- coding: utf-8 -*-
import os
import numpy
from PIL import Image
from pylab import *
import cPickle

def get_imlist(path):   #此函数读取特定文件夹下的png格式图像
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

def imgfile2array(data_path, data_label_bound, data_shape):
    """
    :param data_path: 
    :param data_label_bound: 
    :param data_shape: 
    :return: data, data_label
    """

    x=[]    #存放图片路径
    len_data=0  #图片
    img_path=get_imlist(data_path)  #字符串连接
    #print neutral  #这里以list形式输出bmp格式的所有图像（带路径）
    for j in range(len(img_path)):
        x.append(img_path[j])
    #   print len(img_path)
    len_data=len_data+len(img_path) #这可以以输出图像个数
    # print len_data
    #print len(x)
    d = len_data
    data = numpy.empty((d, data_shape[0]*data_shape[1])) #建立d*（64*64）的矩阵
    while d>0:
        img=Image.open(x[d-1])  #打开图像
        img = img.resize((data_shape[0],data_shape[1])).convert('L')
        img_ndarray_raw=numpy.asarray(img)
        img_ndarray=numpy.asarray(img,dtype='float64')/255  #将图像转化为数组并将像素转化到0-1之间
        data[d-1]=numpy.ndarray.flatten(img_ndarray)    #将图像的矩阵形式转化为一维数组保存到data中
        d=d-1
    # print len_data
    # print shape(data)[1]  #输出矩阵大小


    data_label=numpy.empty(len_data)
    class_num = len(data_label_bound) - 1

    for label in range(len_data):
        for class_bound_id in range(len(data_label_bound)):
            if label>=data_label_bound[class_bound_id] and label<data_label_bound[class_bound_id+1]:
                data_label[label] = class_bound_id
                break

    data_label=data_label.astype(numpy.int)  #将标签转化为int类型
    # print data_label[924]
    # print data_label[1638]

    return data, data_label

test_data, test_label = imgfile2array('..\\real_data\\test',[0,535,1336],[28,28])
train_data, train_label = imgfile2array('..\\real_data\\train',[0,2468,9691],[28,28])

train_set = (train_data, train_label)
test_set  = (test_data, test_label)
write_data = (train_set,test_set)
write_file=open('..\\real_data\\data.pkl','wb')
cPickle.dump(write_data,write_file,-1)
write_file.close()


#保存data以及data_label到data.pkl文件
# write_file=open('D:\\CK\\data.pkl','wb')
# cPickle.dump(data,write_file,-1)
# cPickle.dump(data_label,write_file,-1)
# write_file.close()