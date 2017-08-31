# -*- coding: utf-8 -*-

import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
import shutil
import time
import caffe

#因为RGB和BGR需要调换一下才能显示
def showimage(im):
    if im.ndim == 3:
        im = im[:, :, ::-1]
    plt.set_cmap('jet')
    plt.imshow(im)
    plt.show()

# 特征可视化显示，padval用于调整亮度
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    # 因为我们要把某一层的特征图都显示到一个figure上，因此需要计算每个图片占用figure多少比例，以及绘制的位置
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    showimage(data)

# 加载均值文件
mean_filename = r"./caffe_model/imagenet_mean.binaryproto"
proto_data = open(mean_filename, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean = caffe.io.blobproto_to_array(a)[0]

#创建网络，并加载已经训练好的模型文件
gender_net_pretrained = r"./caffe_model/finetune_flickr_style_iter_10000.caffemodel"
gender_net_model_file = r"./caffe_model/deploy.prototxt"
gender_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained, mean=mean,
                       channel_swap=(2, 1, 0),#RGB通道与BGR
                       raw_scale=255,#把图片归一化到0~1之间
                       image_dims=(256, 256))#设置输入图片的大小

#预测分类及其可特征视化
gender_list = ['car', 'construction', 'farm', 'pole', 'road']
input_image = caffe.io.load_image(r"./test_img/road.jpg")#读取图片

prediction_gender=gender_net.predict([input_image])#预测图片性别
#打印我们训练每一层的参数形状
print 'params:'
for k, v in gender_net.params.items():
    print 'weight:'
    print (k, v[0].data.shape)#在每一层的参数blob中，caffe用vector存储了两个blob变量，用v[0]表示weight
    print 'b:'
    print (k, v[1].data.shape)#用v[1]表示偏置参数
#conv1滤波器可视化
filters = gender_net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1))
#conv2滤波器可视化
'''filters = gender_net.params['conv2'][0].data
vis_square(filters[:48].reshape(48**2, 5, 5))'''
#特征图
print 'feature maps:'
for k, v in gender_net.blobs.items():
    print (k, v.data.shape)
    feat = gender_net.blobs[k].data[0, 0:4]
    if 1 == len(feat.shape):
        continue
    vis_square(feat, padval=1)

#显示原图片，以及分类预测结果
str_gender = gender_list[prediction_gender[0].argmax()]
print str_gender

plt.imshow(input_image)
plt.title(str_gender)
plt.show()
