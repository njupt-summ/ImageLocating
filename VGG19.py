import numpy as np
import sys
import cv2
import os
# import imageio
sys.path.append("/home/cjq/caffe/python")
import caffe


 
caffe.set_mode_gpu()
 
model_def = '../models/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.prototxt'
model_weights = '../models/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.caffemodel'
 
net = caffe.Net(model_def,      # 定义模型结构
                model_weights,  # 包含了模型的训练权值
                caffe.TEST)     # 使用测试模式(不执行dropout)

 
    # 加载ImageNet图像均值 (随着Caffe一起发布的)
mu = np.load('../imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  #对所有像素值取平均以此获取BGR的均值像素值
# print('mean-subtracted values:', zip('BGR', mu))
 
# 对输入数据进行变换
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  #将图像的通道数设置为outermost的维数
transformer.set_mean('data', mu)            #对于每个通道，都减去BGR的均值像素值
transformer.set_raw_scale('data', 255)      #将像素值从[0,255]变换到[0,1]之间
transformer.set_channel_swap('data', (2,1,0))  #交换通道，从RGB变换到BGR

def get_imgs_fea(fold_name):
    dir =fold_name
    files = os.listdir(dir)
    files.sort(key=lambda x:int(x[40:]), reverse = True)    # 图片顺序调整
    imgs_feature = list()
    for f in files:
        image_path = os.path.join(dir,f)
        fea = extra_fea(image_path)
        if fea:
            imgs_feature.append(fea)
    return imgs_feature

def extra_fea(image_path):
    img = cv2.imread(image_path)
    if img is None :
        return False
    image=cv2.resize(img,(224,224))
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    output = net.forward(end = 'fc7')
    output_fc = output['fc7'][0]
    return list(output_fc)


# imgs_fold = '../../../../dailymail/images'
# epoch = 0 
# for line in open('../../../../dailymail/trains','r'):
#     if epoch>40300:
#         print(epoch)
#         filename = line.strip()
#         print(filename)
#         imgs_path = os.path.join(imgs_fold,filename)
#         get_imgs_fea(imgs_path)
#     epoch = epoch+1
# t = get_imgs_fea('../../../../dailymail/images/cc88afe9785f8c7f0af922534da00c4f03e83e2d')
# print(np.array(t).shape)
# file = os.listdir('../../../../dailymail/images/93e120a18f19f81445320d5a34906604135dc590')
# print(file)
# print(file.sort(key=lambda x:int(x[41:])))
# 删除了 93e120a18f19f81445320d5a34906604135dc590-19

