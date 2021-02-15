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
 
net = caffe.Net(model_def,      # define model_withoutcap structure
                model_weights,  # includes model_withoutcap's training weights
                caffe.TEST)     # use test mode (without dropout)

 
# load Imagenet image mean (released with Cafe)
mu = np.load('../imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average all the pixel values to get the average pixel value of BGR

 
# transform the input data
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

# set the number of channels of the image to the dimension of outermost
transformer.set_transpose('data', (2,0,1))  
# for each channel, the average pixel value of BGR is subtracted
transformer.set_mean('data', mu)
# transform the pixel value from [0, 255] to [0,1]
transformer.set_raw_scale('data', 255)
# switching channel, from RGB to BGR
transformer.set_channel_swap('data', (2,1,0))

def get_imgs_fea(fold_name):
    dir =fold_name
    files = os.listdir(dir)
#     the pictures are out of order in folder, so we sort them by file name
    files.sort(key=lambda x:int(x[40:]), reverse = True)
    imgs_feature = list()
    for f in files:
        image_path = os.path.join(dir,f)
        fea = extra_fea(image_path)
#         some pictures are invalid
        if fea:
            imgs_feature.append(fea)
    return imgs_feature

def extra_fea(image_path):
#         some pictures are invalid, such as ***.gif
    img = cv2.imread(image_path)
    if img is None :
        return False
#     reshape the size of image to fit the input size of VGGNet
    image=cv2.resize(img,(224,224))
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
#     get vector by layer name
    output = net.forward(end = 'fc7')
    output_fc = output['fc7'][0]
    return list(output_fc)