
# coding: utf-8

# # Extracte features via pretrain models, e.g. vgg, resnet...
# 
# more can see the [gluon model_zoo](https://mxnet.incubator.apache.org/versions/master/api/python/gluon/model_zoo.html)

# In[ ]:


# import packages
import mxnet as mx
from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import ndarray as nd
from mxnet.gluon import nn
from mxnet.gluon.data import vision
from mxnet.gluon.model_zoo import vision as models
import pandas as pd
import numpy as np
import shutil
import h5py
import os
import gc
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# In[ ]:


# try run via gpu
ctx = mx.gpu()


# In[ ]:


transformers = [
    # 强制resize成pretrain模型的输入大小
    image.ForceResizeAug((224,224)),
    # 标准化处理
    image.ColorNormalizeAug(mean=nd.array([0.485, 0.456, 0.406]), std=nd.array([0.229, 0.224, 0.225]))
]


# Transoform image and label to our target format data
# 
# 为避免过拟合，我们在这里使用`image.CreateAugmenter`来加强数据集。例如我们设`rand_mirror=True`即可随机对每张图片做镜面反转。我们也通过`mean`和`std`对彩色图像RGB三个通道分别做标准化。以下我们列举了该函数里的所有参数，这些参数都是可以调的。

# In[ ]:


train_augs_params = {
    "resize":1, 
    "rand_crop":False, 
    "rand_resize":False, 
    "rand_mirror":True,
    "mean":np.array([0.4914, 0.4822, 0.4465]), 
    "std":np.array([0.2023, 0.1994, 0.2010]), 
    "brightness":0, 
    "contrast":0,
    "saturation":0, 
    "hue":0,
    "pca_noise":0, 
    "rand_gray":0, 
    "inter_method":2
}

test_augs_params = {
    "resize":1,
    "mean":np.array([0.4914, 0.4822, 0.4465]), 
    "std":np.array([0.2023, 0.1994, 0.2010])
}

def get_image_augs(shape, scope="train"):
    augs_params = train_augs_params if scope == 'train' else test_augs_params
    augs_params['data_shape'] =  (0, shape, shape)
    augs = image.CreateAugmenter(**augs_params)
    return augs

augs = get_image_augs(244)

def transform_callback(data, label):
    im = data.astype('float32') / 255
    for aug in augs:
        im = aug(im)
    # 将数据格式从"高*宽*通道"改为"通道*高*宽"。
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar().astype('float32'))


# In[ ]:


from mxnet.gluon.model_zoo.vision import *


# In[ ]:


net_list_dict = {
    # net_input_shape -> pretrained net
    244: [resnet152_v1, densenet161],
    299: [inception_v3]
}


# In[ ]:


models = {}
for net_list in net_list_dict.values():
    for net in net_list:
        models[net.__name__] = net(pretrained=True, ctx=ctx)


# definition a function for extractor feature from pretrain model, then concat them together.

# In[ ]:


def features_extract(net_list, net_input_shape, X, y, batch_num, scope = "train"):
    feature_dir = 'features/%s/batch_%05d' % (scope, batch_num)
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    feature_file = '%s/feature_shape_%03d.h5' % (feature_dir, net_input_shape)
    if os.path.exists(feature_file):
        return
    
    features = {}
    labels = None
    for net in net_list: 
        net_name = net.__name__
        net = models[net_name]
        net.collect_params().reset_ctx(ctx) # make sure or network's params on gpu
        feature = net.features(X.as_in_context(ctx)).asnumpy()
        labels = y.asnumpy()
        features[net_name] = feature

    # save file
    with h5py.File(feature_file, 'w') as f:
        for net in net_list:
            f[net.__name__] = features[net.__name__]
        if scope == 'train':
            f['labels'] = labels


# In[ ]:


def build_features_with_net_input_shape(scope="train", batch_size=32):
    imgs = vision.ImageFolderDataset('input/%s' % scope, flag=1, transform=transform_callback)
    data = gluon.data.DataLoader(imgs, batch_size)
    for net_shape in net_list_dict:
        augs = get_image_augs(net_shape, scope)
        net_list = net_list_dict[net_shape]
        batch_num = 0
        for X, y in tqdm(data):
            features_extract(net_list, net_shape, X, y, batch_num, scope)
            batch_num += 1


# In[ ]:


build_features_with_net_input_shape("train")


# In[ ]:


build_features_with_net_input_shape("test")

