
# coding: utf-8

# # 特征抽取
# 
# Gluon中的model_zoo中模型的一般结构都是分为 feature 和 classifier部分。
# 
# 这里使用gluon中已经Pertrained好的network抽取特征，然后再接下来的分类器部分，我们用来组合特征进行分类。

# In[ ]:

# import packages

import mxnet as mx

from mxnet import image
from mxnet import init
from mxnet import gluon
from mxnet import ndarray as nd
from tqdm import tqdm
from mxnet import autograd
from mxnet.gluon.model_zoo.vision import *
import numpy as np
import time
import h5py
import os
import gc
from time import time


# In[ ]:

# 尝试在gpu上运行，cpu也是可以的，只是比较耗时
ctx = mx.gpu()


# ## 定义图片增广

# In[ ]:

def get_augs(shape):
    train_augs = [
        image.HorizontalFlipAug(.5),
        image.RandomCropAug((shape,shape))
    ]
    valid_augs = train_augs
    tests_augs = [
        image.CenterCropAug((shape,shape))
    ]
    return train_augs, valid_augs, tests_augs

def transform(data, label, augs):
    data = data.astype('float32')
    for aug in augs:
        data = aug(data)
    data = nd.transpose(data, (2,0,1))
    return data, nd.array([label]).asscalar().astype('float32')


# In[ ]:

loader = gluon.data.DataLoader


# ## 使用gluon中非常好用的 **ImageFolderDataset** 加载数据，实际上就是一个迭代器
# 通过aug支持reshape成network对应需要的input_shape大小

# In[ ]:

def get_images_data_loader(batch_size, shape):
    train_augs, valid_augs, tests_augs = get_augs(shape)
    train_imgs = gluon.data.vision.ImageFolderDataset('input/train',transform=lambda X, y: transform(X, y, train_augs))
    valid_imgs = gluon.data.vision.ImageFolderDataset('input/valid',transform=lambda X, y: transform(X, y, valid_augs))
    tests_imgs = gluon.data.vision.ImageFolderDataset('input/tests',transform=lambda X, y: transform(X, y, tests_augs))
    train_data = loader(train_imgs, batch_size, shuffle=True)
    valid_data = loader(valid_imgs, batch_size, shuffle=True)
    tests_data = loader(tests_imgs, batch_size)
    return train_data, valid_data, tests_data


# In[ ]:

batch_size = 64
train_224_data, valid_224_data, tests_224_data = get_images_data_loader(batch_size, 224)
train_299_data, valid_299_data, tests_299_data = get_images_data_loader(batch_size, 299)


# 尝试resnet和densent网络模型

# In[ ]:

# input shape 为224 的网络模型
net_224_list = ['resnet18_v2','resnet34_v2', 'resnet50_v2','resnet101_v1', 'resnet152_v1',
                'densenet121','densenet161','densenet169','densenet201']
# input shape 为299 的网络模型
net_299_list = ['inception_v3']


# 定义一个可以通过模型名字获得微调后的模型方法

# In[ ]:

def get_tuning_net(net_name, ctx):
    net_function = globals()[net_name] # 动态调用
    pretrained_net = net_function(pretrained=True, ctx=ctx)
    finetune_net = net_function(classes=120, ctx=ctx)
    finetune_net.features = pretrained_net.features
    finetune_net.classifier.initialize(init.Xavier())
    return finetune_net


# In[ ]:

def get_data_loader(net_input_shape, data_scope):
    '''
    通过network的输入shape和数据类型（train,vaild,tests）
    '''
    return globals()[data_scope + "_" + str(net_input_shape) + "_data"]


# 下面定义从pretrained 网络模型中抽取特征的方法，后边可以重复利用

# In[ ]:

def extract_features_by_pretrain_network(net_input_shape, data_scope, rebuild=False):
    
    net_list = globals()["net_%d_list" % net_input_shape]
    for name in net_list:
        
        features_file_name = "features/%s_%d_%s_features.h5" % (name, net_input_shape, data_scope)
        labels_file_name = "features/%d_%s_labels.h5" % (net_input_shape, data_scope)
        
        if not rebuild and os.path.exists(features_file_name) and os.path.exists(labels_file_name):
            print("%s and %s have exsits!" % (features_file_name, labels_file_name))
            continue
        
        print("starting extract features for network:%s, data_scope:%s, input_shape:%d" % (name, data_scope, net_input_shape))
        net = get_tuning_net(name, ctx)
        data_loader = get_data_loader(net_input_shape, data_scope)
        
        features = []
        labels = []
        for X, y in tqdm(data_loader):
            if not os.path.exists(features_file_name):
                feature = net.features(X.as_in_context(ctx))
                features.append(feature.asnumpy())
            if not os.path.exists(labels_file_name):
                labels.append(y.asnumpy())
        
        if len(features) > 0:
            print("saving features to file: %s" % features_file_name)
            features = np.concatenate(features, axis=0)
            with h5py.File(features_file_name, "w") as f:
                f["features"] = features
        
        if os.path.exists(labels_file_name):
            continue
        print("saving labels to file: %s" % labels_file_name)
        with h5py.File(labels_file_name, "w") as f:
            labels = np.concatenate(labels, axis=0)
            f["labels"] = labels  


# ## 开始抽取各个数据的特征

# In[ ]:

extract_features_by_pretrain_network(224, 'train')
extract_features_by_pretrain_network(224, 'valid')
extract_features_by_pretrain_network(224, 'tests')
#extract_features_by_pretrain_network(299, 'train')
#extract_features_by_pretrain_network(299, 'valid')
#extract_features_by_pretrain_network(299, 'tests')

