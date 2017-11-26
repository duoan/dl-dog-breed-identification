
# coding: utf-8

# In[1]:


import mxnet as mx

from mxnet import image
from mxnet import init
from mxnet import gluon
from mxnet import ndarray as nd
from tqdm import tqdm
from mxnet import autograd
from mxnet.gluon.model_zoo import vision as models
import time
import h5py
import os


# In[2]:


ctx = mx.gpu()


# In[3]:


models.alexnet(pretrained=True, ctx=ctx)
net_functions = [models.alexnet, 
        models.densenet121,
        models.densenet161,
        models.densenet169,
        models.densenet201,
        models.inception_v3,
        models.resnet18_v2,
        models.resnet34_v2,
        models.resnet50_v1,
        models.vgg11_bn,
        models.vgg13_bn,
        models.vgg16_bn,
        models.vgg19_bn]


# In[4]:


net_dict = {}
for net_function in net_functions:
    pretrained_net = net_function(pretrained=True, ctx=ctx)
    finetune_net = net_function(classes=120, ctx=ctx)
    finetune_net.features = pretrained_net.features
    finetune_net.classifier.initialize(init.Xavier())
    net_dict[net_function.__name__] = finetune_net


# In[5]:


train_augs = [
    image.HorizontalFlipAug(.5),
    image.RandomCropAug((224,224))
]
valid_augs = train_augs
tests_augs = [
    image.CenterCropAug((224,224))
]

def transform(data, label, augs):
    data = data.astype('float32')
    for aug in augs:
        data = aug(data)
    data = nd.transpose(data, (2,0,1))
    return data, nd.array([label]).asscalar().astype('float32')


# In[6]:


train_imgs = gluon.data.vision.ImageFolderDataset('input/train',transform=lambda X, y: transform(X, y, train_augs))
valid_imgs = gluon.data.vision.ImageFolderDataset('input/valid',transform=lambda X, y: transform(X, y, valid_augs))
tests_imgs = gluon.data.vision.ImageFolderDataset('input/tests',transform=lambda X, y: transform(X, y, tests_augs))


# In[7]:


loader = gluon.data.DataLoader


# In[8]:


def features_extract(scope="train", batch_size=32):
    if scope == 'train':
        data_iterator = loader(train_imgs, batch_size, shuffle=True)
    elif scope == 'valid':
        data_iterator = loader(valid_imgs, batch_size, shuffle=True)
    else:
        data_iterator = loader(tests_imgs, batch_size, shuffle=True)
    
    batch_num = 0
    for X, y in tqdm(data_iterator):
        for net_name in net_dict:
            print("start %s extract features via %s" % (scope, net_name))
            net = net_dict[net_name]
            feats_file = 'features/%s/feats_%03d_%03d_%s.h5' % (scope, batch_size, batch_num, net_name)
            labls_file = 'features/%s/labls_%03d_%03d.h5' % (scope, batch_size, batch_num)

            if not os.path.exists(feats_file):
                net.collect_params().reset_ctx(ctx) # make sure or network's params on gpu
                features = net.features(X.as_in_context(ctx)).asnumpy()
                with h5py.File(feats_file, 'w') as f:
                    f["features"] = features

            if scope != "tests" and not os.path.exists(labls_file):
                labels = y.asnumpy()
                with h5py.File(labls_file, 'w') as f:
                    f['labels'] = labels
        
        batch_num += 1
    print("finished extract %s features via %s" % (scope, net_name))


# In[ ]:


features_extract('train')
features_extract('valid')
features_extract('tests')


# In[ ]:


# def accuracy(output, labels):
#     return nd.mean(nd.argmax(output, axis=1) == labels).asscalar()

# def evaluate(net, data):
#     loss, acc, n = 0., 0., 0.
#     steps = len(data_iter_val)
#     for data, label in data:
#         data, label = data.as_in_context(ctx), label.as_in_context(ctx)
#         output = net(data)
#         acc += accuracy(output, label)
#         loss += nd.mean(softmax_cross_entropy(output, label)).asscalar()
#     return loss/steps, acc/steps


# In[ ]:




# softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
# def fit(net, ctx, batch_size=64, epochs=10, valid_ratio=0.3, learning_rate=0.01, wd=0.001):
#     train_data = loader(train_imgs, batch_size, shuffle=True)
#     valid_data = loader(valid_imgs, batch_size, shuffle=True)
#     # 确保net的初始化在ctx上
#     net.collect_params().reset_ctx(ctx)
#     net.hybridize()
#     loss = gluon.loss.SoftmaxCrossEntropyLoss()
#     # 训练
#     trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 1e-4, 'wd': 1e-5})
#     for epoch in range(30):
#         train_loss = 0.
#         train_acc = 0.
#         steps = len(train_data)
#         for data, label in tqdm(train_data):
#             data, label = data.as_in_context(ctx), label.as_in_context(ctx)
#             with autograd.record():
#                 output = net(data)
#                 loss = softmax_cross_entropy(output, label)

#             loss.backward()
#             trainer.step(batch_size)
            
#             train_loss += nd.mean(loss).asscalar()
#             train_acc += accuracy(output, label)

#         val_loss, val_acc = evaluate(net, valid_data)
#         print("Epoch %d. loss: %.4f, acc: %.2f%%, val_loss %.4f, val_acc %.2f%%" % 
#               (epoch+1, train_loss/steps, train_acc/steps*100, val_loss, val_acc*100))
    
# for net_name in net_dict:
#     net = net_dict[net_name]
#     fit(net,ctx)
#     break

