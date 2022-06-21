#%%
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dataset import *
from model import custom_DeepLabv3
from util import *

#%%
lr = 1e-3
batch_size = 4
num_epoch = 100
img_dir = '/home/h/Desktop/data/random/train/m_label'
label_dir = '/home/h/Desktop/data/random/train/t_label'
ckpt_dir = 'ckpt'
result_dir = 'result/d'

print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)
print("img dir: %s" % img_dir)
print("label dir: %s" % label_dir)
print("ckpt dir: %s" % ckpt_dir)
print("result dir: %s" % result_dir)

# make folder if doesn't exist
if not os.path.exists(result_dir) : os.makedirs(result_dir)
#%%
# network train
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomAutocontrast(p = 1),
    transforms.ToTensor(),
    transforms.Normalize(mean = 0.5, std = 0.5)
])
transform_label = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])
test_dataset = CustomDataset(
    img_dir = img_dir,
    label_dir = label_dir,
    transform = transform,
    transform_l = transform_label
)
test_loader = DataLoader(
    test_dataset,
    batch_size = batch_size,
    shuffle = False
)


# network generate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = custom_DeepLabv3().to(device)

# loss function, optimizer
fn_loss=nn.BCEWithLogitsLoss().to(device)
optim=torch.optim.Adam(net.parameters(), lr=lr)

# variables
num_data_test=len(test_dataset)

num_batch_test=np.ceil(num_data_test/batch_size)

# functions
fn_denorm=lambda x, mean, std:(x*std)+mean
fn_class=lambda x:1.0*(x>0.5) # network output image->binary class로 분류
tensor2PIL=transforms.ToPILImage()

#%%
# test network
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

with torch.no_grad(): # no backward pass 
    net.eval()
    loss_arr=[]

    for batch, data in enumerate(test_loader, 1):
        # forward pass
        input=data[0].to(device)
        label=data[1].to(device)
        output=net(input)['out']
        
        # loss function
        loss = fn_loss(output, label)
        loss_arr+=[loss.item()]
        print("TEST: BATCH %04d / %04d | LOSS %.4f" %
                      (batch, num_batch_test, np.mean(loss_arr)))
        
        for i in range(input.shape[0]):
            inputimg=tensor2PIL(fn_denorm(input[i], mean=0.5, std=0.5))
            outputimg=tensor2PIL(fn_class(output[i]))
            bg= Image.open('transparence.png').resize((512, 512)) 
            bg.paste(inputimg,outputimg)
            
            name=data[2][i].split('/')[-1].replace('m_label', 'eval').replace('jpg','png')

            new_image = Image.new('RGB',(1024,512), (250,250,250))
            new_image.paste(inputimg,(0,0))
            new_image.paste(bg,(512,0))
            new_image.save(os.path.join(os.path.join(result_dir), name))
print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %(batch, num_batch_test, np.mean(loss_arr)))

# %%
