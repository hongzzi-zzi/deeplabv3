#%%
import argparse
import gc
import os
from datetime import datetime
from tabnanny import verbose

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from dataset import *
from model import custom_DeepLabv3
from util import *

#%%
# training parameter
lr = 1e-3
batch_size = 4 # 6이 최대
train_continue = 'off'
num_epoch=100
img_dir='/home/h/Desktop/data/random/test/m_label'
label_dir='/home/h/Desktop/data/random/test/t_mask'
ckpt_dir = 'ckpt'
log_dir = 'log'

print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)
print("img_dir : %s" % img_dir)
print("label_dir : %s" % label_dir)
print("train_continue: %s" % train_continue)
print("ckpt_dir: %s" % ckpt_dir)
print("log_dir: %s" % log_dir)
#%%
# data aug & custom dataset
transform=transforms.Compose([transforms.Resize((512, 512)),
                              transforms.RandomHorizontalFlip(), 
                              transforms.RandomVerticalFlip(), 
                              transforms.RandomAffine([-60, 60]),
                              transforms.ColorJitter(brightness=0.3, contrast=0.3),
                              transforms.RandomAutocontrast(p=1),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=0.5, std=0.5)
                              ])
transform_label=transforms.Compose([transforms.Resize((512, 512)),
                              transforms.RandomHorizontalFlip(), 
                              transforms.RandomVerticalFlip(), 
                              transforms.RandomAffine([-60, 60]),
                              transforms.ToTensor(),
                              ])
dataset=CustomDataset(img_dir,label_dir , transform=transform,transform_l= transform_label)


validation_split=.1
shuffle_dataset=True
random_seed=42

dataset_size=len(dataset)
indices=list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)# 900
valid_sampler = SubsetRandomSampler(val_indices)

# dataset=dataset.type(torch.LongTensor)

training_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, drop_last=True)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=valid_sampler, drop_last=True)
#%%
# network generate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = custom_DeepLabv3().to(device)

# print(model)

# loss function, optimizer
# loss_fn = nn.BCEWithLogitsLoss().to(device)
loss_fn = torch.nn.BCEWithLogitsLoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# variables
num_data_train = len(training_loader)
num_data_val = len(validation_loader)

num_batch_train = np.ceil(num_data_train / batch_size)
num_batch_val = np.ceil(num_data_val / batch_size)

# functions
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5) # network output image->binary class로 분류

# set summarywriter to use tensorboard
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

# training network
st_epoch=0
#%%
'''def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    

    for i, data in enumerate(training_loader):
        inputs=data[0].to(device)# <class 'torch.Tensor'>
        labels=data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5
# EPOCHS = num_epoch

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    for i, vdata in enumerate(validation_loader):
        vinputs=vdata[0].to(device)
        vlabels=vdata[1].to(device)
        voutputs = model(vinputs)['out']
        vloss = loss_fn(voutputs.to(device), vlabels.squeeze().type(torch.LongTensor).to(device))
        running_vloss += vloss
    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1
    torch.cuda.empty_cache()
    gc.collect()
    if epoch % 5 == 0:
        save(ckpt_dir=ckpt_dir, net=model, optim=optimizer, epoch=epoch)'''
# %%
if train_continue == "on":
    model, optimizer, st_epoch = load(ckpt_dir=ckpt_dir, net=model, optim=optimizer)

for epoch in range(st_epoch + 1, num_epoch + 1):
    model.train()
    loss_arr = []

    for batch, data in enumerate(training_loader, 1):
        # forward pass
        
        input=data[0].to(device) # torch.Size([4, 3, 512, 512])
        label=data[1].to(device) # torch.Size([4, 1, 512, 512])
        output = model(input)['out'] # torch.Size([4, 1, 512, 512])

        # backward pass
        optimizer.zero_grad()
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        
        # loss function
        loss_arr += [loss.item()]
        print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %(epoch, num_epoch, batch, num_data_train, np.mean(loss_arr)))

        # save to tensorboard
        input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
        label = fn_tonumpy(label)
        # output = fn_tonumpy(output)
        output = fn_tonumpy(fn_class(output))
        # output = fn_tonumpy((output))
        
        writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
        writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
        writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

    writer_train.add_scalar('loss', np.mean(loss_arr), epoch)
        
    with torch.no_grad():
        model.eval()
        loss_arr = []

        for batch, data in enumerate(validation_loader, 1):
            # forward pass
            input=data[0].to(device)
            label=data[1].to(device)

            output = model(input)['out']

            # 손실함수 계산하기
            loss = loss_fn(output, label)

            loss_arr += [loss.item()]

            print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %(epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

            # save to tensorboard
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            label = fn_tonumpy(label)
            output = fn_tonumpy(fn_class(output))
            
            writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
            writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
            writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
    writer_val.add_scalar('loss', np.mean(loss_arr), epoch)
    
    if epoch % 50 == 0:
        save(ckpt_dir=ckpt_dir, net=model, optim=optimizer, epoch=epoch)
writer_train.close()
writer_val.close()
# %%
