#%%
# package
import os
import shutil
import random 
from glob import glob
from PIL import Image
from torchvision import transforms
#%%
dir_input=glob('/home/h/Desktop/data/random/*/m_label/*')
dir_label=glob('/home/h/Desktop/data/random/*/t_mask/*')
# print(len(dir_input))
# print(len(dir_label))
lst_input=sorted(dir_input)
lst_label=sorted(dir_label)
# print(lst_input[300])
# print(lst_label[300])
lst_all=[[i, l]for i, l in zip(lst_input, lst_label)]
img_cnt=len(lst_all)
# print(img_cnt)
# print(lst_all)
#%%
def invert(image):
    return image.point(lambda p: 255 - p)
# %%
for i in lst_all:
    origin_img=Image.open(i[0]).resize((512, 512))
    teeth_img=Image.open(i[1]).resize((512, 512))
    teeth_mask=teeth_img.split()[-1]
    
    name=i[1].replace('t_mask','b_mask')
    invert(teeth_mask).save(name)
# %%
