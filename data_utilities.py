#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function, division, absolute_import, unicode_literals


# In[ ]:


__idea__ = 'IFSS-Net'
__author__ = 'Dawood AL CHANTI'
__affiliation__ = 'LS2N-ECN'


# In[ ]:


from PIL import Image
import numpy as np
import glob
import cv2
import os
from os.path import join
import random
from medpy.io import load


# In[ ]:


def crop_pad(img):    
    H,W = img.shape
    if H>=512 and W>=512:
        PadEdgesSize1_H = int(abs(512-H)/2.)
        PadEdgesSize2_H = int(abs(512-H) - PadEdgesSize1_H)
        PadEdgesSize1_W = int(abs(512-W)/2.)
        PadEdgesSize2_W = int(abs(512-W) - PadEdgesSize1_W)
        new=img[PadEdgesSize1_H:H-PadEdgesSize2_H,PadEdgesSize1_W:W-PadEdgesSize2_W]
        
    elif H<512 and W>=512:
        new = np.vstack((img[:,:512],np.zeros_like(img[:,:512])))
        HH,WW = new.shape
        PadEdgesSize1_H = int(abs(512-HH)/2.)
        PadEdgesSize2_H = int(abs(512-HH) - PadEdgesSize1_H)
        PadEdgesSize1_W = int(abs(512-WW)/2.)
        PadEdgesSize2_W = int(abs(512-WW) - PadEdgesSize1_W)
        new=new[:512,PadEdgesSize1_W:WW-PadEdgesSize2_W]
    
    elif H>=512 and W<512:    
        new = np.hstack((img[:512,:],np.zeros_like(img[:512,:])))
        HH,WW = new.shape
        PadEdgesSize1_H = int(abs(512-HH)/2.)
        PadEdgesSize2_H = int(abs(512-HH) - PadEdgesSize1_H)
        PadEdgesSize1_W = int(abs(512-WW)/2.)
        PadEdgesSize2_W = int(abs(512-WW) - PadEdgesSize1_W)
        new=new[PadEdgesSize1_H:HH-PadEdgesSize2_H,:512]
    
    elif H<512 and W<512:   
        new = np.hstack((img[:,:],np.zeros_like(img[:,:])))
        new = np.vstack((new[:,:512],np.zeros_like(new[:,:512])))
        new = new[:512,:512]
    return new


# In[ ]:

def ReturnIndicesOfFullMask(sequence):
    '''
    The input sequence is of shape (1583, 512, 512)
    The output is a list of 2 indices, the begining and the end of thesequence with full masks
    Return the begining and the end of the sequence
    '''
    result = list(map(lambda img: img.sum() ,sequence ))
    resultIndex = list(map(lambda element: element>2950 ,result))
    Indices = [i for i, x in enumerate(resultIndex) if x]
    return Indices[0],Indices[-1]


def compute_mean_iou(pred, label):
    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels);
    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)
    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val
        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))
    mean_iou = np.mean(I / U)
    return mean_iou



def process_mask(mask_file_path):
    '''
    Return 3 masks in order SOL, GL and GM.
    Input: main path for the dataset: either Train/Val/or Testm i.e. '/tf/volumes/train/CAT_TH/masksX1.mha'
    '''
    
    #Read the data of formate (528, 640, 1574)
    mask_data,_ = load(mask_file_path)
    # Adjust the formate to (640, 528, 1574)
    mask_data = mask_data.transpose([1,0,2])

    # do some pre_processing : resize into 640 x 512 x 1574
    #Mask= list(map(lambda mask_time_step: cv2.resize(mask_time_step,(512,640),
         #                                      interpolation = cv2.INTER_AREA),
                   #mask_data.transpose(2,0,1)))
            
#     Mask= list(map(lambda mask_time_step: cv2.resize(mask_time_step,(512,512),
#                                               interpolation = cv2.INTER_AREA),
#                    mask_data.transpose(2,0,1)))#,(512,640),

    Mask= list(map(lambda mask_time_step:  crop_pad(mask_time_step),
                   mask_data.transpose(2,0,1)))#,(512,640),
    
    
    #Mask= list(map(lambda mask_time_step: mask_time_step[:640,:512],
        #           mask_data.transpose(2,0,1)))
    
    # get the output for each muscle of formate timesteps x H x W 
    # data clip to replace values of 100 150 and 200 to 1
    mask_sol= np.array(list(map(lambda mask_time_step: np.clip(np.where(mask_time_step == 100, mask_time_step, 0), 
                                                               0, 1,
                                                               np.where(mask_time_step == 100, mask_time_step, 0))
                                ,Mask)))
    mask_gl= np.array(list(map(lambda mask_time_step:np.clip(np.where(mask_time_step == 200, mask_time_step, 0), 
                                                             0, 1,np.where(mask_time_step == 200, mask_time_step, 
                                                                           0)),Mask)))
    mask_gm= np.array(list(map(lambda mask_time_step: np.clip(np.where(mask_time_step == 150, 
                                                                       mask_time_step, 0), 0, 1,
                                                              np.where(mask_time_step == 150, mask_time_step, 0)),
                               Mask)))
    
    mask_sol = np.expand_dims(mask_sol,-1)
    mask_gl = np.expand_dims(mask_gl,-1)
    mask_gm = np.expand_dims(mask_gm,-1)
    
    
    # return the whole muscles on channel axis of order SOL GL and GM
    return np.concatenate([mask_sol,mask_gl,mask_gm],-1)


# In[ ]:


#Only for SOL (Foregorund and Background)
def process_mask_SOL(mask_file_path):
    '''
    Return 3 masks in order SOL, GL and GM.
    Input: main path for the dataset: either Train/Val/or Testm i.e. '/tf/volumes/train/CAT_TH/masksX1.mha'
    '''
    
    #Read the data of formate (528, 640, 1574)
    mask_data,_ = load(mask_file_path)
    # Adjust the formate to (640, 528, 1574)
    mask_data = mask_data.transpose([1,0,2])

    Mask= list(map(lambda mask_time_step:  crop_pad(mask_time_step),
                   mask_data.transpose(2,0,1)))#,(512,640),
    
   #resize(mask_time_step,(512,640),

    #Mask= list(map(lambda mask_time_step: mask_time_step[:640,:512],
                   #mask_data.transpose(2,0,1)))
        
        
    # get the output for each muscle of formate timesteps x H x W 
    # data clip to replace values of 100 150 and 200 to 1
    mask_sol= np.array(list(map(lambda mask_time_step: np.clip(np.where(mask_time_step == 100, mask_time_step, 0), 
                                                               0, 1,
                                                               np.where(mask_time_step == 100, mask_time_step, 0))
                                ,Mask)))

    mask_sol = np.expand_dims(mask_sol,-1)
 
    # Get the back ground of the annotated mask using the foreground annotation
    mask_sol=np.concatenate([mask_sol,1-mask_sol],-1)

    return mask_sol


# In[ ]:


def process_data(data_file_path):
    '''
    Return 3 masks in order SOL, GL and GM.
    Input: main path for the dataset: either Train/Val/or Testm i.e. '/tf/volumes/train/CAT_TH/masksX1.mha'
    '''
    
    #Read the data of formate (528, 640, 1574)
    image_data,_ = load(data_file_path)
    # Adjust the formate to (640, 528, 1574)
    image_data = image_data.transpose([1,0,2])

    image_data= list(map(lambda mask_time_step:  crop_pad(mask_time_step),
                   image_data.transpose(2,0,1))) #,(512,640),
    
    #image_data= list(map(lambda image_data_step: image_data_step[:640,:512],image_data.transpose(2,0,1)))    
    
    return np.array(image_data)


# In[ ]:


def Pull_data_from_path(path):
    data = process_data(path)
    # return normalized data
    # values from whole data
    mean_val = 19.027262640214904
    std_val = 34.175155632916
    
    data = (data-mean_val) / std_val
    # reshape to t,h,w,1
    return  np.expand_dims(data,-1)


# In[ ]:


def Pull_data_from_path_Complete(path):
    data = process_data(path)
    # return normalized data
    # values from whole data
    mean_val = 19.027262640214904
    std_val = 34.175155632916
    
    data = (data-mean_val) / std_val
    # reshape to t,h,w,1
    return  np.expand_dims(data,-1)


# In[ ]:


#Only for SOL (Foregorund and Background)
def process_mask_SOL_Complete(mask_file_path):
    '''
    Return 3 masks in order SOL, GL and GM.
    Input: main path for the dataset: either Train/Val/or Testm i.e. '/tf/volumes/train/CAT_TH/masksX1.mha'
    '''
    
    #Read the data of formate (528, 640, 1574)
    mask_data,_ = load(mask_file_path)
    # Adjust the formate to (640, 528, 1574)
    mask_data = mask_data.transpose([1,0,2])

    # do some pre_processing : resize into 640 x 512 x 1574
    #Mask= list(map(lambda mask_time_step: cv2.resize(mask_time_step,(512,640),
         #                                      interpolation = cv2.INTER_AREA),
                   #mask_data.transpose(2,0,1)))
            
    Mask= list(map(lambda mask_time_step:  crop_pad(mask_time_step),
                   mask_data.transpose(2,0,1))) #(512,640)

    #Mask= list(map(lambda mask_time_step: mask_time_step[:640,:512],
               #    mask_data.transpose(2,0,1)))
    
    # get the output for each muscle of formate timesteps x H x W 
    # data clip to replace values of 100 150 and 200 to 1
    mask_sol= np.array(list(map(lambda mask_time_step: np.clip(np.where(mask_time_step == 100, mask_time_step, 0), 
                                                               0, 1,
                                                               np.where(mask_time_step == 100, mask_time_step, 0))
                                ,Mask)))

    mask_sol = np.expand_dims(mask_sol,-1)
 
    # Get the back ground of the annotated mask using the foreground annotation
    mask_sol=np.concatenate([mask_sol,1-mask_sol],-1)

    return mask_sol


# In[ ]:


def Pull_mask_from_path(path):
    return process_mask(path)


# In[ ]:


def Patient_name(diretoryPathforOnePatient):
    return diretoryPathforOnePatient.split('/')[4]

