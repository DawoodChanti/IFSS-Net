#!/usr/bin/env python
# coding: utf-8

# In[ ]:


__idea__ = 'IFSS-Net'
__author__ = 'Dawood AL CHANTI'
__affiliation__ = 'LS2N-ECN'


# In[ ]:


import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib import rnn


# In[ ]:


def Conv3DBlock(inputs, n_filters, kernel_size=[3, 3, 3], stride = [1,1,1],activation_fn=None):
        """
        Builds the 3d conv block 
        Apply successivly a 3D convolution, BatchNormalization and relu
        """
        # Skip pointwise by setting num_outputs=Non
        net = slim.conv3d(inputs, n_filters, kernel_size=kernel_size,stride=stride, activation_fn=activation_fn)
        net =  slim.layer_norm(net) #slim.batch_norm(net, fused=True)
        net = tf.nn.relu(net)
        return net


# In[ ]:


def Conv3DBlockTranspose(inputs, n_filters, kernel_size=[3, 3, 3], stride = [1,1,1],activation_fn=None):
        """
        Builds the 3d conv transpose block 
        Apply successivly a 3D transpose convolution, BatchNormalization and relu
        """
        # Skip pointwise by setting num_outputs=Non
        net = slim.conv3d_transpose(inputs, n_filters, kernel_size=kernel_size,
                                    stride=stride, activation_fn=activation_fn)
        net =  slim.layer_norm(net) #slim.batch_norm(net, fused=True)
        net = tf.nn.relu(net)
        return net


# In[ ]:


def AtrousSpatialPyramidPoolingModule_3D(inputs, depth=256):
    
    '''
    5D Tensor: batch, time, H, W, C
    '''
    feature_map_size = tf.shape(inputs)

    # Global average pooling
    image_features = tf.reduce_mean(inputs, [2, 3], keep_dims=True)

    image_features = slim.conv3d(image_features, depth, [1,1,1], activation_fn=None)
  

#     image_features = tf.transpose(tf.stack(list(map(lambda a: 
#                                                     tf.image.resize_bilinear(a, (feature_map_size[2], 
#                                                                                  feature_map_size[3])), 
#                                                     tf.unstack(tf.transpose(image_features,[1,0,2,3,4])))),0),
#                                   [1,0,2,3,4])
    
    image_features = tf.keras.layers.UpSampling3D(size=(1, int(inputs.shape[2]), 
                                                        int(inputs.shape[3])))(image_features)
    
    atrous_pool_block_1 = slim.conv3d(inputs, depth, [1, 1,1], activation_fn=None) # 3x3 filter reciptive field

    atrous_pool_block_6 = slim.conv3d(inputs, depth, [3, 3,3], rate=6, activation_fn=None)# 9x9

    atrous_pool_block_12 = slim.conv3d(inputs, depth, [3, 3,3], rate=12, activation_fn=None)# 15x15

    atrous_pool_block_18 = slim.conv3d(inputs, depth, [3, 3,3], rate=18, activation_fn=None)# 21x21

    net = tf.concat((image_features, atrous_pool_block_1, atrous_pool_block_6, 
                     atrous_pool_block_12, atrous_pool_block_18), axis=-1)

    return net


# In[ ]:


def AtrousSpatialPyramidPoolingModule_3D_rate_9(inputs, depth=256):
    
    '''
    5D Tensor: batch, time, H, W, C
    '''
    feature_map_size = tf.shape(inputs)

    # Global average pooling
    image_features = tf.reduce_mean(inputs, [2, 3], keep_dims=True)

    image_features = slim.conv3d(image_features, depth, [1,1,1], activation_fn=None)
  

#     image_features = tf.transpose(tf.stack(list(map(lambda a: 
#                                                     tf.image.resize_bilinear(a, (feature_map_size[2], 
#                                                                                  feature_map_size[3])), 
#                                                     tf.unstack(tf.transpose(image_features,[1,0,2,3,4])))),0),
#                                   [1,0,2,3,4])
    
    
    image_features = tf.keras.layers.UpSampling3D(size=(1, int(inputs.shape[2]), 
                                                        int(inputs.shape[3])))(image_features)
    
    
    atrous_pool_block_1 = slim.conv3d(inputs, depth, [1, 1,1], activation_fn=None)# 3x3

    atrous_pool_block_6 = slim.conv3d(inputs, depth, [3, 3,3], rate=2, activation_fn=None)#5x5

    atrous_pool_block_12 = slim.conv3d(inputs, depth, [3, 3,3], rate=4, activation_fn=None)# 7x7

    atrous_pool_block_18 = slim.conv3d(inputs, depth, [3, 3,3], rate=6, activation_fn=None)# 9x9

    net = tf.concat((image_features, atrous_pool_block_1, atrous_pool_block_6, 
                     atrous_pool_block_12, atrous_pool_block_18), axis=-1)

    return net


# In[ ]:


def AtrousSpatialPyramidPoolingModule_3D_rate_11(inputs, depth=256):
    
    '''
    5D Tensor: batch, time, H, W, C
    '''
    feature_map_size = tf.shape(inputs)

    # Global average pooling
    image_features = tf.reduce_mean(inputs, [2, 3], keep_dims=True)

    image_features = slim.conv3d(image_features, depth, [1,1,1], activation_fn=None)
  

#     image_features = tf.transpose(tf.stack(list(map(lambda a: 
#                                                     tf.image.resize_bilinear(a, (feature_map_size[2], 
#                                                                                  feature_map_size[3])), 
#                                                     tf.unstack(tf.transpose(image_features,[1,0,2,3,4])))),0),
#                                   [1,0,2,3,4])
    
    image_features = tf.keras.layers.UpSampling3D(size=(1, int(inputs.shape[2]), 
                                                        int(inputs.shape[3])))(image_features)
    
    
    atrous_pool_block_1 = slim.conv3d(inputs, depth, [1, 1,1], activation_fn=None)

    atrous_pool_block_6 = slim.conv3d(inputs, depth, [3, 3,3], rate=4, activation_fn=None) #7x7

    atrous_pool_block_12 = slim.conv3d(inputs, depth, [3, 3,3], rate=6, activation_fn=None)# 9x9

    atrous_pool_block_18 = slim.conv3d(inputs, depth, [3, 3,3], rate=8, activation_fn=None)# 11x11

    net = tf.concat((image_features, atrous_pool_block_1, atrous_pool_block_6, 
                     atrous_pool_block_12, atrous_pool_block_18), axis=-1)

    return net


# In[ ]:


def AttentionRefinementModule_3D(inputs, n_filters):
    'for 3d data'
    # 3D Global average pooling
    net = tf.reduce_mean(inputs, [2, 3], keep_dims=True)
    net = slim.conv3d(net, n_filters, kernel_size=[1,1,1])
    net = slim.layer_norm(net)
    #net = slim.batch_norm(net, fused=True)
    net = tf.sigmoid(net)
    net = tf.multiply(inputs, net)
    return net


# In[ ]:


def FeatureFusionModule(input_1, input_2, n_filters):
    
    inputs = tf.concat([input_1, input_2], axis=-1)
    inputs =Conv3DBlock(inputs, n_filters, kernel_size=[3, 3, 3], stride = [1,1,1],activation_fn=None)
    

    # Global average pooling
    net = tf.reduce_mean(inputs, [2, 3], keep_dims=True)
    
    net = slim.conv3d(net, n_filters, kernel_size=[1,1,1])
    net = tf.nn.relu(net)
    
    net = slim.conv3d(net, n_filters, kernel_size=[1,1,1])
    net = tf.sigmoid(net)

    net = tf.multiply(inputs, net)

    net = tf.add(inputs, net)
    return net


# In[ ]:


def FeatureFusionModule_with_Stirde(input_1, input_2, n_filters):
    
    inputs = tf.concat([input_1, input_2], axis=-1)
    inputs =Conv3DBlock(inputs, n_filters, kernel_size=[3, 3, 3], stride = [1,2,2],activation_fn=None)
    

    # Global average pooling
    net = tf.reduce_mean(inputs, [2, 3], keep_dims=True)
    
    net = slim.conv3d(net, n_filters, kernel_size=[1,1,1])
    net = tf.nn.relu(net)
    
    net = slim.conv3d(net, n_filters, kernel_size=[1,1,1])
    net = tf.sigmoid(net)

    net = tf.multiply(inputs, net)

    net = tf.add(inputs, net)
    return net


# In[ ]:


def Tversky_sigmoid(prediction,ground_truth,alpha = 0.25, axis=(1, 2, 3,4)):
    '''
    prediction is the probabilities from sigmoid. Only one channel is provided and backgorund is estimated
    '''
    
    P_foreground = prediction
    P_background = 1-prediction

    g_foreground = ground_truth
    g_background = 1-ground_truth

    true_positive = tf.reduce_sum(P_foreground * g_foreground, axis=axis)
    false_pos = tf.reduce_sum(P_foreground * g_background, axis=axis)
    false_neg = tf.reduce_sum(P_background * g_foreground, axis=axis)

    Tversky = tf.divide(true_positive+smooth, 
                       true_positive + (alpha*false_pos) + ((1-alpha)*false_neg) + smooth)

    return 1-tf.reduce_mean(Tversky)


def focal_tversky_sigmoid(prediction,ground_truth,alpha):
    Tversly_loss = Tversky_sigmoid(prediction,ground_truth,alpha)
    gamma = 2
    return tf.pow(-tf.math.log(Tversly_loss), gamma)


def Tversky_softmax(prediction,ground_truth,keep_prob_tp,keep_prob_alpha,keep_prob_beta, axis=(1, 2, 3,4)):
    '''
    prediction is the probabilities from softmax
    '''    
    smooth=1
    P_foreground = prediction[...,:1]
    P_background = prediction[...,1:]
    g_foreground = ground_truth[...,:1]
    g_background = ground_truth[...,1:]

    P_foreground = tf.nn.dropout(P_foreground,keep_prob_tp,name="P_foreground_drop_out")
    #g_foreground = tf.nn.dropout(g_foreground,keep_prob_tp,name="g_foreground_drop_out")

    #P_background = tf.nn.dropout(P_background,keep_prob_tp,name="P_background_drop_out")
    #g_background = tf.nn.dropout(g_background,keep_prob_tp,name="g_background_drop_out")
   
    true_positive= P_foreground * g_foreground
    true_positive = tf.reduce_sum(true_positive, axis=axis)
   
    false_pos = P_foreground * g_background
    false_pos = keep_prob_alpha*tf.reduce_sum(false_pos, axis=axis)
    
    
    false_neg = P_background * g_foreground
    false_neg = keep_prob_beta*tf.reduce_sum(false_neg, axis=axis)
    
    Tversky = tf.divide(true_positive+smooth, 
                        (true_positive + false_pos + false_neg + smooth))
    
    return tf.reduce_mean(Tversky) 

def focal_tversky_softmax(prediction,ground_truth,keep_prob_tp,keep_prob_alpha,keep_prob_beta):
    Tversly_loss = Tversky_softmax(prediction,ground_truth,keep_prob_tp,keep_prob_alpha,keep_prob_beta)
    gamma = 2
    return tf.pow(1-Tversly_loss, gamma)


# In[ ]:


def Tversky_Index(prediction,ground_truth,axis=(1, 2, 3,4)):
    '''
    prediction is the probabilities from softmax, here it act as Dice Coef, as alpha and beta 
    '''    
    smooth=1
    P_foreground = prediction[...,:1]
    P_background = prediction[...,1:]
    g_foreground = ground_truth[...,:1]
    g_background = ground_truth[...,1:]

   
    true_positive= P_foreground * g_foreground
    true_positive = tf.reduce_sum(true_positive, axis=axis)
   
    false_pos = P_foreground * g_background
    false_pos = 0.5*tf.reduce_sum(false_pos, axis=axis)
    
    
    false_neg = P_background * g_foreground
    false_neg = 0.5*tf.reduce_sum(false_neg, axis=axis)

    Tversky = tf.divide(true_positive+smooth, 
                        (true_positive + false_pos + false_neg + smooth))
    
    return tf.reduce_mean(Tversky) 


# In[ ]:


def dice_coe(output, target, loss_type='sorensen', axis=(1, 2, 3,4), smooth=1e-5):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity

    """
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    # old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    # new haodong
    dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    dice = tf.reduce_mean(dice, name='dice_coe')
    return dice   


# In[ ]:


def iou_coe(output, target, threshold=0.5, axis=(1, 2, 3,4), smooth=1e-5):
    """Non-differentiable Intersection over Union (IoU) for comparing the similarity 
    """
    pre = tf.cast(output > threshold, dtype=tf.float32)
    truth = tf.cast(target > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(pre, truth), axis=axis)  # AND
    union = tf.reduce_sum(tf.cast(tf.add(pre, truth) >= 1, dtype=tf.float32), axis=axis)  # OR
    # old axis=[0,1,2,3]
    # epsilon = 1e-5
    # batch_iou = inse / (union + epsilon)
    # new haodong
    batch_iou = (inse + smooth) / (union + smooth)
    iou = tf.reduce_mean(batch_iou, name='iou_coe')
    return iou  # , pre, truth, inse, union


# In[ ]:


def dice_hard_coe(output, target, threshold=0.5, axis=(1, 2, 3,4), smooth=1e-5):
    """Non-differentiable Sørensen–Dice coefficient for comparing the similarity
    """
    output = tf.cast(output > threshold, dtype=tf.float32)
    target = tf.cast(target > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(output, target), axis=axis)
    l = tf.reduce_sum(output, axis=axis)
    r = tf.reduce_sum(target, axis=axis)
    hard_dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    hard_dice = tf.reduce_mean(hard_dice, name='hard_dice')
    return hard_dice


# In[ ]:


K = tf.keras.backend

def confusion(y_true, y_pred):
    smooth=1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg) 
    prec = (tp + smooth)/(tp+fp+smooth)
    recall = (tp+smooth)/(tp+fn+smooth)
    return prec, recall


# In[ ]:


def Downsampling_2D(inputs,scale1,scale2):
    return tf.image.resize_bilinear(inputs, size=[scale1,scale2])

def Down_sample_3D(input_layer,scale1,scale2):
    '''
    TODO : 10 should be changed to new time scale 10 30 or 200 
    '''
    unpol_layer = list(map(lambda layer: Downsampling_2D(layer,scale1,scale2), tf.unstack(input_layer, 
                                                                                  int(input_layer.get_shape()[1]), 
                                                                                  1)))
    return tf.transpose(tf.stack(unpol_layer),[1,0,2,3,4])


# In[ ]:


def skip(layer, end_point): return tf.concat([layer, end_point], axis=4)
def unpool(layer): return tf.image.resize_nearest_neighbor(layer, 
                                                           [2*int(layer.get_shape()[1]), 
                                                            2*int(layer.get_shape()[2])])
# unpool based on 2D data
def unpool_3D_2(input_layer):
    unpol_layer = list(map(lambda layer: unpool(layer), tf.unstack(input_layer, 
                                                                   int(input_layer.get_shape()[1]), 1)))
    return tf.transpose(tf.stack(unpol_layer),[1,0,2,3,4])


def unpool_3D(input_layer):
    
    unpol_layer = tf.keras.layers.UpSampling3D(size=(1, 2,2))(input_layer)
    return unpol_layer


# In[ ]:


# Re-define for 5D Tensor, here we are dealing with 5D: axis=3 become axis=4 last channel
def pixel_wise_softmax(output_map):
    return tf.nn.softmax(output_map)


# In[ ]:


from tensorflow.contrib.layers.python.layers import initializers
def weight_variable(shape):
    weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.float32)
    Weight = tf.Variable(weights_initializer(shape=shape))    
    return Weight


# In[ ]:


def weight_variable2(shape):
    weights_initializer=tf.random_uniform_initializer(minval=0.5, maxval=0.5, seed=None)
    Weight = tf.Variable(weights_initializer(shape=shape))    
    return Weight


# In[ ]:





# In[ ]:


def create_folder_subfolders(proposedPath):
    '''
    
    Create a folder and sub folders for each of the foreground and background using a proposed path.
    '''
    
    # GL
    newpathGLL = proposedPath + '/GL'
    if os.path.exists(newpathGLL) is True:
        pass
    else:
        os.mkdir(newpathGLL)
        
    newpathGLFF = newpathGLL +  '/F'
    newpathGLBB = newpathGLL +  '/B'
    
    if os.path.exists(newpathGLFF) is True:
        pass
    elif os.path.exists(newpathGLBB) is True:
        pass
    else:
        os.mkdir(newpathGLFF)
        os.mkdir(newpathGLBB)
                    

    # GM
    newpathGMM = proposedPath + '/GM'
    if os.path.exists(newpathGMM) is True:
        pass
    else:
        os.mkdir(newpathGMM)
        
    newpathGMFF = newpathGMM +  '/F'
    newpathGMBB = newpathGMM +  '/B'
    
    if os.path.exists(newpathGMFF) is True:
        pass
    elif os.path.exists(newpathGMBB) is True:
        pass
    else:
        os.mkdir(newpathGMFF)
        os.mkdir(newpathGMBB)
                      
            
     # SOL
    newpathSOLL = proposedPath + '/SOL'
    if os.path.exists(newpathSOLL) is True:
        pass
    else:
        os.mkdir(newpathSOLL)
        
    newpathSOLFF = newpathSOLL +  '/F'
    newpathSOLBB = newpathSOLL +  '/B'
    
    if os.path.exists(newpathSOLFF) is True:
        pass
    elif os.path.exists(newpathSOLBB) is True:
        pass
    else:
        os.mkdir(newpathSOLFF)
        os.mkdir(newpathSOLBB)
        
    return newpathGLFF,newpathGLBB,newpathGMFF,newpathGMBB,newpathSOLFF, newpathSOLBB     


# In[ ]:


def create_folder_subfoldersMASK(proposedPath):
    '''
    
    Create a folder and sub folders for each of the foreground and background using a proposed path.
    '''
    
    # GL
    newpathGLL = proposedPath + '/GL'
    if os.path.exists(newpathGLL) is True:
        pass
    else:
        os.mkdir(newpathGLL)
        
    newpathGLFF = newpathGLL +  '/F_mask'
    newpathGLBB = newpathGLL +  '/B_mask'
    
    if os.path.exists(newpathGLFF) is True:
        pass
    elif os.path.exists(newpathGLBB) is True:
        pass
    else:
        os.mkdir(newpathGLFF)
        os.mkdir(newpathGLBB)
                    

    # GM
    newpathGMM = proposedPath + '/GM'
    if os.path.exists(newpathGMM) is True:
        pass
    else:
        os.mkdir(newpathGMM)
        
    newpathGMFF = newpathGMM +  '/F_mask'
    newpathGMBB = newpathGMM +  '/B_mask'
    
    if os.path.exists(newpathGMFF) is True:
        pass
    elif os.path.exists(newpathGMBB) is True:
        pass
    else:
        os.mkdir(newpathGMFF)
        os.mkdir(newpathGMBB)
                      
            
     # SOL
    newpathSOLL = proposedPath + '/SOL'
    if os.path.exists(newpathSOLL) is True:
        pass
    else:
        os.mkdir(newpathSOLL)
        
    newpathSOLFF = newpathSOLL +  '/F_mask'
    newpathSOLBB = newpathSOLL +  '/B_mask'
    
    if os.path.exists(newpathSOLFF) is True:
        pass
    elif os.path.exists(newpathSOLBB) is True:
        pass
    else:
        os.mkdir(newpathSOLFF)
        os.mkdir(newpathSOLBB)
        
    return newpathGLFF,newpathGLBB,newpathGMFF,newpathGMBB,newpathSOLFF, newpathSOLBB     


# In[ ]:


def iou_coe_Slice_by_Slice(output, target, threshold=0.5, axis=(2, 3,4), smooth=1e-5):
    """Non-differentiable Intersection over Union (IoU) for comparing the similarity 
    """
    pre = tf.cast(output > threshold, dtype=tf.float32)
    truth = tf.cast(target > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(pre, truth), axis=axis)  # AND
    union = tf.reduce_sum(tf.cast(tf.add(pre, truth) >= 1, dtype=tf.float32), axis=axis)  # OR
    # old axis=[0,1,2,3]
    # epsilon = 1e-5
    # batch_iou = inse / (union + epsilon)
    # new haodong
    batch_iou = (inse + smooth) / (union + smooth)
    iou = tf.reduce_mean(batch_iou,axis=0, name='iou_coe')
    return iou  # , pre, truth, inse, union


# In[ ]:


def Tversky_Index_Slice_by_Slice(prediction,ground_truth,axis=(2, 3,4)):
    '''
    prediction is the probabilities from softmax, here it act as Dice Coef, as alpha and beta 
    '''    
    smooth=1
    P_foreground = prediction[...,:1]
    P_background = prediction[...,1:]
    g_foreground = ground_truth[...,:1]
    g_background = ground_truth[...,1:]

   
    true_positive= P_foreground * g_foreground
    true_positive = tf.reduce_sum(true_positive, axis=axis)
   
    false_pos = P_foreground * g_background
    false_pos = 0.5*tf.reduce_sum(false_pos, axis=axis)
    
    
    false_neg = P_background * g_foreground
    false_neg = 0.5*tf.reduce_sum(false_neg, axis=axis)

    Tversky = tf.divide(true_positive+smooth, 
                        (true_positive + false_pos + false_neg + smooth))
    
    return tf.reduce_mean(Tversky,axis=0)   


# In[ ]:


import numpy as np
from numpy.core.umath_tests import inner1d

# Hausdorff Distance
def HausdorffDist(A,B):

    D_mat = np.sqrt(inner1d(A,A)[np.newaxis].T + inner1d(B,B)-2*(np.dot(A,B.T)))
    # Find DH
    dH = np.max(np.array([np.max(np.min(D_mat,axis=0)),np.max(np.min(D_mat,axis=1))]))
    return(dH)


def ComputeHDD(A, B):
    ''' A and B has the following Shape
    tensor of 3d: batch, H, W
    '''
    return list(map(lambda a,b:HausdorffDist(a,b) , A,B))


# In[ ]:


def process_data_with_spacing(data_file_path):
    '''
    Return 3 masks in order SOL, GL and GM.
    Input: main path for the dataset: either Train/Val/or Testm i.e. '/tf/volumes/train/CAT_TH/masksX1.mha'
    '''
    
    #Read the data of formate (528, 640, 1574)
    image_data,HEADER = load(data_file_path)
    # Adjust the formate to (640, 528, 1574)
    image_data = image_data.transpose([1,0,2])

    image_data= list(map(lambda mask_time_step:  crop_pad(mask_time_step),
                   image_data.transpose(2,0,1))) #,(512,640),
    
    #image_data= list(map(lambda image_data_step: image_data_step[:640,:512],image_data.transpose(2,0,1)))    
    
    return np.array(image_data),HEADER.spacing


def Pull_data_from_path_with_spacing(path):
    data,SPACING = process_data_with_spacing(path)
    # return normalized data
    # values from whole data
    mean_val = 19.027262640214904
    std_val = 34.175155632916
    
    data = (data-mean_val) / std_val
    # reshape to t,h,w,1
    return  np.expand_dims(data,-1),SPACING

