#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[2]:


__idea__ = 'IFSS-Net'
__author__ = 'Dawood AL CHANTI'
__affiliation__ = 'LS2N-ECN'


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams['image.cmap'] = 'gist_earth'


# In[4]:


#Ignore warning in Jupyter
import warnings
warnings.filterwarnings('ignore')


# In[5]:


from __future__ import print_function, division, absolute_import, unicode_literals


# In[6]:


# define the location of .py helper module that we import
import sys
sys.path.insert(1, '/tf/JournalWork/')
sys.path.insert(1, '/tf/')


# In[7]:


import tensorflow as tf


# In[8]:


from data_utilities import *


# In[9]:


# Define the path 
x_y_path = '/tf/volumes/train/'
# define the full path for each patient
Patient_folder_path = sorted(os.listdir(x_y_path))
Patient_folder_full_path = list(map(lambda v : str(join(x_y_path,v)) + '/', Patient_folder_path))

# Get the full path for the volume and the mask
DataTrainPath = list(map(lambda s : s+'x1.mha' , Patient_folder_full_path))
MasksTrainPath = list(map(lambda s : s+'masksX1.mha' , Patient_folder_full_path))


# In[10]:


print(DataTrainPath[0])
print(MasksTrainPath[0])


# In[11]:


# Return the patient Name
Patient_name(DataTrainPath[0])


# In[12]:


locals()[Patient_name(DataTrainPath[0]) + '_data']=Pull_data_from_path(DataTrainPath[0])
print(locals()[Patient_name(DataTrainPath[0]) + '_data'].shape)
locals()[Patient_name(DataTrainPath[0]) + '_mask']= process_mask(MasksTrainPath[0])
print(locals()[Patient_name(DataTrainPath[0]) + '_mask'].shape)


# In[13]:


# Plot each muscle alone
fig, ax = plt.subplots(1,4, sharey=True, figsize=(12,4))

ax[0].imshow(locals()[Patient_name(DataTrainPath[0]) + '_data'][700][:,:,0] , aspect="auto",cmap='gray')
ax[1].imshow(locals()[Patient_name(DataTrainPath[0]) + '_mask'][700][:,:,0] , aspect="auto",cmap='gray')
ax[2].imshow(locals()[Patient_name(DataTrainPath[0]) + '_mask'][700][:,:,1], aspect="auto",cmap='gray') 
ax[3].imshow(locals()[Patient_name(DataTrainPath[0]) + '_mask'][700][:,:,2] , aspect="auto",cmap='gray') 

ax[0].set_title('Volume')
ax[1].set_title('SOL')
ax[2].set_title('GL')
ax[3].set_title('GM')


# In[14]:


# only for SOL


# In[15]:


locals()[Patient_name(DataTrainPath[0]) + '_mask_SOL']= process_mask_SOL(MasksTrainPath[0])
print(locals()[Patient_name(DataTrainPath[0]) + '_mask_SOL'].shape)


# In[16]:


# Plot each muscle alone
fig, ax = plt.subplots(1,2, sharey=True, figsize=(12,4))

ax[0].imshow(locals()[Patient_name(DataTrainPath[0]) + '_mask_SOL'][700][:,:,0] , aspect="auto",cmap='gray')
ax[1].imshow(locals()[Patient_name(DataTrainPath[0]) + '_mask_SOL'][700][:,:,1] , aspect="auto",cmap='gray')

ax[0].set_title('F')
ax[1].set_title('B')


# # Data flip from the back to front (as a way of Data Augmentation and mimicing Bi directional)

# In[17]:


locals()[Patient_name(DataTrainPath[0]) + '_data_flip']=np.flip(Pull_data_from_path(DataTrainPath[0]),0)

locals()[Patient_name(DataTrainPath[0]) + '_mask_flip']= np.flip(process_mask(MasksTrainPath[0]),0)


# In[18]:


# Plot each muscle alone
fig, ax = plt.subplots(1,4, sharey=True, figsize=(12,4))

ax[0].imshow(locals()[Patient_name(DataTrainPath[0]) + '_data_flip'][700][:,:,0] , aspect="auto",cmap='gray')
ax[1].imshow(locals()[Patient_name(DataTrainPath[0]) + '_mask_flip'][700][:,:,0] , aspect="auto",cmap='gray')
ax[2].imshow(locals()[Patient_name(DataTrainPath[0]) + '_mask_flip'][700][:,:,1], aspect="auto",cmap='gray') 
ax[3].imshow(locals()[Patient_name(DataTrainPath[0]) + '_mask_flip'][700][:,:,2] , aspect="auto",cmap='gray') 

ax[0].set_title('Volume')
ax[1].set_title('SOL')
ax[2].set_title('GL')
ax[3].set_title('GM')


# In[19]:


# Plot each muscle alone
fig, ax = plt.subplots(1,4, sharey=True, figsize=(12,4))

ax[0].imshow(locals()[Patient_name(DataTrainPath[0]) + '_data'][700][:,:,0] -locals()[Patient_name(DataTrainPath[0]) + '_data_flip'][700][:,:,0], aspect="auto",cmap='gray')
ax[1].imshow(locals()[Patient_name(DataTrainPath[0]) + '_mask'][700][:,:,0]-locals()[Patient_name(DataTrainPath[0]) + '_mask_flip'][700][:,:,0] , aspect="auto",cmap='gray')
ax[2].imshow(locals()[Patient_name(DataTrainPath[0]) + '_mask'][700][:,:,1]-locals()[Patient_name(DataTrainPath[0]) + '_mask_flip'][700][:,:,1], aspect="auto",cmap='gray') 
ax[3].imshow(locals()[Patient_name(DataTrainPath[0]) + '_mask'][700][:,:,2]-locals()[Patient_name(DataTrainPath[0]) + '_mask_flip'][700][:,:,2] , aspect="auto",cmap='gray') 

ax[0].set_title('Volume')
ax[1].set_title('SOL')
ax[2].set_title('GL')
ax[3].set_title('GM')


# In[20]:


# Free Memory
del(locals()[Patient_name(DataTrainPath[0]) + '_data_flip'])
del(locals()[Patient_name(DataTrainPath[0]) + '_data'])
del(locals()[Patient_name(DataTrainPath[0]) + '_mask_flip'])
del(locals()[Patient_name(DataTrainPath[0]) + '_mask'])


# # Graph Implementation of the Model

# ### Rest the Graph and build it

# In[21]:


from IFSSNet_utilities import *


# In[22]:


tf.reset_default_graph()


# ## Define the parameters

# In[23]:


time_step = None # whatever the depth of the volume, we use a sliding window of T
H=512
W=512
C = 1 # number of input channels
num_classes = 2 # related to SOl or GL or GM muscles
n_class = 2


# ## Define the place holder that takes in the data


# define the input stracture: 5D Tensor

x = tf.placeholder("float", shape=[None,time_step, H, W, C ], name="x")
y = tf.placeholder("float", shape=[None,time_step,H, W,num_classes], name="y")


x_target = tf.placeholder("float", shape=[None,time_step, H, W, C ], name="x_target")
x_ref = tf.placeholder("float", shape=[None,time_step, H, W, C ], name="x_ref")


y_reference = tf.placeholder("float", shape=[None,time_step,H, W,num_classes], name="y_reference")
y_estimated = tf.placeholder("float", shape=[None,time_step,H, W,num_classes], name="y_estimated")


# define the placeholder for dropout
keep_prob = tf.placeholder(tf.float32, name="dropout_probability")
keep_prob_input = tf.placeholder(tf.float32, name="dropout_probability_input")
keep_prob_skip = tf.placeholder(dtype=tf.float32,name='SkipDropout')

print('Input Structure: ',x)
print('Target Structure: ',y)


# In[25]:


keep_prob_alpha = tf.placeholder(tf.float32, name="drop_alpha")
keep_prob_beta = tf.placeholder(tf.float32, name="drop_beta")
keep_prob_tp = tf.placeholder(tf.float32, name="drop_tp")


# In[26]:


# defined as variables ?
Batch_size = tf.shape(x)[0]
time_size = tf.shape(x)[1]


# # Bi-directional spatio-temporal smoothness and Interactive

# In[28]:


reuseflagEncoder_target = False # to train flag must be set to zero

with slim.arg_scope([slim.conv3d],
                    padding='SAME', 
                    normalizer_fn=None, #slim.layer_norm
                    normalizer_params=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.float32), #tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_AVG'), 
                    weights_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                    biases_initializer=None,
                    activation_fn=None) as Encoder_scp:
    
    ################# Layer 1
    # conv3d
    
    with tf.variable_scope("layer_1",reuse=reuseflagEncoder_target) as scope:
        
        Layer_1 = Conv3DBlock(x_target, 30, kernel_size=[3,3,3], stride = [1,1,1],activation_fn=None)
        Layer_1 = tf.nn.max_pool3d(Layer_1, strides=[1, 1, 2, 2, 1], 
                                        ksize=[1, 3, 3, 3, 1], padding='SAME', name='Layer_1p')

        Layer_1 = AtrousSpatialPyramidPoolingModule_3D(Layer_1, depth=6)
        print(Layer_1)
        #Layer_1 = Conv3DBlock(Layer_1, 16, kernel_size=[3,3,3], stride = [1,1,1],activation_fn=None)
        #print(Layer_1)
        Layer_1 =  tf.nn.dropout(Layer_1,keep_prob_skip)
        print(Layer_1)


    # Conv3d
    with tf.variable_scope("layer_2",reuse=reuseflagEncoder_target) as scope:

        Layer_2 = Conv3DBlock(Layer_1, 30, kernel_size=[3,3,3], stride = [1,1,1],activation_fn=None)
        Layer_2 = tf.nn.max_pool3d(Layer_2, strides=[1, 1, 2, 2, 1], 
                                        ksize=[1, 3, 3, 3, 1], padding='SAME', name='Layer_2p')
            
        print(Layer_2)
        Layer_2 = AtrousSpatialPyramidPoolingModule_3D(Layer_2, depth=6)
        print(Layer_2)
        Layer_2 =  tf.nn.dropout(Layer_2,keep_prob_skip)
        print(Layer_2)
    

    with tf.variable_scope("layer_3",reuse=reuseflagEncoder_target) as scope:

        Layer_3 = Conv3DBlock(Layer_2, 60, kernel_size=[3,3,3], stride = [1,1,1],activation_fn=None)
        Layer_3 = tf.nn.max_pool3d(Layer_3, strides=[1, 1, 2, 2, 1], 
                                        ksize=[1, 3, 3, 3, 1], padding='SAME', name='Layer_3p')
        print(Layer_3)
        
        Layer_3 = AtrousSpatialPyramidPoolingModule_3D_rate_11(Layer_3, depth=12)
        print(Layer_3)
        Layer_3 =  tf.nn.dropout(Layer_3,keep_prob_skip)
        print(Layer_3)

    with tf.variable_scope("layer_4",reuse=reuseflagEncoder_target) as scope:
        Layer_4 = Conv3DBlock(Layer_3, 120, kernel_size=[3,3,3], stride = [1,1,1],activation_fn=None)
        Layer_4 = tf.nn.max_pool3d(Layer_4, strides=[1, 1, 2, 2, 1], 
                                        ksize=[1, 3, 3, 3, 1], padding='SAME', name='Layer_4p')
        print(Layer_4)
        Layer_4 = AtrousSpatialPyramidPoolingModule_3D_rate_9(Layer_4, depth=24)
        print(Layer_4)
        Layer_4 =  tf.nn.dropout(Layer_4,keep_prob_skip)
        print(Layer_4)
    
    with tf.variable_scope("layer_5",reuse=reuseflagEncoder_target) as scope:
        Layer_5 = Conv3DBlock(Layer_4, 120, kernel_size=[3,3,3], stride = [1,1,1],activation_fn=None)
        Layer_5 = tf.nn.max_pool3d(Layer_5, strides=[1, 1, 2, 2, 1], 
                                        ksize=[1, 3, 3, 3, 1], padding='SAME', name='Layer_5p')
        Layer_5 = AtrousSpatialPyramidPoolingModule_3D_rate_9(Layer_5, depth=24)
        print(Layer_5)


# # Encoder Stream for Estimated Predictions (reuse features from x)

# In[29]:


reuseflagEncoder_target = True # to train flag must be set to zero

with slim.arg_scope([slim.conv3d],
                    padding='SAME', 
                    normalizer_fn=None, #slim.layer_norm
                    normalizer_params=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.float32), #tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_AVG'), 
                    weights_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                    biases_initializer=None,
                    activation_fn=None) as Encoder_scp:
    
    ################# Layer 1
    # conv3d
    
    with tf.variable_scope("layer_1",reuse=reuseflagEncoder_target) as scope:
        
        Layer_1y = Conv3DBlock(y_estimated[:,:,:,:,:1], 30, kernel_size=[3,3,3], stride = [1,1,1],activation_fn=None)
        Layer_1y = tf.nn.max_pool3d(Layer_1y, strides=[1, 1, 2, 2, 1], 
                                        ksize=[1, 3, 3, 3, 1], padding='SAME', name='Layer_1p')

        Layer_1y = AtrousSpatialPyramidPoolingModule_3D(Layer_1y, depth=6)
        print(Layer_1y)
        #Layer_1 = Conv3DBlock(Layer_1, 16, kernel_size=[3,3,3], stride = [1,1,1],activation_fn=None)
        #print(Layer_1)
        Layer_1y =  tf.nn.dropout(Layer_1y,keep_prob_skip)
        print(Layer_1y)


    # Conv3d
    with tf.variable_scope("layer_2",reuse=reuseflagEncoder_target) as scope:

        Layer_2y = Conv3DBlock(Layer_1y, 30, kernel_size=[3,3,3], stride = [1,1,1],activation_fn=None)
        Layer_2y = tf.nn.max_pool3d(Layer_2y, strides=[1, 1, 2, 2, 1], 
                                        ksize=[1, 3, 3, 3, 1], padding='SAME', name='Layer_2p')
            
        print(Layer_2y)
        Layer_2y = AtrousSpatialPyramidPoolingModule_3D(Layer_2y, depth=6)
        print(Layer_2y)
        Layer_2y =  tf.nn.dropout(Layer_2y,keep_prob_skip)
        print(Layer_2y)
    

    with tf.variable_scope("layer_3",reuse=reuseflagEncoder_target) as scope:

        Layer_3y = Conv3DBlock(Layer_2y, 60, kernel_size=[3,3,3], stride = [1,1,1],activation_fn=None)
        Layer_3y = tf.nn.max_pool3d(Layer_3y, strides=[1, 1, 2, 2, 1], 
                                        ksize=[1, 3, 3, 3, 1], padding='SAME', name='Layer_3p')
        print(Layer_3y)
        
        Layer_3y = AtrousSpatialPyramidPoolingModule_3D_rate_11(Layer_3y, depth=12)
        print(Layer_3y)
        Layer_3y =  tf.nn.dropout(Layer_3y,keep_prob_skip)
        print(Layer_3y)

    with tf.variable_scope("layer_4",reuse=reuseflagEncoder_target) as scope:
        Layer_4y = Conv3DBlock(Layer_3y, 120, kernel_size=[3,3,3], stride = [1,1,1],activation_fn=None)
        Layer_4y = tf.nn.max_pool3d(Layer_4y, strides=[1, 1, 2, 2, 1], 
                                        ksize=[1, 3, 3, 3, 1], padding='SAME', name='Layer_4p')
        print(Layer_4y)
        Layer_4y = AtrousSpatialPyramidPoolingModule_3D_rate_9(Layer_4y, depth=24)
        print(Layer_4y)
        Layer_4y =  tf.nn.dropout(Layer_4y,keep_prob_skip)
        print(Layer_4y)
    
    with tf.variable_scope("layer_5",reuse=reuseflagEncoder_target) as scope:
        Layer_5y = Conv3DBlock(Layer_4y, 120, kernel_size=[3,3,3], stride = [1,1,1],activation_fn=None)
        Layer_5y = tf.nn.max_pool3d(Layer_5y, strides=[1, 1, 2, 2, 1], 
                                        ksize=[1, 3, 3, 3, 1], padding='SAME', name='Layer_5p')
        Layer_5y = AtrousSpatialPyramidPoolingModule_3D_rate_9(Layer_5y, depth=24)
        print(Layer_5y)


# # Fusion and SpatioTemporal Correlations

# In[27]:


ConvLSTM2D = tf.keras.layers.ConvLSTM2D 


# In[30]:


reuseflag_Temporal_target_y = False # to train flag must be set to zero
with tf.variable_scope("Temporal",reuse=reuseflag_Temporal_target_y) as scope:
    
    Fused_x_y = FeatureFusionModule(Layer_5, Layer_5y, 120)
        
    BiSpatioTemporal_x_y_F = ConvLSTM2D(filters = 120, kernel_size=(1, 1), padding='same', return_sequences = True, 
                        go_backwards = False,kernel_initializer = 'he_normal',
                          recurrent_dropout=0.3,activation=tf.nn.tanh)(Fused_x_y)
    print(BiSpatioTemporal_x_y_F)
    


# In[31]:


reuseflag_Temporal_target_y = True # to train flag must be set to zero
with tf.variable_scope("Temporal",reuse=reuseflag_Temporal_target_y) as scope:
    
    Fused_x_y_Reversed = tf.reverse(Fused_x_y,[1])
        
    BiSpatioTemporal_x_y_B = ConvLSTM2D(filters = 120, kernel_size=(1, 1), padding='same', return_sequences = True, 
                        go_backwards = False,kernel_initializer = 'he_normal',
                          recurrent_dropout=0.3,activation=tf.nn.tanh)(Fused_x_y_Reversed)
    print(BiSpatioTemporal_x_y_B)
    


# In[32]:


with tf.variable_scope("BiCLSTM",reuse=reuseflag_Temporal_target_y) as scope:
    BiSpatioTemporal_x_y = BiSpatioTemporal_x_y_F+tf.reverse(BiSpatioTemporal_x_y_B,[1])


# # Build the Decoder

# In[33]:


# Do not reuse the features, train them
reuseflagDecoder = False

# Building the Decoder Layers
with slim.arg_scope([slim.conv3d_transpose],
                    padding='SAME',
                    normalizer_fn=None,
                    normalizer_params={'scale': False},
                    weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True,
                                                                             seed=None,
                                                                             dtype=tf.float32),#tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_AVG'), 
                    weights_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                    biases_initializer=None,
                    activation_fn=None) as Decoder_scp:
    
    with tf.variable_scope("layer_6",reuse=reuseflagDecoder) as scope:
        Layer_6 = Conv3DBlockTranspose(BiSpatioTemporal_x_y,
                                       120, kernel_size=[3,3,3], stride = [1,1,1],activation_fn=None)
        print(Layer_6)
        Layer_6 = Conv3DBlock(Layer_6, 120, kernel_size=[3,3,3], stride = [1,1,1],activation_fn=None)
        print(Layer_6)
        Layer_6 =FeatureFusionModule(input_1=Layer_6,
                                input_2=BiSpatioTemporal_x_y, 
                                n_filters=120)
        print(Layer_6)
        Layer_6 = unpool_3D(Layer_6)
        Layer_6 =  tf.nn.dropout(Layer_6,keep_prob_skip)
        print(Layer_6)
        
    with tf.variable_scope("layer_7",reuse=reuseflagDecoder) as scope:
        Layer_7 = Conv3DBlockTranspose(Layer_6, 120, kernel_size=[3,3,3], stride = [1,1,1],activation_fn=None)
        print(Layer_7)
    
        Layer_7 = Conv3DBlock(Layer_7, 120, kernel_size=[3,3,3], stride = [1,1,1],activation_fn=None)
        print(Layer_7)
        Layer_7 = FeatureFusionModule(input_1=Layer_7,
                                input_2=Layer_4, 
                                n_filters=120)
        print(Layer_7)
        Layer_7 = unpool_3D(Layer_7)
        
        Layer_7 =  tf.nn.dropout(Layer_7,keep_prob_skip)
        print(Layer_7)
        
    with tf.variable_scope("layer_8",reuse=reuseflagDecoder) as scope:
        Layer_8 = Conv3DBlockTranspose(Layer_7, 60, kernel_size=[3,3,3], stride = [1,1,1],activation_fn=None)
        print(Layer_8)
        Layer_8 = Conv3DBlock(Layer_8, 60, kernel_size=[3,3,3], stride = [1,1,1],activation_fn=None)
        print(Layer_8)
        Layer_8 = FeatureFusionModule(input_1=Layer_8,
                                input_2=Layer_3, 
                                n_filters=60)
        print(Layer_8)
        Layer_8 = unpool_3D(Layer_8)
        Layer_8 =  tf.nn.dropout(Layer_8,keep_prob_skip)
        print(Layer_8)

    with tf.variable_scope("layer_9",reuse=reuseflagDecoder) as scope:
        Layer_9= Conv3DBlockTranspose(Layer_8, 60, kernel_size=[3,3,3], stride = [1,1,1],activation_fn=None)
        print(Layer_9)
        Layer_9 = Conv3DBlock(Layer_9, 60, kernel_size=[3,3,3], stride = [1,1,1],activation_fn=None)
        print(Layer_9)
        Layer_9 = FeatureFusionModule(input_1=Layer_9,
                                input_2=Layer_2, 
                                n_filters=60)
        print(Layer_9)
        Layer_9 = unpool_3D(Layer_9)
        Layer_9 =  tf.nn.dropout(Layer_9,keep_prob_skip)
        print(Layer_9)

    with tf.variable_scope("layer_10",reuse=reuseflagDecoder) as scope:
        Layer_10= Conv3DBlockTranspose(Layer_9, 16, kernel_size=[3,3,3], stride = [1,1,1],activation_fn=None)
        print(Layer_10)
        Layer_10 = Conv3DBlock(Layer_10, 16, kernel_size=[3,3,3], stride = [1,1,1],activation_fn=None)
        print(Layer_10)
        Layer_10 = FeatureFusionModule(input_1=Layer_10,
                                input_2=Layer_1, 
                                n_filters=16)
        Layer_10 = unpool_3D(Layer_10)
        Layer_10 =  tf.nn.dropout(Layer_10,keep_prob_skip)
        print(Layer_10) 
        

    with tf.variable_scope("layer_11",reuse=reuseflagDecoder) as scope:
        Layer_11= Conv3DBlockTranspose(Layer_10, 8, kernel_size=[3,3,3], stride = [1,1,1],activation_fn=None)
        print(Layer_11)
        Layer_11 = Conv3DBlock(Layer_11, 8, kernel_size=[1,1,1], stride = [1,1,1],activation_fn=None)
        print(Layer_11)
        #Layer_11 = unpool_3D(Layer_11)
        Layer_11 =  tf.nn.dropout(Layer_11,keep_prob_skip)
        print(Layer_11) 


# In[34]:


# Output Map for mask
reuseflagOutputMap = False

with slim.arg_scope([slim.conv3d_transpose],
                    padding='SAME',
                    normalizer_fn=None,
                    normalizer_params={'scale': False},
                    weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True,
                                                                             seed=None,
                                                                             dtype=tf.float32),#tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_AVG'), 
                    weights_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                    biases_initializer=None,
                    activation_fn=None)as output_scp:

    
    with tf.variable_scope("layer_output",reuse=reuseflagOutputMap) as scope:
        net_output_256_logits = slim.conv3d(Layer_11, num_classes, [1,1,1], 
                                                activation_fn=tf.nn.relu, scope='output_map')
        print(net_output_256_logits)


# In[35]:


reuseflagOutputHM= False
with tf.variable_scope("output",reuse=reuseflagOutputHM) as scope:
    net_output_256 = pixel_wise_softmax(net_output_256_logits)
    print(net_output_256)


# # Define the Loss based on Tversky Index and Learned Alpha and Beta

# # compare with pseduo Code which is our previous estimation

# In[36]:


smooth=1
# Current Prediction
P_foreground_SOL = net_output_256[...,:1]
P_background_SOL  = net_output_256[...,1:]
# Reference or previously estimated tensor
g_foreground_SOL  = y_estimated[...,:1]
g_background_SOL  = y_estimated[...,1:]
print(P_foreground_SOL)
print(P_background_SOL)


# # Compute the TP, FP, and FN for each Muscle

# In[37]:


#SOL
true_positive_SOL= P_foreground_SOL * g_foreground_SOL
P_foreground_SOL = tf.nn.dropout(P_foreground_SOL,keep_prob_tp,name="P_foreground_drop_out_SOL")

false_pos_SOL = P_foreground_SOL * g_background_SOL
false_neg_SOL = P_background_SOL * g_foreground_SOL

print(true_positive_SOL)
print(false_pos_SOL)
print(false_neg_SOL)


# In[38]:


# Penalties Weights


# In[39]:


W_tanh = weight_variable2([1, 2])
W_tanh = tf.nn.softmax(W_tanh)
print(W_tanh)


# # Penealise the FP and the FN by learned alpha and beta

# In[40]:


true_positive_SOL=tf.reduce_sum(tf.reduce_sum(true_positive_SOL, axis=(2,3,4)),axis=(1))
false_pos_SOL=tf.reduce_sum(W_tanh[...,0]*tf.reduce_sum(false_pos_SOL, axis=(2,3,4)),axis=(1))
false_neg_SOL=tf.reduce_sum(W_tanh[...,1]*tf.reduce_sum(false_neg_SOL, axis=(2,3,4)),axis=(1))
print(true_positive_SOL)
print(false_pos_SOL)
print(false_neg_SOL)


# # Compute Tversky Index for each of the muscle

# In[41]:


TverskyIndex_SOL = tf.divide(true_positive_SOL+smooth, (true_positive_SOL + false_pos_SOL  + false_neg_SOL  + smooth))
print(TverskyIndex_SOL)    


# In[42]:


with tf.name_scope("loss"):
    dice_loss_SOL = 1-tf.reduce_mean(TverskyIndex_SOL)
print(dice_loss_SOL)


# # Schaduale Learning rate

# In[43]:


# Schaduale Learning rate
global_step = tf.Variable(0, trainable=False)
boundaries = [8000]
values = [0.0001,0.00001]
learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)


# In[44]:


with tf.name_scope('Optimizer'):
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(dice_loss_SOL,global_step=global_step)


# # Computing Some Measures

# In[45]:


# Compute Dice Value
IoU_SOL = iou_coe(net_output_256,y)      
print(IoU_SOL)


# In[46]:


# Compute Dice Value
TverskyIndexValue_SOL = Tversky_Index(net_output_256,y)      
print(TverskyIndexValue_SOL)


# In[47]:


#MeanDice = tf.reduce_mean(TverskyIndexValue_SOL+TverskyIndexValue_GL+TverskyIndexValue_GM)
#print(MeanDice)


# In[48]:


with tf.name_scope("Precision_Recall"):
    output_thres_SOL = tf.cast(tf.squeeze(net_output_256[...,:1],0)> 0.5, dtype=tf.float32)
    
    target_thres_SOL = tf.cast(tf.squeeze(y[...,:1],0)> 0.5, dtype=tf.float32)
    
    precVSOL, recallVSOL = confusion(target_thres_SOL,output_thres_SOL)


# In[49]:


#MeanP = tf.reduce_mean(precVSOL+precVGL+precVGM)
#MeanR = tf.reduce_mean(recallVSOL+recallVGL+recallVGM)


# In[ ]:





# In[ ]:


saver = tf.train.Saver()
model_path = '/tf/JournalWork/Models//modelSOL/model/'


# In[ ]:


import time
from IPython import display


# In[ ]:


init_o = tf.global_variables_initializer()
Modelsummary={}
isRestor=False

import random # to generate a random number for dropout
# # Run the Session

# In[50]:


import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pickle
from random import randint


# # run Session

# In[ ]:


'''
incremental learning over the subjects with 50% 40 30 20 10 ...
'''


x_y_path = '/tf/volumes/test/'
Patient_folder_path = sorted(os.listdir(x_y_path))
Patient_folder_full_path = list(map(lambda v : str(join(x_y_path,v)) + '/', Patient_folder_path))

# Get the full path for the volume and the mask
DataTestPath = list(map(lambda s : s+'x1.mha' , Patient_folder_full_path))
MasksTestPath = list(map(lambda s : s+'masksX1.mha' , Patient_folder_full_path))



#ValidationSet


# Define the path 
x_y_path = '/tf/volumes/val/'
# define the full path for each patient
Patient_folder_path = sorted(os.listdir(x_y_path))
Patient_folder_full_path = list(map(lambda v : str(join(x_y_path,v)) + '/', Patient_folder_path))

# Get the full path for the volume and the mask
DataValPath = list(map(lambda s : s+'x1.mha' , Patient_folder_full_path))
MasksValPath = list(map(lambda s : s+'masksX1.mha' , Patient_folder_full_path))


# # Learn with overlap 0 1 2, 1 2 3, ...


fig, axs = plt.subplots(2,3,figsize=(12,6))
fig.tight_layout(pad=1.5)

isRestor=False
file_path = model_path + 'model'

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.999)

#with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
with tf.Session() as sess:
    if isRestor:
        sess.run(init_o) 
        saver.restore(sess,file_path)
        print("Model restored.")
    else:
        sess.run(init_o) 
        print('Model Training from Scratch')
   
    start_time = time.process_time()

    itCounter=0
    
    for iterations in range(10):
        
        annotationAmounts=30
        for subj in range(len(TrainSetDataPath)):
         
            #Data
            x_train=Pull_data_from_path(TrainSetDataPath[subj])
            #Mask 
            y_train= process_mask_SOL(TrainSetMaskPath[subj])
            
            StartIndex,EndIndex = ReturnIndicesOfFullMask(y_train[:,:,:,0])
            
            print('Subject # {1}, {0} is being processed of start {2} and end {3}.' 
                  .format(Patient_name(TrainSetDataPath[subj]),subj,StartIndex,EndIndex))

            x_train=x_train[StartIndex:EndIndex]
            y_train= y_train[StartIndex:EndIndex]
            
            # randome indicies that generates random image sequence
            maxlength=200
            minlength=int(maxlength/2)
            lengthSeq = random.sample(range(minlength,maxlength), 1)[0]
            list_indices = np.array([xx for xx in range(0, x_train.shape[0])])
            indexNumber = random.sample(range(0+lengthSeq, x_train.shape[0]-lengthSeq), 1)
            list_indices = list_indices[indexNumber[0]:indexNumber[0]+lengthSeq]
            
            #Data
            x_train=x_train[list_indices]
            #Mask 
            y_train= y_train[list_indices]
            
            # Sequence Length for the current patient
            SequenceLength= lengthSeq #randomly sampled over each iteration to make the network robust
        
            # Return the patient Name
            print('Subject # {1}, {0} is being processed of length {2}.' .format(Patient_name(TrainSetDataPath[subj]),
                                                                                 subj,
                                                                                 SequenceLength))
            # if 1 keep data same order, otherwise 2 or 3, flip it to go in the other direction
            num=randint(1, 2)
            if num==1:
                pass
            else:
                # Flip Data Based on a random variable Choice
                print('Data Flipped into the other direction')
                x_train = np.flip(x_train,0)
                y_train = np.flip(y_train,0)
                                
            #now for the target we will go through a sliding window of step size = 10
            #step=1
    
            indices_target = [xx for xx in range(0, SequenceLength)]
            j=0
            k=0
                        
            train_Dice_Coef_list=[] 
            train_loss_list=[]      
            train_mIoU_list=[] 
            train_P_list=[] 
            train_R_list=[] 
            train_alpha_list = []
            train_beta_list = []
            
            step=random.sample(range(3,7), 1)[0] ## random step size at each iterationm could be fixed to 3 also
            s=0
            e=step
            for i in range(0,int(len(indices_target))): 
                    j=step*i
                    if (s<SequenceLength-step-1):
                        if i%annotationAmounts==0:
                            next_indices_ref = indices_target[s:e]
                            print('*************Reference Update {0}**********' .format(next_indices_ref))
                            # Reference
                            x_train_ref = x_train[next_indices_ref] 
                            y_train_ref = y_train[next_indices_ref] 
                            # Reshape into 5D Tensor
                            x_train_ref = np.expand_dims(x_train_ref, axis=0)
                            y_train_ref = np.expand_dims(y_train_ref, axis=0)
                            
                            # Feed the network to train with the Reference and the Target
                            _, dice_losst, MeanDicet, MeanPt, MeanRt,                            alphat,betat,current_output=sess.run([train_step,dice_loss_SOL, 

                                                                      TverskyIndexValue_SOL, 
                                                                      precVSOL,
                                                                      recallVSOL,W_tanh[0,0],W_tanh[0,1], 
                                                                      net_output_256],
                                                                     feed_dict={
                                                                         x_target: x_train_ref,
                                                                         y: y_train_ref,
                                                                         y_estimated: y_train_ref,
                                                                         keep_prob:random.uniform(0.85, 1.),
                                                    keep_prob_input:random.uniform(0.9, 1.),
                                                    keep_prob_skip:random.uniform(0.75, 1.),
                                                    keep_prob_alpha :1.0,
                                                    keep_prob_beta :1.0,
                                                    keep_prob_tp:0.95
                                                   })

                            previous_estimated_y =current_output
                        else:
                            next_indices_target = indices_target[s:e]
                            print('Target Update {0}' .format(next_indices_target))

                            # Target
                            x_train_target_current = x_train[next_indices_target] 
                            y_train_target_current = y_train[next_indices_target] 
                            # Reshape into 5D Tensor
                            x_train_target_current = np.expand_dims(x_train_target_current, axis=0)
                            y_train_target_current = np.expand_dims(y_train_target_current, axis=0)

                            # Feed the network to train with the Reference and the Target
                            _, dice_losst, MeanDicet, MeanPt, MeanRt,                            alphat,betat,current_output=sess.run([train_step,dice_loss_SOL, 

                                                                      TverskyIndexValue_SOL, 
                                                                      precVSOL,
                                                                      recallVSOL,W_tanh[0,0],W_tanh[0,1], 
                                                                      net_output_256],
                                                                     feed_dict={
                                                                         x_target: x_train_target_current,
                                                                         y: y_train_target_current,
                                                                         y_estimated: previous_estimated_y,

                                                                                                                    keep_prob:random.uniform(0.85, 1.),
                                                    keep_prob_input:random.uniform(0.9, 1.),
                                                    keep_prob_skip:random.uniform(0.75, 1.),
                                                    keep_prob_alpha :1.0,
                                                    keep_prob_beta :1.0,
                                                    keep_prob_tp:0.95
                                                   })

                                    # Update the previous estimation with the current output
                            previous_estimated_y = current_output
                            im1 = previous_estimated_y[0][-1][...,0].reshape(512,512)

                            MeanIoUt = list(map(lambda a,b: compute_mean_iou(a>0.5,b), 
                                                            current_output[0][...,0],
                                                    y_train_target_current[0][...,0]))


                            MeanIoUt= (np.array(MeanIoUt)).mean()

                                # append the values over T segment and compute their mean for each patient   
                            train_loss_list.extend([dice_losst])
                            train_mIoU_list.extend([MeanIoUt])
                            train_Dice_Coef_list.extend([MeanDicet])
                            train_P_list.extend([MeanPt])
                            train_R_list.extend([MeanRt])
                            train_alpha_list.extend([alphat])
                            train_beta_list.extend([betat])

                            s=s+1
                            e=e+1
                            print(s,e)   
                    else:     
                        print('break')
                        break
            print('compute the mean of the results')

            # Compute the mean of the metrics                       
            lossnp = np.mean(train_loss_list)
            mIoUnp = np.mean(train_mIoU_list)
            DiceCoefnp = np.mean(train_Dice_Coef_list)
            Pnp = np.mean(train_P_list)
            Rnp = np.mean(train_R_list)
            alph = np.mean(train_alpha_list)
            betta = np.mean(train_beta_list)

            
            axs[0, 0].set_title('Tversky loss')
            axs[0, 1].set_title('mIoU')
            axs[0, 2].set_title('Dice Score')

            axs[1, 0].set_title('Alpha and Beta')
            
            axs[1, 1].set_title('P-R')
            axs[1, 2].set_title('Heat Map')
            
            axs[0, 0].plot([itCounter],lossnp,'b*')
            axs[0, 1].plot([itCounter],mIoUnp,'b*')
            axs[0, 2].plot([itCounter],DiceCoefnp,'k*')       # Dice Coef                            
                                                
    
            axs[1, 0].plot([itCounter],alph,'b*')
            axs[1, 0].plot([itCounter],betta,'g*')
            
            axs[1, 1].plot([itCounter],Pnp,'b*')
            axs[1, 1].plot([itCounter],Rnp,'k*')

            axs[1, 2].imshow(im1, aspect="auto",cmap='gist_earth')
            
            display.clear_output(wait=True)
            display.display(plt.gcf())   
            sys.stdout.flush()
            
            Modelsummary[itCounter]=[lossnp, mIoUnp,DiceCoefnp,Pnp,Rnp,alph,betta]  
            
            itCounter=itCounter+1
            
            print('next subject')
            print('>> Model Saved and can be restored during another session!')
            saver.save(sess, file_path, global_step=iterations) 

            with open(model_path+'learningCurve.pickle', 'wb') as handle:
                pickle.dump(Modelsummary, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save the model for the use later
    print('>> Model Saved and can be restored during another session!')
    saver.save(sess, file_path, global_step=iterations) 
            
    with open(model_path+'learningCurve.pickle', 'wb') as handle:
        pickle.dump(Modelsummary, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('Done')


# # Model Analysis after training

# In[ ]:
save_path = '/tf/JournalWork/Models/model2SOLSMOOTH14July/analysis/'
model_path = '/tf/JournalWork/Models//model2SOLSMOOTH14July/modelv66/'

file_path = model_path + 'model-4-4-4'


# In[ ]:


import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


# In[ ]:


with open(model_path+'learningCurve.pickle', 'rb') as handle:
    summary = pickle.load(handle)  


# In[ ]:


lists = sorted(summary.items()) # sorted by key, return a list of tuples
xx,yy =zip(*lists)
y1,y2,y3,y4,y5,y6,y7= zip(*yy)


# ### Compute based on slice based 

# In[ ]:


Tversky_Index_slice = Tversky_Index_Slice_by_Slice(net_output_256,y)
iou_coe_slice = iou_coe_Slice_by_Slice(net_output_256, y, threshold=0.5, smooth=1e-5)

print(Tversky_Index_slice)

print(iou_coe_slice)


# 
# 
# ### Compute the Precision and Recall Based on Slice Based

# In[ ]:


P_R_1,P_R_2,P_R_3 = list(map(lambda a,b:confusion(tf.cast( a> 0.5, dtype=tf.float32),tf.cast( b> 0.5, dtype=tf.float32))
                             ,tf.unstack(net_output_256,3,1),tf.unstack(y,3,1)))

s1 = tf.unstack(P_R_1,2,0)
s2 =tf.unstack(P_R_2,2,0)
s3 =tf.unstack(P_R_3,2,0)

Precsion_slice_by_slice = tf.stack([s1[0],s2[0],s3[0]])
Recal_slice_by_slice = tf.stack([s1[1],s2[1],s3[1]])


print(Precsion_slice_by_slice)
print(Recal_slice_by_slice)


# # Run the evluation over the Validation Set

# ### Prepare to read the Validation Data

# In[ ]:


# Define the path 
x_y_path = '/tf/volumes/val/'
# define the full path for each patient
Patient_folder_path = sorted(os.listdir(x_y_path))
Patient_folder_full_path = list(map(lambda v : str(join(x_y_path,v)) + '/', Patient_folder_path))

# Get the full path for the volume and the mask
DataValPath = list(map(lambda s : s+'x1.mha' , Patient_folder_full_path))
MasksValPath = list(map(lambda s : s+'masksX1.mha' , Patient_folder_full_path))


# In[ ]:


x_val=Pull_data_from_path_Complete(DataValPath[0])
y_val= process_mask_SOL_Complete(MasksValPath[0])       


# In[ ]:


# Plot each muscle alone
fig, ax = plt.subplots(1,3, sharey=True, figsize=(12,4))

ax[0].imshow(x_val[400][:,:,0] , aspect="auto",cmap='gray')
ax[1].imshow(y_val[400][:,:,0] , aspect="auto",cmap='gray')
ax[2].imshow(y_val[400][:,:,1] , aspect="auto",cmap='gray')

ax[0].set_title('Image')
ax[1].set_title('Mask F')
ax[2].set_title('Mask B')


# In[ ]:


#Data and mask
x_val=Pull_data_from_path_Complete(DataValPath[0])
y_val= process_mask_SOL_Complete(MasksValPath[0])


# In[ ]:


# Plot each muscle alone
fig, ax = plt.subplots(1,3, sharey=True, figsize=(12,4))

ax[0].imshow(x_val[700][:,:,0] , aspect="auto",cmap='gray')
ax[1].imshow(y_val[1000][:,:,0] , aspect="auto",cmap='gray') 

ax[2].imshow(x_val[700][:,:,0]*y_val[700][:,:,0] , aspect="auto",cmap='gray') 


ax[0].set_title('Volume')
ax[1].set_title('SOL')


# # Validation over the whole Validation Set

# In[ ]:


save_path = '/tf/JournalWork/Models/model2SOLSMOOTH14July/'
model_path = '/tf/JournalWork/Models//model2SOLSMOOTH14July/modelv66/'
file_path = model_path + 'model-4-4-4'
SavePathVal = save_path + '/patientAnalysisVal/'


# In[ ]:


from medpy import metric


# In[ ]:


isRestor=True
with tf.Session() as sess:
    if isRestor:
        sess.run(init_o) 
        saver.restore(sess,file_path)
        print("Model restored.")
    else:
        sess.run(init_o) 
        print('Model Training from Scratch')
   
    start_time = time.process_time()
    itCounter=0
   
    generatedSeqLength = 3
    for subj in range(len(DataValPath)):
            proposedPath = SavePathVal + str(Patient_name(DataValPath[subj]))
        
            # Check if this directory exist and pass, otherwise create it
            if os.path.exists(proposedPath) is True:
                pass
            else:
                os.mkdir(proposedPath)

            # create new path for folder and subfolder   
            newpathGLF,newpathGLB,newpathGMF,newpathGMB,            newpathSOLF,newpathSOLB=create_folder_subfolders(proposedPath)

            
            #Data, mask and spacing value
            x_val_o,SpacingValue=Pull_data_from_path_with_spacing(DataValPath[subj])
            y_val_o= process_mask_SOL_Complete(MasksValPath[subj])
        
            SeqLengthOriginal = y_val_o.shape[0]
            StartIndex,EndIndex = ReturnIndicesOfFullMask(y_val_o[:,:,:,0])
            
            
            
            # save the black images just for order of the volume in image visualization
            StartToPrint = [v for v in range(StartIndex)]
            EndToPrint = [v for v in range(EndIndex,SeqLengthOriginal)]

            if not StartToPrint:
                pass
            else:
                for item in StartToPrint:
                    matplotlib.image.imsave(newpathSOLF + '/mask_'+str(item)+'.png',np.zeros_like(y_val_o[0,...,0]))
                    matplotlib.image.imsave(newpathSOLF + '/annotation_'+str(item)+'.png',y_val_o[item,...,0])

            if not EndToPrint:
                pass
            else:     
                for item in EndToPrint:
                    matplotlib.image.imsave(newpathSOLF + '/mask_'+str(item)+'.png',np.zeros_like(y_val_o[0,...,0]))
                    matplotlib.image.imsave(newpathSOLF + '/annotation_'+str(item)+'.png',y_val_o[item,...,0])

            
            # Select without black area
            x_val=x_val_o[StartIndex:EndIndex]
            y_val=y_val_o[StartIndex:EndIndex]
            
            # Sequence Length for the current patient
            SequenceLength=x_val.shape[0]

            print('Subject # {1}, {0} is being processed of length {2}.' .format(Patient_name(DataValPath[subj]),
                                                                                 subj,SequenceLength))

            #now for the target we will go through a sliding window of step size = 10
            indices_target = [xx for xx in range(0, SequenceLength)]
            j=0
            k=0
            
            step=3
        
            s=0
            e=step
            counter = StartIndex  # start the with the starting index to know the order
            annotationAmounts=100
            
            locals()['VOL_Pred'+str(subj)]=[]
            locals()['VOL_GT'+str(subj)]=[]
            
            for i in range(0,int(len(indices_target))): 
                    j=step*i
                    if (s<SequenceLength-step-1):
                        if i%annotationAmounts==0:
                            next_indices_ref = indices_target[s:e]

                            x_val_ref = x_val[next_indices_ref] 
                            y_val_ref = y_val[next_indices_ref] 
                            # Reshape into 5D Tensor
                            x_val_ref = np.expand_dims(x_val_ref, axis=0)
                            y_val_ref = np.expand_dims(y_val_ref, axis=0)
                            
                            current_output_val  =sess.run(net_output_256,
                                                      feed_dict={
                                                          x_target: x_val_ref,
                                                          y: y_val_ref,
                                                          y_estimated:y_val_ref,
                                                          keep_prob:1.,
                                                keep_prob_input:1.,
                                                keep_prob_skip:1.,
                                                keep_prob_alpha :1.0,
                                                keep_prob_beta :1.0,
                                                keep_prob_tp:1.
                                               })
                            
                        
                            previous_estimated_y =y_val_ref
                            val_predicted_label = np.squeeze(current_output_val,0)
                            
                            _,im1 = cv2.threshold(val_predicted_label[0][...,0],0.5,1,cv2.THRESH_BINARY)
                            im1=im1.astype('uint8')
                                
                            matplotlib.image.imsave(newpathSOLF + '/mask_'+str(counter)+'.png',im1)
                            matplotlib.image.imsave(newpathSOLF + '/annotation_'+str(counter)+'.png',y_val_ref[0][0][...,0])
                            counter +=1
                            
                            locals()['VOL_Pred'+str(subj)].append(np.sum(im1))
                            locals()['VOL_GT'+str(subj)].append(np.sum(y_val_ref[0][0][...,0]))

                        else:
                            
                            #change flag of update
                            #flag=0
                            next_indices_target = indices_target[s:e]

                            # Target
                            x_val_target_current = x_val[next_indices_target] 
                            y_val_target_current = y_val[next_indices_target] 
                            # Reshape into 5D Tensor
                            x_val_target_current = np.expand_dims(x_val_target_current, axis=0)
                            y_val_target_current = np.expand_dims(y_val_target_current, axis=0)
                        
                             # Feed the network to train with the Reference and the Target
                            current_output_val  =sess.run(net_output_256,
                                                      feed_dict={
                                                          x_target: x_val_target_current,
                                                          y: y_val_target_current,
                                                          y_estimated:previous_estimated_y,
                                                          keep_prob:1.,
                                                keep_prob_input:1.,
                                                keep_prob_skip:1.,
                                                keep_prob_alpha :1.0,
                                                keep_prob_beta :1.0,
                                                keep_prob_tp:1.
                                               })

                            # Update the previous estimation with the current output
                            previous_estimated_y =current_output_val
                            val_predicted_label = np.squeeze(current_output_val,0)
                            _,im1 = cv2.threshold(val_predicted_label[0][...,0],0.5,1,cv2.THRESH_BINARY)
                            im1=im1.astype('uint8')
                            
                            # SOL
                            matplotlib.image.imsave(newpathSOLF + '/mask_'+str(counter)+'.png',im1)
                            matplotlib.image.imsave(newpathSOLF + '/annotation_'+str(counter)+'.png',y_val_target_current[0][0][...,0])
                            counter +=1
                            
                            locals()['VOL_Pred'+str(subj)].append(np.sum(im1))
                            locals()['VOL_GT'+str(subj)].append(np.sum(y_val_target_current[0][0][...,0]))

                        s=s+1
                        e=e+1
                       
                            
                    else:     
                        print('break')
                        break
                           
            Predictionvolume = SpacingValue[0]*SpacingValue[1]*SpacingValue[2]*np.sum(locals()['VOL_Pred'+str(subj)])
            Predictionvolume = Predictionvolume/1000.

            GTVolume = SpacingValue[0]*SpacingValue[1]*SpacingValue[2]*np.sum(locals()['VOL_GT'+str(subj)])
            GTVolume = GTVolume/1000.
            

            print('Subject # {1}, {0} is being processed of Predicted Volume {2} and GT Volume {3} in cm cube' .format(Patient_name(DataValPath[subj]),
                                                                                 subj,Predictionvolume,GTVolume))
            np.save(proposedPath+'/Volume_Prediction_SOL'+Patient_name(DataValPath[subj]), Predictionvolume)
            np.save(proposedPath+'/Volume_GT_SOL'+Patient_name(DataValPath[subj]), GTVolume)
            
            print('next subject')
    print('Done')


# # Full Supervison VOL

# In[ ]:


save_path = '/tf/JournalWork/Models/model2SOLSMOOTH14July/'
model_path = '/tf/JournalWork/Models//model2SOLSMOOTH14July/modelv66/'
file_path = model_path + 'model-4-4-4'
SavePathVal = save_path + '/patientAnalysisValFS/'


# In[ ]:


isRestor=True
with tf.Session() as sess:
    if isRestor:
        sess.run(init_o) 
        saver.restore(sess,file_path)
        print("Model restored.")
    else:
        sess.run(init_o) 
        print('Model Training from Scratch')
   
    start_time = time.process_time()
    itCounter=0
   
    generatedSeqLength = 3
    for subj in range(len(DataValPath)):
            proposedPath = SavePathVal + str(Patient_name(DataValPath[subj]))
        
            # Check if this directory exist and pass, otherwise create it
            if os.path.exists(proposedPath) is True:
                pass
            else:
                os.mkdir(proposedPath)

            # create new path for folder and subfolder   
            newpathGLF,newpathGLB,newpathGMF,newpathGMB,            newpathSOLF,newpathSOLB=create_folder_subfolders(proposedPath)

            
            #Data, mask and spacing value
            x_val_o,SpacingValue=Pull_data_from_path_with_spacing(DataValPath[subj])
            y_val_o= process_mask_SOL_Complete(MasksValPath[subj])
        
            SeqLengthOriginal = y_val_o.shape[0]
            StartIndex,EndIndex = ReturnIndicesOfFullMask(y_val_o[:,:,:,0])
            
            
            
            # save the black images just for order of the volume in image visualization
            StartToPrint = [v for v in range(StartIndex)]
            EndToPrint = [v for v in range(EndIndex,SeqLengthOriginal)]

            if not StartToPrint:
                pass
            else:
                for item in StartToPrint:
                    matplotlib.image.imsave(newpathSOLF + '/mask_'+str(item)+'.png',np.zeros_like(y_val_o[0,...,0]))
                    matplotlib.image.imsave(newpathSOLF + '/annotation_'+str(item)+'.png',y_val_o[item,...,0])

            if not EndToPrint:
                pass
            else:     
                for item in EndToPrint:
                    matplotlib.image.imsave(newpathSOLF + '/mask_'+str(item)+'.png',np.zeros_like(y_val_o[0,...,0]))
                    matplotlib.image.imsave(newpathSOLF + '/annotation_'+str(item)+'.png',y_val_o[item,...,0])

            
            # Select without black area
            x_val=x_val_o[StartIndex:EndIndex]
            y_val=y_val_o[StartIndex:EndIndex]
            
            # Sequence Length for the current patient
            SequenceLength=x_val.shape[0]

            print('Subject # {1}, {0} is being processed of length {2}.' .format(Patient_name(DataValPath[subj]),
                                                                                 subj,SequenceLength))

            #now for the target we will go through a sliding window of step size = 10
            indices_target = [xx for xx in range(0, SequenceLength)]
            j=0
            k=0
            
            step=3
        
            s=0
            e=step
            counter = StartIndex  # start the with the starting index to know the order
            annotationAmounts=1
            
            locals()['VOL_Pred'+str(subj)]=[]
            locals()['VOL_GT'+str(subj)]=[]
            
            for i in range(0,int(len(indices_target))): 
                    j=step*i
                    if (s<SequenceLength-step-1):
                        if i%annotationAmounts==0:
                            next_indices_ref = indices_target[s:e]

                            x_val_ref = x_val[next_indices_ref] 
                            y_val_ref = y_val[next_indices_ref] 
                            # Reshape into 5D Tensor
                            x_val_ref = np.expand_dims(x_val_ref, axis=0)
                            y_val_ref = np.expand_dims(y_val_ref, axis=0)
                            
                            current_output_val  =sess.run(net_output_256,
                                                      feed_dict={
                                                          x_target: x_val_ref,
                                                          y: y_val_ref,
                                                          y_estimated:y_val_ref,
                                                          keep_prob:1.,
                                                keep_prob_input:1.,
                                                keep_prob_skip:1.,
                                                keep_prob_alpha :1.0,
                                                keep_prob_beta :1.0,
                                                keep_prob_tp:1.
                                               })
                            
                        
                            previous_estimated_y =y_val_ref
                            val_predicted_label = np.squeeze(current_output_val,0)
                            
                            _,im1 = cv2.threshold(val_predicted_label[0][...,0],0.5,1,cv2.THRESH_BINARY)
                            im1=im1.astype('uint8')
                                
                            matplotlib.image.imsave(newpathSOLF + '/mask_'+str(counter)+'.png',im1)
                            matplotlib.image.imsave(newpathSOLF + '/annotation_'+str(counter)+'.png',y_val_ref[0][0][...,0])
                            counter +=1
                            
                            locals()['VOL_Pred'+str(subj)].append(np.sum(im1))
                            locals()['VOL_GT'+str(subj)].append(np.sum(y_val_ref[0][0][...,0]))

                        else:
                            
                            #change flag of update
                            #flag=0
                            next_indices_target = indices_target[s:e]

                            # Target
                            x_val_target_current = x_val[next_indices_target] 
                            y_val_target_current = y_val[next_indices_target] 
                            # Reshape into 5D Tensor
                            x_val_target_current = np.expand_dims(x_val_target_current, axis=0)
                            y_val_target_current = np.expand_dims(y_val_target_current, axis=0)
                        
                             # Feed the network to train with the Reference and the Target
                            current_output_val  =sess.run(net_output_256,
                                                      feed_dict={
                                                          x_target: x_val_target_current,
                                                          y: y_val_target_current,
                                                          y_estimated:previous_estimated_y,
                                                          keep_prob:1.,
                                                keep_prob_input:1.,
                                                keep_prob_skip:1.,
                                                keep_prob_alpha :1.0,
                                                keep_prob_beta :1.0,
                                                keep_prob_tp:1.
                                               })

                            # Update the previous estimation with the current output
                            previous_estimated_y =current_output_val
                            val_predicted_label = np.squeeze(current_output_val,0)
                            _,im1 = cv2.threshold(val_predicted_label[0][...,0],0.5,1,cv2.THRESH_BINARY)
                            im1=im1.astype('uint8')
                            
                            # SOL
                            matplotlib.image.imsave(newpathSOLF + '/mask_'+str(counter)+'.png',im1)
                            matplotlib.image.imsave(newpathSOLF + '/annotation_'+str(counter)+'.png',y_val_target_current[0][0][...,0])
                            counter +=1
                            
                            locals()['VOL_Pred'+str(subj)].append(np.sum(im1))
                            locals()['VOL_GT'+str(subj)].append(np.sum(y_val_target_current[0][0][...,0]))

                        s=s+1
                        e=e+1
                       
                            
                    else:     
                        print('break')
                        break
                           
            Predictionvolume = SpacingValue[0]*SpacingValue[1]*SpacingValue[2]*np.sum(locals()['VOL_Pred'+str(subj)])
            Predictionvolume = Predictionvolume/1000.

            GTVolume = SpacingValue[0]*SpacingValue[1]*SpacingValue[2]*np.sum(locals()['VOL_GT'+str(subj)])
            GTVolume = GTVolume/1000.
            

            print('Subject # {1}, {0} is being processed of Predicted Volume {2} and GT Volume {3} in cm cube' .format(Patient_name(DataValPath[subj]),
                                                                                 subj,Predictionvolume,GTVolume))
            np.save(proposedPath+'/Volume_Prediction_SOL'+Patient_name(DataValPath[subj]), Predictionvolume)
            np.save(proposedPath+'/Volume_GT_SOL'+Patient_name(DataValPath[subj]), GTVolume)
            
            print('next subject')
    print('Done')


# # Validation: Quantitaive and Qualitative Analysis

# # Tversky Loss

# In[ ]:


save_path = '/tf/JournalWork/Models/model2SOLSMOOTH14July/'
model_path = '/tf/JournalWork/Models//model2SOLSMOOTH14July/modelv66/'
file_path = model_path + 'model-4-4-4'

SavePathVal = save_path + '/patientAnalysisVal/'
SavePathValAnalysisQuantitative = save_path + 'QAnalysisVal/'


# # dice coefficient

# In[ ]:


# Violin Plot trial

SavePathVal = save_path + '/patientAnalysisVal/'
SavePathValAnalysisQuantitative = save_path + 'QAnalysisVal/'

listdata=[]
meanavg=[]
for subj in range(len(DataValPath)):
    pathRead = SavePathVal + str(Patient_name(DataValPath[subj]))+'/DICE_SOL_'+Patient_name(DataValPath[subj])+'.npy'
    data  = np.load(pathRead)
    
    listdata.append(data)
    meanavg.append(data.mean())

fig = plt.figure()
ax = plt.subplot(111)

my_xticks = list(map(lambda a: str(Patient_name(DataValPath[a])), range(len(DataValPath))))
           
ax.violinplot(listdata,
                   showmeans=True,
                   showmedians=False)
    
    
# add x-tick labels
plt.setp(ax, xticks=[y+1 for y in range(len(listdata))],
         xticklabels=list(map(lambda a: str(Patient_name(DataValPath[a])), range(len(DataValPath)))))


plt.xlabel('Patients')
plt.ylabel('Dice Coefficient Scores Distribution')
plt.title('Dice Coefficient scores over SOL muscle for all validation set patients')

chartBox = ax.get_position()

ax.set_position([2*chartBox.x0, 0.5*chartBox.y0, 1.5*chartBox.width, 1.5*chartBox.height])
ax.legend(loc='upper center', bbox_to_anchor=(1.10, 0.90), shadow=False, ncol=1)

fig1 = plt.gcf()
plt.show()

fig1.savefig(SavePathValAnalysisQuantitative +'/dice_coef_plot'+'.png', dpi=200,bbox_inches ='tight')            
        


# In[ ]:


print('Average Dice over Validation Set {0}'.format(np.array(meanavg).mean()))


# # Precision

# In[ ]:


# Violin Plot trial

SavePathVal = save_path + '/patientAnalysisVal/'
SavePathValAnalysisQuantitative = save_path + 'QAnalysisVal/'

listdata=[]
meanavg=[]
for subj in range(len(DataValPath)):
    pathRead = SavePathVal + str(Patient_name(DataValPath[subj]))+'/P_SOL_'+Patient_name(DataValPath[subj])+'.npy'
    data  = np.load(pathRead)
    
    listdata.append(data)
    meanavg.append(data.mean())

        

fig = plt.figure()
ax = plt.subplot(111)

my_xticks = list(map(lambda a: str(Patient_name(DataValPath[a])), range(len(DataValPath))))
           
ax.violinplot(listdata,
                   showmeans=True,
                   showmedians=False)
    
    
# add x-tick labels
plt.setp(ax, xticks=[y+1 for y in range(len(listdata))],
         xticklabels=list(map(lambda a: str(Patient_name(DataValPath[a])), range(len(DataValPath)))))


plt.xlabel('Patients')
plt.ylabel('Precision Distribution')
plt.title('Precision scores over SOL muscle for all validation set patients')

chartBox = ax.get_position()

ax.set_position([2*chartBox.x0, 0.5*chartBox.y0, 1.5*chartBox.width, 1.5*chartBox.height])
ax.legend(loc='upper center', bbox_to_anchor=(1.10, 0.90), shadow=False, ncol=1)

fig1 = plt.gcf()
plt.show()

fig1.savefig(SavePathValAnalysisQuantitative +'/Precision_plot'+'.png', dpi=200,bbox_inches ='tight')            
        


# In[ ]:


print('Average Precision over Validation Set {0}'.format(np.array(meanavg).mean()))


# In[ ]:





# In[ ]:


# Violin Plot trial

SavePathVal = save_path + '/patientAnalysisVal/'
SavePathValAnalysisQuantitative = save_path + 'QAnalysisVal/'

listdata=[]
meanavg=[]
for subj in range(len(DataValPath)):
    pathRead = SavePathVal + str(Patient_name(DataValPath[subj]))+'/R_SOL_'+Patient_name(DataValPath[subj])+'.npy'
    data  = np.load(pathRead)
    
    listdata.append(data)
    meanavg.append(data.mean())

        
        

fig = plt.figure()
ax = plt.subplot(111)

my_xticks = list(map(lambda a: str(Patient_name(DataValPath[a])), range(len(DataValPath))))
           
ax.violinplot(listdata,
                   showmeans=True,
                   showmedians=False)
    
    
# add x-tick labels
plt.setp(ax, xticks=[y+1 for y in range(len(listdata))],
         xticklabels=list(map(lambda a: str(Patient_name(DataValPath[a])), range(len(DataValPath)))))


plt.xlabel('Patients')
plt.ylabel('Recall Distribution')
plt.title('Recall scores over SOL muscle for all validation set patients')

chartBox = ax.get_position()

ax.set_position([2*chartBox.x0, 0.5*chartBox.y0, 1.5*chartBox.width, 1.5*chartBox.height])
ax.legend(loc='upper center', bbox_to_anchor=(1.10, 0.90), shadow=False, ncol=1)

fig1 = plt.gcf()
plt.show()

fig1.savefig(SavePathValAnalysisQuantitative +'/Recall_plot'+'.png', dpi=200,bbox_inches ='tight')            
        


# In[ ]:


print('Average Recall over Validation Set {0}'.format(np.array(meanavg).mean()))


# # mIoU

# In[ ]:


# Violin Plot trial

SavePathVal = save_path + '/patientAnalysisVal/'
SavePathValAnalysisQuantitative = save_path + 'QAnalysisVal/'

listdata=[]
meanavg=[]
for subj in range(len(DataValPath)):
    pathRead = SavePathVal + str(Patient_name(DataValPath[subj]))+'/IOU_SOL_'+Patient_name(DataValPath[subj])+'.npy'
    data  = np.load(pathRead)
    
    listdata.append(data)
    meanavg.append(data.mean())

        

fig = plt.figure()
ax = plt.subplot(111)

my_xticks = list(map(lambda a: str(Patient_name(DataValPath[a])), range(len(DataValPath))))
           
ax.violinplot(listdata,
                   showmeans=True,
                   showmedians=False)
    
    
# add x-tick labels
plt.setp(ax, xticks=[y+1 for y in range(len(listdata))],
         xticklabels=list(map(lambda a: str(Patient_name(DataValPath[a])), range(len(DataValPath)))))


plt.xlabel('Patients')
plt.ylabel('mIoU Distribution')
plt.title('mIoU scores over SOL muscle for all validation set patients')

chartBox = ax.get_position()

ax.set_position([2*chartBox.x0, 0.5*chartBox.y0, 1.5*chartBox.width, 1.5*chartBox.height])
ax.legend(loc='upper center', bbox_to_anchor=(1.10, 0.90), shadow=False, ncol=1)

fig1 = plt.gcf()
plt.show()

fig1.savefig(SavePathValAnalysisQuantitative +'/mIoU_plot'+'.png', dpi=200,bbox_inches ='tight')            
        


# In[ ]:


print('Average IOU over Validation Set {0}'.format(np.array(meanavg).mean()))


# # HDD

# In[ ]:


# Violin Plot trial

SavePathVal = save_path + '/patientAnalysisVal/'
SavePathValAnalysisQuantitative = save_path + 'QAnalysisVal/'

listdata=[]
meanavg=[]
for subj in range(len(DataValPath)):
    pathRead = SavePathVal + str(Patient_name(DataValPath[subj]))+'/HDD_SOL_'+Patient_name(DataValPath[subj])+'.npy'
    data  = np.load(pathRead)
    
    listdata.append(data)
    meanavg.append(data.mean())


fig = plt.figure()
ax = plt.subplot(111)

my_xticks = list(map(lambda a: str(Patient_name(DataValPath[a])), range(len(DataValPath))))
           
ax.violinplot(listdata,
                   showmeans=True,
                   showmedians=False)
    
    
# add x-tick labels
plt.setp(ax, xticks=[y+1 for y in range(len(listdata))],
         xticklabels=list(map(lambda a: str(Patient_name(DataValPath[a])), range(len(DataValPath)))))


plt.xlabel('Patients')
plt.ylabel('HDD Distribution')
plt.title('HDD scores over SOL muscle for all validation set patients')

chartBox = ax.get_position()

ax.set_position([2*chartBox.x0, 0.5*chartBox.y0, 1.5*chartBox.width, 1.5*chartBox.height])
ax.legend(loc='upper center', bbox_to_anchor=(1.10, 0.90), shadow=False, ncol=1)

fig1 = plt.gcf()
plt.show()

fig1.savefig(SavePathValAnalysisQuantitative +'/HDD_plot'+'.png', dpi=200,bbox_inches ='tight')            
        


# In[ ]:


print('Average HDD over Validation Set {0} mm cube'.format(np.array(meanavg).mean()))


# # Average symmetric surface distance.

# In[ ]:


# Violin Plot trial

SavePathVal = save_path + '/patientAnalysisVal/'
SavePathValAnalysisQuantitative = save_path + 'QAnalysisVal/'

listdata=[]
meanavg=[]
for subj in range(len(DataValPath)):
    pathRead = SavePathVal + str(Patient_name(DataValPath[subj]))+'/Assd_SOL_'+Patient_name(DataValPath[subj])+'.npy'
    data  = np.load(pathRead)
    
    listdata.append(data)
    meanavg.append(data.mean())


fig = plt.figure()
ax = plt.subplot(111)

my_xticks = list(map(lambda a: str(Patient_name(DataValPath[a])), range(len(DataValPath))))
           
ax.violinplot(listdata,
                   showmeans=True,
                   showmedians=False)
    
    
# add x-tick labels
plt.setp(ax, xticks=[y+1 for y in range(len(listdata))],
         xticklabels=list(map(lambda a: str(Patient_name(DataValPath[a])), range(len(DataValPath)))))


plt.xlabel('Patients')
plt.ylabel('Average symmetric surface distance distribution')
plt.title('Assd scores over SOL muscle for all validation set patients')

chartBox = ax.get_position()

ax.set_position([2*chartBox.x0, 0.5*chartBox.y0, 1.5*chartBox.width, 1.5*chartBox.height])
ax.legend(loc='upper center', bbox_to_anchor=(1.10, 0.90), shadow=False, ncol=1)

fig1 = plt.gcf()
plt.show()

fig1.savefig(SavePathValAnalysisQuantitative +'/assd_plot'+'.png', dpi=200,bbox_inches ='tight')            
        


# In[ ]:


print('Average ASSD over Validation Set {0} mm cube'.format(np.array(meanavg).mean()))


#  # Average surface distance metric.

# In[ ]:


# Violin Plot trial

SavePathVal = save_path + '/patientAnalysisVal/'
SavePathValAnalysisQuantitative = save_path + 'QAnalysisVal/'

listdata=[]
meanavg=[]
for subj in range(len(DataValPath)):
    pathRead = SavePathVal + str(Patient_name(DataValPath[subj]))+'/Asd_SOL_'+Patient_name(DataValPath[subj])+'.npy'
    data  = np.load(pathRead)
    
    listdata.append(data)
    meanavg.append(data.mean())


fig = plt.figure()
ax = plt.subplot(111)

my_xticks = list(map(lambda a: str(Patient_name(DataValPath[a])), range(len(DataValPath))))
           
ax.violinplot(listdata,
                   showmeans=True,
                   showmedians=False)
    
    
# add x-tick labels
plt.setp(ax, xticks=[y+1 for y in range(len(listdata))],
         xticklabels=list(map(lambda a: str(Patient_name(DataValPath[a])), range(len(DataValPath)))))


plt.xlabel('Patients')
plt.ylabel('Average surface distance distribution')
plt.title('Assd scores over SOL muscle for all validation set patients')

chartBox = ax.get_position()

ax.set_position([2*chartBox.x0, 0.5*chartBox.y0, 1.5*chartBox.width, 1.5*chartBox.height])
ax.legend(loc='upper center', bbox_to_anchor=(1.10, 0.90), shadow=False, ncol=1)

fig1 = plt.gcf()
plt.show()

fig1.savefig(SavePathValAnalysisQuantitative +'/asd_plot'+'.png', dpi=200,bbox_inches ='tight')            
        


# In[ ]:


print('Average ASD over Validation Set {0} mm cube'.format(np.array(meanavg).mean()))


# In[ ]:





# In[ ]:


# Violin Plot trial
SavePathVal = save_path + '/patientAnalysisVal/'
SavePathValAnalysisQuantitative = save_path + 'QAnalysisVal/'

for subj in range(len(DataValPath)):
    pathRead = SavePathVal + str(Patient_name(DataValPath[subj]))+'/HDD_SOL_'+Patient_name(DataValPath[subj])+'.npy'
    y1  = np.load(pathRead)
 
    fig = plt.figure()
    ax = plt.subplot(111)
    
    ax.plot(y1 ,'r-*')

    plt.xlabel('depth')
    plt.ylabel('HDD')
    plt.title('HDD over SOL muscle for patient ' + str(Patient_name(DataValPath[subj])))

    chartBox = ax.get_position()

#     ax.set_position([2*chartBox.x0, 0.5*chartBox.y0, 1.5*chartBox.width, 1.5*chartBox.height])
#     ax.legend(loc='upper center', bbox_to_anchor=(1.10, 0.90), shadow=False, ncol=1)

    fig1 = plt.gcf()
    plt.show()
    
    proposedPath = SavePathVal + str(Patient_name(DataValPath[subj]))
        
    fig1.savefig(proposedPath +'/HDD_'+str(Patient_name(DataValPath[subj]))+'.png', dpi=200,bbox_inches ='tight')    
    
    
#'/mIoU'
#'/DiceCoef'


# In[ ]:





# In[ ]:


# Violin Plot trial
SavePathVal = save_path + '/patientAnalysisVal/'
SavePathValAnalysisQuantitative = save_path + 'QAnalysisVal/'
for subj in range(len(DataValPath)):
    pathRead = SavePathVal + str(Patient_name(DataValPath[subj]))+'/IOU_SOL_'+Patient_name(DataValPath[subj])+'.npy'
    y1  = np.load(pathRead)
 
    fig = plt.figure()
    ax = plt.subplot(111)
    
    ax.plot(y1 ,'r-*')


    plt.xlabel('depth')
    plt.ylabel('mIoU')
    plt.title('mIoU over SOL muscle for patient ' + str(Patient_name(DataValPath[subj])))

    chartBox = ax.get_position()

#     ax.set_position([2*chartBox.x0, 0.5*chartBox.y0, 1.5*chartBox.width, 1.5*chartBox.height])
#     ax.legend(loc='upper center', bbox_to_anchor=(1.10, 0.90), shadow=False, ncol=1)

    fig1 = plt.gcf()
    plt.show()
    
    proposedPath = SavePathVal + str(Patient_name(DataValPath[subj]))
        
    fig1.savefig(proposedPath +'/mIoU_'+str(Patient_name(DataValPath[subj]))+'.png', dpi=200,bbox_inches ='tight')    
    


# In[ ]:


# Violin Plot trial
SavePathVal = save_path + '/patientAnalysisVal/'
SavePathValAnalysisQuantitative = save_path + 'QAnalysisVal/'

for subj in range(len(DataValPath)):
    pathRead = SavePathVal + str(Patient_name(DataValPath[subj]))+'/DICE_SOL_'+Patient_name(DataValPath[subj])+'.npy'
    y1  = np.load(pathRead)
 
    fig = plt.figure()
    ax = plt.subplot(111)
    
    ax.plot(y1 ,'r-*')


    plt.xlabel('depth')
    plt.ylabel('Dice Coefficient')
    plt.title('Dice Coefficient over SOL muscle for patient ' + str(Patient_name(DataValPath[subj])))

    chartBox = ax.get_position()

#     ax.set_position([2*chartBox.x0, 0.5*chartBox.y0, 1.5*chartBox.width, 1.5*chartBox.height])
#     ax.legend(loc='upper center', bbox_to_anchor=(1.10, 0.90), shadow=False, ncol=1)

    fig1 = plt.gcf()
    plt.show()
    
    proposedPath = SavePathVal + str(Patient_name(DataValPath[subj]))
   
    fig1.savefig(proposedPath +'/DiceCoef_'+str(Patient_name(DataValPath[subj]))+'.png', dpi=200,bbox_inches ='tight')    
    


# In[ ]:


# Violin Plot trial
SavePathVal = save_path + '/patientAnalysisVal/'
SavePathValAnalysisQuantitative = save_path + 'QAnalysisVal/'

for subj in range(len(DataValPath)):
    pathRead = SavePathVal + str(Patient_name(DataValPath[subj]))+'/Assd_SOL_'+Patient_name(DataValPath[subj])+'.npy'
    y1  = np.load(pathRead)
 
    fig = plt.figure()
    ax = plt.subplot(111)
    
    ax.plot(y1 ,'r-*')


    plt.xlabel('depth')
    plt.ylabel('ASSD Scores')
    plt.title('ASSD scores over SOL muscle for patient ' + str(Patient_name(DataValPath[subj])))

    chartBox = ax.get_position()

    #ax.set_position([2*chartBox.x0, 0.5*chartBox.y0, 1.5*chartBox.width, 1.5*chartBox.height])
    #ax.legend(loc='upper center', bbox_to_anchor=(1.10, 0.90), shadow=False, ncol=1)

    fig1 = plt.gcf()
    plt.show()
    
    proposedPath = SavePathVal + str(Patient_name(DataValPath[subj]))
   
    fig1.savefig(proposedPath +'/Assd_SOL_'+str(Patient_name(DataValPath[subj]))+'.png', dpi=200,bbox_inches ='tight')    
    


# In[ ]:


# Violin Plot trial
SavePathVal = save_path + '/patientAnalysisVal/'
SavePathValAnalysisQuantitative = save_path + 'QAnalysisVal/'

for subj in range(len(DataValPath)):
    pathRead = SavePathVal + str(Patient_name(DataValPath[subj]))+'/Asd_SOL_'+Patient_name(DataValPath[subj])+'.npy'
    y1  = np.load(pathRead)
 
    fig = plt.figure()
    ax = plt.subplot(111)
    
    ax.plot(y1 ,'r-*')


    plt.xlabel('depth')
    plt.ylabel('ASD Scores')
    plt.title('ASD scores over SOL muscle for patient ' + str(Patient_name(DataValPath[subj])))

    chartBox = ax.get_position()

#     ax.set_position([2*chartBox.x0, 0.5*chartBox.y0, 1.5*chartBox.width, 1.5*chartBox.height])
#     ax.legend(loc='upper center', bbox_to_anchor=(1.10, 0.90), shadow=False, ncol=1)

    fig1 = plt.gcf()
    plt.show()
    
    proposedPath = SavePathVal + str(Patient_name(DataValPath[subj]))
   
    fig1.savefig(proposedPath +'/Asd_SOL_'+str(Patient_name(DataValPath[subj]))+'.png', dpi=200,bbox_inches ='tight')    
    


# In[ ]:





# # TestSet Evaluation

# In[ ]:


del(y_val)
del(x_val)     


# In[ ]:


x_y_path = '/tf/volumes/test/'
Patient_folder_path = sorted(os.listdir(x_y_path))
Patient_folder_full_path = list(map(lambda v : str(join(x_y_path,v)) + '/', Patient_folder_path))

# Get the full path for the volume and the mask
DataTestPath = list(map(lambda s : s+'x1.mha' , Patient_folder_full_path))
MasksTestPath = list(map(lambda s : s+'masksX1.mha' , Patient_folder_full_path))


# In[ ]:


'''
Similar to Validation, keep everything Similar and Change only the Pah to the test
'''


# In[ ]:


x_y_path = '/tf/volumes/test/'
Patient_folder_path = sorted(os.listdir(x_y_path))
Patient_folder_full_path = list(map(lambda v : str(join(x_y_path,v)) + '/', Patient_folder_path))


# In[ ]:


# Get the full path for the volume and the mask
DataTestPath = list(map(lambda s : s+'x1.mha' , Patient_folder_full_path))
MasksTestPath = list(map(lambda s : s+'masksX1.mha' , Patient_folder_full_path))


# In[ ]:


x_test=Pull_data_from_path_Complete(DataTestPath[0])
y_test= process_mask_SOL_Complete(MasksTestPath[0])


# In[ ]:


# Plot each muscle alone
fig, ax = plt.subplots(1,3, sharey=True, figsize=(12,4))

ax[0].imshow(x_test[700][:,:,0] , aspect="auto",cmap='gray')
ax[1].imshow(y_test[700][:,:,0] , aspect="auto",cmap='gray')
ax[2].imshow(y_test[700][:,:,1] , aspect="auto",cmap='gray')

ax[0].set_title('Image')
ax[1].set_title('Mask F')
ax[2].set_title('Mask B')

