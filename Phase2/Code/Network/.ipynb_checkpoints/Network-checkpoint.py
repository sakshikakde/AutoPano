"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import tensorflow as tf
import sys
import numpy as np
from Misc.MiscUtils import *
from Misc.TFSpatialTransformer import transformer
from keras import backend as K

from keras import optimizers
from keras.callbacks import ModelCheckpoint

from keras.utils import Sequence
from keras.initializers import VarianceScaling
from keras.models import Sequential
from keras.layers import  Activation, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, InputLayer
# Don't generate pyc codes
sys.dont_write_bytecode = True


###################################################### Keras Model for Supervised model

def supervised_HomographyNet():

#     hidden_layer_size, num_classes = 1000, 8
    input_shape = (128, 128, 2)
    kernel_size = 3
    pool_size = 2
    filters = 64
    dropout = 0.5
    
    model = Sequential()
    model.add(InputLayer(input_shape))
    ## conv2d 128
    model.add(Conv2D(filters=filters,kernel_size = kernel_size, activation ='relu', padding ='same'))
    model.add(BatchNormalization())
    
    ## conv2d 128
    model.add(Conv2D(filters = filters,kernel_size = kernel_size, activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    
    model.add(MaxPooling2D(pool_size))
    
    ## conv2d 64
    model.add(Conv2D(filters=filters,kernel_size=kernel_size, activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    ## conv2d 64
    model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    
    model.add(MaxPooling2D(pool_size))
    
    ## conv2d 32 2x Filters
    model.add(Conv2D(filters=filters*2,kernel_size=kernel_size, activation='relu', padding='same',))
    model.add(BatchNormalization())
    ## conv2d 32 2x Filters
    model.add(Conv2D(filters=filters*2,kernel_size=kernel_size, activation='relu', padding='same',))
    model.add(BatchNormalization())
    
    model.add(MaxPooling2D(pool_size))
    
    ## conv2d 16 2x Filters
    model.add(Conv2D(filters=filters*2, kernel_size=kernel_size, activation='relu', padding='same',))
    model.add(BatchNormalization())
    ## conv2d 16 2x Filters
    model.add(Conv2D(filters=filters*2, kernel_size=kernel_size, activation='relu', padding='same',))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(dropout))
    #for regression model
    model.add(Dense(8))
    return model

#Loss Function using SMSE
def L2_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True))

#################################################################################################################################################### Unsupervised homographyNet#####################

# in memory of 02/22/2021
# def crop_tf(warped_Ia, corners_a):
#     size = tf.constant([128,128], tf.int32)
#     left_offsets = corners_a[:,:2]
#     center_X = left_offsets[:,0]+64
#     center_Y = left_offsets[:,1]+64
#     centers =  tf.stack([center_Y,center_X],axis=1)
#     warped_Pa = tf.image.extract_glimpse(warped_Ia,size,centers,centered=False,normalized=False)
#     return warped_Pa

def TensorDLT(H4, corners_a , batch_size):
    
    corners_a_tile = tf.expand_dims(corners_a, [2]) # batch_size x 8 x 1
    
    # Solve for H using DLT
    pred_h4p_tile = tf.expand_dims(H4, [2]) # batch_size x 8 x 1
    # 4 points on the second image
    pred_corners_b_tile = tf.add(pred_h4p_tile, corners_a_tile)
    
    # obtain 8 auxiliary tensors -> expand dimensions by 1 at first,-> create batch_size number of copies
    tensor_aux_M1 = tf.constant(Aux_M1,tf.float32)
    tensor_aux_M1 =  tf.expand_dims(tensor_aux_M1 ,[0])
    M1_tile = tf.tile(tensor_aux_M1,[batch_size,1,1])

    tensor_aux_M2 = tf.constant(Aux_M2,tf.float32)
    tensor_aux_M2 = tf.expand_dims(tensor_aux_M2,[0])
    M2_tile = tf.tile(tensor_aux_M2,[batch_size,1,1])
    
    tensor_aux_M3 = tf.constant(Aux_M3,tf.float32)
    tensor_aux_M3 = tf.expand_dims(tensor_aux_M3,[0])
    M3_tile = tf.tile(tensor_aux_M3,[batch_size,1,1])
    
    tensor_aux_M4 = tf.constant(Aux_M4,tf.float32)
    tensor_aux_M4 = tf.expand_dims(tensor_aux_M4,[0])
    M4_tile = tf.tile(tensor_aux_M4,[batch_size,1,1])
                      
    tensor_aux_M5 = tf.constant(Aux_M5,tf.float32)
    tensor_aux_M5 = tf.expand_dims(tensor_aux_M5,[0])
    M5_tile = tf.tile(tensor_aux_M5,[batch_size,1,1])
                      
    tensor_aux_M6 = tf.constant(Aux_M6,tf.float32)
    tensor_aux_M6 = tf.expand_dims(tensor_aux_M6,[0])
    M6_tile = tf.tile(tensor_aux_M6,[batch_size,1,1])
    
    tensor_aux_M71 = tf.constant(Aux_M71,tf.float32)
    tensor_aux_M71 = tf.expand_dims(tensor_aux_M71,[0])
    M71_tile = tf.tile(tensor_aux_M71,[batch_size,1,1])
                      
    tensor_aux_M72 = tf.constant(Aux_M72,tf.float32)
    tensor_aux_M72 = tf.expand_dims(tensor_aux_M72,[0])
    M72_tile = tf.tile(tensor_aux_M72,[batch_size,1,1])
                      
    tensor_aux_M8 = tf.constant(Aux_M8,tf.float32)
    tensor_aux_M8 = tf.expand_dims(tensor_aux_M8,[0])
    M8_tile = tf.tile(tensor_aux_M8,[batch_size,1,1])
                      
    tensor_aux_Mb = tf.constant(Aux_Mb,tf.float32)
    tensor_aux_Mb = tf.expand_dims(tensor_aux_Mb,[0])
    Mb_tile = tf.tile(tensor_aux_Mb,[batch_size,1,1])
    
    # Form the equations Ax = b to compute H
        # Build A matrix
    A1 = tf.matmul(M1_tile, corners_a_tile)                                              # Column 1
    A2 = tf.matmul(M2_tile, corners_a_tile)                                              # Column 2
    A3 = M3_tile                                                                         # Column 3
    A4 = tf.matmul(M4_tile, corners_a_tile)                                              # Column 4
    A5 = tf.matmul(M5_tile, corners_a_tile)                                              # Column 5
    A6 = M6_tile                                                                         # Column 6
    A7 = tf.matmul(M71_tile, pred_corners_b_tile) *  tf.matmul(M72_tile, corners_a_tile) # Column 7
    A8 = tf.matmul(M71_tile, pred_corners_b_tile) *  tf.matmul(M8_tile, corners_a_tile)  # Column 8
                      
                      
    # reshape A1-A8 in as 8x1 and stack them column wise                    
    A = tf.stack([tf.reshape(A1,[-1,8]),tf.reshape(A2,[-1,8]), tf.reshape(A3,[-1,8]),tf.reshape(A4,[-1,8]),
                 tf.reshape(A5,[-1,8]),tf.reshape(A6,[-1,8]), tf.reshape(A7,[-1,8]),tf.reshape(A8,[-1,8])],axis=1)
    A = tf.transpose(A, perm=[0,2,1]) 

    # Build b matrix
    b = tf.matmul(Mb_tile, pred_corners_b_tile)

#     print('shape of A:', A.get_shape().as_list())
#     print('shape of B:', b.get_shape().as_list())

    # Solve the Ax = b to get h11 - h32 as H8 matrix
    H_8 = tf.matrix_solve(A , b)  # batch_size x 8. -  has values from H11-H32
#     print('shape of H_8', H_8)

    # Add h33 = ones to the last cols to complete H matrix
    
    h_33 = tf.ones([batch_size, 1, 1]) 
    H_9 = tf.concat([H_8,h_33],1) 
    H_flat = tf.reshape(H_9, [-1,9])
    H = tf.reshape(H_flat,[-1,3,3])   # batch_size x 3 x 3

    return H


def homographyNet(Img):
   
    # Convolutional Layers     
    x = tf.layers.conv2d(inputs=Img, name='Conv2D1', padding='same',filters=64, kernel_size=[3,3], activation=None)
    x = tf.layers.batch_normalization(x, name='BatchNorm1')
    x = tf.nn.relu(x, name='Relu1')

    x = tf.layers.conv2d(inputs=x, name='Conv2D2', padding='same',filters=64, kernel_size=[3,3], activation=None) 
    x = tf.layers.batch_normalization(x, name='BatchNorm2')
    x = tf.nn.relu(x, name='Relu2')

    x = tf.layers.max_pooling2d(inputs=x, pool_size=[2,2], strides=2)

    x = tf.layers.conv2d(inputs=x, name='Conv2D3', padding='same',filters=64, kernel_size=[3,3], activation=None)
    x = tf.layers.batch_normalization(x, name='BatchNorm3')
    x = tf.nn.relu(x, name='Relu3')

    x = tf.layers.conv2d(inputs=x, name='Conv2D4', padding='same',filters=64, kernel_size=[3,3], activation=None)
    x = tf.layers.batch_normalization(x, name='BatchNorm4')
    x = tf.nn.relu(x, name='Relu4')

    x = tf.layers.max_pooling2d(inputs=x, pool_size=[2,2], strides=2)

    x = tf.layers.conv2d(inputs=x, name='Conv2D5', padding='same',filters=128, kernel_size=[3,3], activation=None)
    x = tf.layers.batch_normalization(x, name='BatchNorm5')
    x = tf.nn.relu(x, name='Relu5')

    x = tf.layers.conv2d(inputs=x, name='Conv2D6', padding='same',filters=128, kernel_size=[3,3], activation=None)
    x = tf.layers.batch_normalization(x, name='BatchNorm6')
    x = tf.nn.relu(x, name='Relu6')

    x = tf.layers.max_pooling2d(inputs=x, pool_size=[2,2], strides=2)
    
    x = tf.layers.conv2d(inputs=x, name='Conv2D7', padding='same',filters=128, kernel_size=[3,3], activation=None)
    x = tf.layers.batch_normalization(x, name='BatchNorm7')
    x = tf.nn.relu(x, name='Relu7')

    x = tf.layers.conv2d(inputs=x, name='Conv2D8', padding='same',filters=128, kernel_size=[3,3], activation=None)
    x = tf.layers.batch_normalization(x, name='BatchNorm8')
    x = tf.nn.relu(x, name='Relu8')

    #flattening layer
    x = tf.contrib.layers.flatten(x)

    #Fully-connected layers
    x = tf.layers.dense(inputs=x, name='FC1',units=1024, activation=tf.nn.relu)
    x = tf.layers.dropout(x,rate=0.5, training=True,name='Dropout')
    x = tf.layers.batch_normalization(x, name='BatchNorm9')
    H4 = tf.layers.dense(inputs=x, name='FCfinal',units=8, activation=None)

    return H4

def unsupervised_HomographyNet(patch_batches, corners_a,  patch_b, image_a, patch_indices, batch_size =64 ) :

    # note : corners_a is in shape 4,2 [[x1,y1][x2,y2][x3,y3][x4,y4]]

    batch_size,h,w,channels = image_a.get_shape().as_list()

    H4_batches = homographyNet(patch_batches) # H4 = [dx1,dy1,dx2,dy2,dx3,dy3,dx4,dy4] 

    corners_a = tf.reshape(corners_a,[batch_size,8]) # convert to 8x1 [x1,y1,x2,y2,x3,y3,x4,y4]

    H_batches = TensorDLT(H4_batches , corners_a, batch_size)
    
    # compute M
    M = np.array([[w/2.0, 0., w/2.0],
                  [0., h/2.0, h/2.0],
                  [0., 0., 1.]]).astype(np.float32)
    
    tensor_M = tf.constant(M, tf.float32)
    tensor_M = tf.expand_dims(tensor_M, [0])
    M_batches   = tf.tile(tensor_M, [batch_size, 1,1]) #make 'batch_size' number of copies. 
    
    #compute M_inv
    M_inv = np.linalg.inv(M)
    tensor_M_inv = tf.constant(M_inv, tf.float32)
    tensor_M_inv = tf.expand_dims(tensor_M_inv, [0])
    M_inv_batches   = tf.tile(tensor_M_inv, [batch_size,1,1]) #make 'batch_size' number of copies.
    
    H_scaled = tf.matmul(tf.matmul(M_inv_batches, H_batches), M_batches)

#     Pa = tf.slice(patch_batches,[0,0,0,0],[batch_size,128,128,1])

    warped_Ia, _ = transformer(image_a, H_scaled, (h,w))     

    warped_Ia = tf.reshape(warped_Ia, [batch_size, h,w])
    warped_Pa = tf.gather_nd(warped_Ia, patch_indices, name=None, batch_dims=1)

    warped_Pa = tf.transpose(warped_Pa, perm = [0,2,1])
    
    warped_Pa = tf.reshape(warped_Pa, [batch_size, 128, 128, 1])
    
    return warped_Pa, patch_b, H_batches

