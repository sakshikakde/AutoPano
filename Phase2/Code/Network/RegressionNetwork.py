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
# Don't generate pyc codes
sys.dont_write_bytecode = True

def HomographyModel(Img, ImageSize, MiniBatchSize)

    x = tf.layers.conv2d(inputs=Img, name='conv1', padding='same',filters=64, kernel_size=[3,3], activation=None)
    x = tf.layers.batch_normalization(x, name='bacth_norm1')
    x = tf.nn.relu(x, name='relu_1')

    x = tf.layers.conv2d(inputs=x, name='conv2', padding='same',filters=64, kernel_size=[3,3], activation=None) 
    x = tf.layers.batch_normalization(x, name='bacth_norm2')
    x = tf.nn.relu(x, name='relu_2')

    x = tf.layers.max_pooling2d(inputs=x, pool_size=[2,2], strides=2)

    x = tf.layers.conv2d(inputs=x, name='conv3', padding='same',filters=64, kernel_size=[3,3], activation=None)
    x = tf.layers.batch_normalization(x, name='bacth_norm3')
    x = tf.nn.relu(x, name='relu_3')

    x = tf.layers.conv2d(inputs=x, name='conv4', padding='same',filters=64, kernel_size=[3,3], activation=None)
    x = tf.layers.batch_normalization(x, name='bacth_norm4')
    x = tf.nn.relu(x, name='relu_4')

    x = tf.layers.max_pooling2d(inputs=x, pool_size=[2,2], strides=2)

    x = tf.layers.conv2d(inputs=x, name='conv5', padding='same',filters=128, kernel_size=[3,3], activation=None)
    x = tf.layers.batch_normalization(x, name='bacth_norm5')
    x = tf.nn.relu(x, name='relu_5')

    x = tf.layers.conv2d(inputs=x, name='conv6', padding='same',filters=128, kernel_size=[3,3], activation=None)
    x = tf.layers.batch_normalization(x, name='bacth_norm6')
    x = tf.nn.relu(x, name='relu_6')

    x = tf.layers.max_pooling2d(inputs=x, pool_size=[2,2], strides=2)
    
    x = tf.layers.conv2d(inputs=x, name='conv7', padding='same',filters=128, kernel_size=[3,3], activation=None)
    x = tf.layers.batch_normalization(x, name='bacth_norm7')
    x = tf.nn.relu(x, name='relu_7')

    x = tf.layers.conv2d(inputs=x, name='conv8', padding='same',filters=128, kernel_size=[3,3], activation=None)
    x = tf.layers.batch_normalization(x, name='bacth_norm8')
    x = tf.nn.relu(x, name='relu_8')

    #flattening layer
    x = tf.contrib.layers.flatten(x)

    #Fully-connected layers
    x = tf.layers.dense(inputs=x, name='fc_1',units=1024, activation=tf.nn.relu)
    x = tf.layers.dropout(x,rate=0.5,noise_shape=None,seed=None,training=True,name=None)
    x = tf.layers.batch_normalization(x, name='bacth_norm9')
    H4pt = tf.layers.dense(inputs=x, name='fc_final',units=8, activation=None) 

    return H4Pt

