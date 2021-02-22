import cv2
import sys
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Misc.MiscUtils import *
from Misc.DataUtils import *
import numpy as np
import time
import argparse
import shutil
from io import StringIO
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
import csv
from sklearn.preprocessing import StandardScaler
# Don't generate pyc codes
sys.dont_write_bytecode = True


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow as tf
from keras import backend as K
from keras.utils import Sequence
from keras.initializers import VarianceScaling
from keras.models import Sequential
from keras.layers import  Activation, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, InputLayer
from keras import optimizers
from keras.callbacks import ModelCheckpoint

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

if (len(tf.config.experimental.list_physical_devices('GPU')) > 0) :
       print("################################################ RUNNING IN GPU ")
else:
       print("################################################ NO GPU FOUND" )


class loadDataBatches_gen(Sequence):
    def __init__(self,base_path, files_in_dir, labels_in_dir, batch_size, shuffle=True):
        
        self.labels_in_dir = labels_in_dir
        self.files_in_dir  = files_in_dir
        self.base_path = base_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.files_in_dir))

    def __len__(self):
        # returns the number of batches
        return len(files_in_dir) // self.batch_size

    def __getitem__(self, index):
        # returns one batch
        indices = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        X, y = self.__dataGen(indices)

        return X, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
#         self.indexes = np.arange(len(self.files_in_dir))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)    
    

    def __dataGen(self, indices):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        images_batch = []
        labels_batch = []
        pixel_shift_limit = 30
        for i in indices:

            #Get one row of x,Y
            image1_name = self.base_path + os.sep + "Train_synthetic/PA/" + self.files_in_dir[i, 0]
            image1 = cv2.imread(image1_name, cv2.IMREAD_GRAYSCALE)

            image2_name = self.base_path + os.sep + "Train_synthetic/PB/" + self.files_in_dir[i, 0] 
            image2 = cv2.imread(image2_name, cv2.IMREAD_GRAYSCALE)


            if(image1 is None) or (image2 is None):
                continue
                
#             image1 = cv2.normalize(image1.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
#             image2 = cv2.normalize(image2.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)

            image = np.dstack((image1.astype(np.float32), image2.astype(np.float32)))

            images_batch.append(image)
            labels_batch.append(labels_in_dir[i,:])

#             labels = (labels_in_dir[i,:] / pixel_shift_limit).astype(np.float32) 
#             labels_batch.append(labels)

        return np.array(images_batch), np.array(labels_batch) 
        
    
print("################################################ DataGenerator defined")

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


print("################################################ Model and loss defined")

####################################################### FIRST TRAINING 
print("################################################ Compiling Model, optimizer and loss functions")

model = supervised_HomographyNet()
adam = optimizers.Adam(lr=0.0001)
model.compile(loss= L2_loss, optimizer=adam, metrics=['mean_absolute_error'])

print("Printing model summary ..... \n")
print(model.summary())


print("################################################ Define absolute paths of files .... ")
base_path = "/home/gokul/CMSC733/hgokul_p1/Phase2/Data"
CheckPointPath = "/home/gokul/CMSC733/hgokul_p1/Phase2/Checkpoints/25kmodel/"
files_in_dir, SaveCheckPoint, ImageSize, number_of_training_samples, labels_in_dir = SetupAll(base_path, CheckPointPath, True) # get Train meta data
ckptPath = CheckPointPath+ "weights-{epoch:02d}-{loss:.2f}.ckpt"
checkpoint = ModelCheckpoint(ckptPath, monitor='loss', save_weights_only = True, verbose=1,  save_best_only = True, mode = min)

print("################################################ Define training parameters .... ")
epochs = 100
batch_size = 64
num_iterations_per_epoch = int(number_of_training_samples / batch_size)
print("Number of training samples: ", number_of_training_samples)
       
train_generator = loadDataBatches_gen(base_path, files_in_dir, labels_in_dir, batch_size, True)
X,y = train_generator[1]
print("Batch Shape,:  ", X.shape,y.shape )
       
print('Begin Training .....')                             
history_callback = model.fit_generator(generator = train_generator,steps_per_epoch = num_iterations_per_epoch,  epochs = epochs, callbacks=[checkpoint])

loss_history = history_callback.history["loss"]
np.savetxt("../Results/25kmodel/lossHistory.txt", np.array(loss_history), delimiter=",")

error_history = history_callback.history["mean_absolute_error"]
np.savetxt("../Results/25kmodel/errorHistory.txt", np.array(error_history), delimiter=",")
       
print("################################################ Done Training")

model.save('../Results/25kmodel/model25k.h5')

print("################################################ Model Saved")


################################################################## RELOAD  AND TRAIN

# from keras.models import load_model

# checkpoint = "/home/gokul/CMSC733/hgokul_p1/Phase2/Checkpoints/supervised/weights-49-15.25.hdf5"
# model = load_model(checkpoint, custom_objects={'L2_loss': L2_loss})

# print(model.summary())

# base_path = "/home/gokul/CMSC733/hgokul_p1/Phase2/Data"
# CheckPointPath = "/home/gokul/CMSC733/hgokul_p1/Phase2/Checkpoints/supervised2/"
# files_in_dir, SaveCheckPoint, ImageSize, NumTrainSamples, labels_in_dir = SetupAll(base_path, CheckPointPath)

# ckptPath = CheckPointPath + "weights-{epoch:02d}-{loss:.2f}.hdf5"
# checkpoint = ModelCheckpoint(ckptPath, monitor='loss', verbose=1)
                             
# epochs = 1
# number_of_training_samples = 4985
# batch_size = 64
# num_terations_per_epoch = int(number_of_training_samples / batch_size)

# generator = loadDataBatches_gen(base_path, files_in_dir, labels_in_dir, batch_size, True)
# print('TRAINING...')                             
# model.fit_generator(generator = generator,steps_per_epoch = num_terations_per_epoch,  epochs = epochs, callbacks=[checkpoint])

################################################################## TEST RESULT
print("Begin Testing .....")
n = 1000
all_labels = pd.read_csv("/home/gokul/CMSC733/hgokul_p1/Phase2/Data/Val_synthetic/H4.csv", index_col =False)
all_labels = all_labels.to_numpy()

all_patchNames = pd.read_csv("/home/gokul/CMSC733/hgokul_p1/Phase2/Data/Val_synthetic/ImageFileNames.csv")
all_patchNames = all_patchNames.to_numpy()

testPatches = []
for p in all_patchNames[:n]:
    tPatchA = cv2.imread("/home/gokul/CMSC733/hgokul_p1/Phase2/Data/Val_synthetic/PA/"+ str(p[0]), cv2.IMREAD_GRAYSCALE)
    tPatchB = cv2.imread("/home/gokul/CMSC733/hgokul_p1/Phase2/Data/Val_synthetic/PB/"+ str(p[0]), cv2.IMREAD_GRAYSCALE)
    tPatch = np.dstack((tPatchA, tPatchB))    
    testPatches.append(tPatch)
    
testPatches = np.array(testPatches)

y_preds = model.predict(testPatches)


from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_preds, all_labels[:n])
mse = mean_squared_error(y_preds, all_labels[:n])
np.savetxt("../Results/25kmodel/TestResults.txt", np.array([mae,mse]), delimiter = ",")

# model.save('modelnormalised.h5')

# print("################################################ Model Saved")

##################################################################  END