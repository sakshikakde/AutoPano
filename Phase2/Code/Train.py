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
import pandas as pd
# Don't generate pyc codes
sys.dont_write_bytecode = True

from Network.Network import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow as tf
from keras import backend as K

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
        
        
        
######################################### Keras DataGenerator Function for Supervised Model
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
        return len(self.files_in_dir) // self.batch_size

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
            image1_name = self.base_path + os.sep + "PA/" + self.files_in_dir[i, 0]
            image1 = cv2.imread(image1_name, cv2.IMREAD_GRAYSCALE)

            image2_name = self.base_path + os.sep + "PB/" + self.files_in_dir[i, 0] 
            image2 = cv2.imread(image2_name, cv2.IMREAD_GRAYSCALE)


            if(image1 is None) or (image2 is None):
                continue
                
#             image1 = cv2.normalize(image1.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
#             image2 = cv2.normalize(image2.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)

            image = np.dstack((image1.astype(np.float32), image2.astype(np.float32)))

            images_batch.append(image)
            labels_batch.append(self.labels_in_dir[i,:])

#             labels = (self.labels_in_dir[i,:] / pixel_shift_limit).astype(np.float32) 
#             labels_batch.append(labels)

        return np.array(images_batch), np.array(labels_batch) 
        
###########################################################################################################################

########################################################### Unsupervised model data loader functions

def loadData(folder_name, files_in_dir, points_list, batch_size, shuffle = True):

    patch_pairs = []
    corners1 = []
    patches2 = []
    images1 = []


    if(len(files_in_dir) < batch_size):
        print("The data has only ", len(files_in_dir) , " images and you are trying to get ",batch_size, " images")
        return 0

    for n in range(batch_size):
        index = random.randint(0, len(files_in_dir)-1)  #len(files_in_dir)-1
       
        patch1_name = folder_name + os.sep + "PA/" + files_in_dir[index, 0]
        patch1 = cv2.imread(patch1_name, cv2.IMREAD_GRAYSCALE)

        patch2_name = folder_name + os.sep + "PB/" + files_in_dir[index, 0] 
        patch2 = cv2.imread(patch2_name, cv2.IMREAD_GRAYSCALE)

        image1_name = folder_name + os.sep + "IA/" + files_in_dir[index, 0]
        image1 = cv2.imread(image1_name, cv2.IMREAD_GRAYSCALE)

        if(patch1 is None) or (patch2 is None):
            print(patch1_name, " is empty. Ignoring ...")
            continue

        patch1 = np.float32(patch1)
        patch2 = np.float32(patch2) 
        image1 = np.float32(image1)   

        #combine images along depth
        patch_pair = np.dstack((patch1, patch2))     
        corner1 = points_list[index, :, :, 0]
        
        
        patch_pairs.append(patch_pair)
        corners1.append(corner1)
        patches2.append(patch2.reshape(128, 128, 1))

        images1.append(image1.reshape(image1.shape[0], image1.shape[1], 1))

    patch_indices = getPatchIndices(np.array(corners1))    
    return np.array(patch_pairs), np.array(corners1), np.array(patches2), np.array(images1), patch_indices


def TrainModel(PatchPairsPH, CornerPH, Patch2PH, Image1PH,patchIndicesPH, DirNamesTrain, CornersTrain, NumTrainSamples, ImageSize, NumEpochs, BatchSize, SaveCheckPoint, CheckPointPath, LatestFile, BasePath, LogsPath):

    print("Unsupervised")
    patchb_pred, patchb_true, _ = unsupervised_HomographyNet(PatchPairsPH, CornerPH, Patch2PH, Image1PH,patchIndicesPH, BatchSize)

    with tf.name_scope('Loss'):
        loss = tf.reduce_mean(tf.abs(patchb_pred - patchb_true))


    with tf.name_scope('Adam'):
        Optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)

    # Tensorboard
    # Create a summary to monitor loss tensor
    EpochLossPH = tf.placeholder(tf.float32, shape=None)
    loss_summary = tf.summary.scalar('LossEveryIter', loss)
    epoch_loss_summary = tf.summary.scalar('LossPerEpoch', EpochLossPH)
    # tf.summary.image('Anything you want', AnyImg)

    # Merge all summaries into a single operation
    MergedSummaryOP1 = tf.summary.merge([loss_summary])
    MergedSummaryOP2 = tf.summary.merge([epoch_loss_summary])
    # MergedSummaryOP = tf.summary.merge_all()

    # Setup Saver
    Saver = tf.train.Saver()
    AccOverEpochs=np.array([0,0])

    with tf.Session() as sess:  

        if LatestFile is not None:
            Saver.restore(sess, CheckPointPath + LatestFile + '.ckpt')
          # Extract only numbers from the name
            StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
            print('Loaded latest checkpoint with the name ' + LatestFile + '....')
        else:
            sess.run(tf.global_variables_initializer())
            StartEpoch = 0
            print('New model initialized....')

        # Tensorboard
        Writer = tf.summary.FileWriter(LogsPath, graph=tf.get_default_graph())
        
        L1_loss = []
        for Epochs in tqdm(range(StartEpoch, NumEpochs)):

            NumIterationsPerEpoch = int(NumTrainSamples/BatchSize)
            Loss=[]
            epoch_loss=0

            for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):

                PatchPairsBatch, Corner1Batch, patch2Batch, Image1Batch, patchIndicesBatch = loadData(BasePath, DirNamesTrain, CornersTrain, BatchSize, shuffle = True)
                FeedDict = {PatchPairsPH: PatchPairsBatch, CornerPH: Corner1Batch, Patch2PH: patch2Batch, Image1PH: Image1Batch, patchIndicesPH: patchIndicesBatch}

                _, LossThisBatch, Summary = sess.run([Optimizer, loss, MergedSummaryOP1], feed_dict=FeedDict)
                Loss.append(LossThisBatch)
                epoch_loss = epoch_loss + LossThisBatch
                
                # Save checkpoint every some SaveCheckPoint's iterations
#                 if PerEpochCounter % SaveCheckPoint == 0:
#                   # Save the Model learnt in this epoch
#                     SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
#                     Saver.save(sess,  save_path=SaveName)
#                     print('\n' + SaveName + ' Model Saved...')

          # Tensorboard
            Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
            epoch_loss = epoch_loss/NumIterationsPerEpoch

            print("Printing Epoch:  ",  np.mean(Loss), "\n")
            L1_loss.append(np.mean(Loss))
          # Save model every epoch
            SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
            Saver.save(sess, save_path=SaveName)
            print('\n' + SaveName + ' Model Saved...')
            Summary_epoch = sess.run(MergedSummaryOP2,feed_dict={EpochLossPH: epoch_loss})
            Writer.add_summary(Summary_epoch,Epochs)
            Writer.flush()

        np.savetxt(LogsPath + "losshistory_unsupervised.txt", np.array(L1_loss), delimiter = ",")


####################################################### main function

def main():
    
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default="../Data/Train_synthetic", help='Base path of images, Default: ../Data/Train_synthetic')
    Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--ModelType', default='Unsup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
    
    Parser.add_argument('--NumEpochs', type=int, default=50, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=1, help='Size of the MiniBatch to use, Default:1')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='../Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')
    
    
    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    batch_size = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType
    
    if(not (os.path.isdir(LogsPath))):
        print(LogsPath, "  was not present, creating the folder...")
        os.makedirs(LogsPath)
    
    if ModelType == "Unsup":
        
        print(" UnSupervised  Model Trainer using tensorflow ...")
        
        if(not (os.path.isdir(CheckPointPath))):
            print(CheckPointPath, "  was not present, creating the folder...")
            os.makedirs(CheckPointPath)

        files_in_dir, SaveCheckPoint, ImageSize, NumTrainSamples, _ = SetupAll(BasePath, CheckPointPath)
        
        print("Number of Training Samples:..", NumTrainSamples)

        pointsList = np.load(BasePath+'/pointsList.npy')

        CornerPH = tf.placeholder(tf.float32, shape=(batch_size, 4,2))
        PatchPairsPH = tf.placeholder(tf.float32, shape=(batch_size, 128, 128 ,2))
        Patch2PH = tf.placeholder(tf.float32, shape=(batch_size, 128, 128, 1))
        Images1PH = tf.placeholder(tf.float32, shape=(batch_size, 240, 320, 1))
        patchIndicesPH = tf.placeholder(tf.int32, shape=(batch_size, 128, 128 ,2))

        LatestFile = None

        TrainModel(PatchPairsPH, CornerPH, Patch2PH, Images1PH, patchIndicesPH, files_in_dir, pointsList, NumTrainSamples, ImageSize,NumEpochs, batch_size, SaveCheckPoint, CheckPointPath, LatestFile, BasePath, LogsPath)

    
    else:
        print(" Supervised  Model Trainer using Keras...")
        
        model = supervised_HomographyNet()
        adam = optimizers.Adam(lr=0.0001)
        print("################################################ Model and loss defined")

        model.compile(loss= L2_loss, optimizer=adam, metrics=['mean_absolute_error'])
        print("################################################ Compiling Model, optimizer and loss functions")
        print("Printing model summary ..... \n")
        print(model.summary())


        print("################################################ Define paths of files .... ")

        if(not (os.path.isdir(CheckPointPath))):
            print(CheckPointPath, "  was not present, creating the folder...")
            os.makedirs(CheckPointPath)


        files_in_dir, SaveCheckPoint, ImageSize, number_of_training_samples, labels_in_dir = SetupAll(BasePath, CheckPointPath) # get Train meta data

        ckptPath = CheckPointPath+ "weights-{epoch:02d}-{loss:.2f}.ckpt"
        checkpoint = ModelCheckpoint(ckptPath, monitor='loss', save_weights_only = True, verbose=1,  save_best_only = True, mode = min)

        print("################################################ Define training parameters .... ")
        num_iterations_per_epoch = int(number_of_training_samples / batch_size)
        print("Number of training samples: ", number_of_training_samples)

        train_generator = loadDataBatches_gen(BasePath, files_in_dir, labels_in_dir, batch_size, True)
        
        X,y = train_generator[1]
        print("Batch Shape,:  ", X.shape,y.shape )

        print('Begin Training .....')                             
        history_callback = model.fit_generator(generator = train_generator,steps_per_epoch = num_iterations_per_epoch,  epochs = NumEpochs, callbacks=[checkpoint])

        loss_history = history_callback.history["loss"]
        
        
        np.savetxt(LogsPath+ "lossHistory_supervised.txt", np.array(loss_history), delimiter=",")

        error_history = history_callback.history["mean_absolute_error"]
        np.savetxt(LogsPath+ "errorHistory_supervised.txt", np.array(error_history), delimiter=",")

        print("################################################ Done Training, Saving final model")

        model.save(CheckPointPath+'supervisedModel.h5')

        print("################################################ Model Saved")

        
if __name__ == '__main__':
    main()
