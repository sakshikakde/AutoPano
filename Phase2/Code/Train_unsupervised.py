import tensorflow as tf
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

from Network.Network import homographyNet,unsupervised_HomographyNet
sys.dont_write_bytecode = True



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


if (len(tf.config.experimental.list_physical_devices('GPU')) > 0) :
       print("################################################ RUNNING IN GPU ")
else:
       print("################################################ NO GPU FOUND" )

def getPatchIndices(corners_a):
    patch_indices = []
    for i in range(corners_a.shape[0]):
        xmin,ymin = corners_a[i,0,0], corners_a[i,0,1]
        xmax,ymax = corners_a[i,3,0], corners_a[i,3,1]
#         print(xmin,ymin,xmax,ymax)
        X, Y = np.mgrid[xmin:xmax, ymin:ymax]
        patch_indices.append(np.dstack((Y,X))) 
    return np.array(patch_indices)

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
    pred_patch2, true_pathc2,_ = unsupervised_HomographyNet(PatchPairsPH, CornerPH, Patch2PH, Image1PH,patchIndicesPH, BatchSize)

    with tf.name_scope('Loss'):
        loss = tf.reduce_mean(tf.abs(pred_patch2 - true_pathc2))


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

        np.savetxt(LogsPath + "LossResults.txt", np.array(L1_loss), delimiter = ",")

def main():
    print("######################################## Loaded all functions")

    BasePath = '/home/gokul/CMSC733/hgokul_p1/Phase2/Data/Train_synthetic/'
    CheckPointPath = '/home/gokul/CMSC733/hgokul_p1/Phase2/Checkpoints/unsupervised/'

    files_in_dir, SaveCheckPoint, ImageSize, NumTrainSamples, _ = SetupAll(BasePath, CheckPointPath)
    LogsPath = '/home/gokul/CMSC733/hgokul_p1/Phase2/Logs/'

    print("Total number of Training Samples: ", NumTrainSamples)

    pointsList = np.load(BasePath+'/pointsList.npy')

    batch_size = 64
    CornerPH = tf.placeholder(tf.float32, shape=(batch_size, 4,2))
    PatchPairsPH = tf.placeholder(tf.float32, shape=(batch_size, 128, 128 ,2))
    Patch2PH = tf.placeholder(tf.float32, shape=(batch_size, 128, 128, 1))
    Images1PH = tf.placeholder(tf.float32, shape=(batch_size, 240, 320, 1))
    patchIndicesPH = tf.placeholder(tf.int32, shape=(batch_size, 128, 128 ,2))

    NumEpochs = 100
    LatestFile = None

    TrainModel(PatchPairsPH, CornerPH, Patch2PH, Images1PH, patchIndicesPH, files_in_dir, pointsList, NumTrainSamples, ImageSize,
               NumEpochs, batch_size, SaveCheckPoint, CheckPointPath, LatestFile, BasePath, LogsPath)
    

if __name__ == '__main__':
    main()


