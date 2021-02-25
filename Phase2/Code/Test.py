import cv2
import sys
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Misc.MiscUtils import *
from Misc.DataUtils import *
from Network.Network import *
import numpy as np
import pandas as pd
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

import tensorflow as tf

from keras import backend as K
from keras.models import load_model

from sklearn.metrics import mean_absolute_error, mean_squared_error

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from tensorflow.python.client import device_lib

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if (len(tf.config.experimental.list_physical_devices('GPU')) > 0) :
       print("################################################ RUNNING IN GPU ")
else:
       print("################################################ NO GPU FOUND" )

def Test_supervised(BasePath, ModelPath, SavePath):
    
    SavePath = SavePath+'supervised/'
    if(not (os.path.isdir(SavePath))):
        print(SavePath, "  was not present, creating the folder...")
        os.makedirs(SavePath)

    model = load_model(ModelPath, custom_objects={'L2_loss': L2_loss})

    all_labels = pd.read_csv(BasePath+"/H4.csv", index_col =False)
    all_labels = all_labels.to_numpy()
    all_patchNames = pd.read_csv(BasePath+"/ImageFileNames.csv")
    all_patchNames = all_patchNames.to_numpy()

    X_test = []
    for p in all_patchNames:
    #     print(p)
        tPatchA = cv2.imread(BasePath+"/PA/"+ str(p[0]), cv2.IMREAD_GRAYSCALE)
        tPatchB = cv2.imread(BasePath+"/PB/"+ str(p[0]), cv2.IMREAD_GRAYSCALE)
        tPatch = np.dstack((tPatchA, tPatchB))    
        X_test.append(tPatch)

    X_test = np.array(X_test)    
    Y_true = all_labels

    print("Shape of X_test and Y_test ", X_test.shape,Y_true.shape)

    Y_pred = model.predict(X_test)
    
    np.save(SavePath+"Y_Pred.npy",Y_pred)
    
    mae = mean_absolute_error(Y_true, Y_pred)
    mse = mean_squared_error(Y_true, Y_pred)
     
    print("Mean Absolute Error: ", mae)
    print("Mean Squared Error: ", mse)
    
    return None

########################################################### Unsupervised Functions
def loadData(folder_name, files_in_dir, points_list,NumTestSamples):

    patch_pairs = []
    corners1 = []
    patches2 = []
    images1 = []


    for n in range(NumTestSamples):
#         index = random.randint(0, len(files_in_dir)-1)  #len(files_in_dir)-1
        index = n
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

def Test_unsupervised(PatchPairsPH, CornerPH, Patch2PH, Image1PH,patchIndicesPH, ModelPath, BasePath, files_in_dir, pointsList, SavePath, NumTestSamples):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    DataPath - Paths of all images where testing will be run on
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to ./TxtFiles/PredOut.txt
    """
    
    if(not (os.path.isdir(SavePath))):
        print(SavePath, "  was not present, creating the folder...")
        os.makedirs(SavePath)

    # Create the graph
    # Predict output with forward pass, MiniBatchSize for Test is 1
    _, _, H_batches = unsupervised_HomographyNet(PatchPairsPH, CornerPH, Patch2PH, Image1PH, patchIndicesPH, NumTestSamples)

    # Setup Saver
    # load session and run
    Saver = tf.train.Saver()
    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
            
        PatchPairsBatch, Corner1Batch, patch2Batch, Image1Batch, patchIndicesBatch = loadData(BasePath, files_in_dir, pointsList, NumTestSamples)
        FeedDict = {PatchPairsPH: PatchPairsBatch, CornerPH: Corner1Batch, Patch2PH: patch2Batch, Image1PH: Image1Batch, patchIndicesPH: patchIndicesBatch}            
            
        H_pred = sess.run(H_batches, FeedDict)
        np.save(SavePath+'H_Pred.npy', H_pred)
        
########################################################### Unsupervised Functions
def main():
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelPath', dest='ModelPath', default='../Checkpoints/unsupervised/99model.ckpt', help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--CheckPointPath', dest='CheckPointPath', default= '../Checkpoints/unsupervised/', help='Path to load latest model from, Default:CheckPointPath')
    Parser.add_argument('--BasePath', dest='BasePath', default='../Data/Test_synthetic', help='Path to load images from, Default:BasePath')
    Parser.add_argument('--SavePath', dest='SavePath', default='./Results/', help='Path of labels file, Default: ./Results/')
    Parser.add_argument('--ModelType', default='Unsup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')

    
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath
    CheckPointPath = Args.CheckPointPath
    SavePath = Args.SavePath
    ModelType = Args.ModelType
    
    if ModelType == 'Unsup':
        files_in_dir, SaveCheckPoint, ImageSize, NumTestSamples, _ = SetupAll(BasePath, CheckPointPath)
        
        NumTestSamples = 100
        print(NumTestSamples)
        pointsList = np.load(BasePath+'/pointsList.npy')

        CornerPH = tf.placeholder(tf.float32, shape=(NumTestSamples, 4,2))
        PatchPairsPH = tf.placeholder(tf.float32, shape=(NumTestSamples, 128, 128 ,2))
        Patch2PH = tf.placeholder(tf.float32, shape=(NumTestSamples, 128, 128, 1))
        Images1PH = tf.placeholder(tf.float32, shape=(NumTestSamples, 240, 320, 1))
        patchIndicesPH = tf.placeholder(tf.int32, shape=(NumTestSamples, 128, 128 ,2))

        Test_unsupervised(PatchPairsPH, CornerPH, Patch2PH, Images1PH,patchIndicesPH, ModelPath, BasePath, files_in_dir, pointsList, SavePath+"unsupervised/", NumTestSamples)
         
        rand_i = np.random.randint(0,NumTestSamples-1, size=5)
        for i in rand_i:
            comparison = Visualise_unsupervised(i, BasePath, SavePath)
            cv2.imwrite(SavePath+'unsupervised/comparison'+ str(i)+'.png',comparison)        
        print('Check Results/unsupervised folder..')
        
    else:
        Test_supervised(BasePath, ModelPath, SavePath)
        
        files_in_dir = pd.read_csv(BasePath+"/H4.csv", index_col =False) 
        rand_i = np.random.randint(0,len(files_in_dir)-1, size=5)
        for i in rand_i:
            comparison = Visualise_supervised(i, BasePath, SavePath)
            cv2.imwrite(SavePath+'supervised/comparison'+ str(i)+'.png',comparison)
        
        print('Check Results/supervised folder..')
    
if __name__ == '__main__':
    main()
