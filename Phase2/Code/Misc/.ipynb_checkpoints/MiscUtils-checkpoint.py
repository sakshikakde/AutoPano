"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""


import time
import glob
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import cv2

# Don't generate pyc codes
sys.dont_write_bytecode = True

def tic():
    StartTime = time.time()
    return StartTime

def toc(StartTime):
    return time.time() - StartTime

def remap(x, oMin, oMax, iMin, iMax):
    # Taken from https://stackoverflow.com/questions/929103/convert-a-number-range-to-another-range-maintaining-ratios
    #range check
    if oMin == oMax:
        print("Warning: Zero input range")
        return None

    if iMin == iMax:
        print("Warning: Zero output range")
        return None

     # portion = (x-oldMin)*(newMax-newMin)/(oldMax-oldMin)
    result = np.add(np.divide(np.multiply(x - iMin, oMax - oMin), iMax - iMin), oMin)

    return result

def FindLatestModel(CheckPointPath):
    FileList = glob.glob(CheckPointPath + '*.ckpt.index') # * means all if need specific format then *.csv
    LatestFile = max(FileList, key=os.path.getctime)
    # Strip everything else except needed information
    LatestFile = LatestFile.replace(CheckPointPath, '')
    LatestFile = LatestFile.replace('.ckpt.index', '')
    return LatestFile


def convertToOneHot(vector, n_labels):
    return np.equal.outer(vector, np.arange(n_labels)).astype(np.float)

######################### TF functions and variables...

"""
auxiliaryMatrices to build the A matrix in Tensor DLT  
referred from :  https://github.com/tynguyen/unsupervisedDeepHomographyRAL2018/blob/master/code/utils/utils.py

""" 

Aux_M1  = np.array([
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=np.float64)

Aux_M2  = np.array([
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ]], dtype=np.float64)

Aux_M3  = np.array([
          [0],
          [1],
          [0],
          [1],
          [0],
          [1],
          [0],
          [1]], dtype=np.float64)



Aux_M4  = np.array([
          [-1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 ,-1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  ,-1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 ,-1 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ]], dtype=np.float64)


Aux_M5  = np.array([
          [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 ,-1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ]], dtype=np.float64)

Aux_M6  = np.array([
          [-1 ],
          [ 0 ],
          [-1 ],
          [ 0 ],
          [-1 ],
          [ 0 ],
          [-1 ],
          [ 0 ]], dtype=np.float64)

Aux_M71 = np.array([
          [0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=np.float64)

Aux_M72 = np.array([
          [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [-1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 ,-1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  ,-1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 ,-1 , 0 ]], dtype=np.float64)

Aux_M8  = np.array([
          [0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 ,-1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ]], dtype=np.float64)
Aux_Mb  = np.array([
          [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , -1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=np.float64)


def getPatchIndices(corners_a):
    """
    For a given set of 4 corners, return it's indices inside the region as a mesh grid
    -used in unsupervised model
    """
    
    patch_indices = []
    for i in range(corners_a.shape[0]):
        xmin,ymin = corners_a[i,0,0], corners_a[i,0,1]
        xmax,ymax = corners_a[i,3,0], corners_a[i,3,1]
#         print(xmin,ymin,xmax,ymax)
        X, Y = np.mgrid[xmin:xmax, ymin:ymax]
        patch_indices.append(np.dstack((Y,X))) 
    return np.array(patch_indices)
# functions 

#################################### Visualization Functions .

def getCornersFromH4pt(corner1, H4pt):
    corners1 = np.array(corner1.copy())
    del_corners = H4pt.reshape(2,4).T
    corners2 = corners1 + del_corners
    return corners2

def drawCorners(image, corners, color):

    corners_ = np.array(corners.copy())
    r = corners_[2,:].copy()
    corners_[2,:] = corners_[3,:]
    corners_[3,:] = r
    corners_ = corners_.reshape(-1,1,2)
#     print(corners_)
    corners_ = corners_.astype(int)
    image_corners = cv2.polylines(image.copy(),[corners_],True,color, 4)
    return image_corners

def getHfromH4pt(corners1, H4pt):
#     print("H4pt is: ")
#     print(H4pt.reshape(2,4).T)

    del_corners = H4pt.reshape(2,4).T
    
    corners1 = np.array(corners1)
#     print("corner1 is: ")
#     print(corners1)

    corners2 = corners1 + del_corners
#     print("corner2 is: ")
#     print(corners2)

    H = cv2.getPerspectiveTransform(np.float32(corners1), np.float32(corners2))
#     print("H is:")
#     print(H)
    return H

def warpImage(img, corners, H):
    image = img.copy()
    h, w, _= image.shape

    corners_ = np.array(corners)
    corners_ = corners_.reshape((-1,1,2))

    image_transformed = cv2.warpPerspective(image, H, (w,h))
    corner_transformed = cv2.perspectiveTransform(np.float32(corners_), H)
    corner_transformed = corner_transformed.astype(int)
    
    return image_transformed, corner_transformed


def Visualise_supervised(i, BasePath, SavePath):
    pointsList = np.load(BasePath+'/pointsList.npy')
    Y_Pred = np.load(SavePath+'supervised/Y_Pred.npy')
    
    Y_true  = pd.read_csv(BasePath+"/H4.csv", index_col =False)
    Y_true = Y_true.to_numpy()
    all_patchNames = pd.read_csv(BasePath+"/ImageFileNames.csv")
    all_patchNames = all_patchNames.to_numpy()
    print(len(all_patchNames),len(Y_true))


    corners_a = pointsList[i,:,:,0]
    corners_b = pointsList[i,:,:,1]

    imPathA = BasePath + '/IA/' + all_patchNames[i,0]
    imA = cv2.imread(imPathA)

    H_AB = getHfromH4pt(corners_a, Y_true[i])
    imB, corners_b_cal = warpImage(imA, corners_a, H_AB)

    imA_corners = drawCorners(imA, corners_a, (0,0,255))
    imB_corners = drawCorners(imB, corners_b_cal, (0,0,255))

    mae = mean_absolute_error(Y_Pred[i], Y_true[i])
    print("Mean absolute Error for image at index ",i, ":  ",mae)

    corners_b_pred = getCornersFromH4pt(corners_a, Y_Pred[i])
    # imA_corners = drawCorners(imA, pts1, (0,0,255))
    imB_corners_pred = drawCorners(imB_corners, corners_b_pred, (0,255,0))
    imB_corners_pred = cv2.putText(imB_corners_pred, "MCE: "+str(round(mae,3)),(150,230),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,0,0),2,cv2.LINE_AA)
    return np.hstack((imA_corners,imB_corners_pred))

def Visualise_unsupervised(i,BasePath, SavePath):
    pointsList = np.load(BasePath+'/pointsList.npy')
    Y_Pred = np.load(SavePath+"unsupervised/H_Pred.npy")

    H4Y_true  = pd.read_csv(BasePath+"/H4.csv", index_col =False)
    # Y_true = alternateH4Axis(Y_true)
    H4Y_true = H4Y_true.to_numpy()
    all_patchNames = pd.read_csv(BasePath+"/ImageFileNames.csv")
    all_patchNames = all_patchNames.to_numpy()

    corners_a = pointsList[i,:,:,0]

    imPathA = BasePath + '/IA/' + all_patchNames[i,0]
    imA = cv2.imread(imPathA)

    H_AB = getHfromH4pt(corners_a, H4Y_true[i])
    imB, corners_b_cal = warpImage(imA, corners_a, H_AB)

    imA_corners = drawCorners(imA, corners_a, (0,0,255))
    imB_corners = drawCorners(imB, corners_b_cal, (0,0,255))

    corners_a = corners_a.reshape((-1,1,2))
    corners_b_pred = cv2.perspectiveTransform(np.float32(corners_a), Y_Pred[i])
    corners_b_pred = corners_b_pred.astype(int)

    imB_corners_pred = drawCorners(imB_corners, corners_b_pred, (255,0,0))
    mce = np.mean(np.abs(corners_b_pred -  corners_b_cal)) #mean_corner_error 
    print("Mean corner Error for image at index ",i, ":  ",mce)

    imB_corners_pred = cv2.putText(imB_corners_pred, "MCE: "+str(round(mce,3)),(150,230),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,0,0),2,cv2.LINE_AA)
    return np.hstack((imA_corners,imB_corners_pred))


def getScoresUnsupervised(i,BasePath, SavePath):
    pointsList = np.load(BasePath+'/pointsList.npy')
    Y_Pred = np.load(SavePath+"unsupervised/H_Pred.npy")

    H4Y_true  = pd.read_csv(BasePath+"/H4.csv", index_col =False)
    H4Y_true = H4Y_true.to_numpy()
    all_patchNames = pd.read_csv(BasePath+"/ImageFileNames.csv")
    all_patchNames = all_patchNames.to_numpy()

    corners_a = pointsList[i,:,:,0]

    imPathA = BasePath + '/IA/' + all_patchNames[i,0]
    imA = cv2.imread(imPathA)

    H_AB = getHfromH4pt(corners_a, H4Y_true[i])
    imB, corners_b_cal = warpImage(imA, corners_a, H_AB)

    corners_a = corners_a.reshape((-1,1,2))
    corners_b_pred = cv2.perspectiveTransform(np.float32(corners_a), Y_Pred[i])
    corners_b_pred = corners_b_pred.astype(int)

    mce = np.mean(np.abs(corners_b_pred -  corners_b_cal)) #mean_corner_error 
    return mce
