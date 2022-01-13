# Panorama image stitching
The repository describes in brief our solutions for the [project 1 of CMSC733](https://cmsc733.github.io/2020/proj/p1/). The report is divided into two sections. First section explores the traditional approach to find a homography matrix
between a set of two images. Second section describes the
implementation of a supervised and an unsupervised deep learn-
ing approach of estimating homography between synthetically
generated data.
## Phase 1: Traditional methood
### Sample dataset
### Corners Detection
### Adaptive Non-Maximal Suppression
### Feature Descriptor
### Feature Matching
### RANSAC for outlier rejection and to estimate Robust Homography
### Blending Images

## How to run the code
- Change the location to the root directory      
- Run the following command:
```
python3 Wrapper.py --BasePath ./Phase1/ --ImagesFolder Data/Train/Set3 --SaveFolderName Code/Results/Set3 
```

## Parameters 
- BasePath : Location for Phase 1. Eg. /home/sakshi/courses/CMSC733/sakshi_p1/Phase1/
- ImagesFolder: Location for image folder relative to the BasePath. Eg Data/Test/TestSet2
- SaveFolderName: Location where you want to save the results relativ BasePath. Eg. Code/Results/TestSet2
- ShowImages: If you want to view the step outputs. Set as False by default
- GoSequentially: Go sequentally while stitching or use half split method. Set as false by default.

# Phase 2: Deep learning approach
We implemented two deep learning approaches to estimate the homography between two images. The deep model effectively combines corner detection, ANMS, feature extraction, feature matching, RANSAC and estimate homography all into one. This not only makes the approach faster but also makes it robust if the network is generalizable.

## Data generation
- Copy Train, Val and Phase2 (Test data folder was named so in zip file) Folders, to Phase2/Data/
- cd Phase2/Code
- To generate required patches and labels for training the models, run,
```
    python3 DataGenerator.py
```
## Supervised
![alt](https://github.com/sakshikakde/AutoPano/blob/main/images/sup.png)
### Training
```
python3 Train.py --BasePath ../Data/Train_synthetic --CheckPointPath ../Checkpoints/supervised/ --ModelType sup --NumEpochs 100 --DivTrain 1 --MiniBatchSize 64 --LoadCheckPoint 0 --LogsPath ./Logs/supervised/
```

### Testing
```
python3 Test.py --ModelPath ../Checkpoints/supervised/supervisedModel.h5 --BasePath ../Data/Test_synthetic --SavePath ./Results/ --ModelType sup 
```

## Unsupervised
![alt](https://github.com/sakshikakde/AutoPano/blob/main/images/unsup.png)
### Training
```
python3 Train.py --BasePath ../Data/Train_synthetic --CheckPointPath ../Checkpoints/unsupervised/ --ModelType Unsup --NumEpochs 100 --DivTrain 1 --MiniBatchSize 64 --LoadCheckPoint 0 --LogsPath ./Logs/unsupervised/
```
### Testing
```
python3 Test.py --ModelPath ../Checkpoints/unsupervised/0model.ckpt --BasePath ../Data/Test_synthetic --CheckPointPath ../Checkpoints/unsupervised/ --SavePath ./Results/ --ModelType Unsup
```

## Results
![alt](https://github.com/sakshikakde/AutoPano/blob/main/images/dl_results.png)
