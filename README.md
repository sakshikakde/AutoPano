# Phase 1:

## How to run the code
1) Change the directory      
Eg. cd /home/sakshi/courses/CMSC733/sakshi_p1/Phase1/Code

3) Run the .py script       
python3 Wrapper.py --BasePath /home/sakshi/courses/CMSC733/sakshi_p1/Phase1/ --ImagesFolder Data/Train/Set3 --SaveFolderName Code/Results/Set3 

## Parameters 
1) BasePath : Location for Phase 1. Eg. /home/sakshi/courses/CMSC733/sakshi_p1/Phase1/
2) ImagesFolder: Location for image folder relative to the BasePath. Eg Data/Test/TestSet2
3) SaveFolderName: Location where you want to save the results relativ BasePath. Eg. Code/Results/TestSet2
4) ShowImages: If you want to view the step outputs. Set as False by default
5) GoSequentially: Go sequentally while stitching or use half split method. Set as false by default.

We have created the Results folder inside Code folder, but it is empty since the size became too large.

# Phase 2:

Copy Train, Val and Phase2 (Test data folder was named so in zip file) Folders, to Phase2/Data/

cd Phase2/Code

1) Generate required patches and labels for training the models. Run,

    python3 DataGenerator.py
    
2) Train the models.

Supervised model Training:

    python3 Train.py --BasePath ../Data/Train_synthetic --CheckPointPath ../Checkpoints/supervised/ --ModelType sup --NumEpochs 100 --DivTrain 1 --MiniBatchSize 64 --LoadCheckPoint 0 --LogsPath ./Logs/supervised/

Unsupervised model Training:

    python3 Train.py --BasePath ../Data/Train_synthetic --CheckPointPath ../Checkpoints/unsupervised/ --ModelType Unsup --NumEpochs 100 --DivTrain 1 --MiniBatchSize 64 --LoadCheckPoint 0 --LogsPath ./Logs/unsupervised/


Supervised Model Testing:

    python3 Test.py --ModelPath ../Checkpoints/supervised/supervisedModel.h5 --BasePath ../Data/Test_synthetic --SavePath ./Results/ --ModelType sup 

Unsupervised Model Testing:

    python3 Test.py --ModelPath ../Checkpoints/unsupervised/0model.ckpt --BasePath ../Data/Test_synthetic --CheckPointPath ../Checkpoints/unsupervised/ --SavePath ./Results/ --ModelType Unsup
