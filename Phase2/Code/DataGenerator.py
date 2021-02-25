import cv2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from skimage.feature import peak_local_max
import pandas as pd
import os


def getPatches(image, patch_size = 128, pixel_shift_limit = 32, border_margin = 42):
    
    """
    Inputs: 
    image: image from MS Coco dataset,
    patch_size = size of the patches to be cropped randomly, default: 128
    pixel_shift_limit = radius of pixel neigbourhood for which the corners can be chosen to obtain patch B
    border_margin = margin from the boundaries of image to crop the patch. 
                    (Choose border_margin > pixel_shift_limit)
    
    Returns:
    Patch_a : randomly cropped patch from input image
    Patch_b : patchB cropped from Image B,
    H4: The H4 point homography between Image A and Image B
    pts1,pts2: corner coordinates of Patch_a and Patch_b with respect to Image A and Image B
    """
    
    h,w = image.shape[:2]
    minSize = patch_size+ 2*border_margin+1 # minimuum size of the image to be maintained.
    if ((w > minSize) & (h > minSize)):
        # pixel_shift_limit =  amount of random shift that the corner points can go through
        # make sure border_margin > pixel_shift_limit
         #  leave some margin space along borders 
        end_margin = patch_size + border_margin # to get right/bottom most pixel within image frame
    
    # choose left-top most point within the defined bordera
        x = np.random.randint(border_margin, w-end_margin) # left x pixel of the patch  
        y = np.random.randint(border_margin, h-end_margin) # top y pixel of the patch
    
        # choose left-top most point within the defined border
        pts1 = np.array([[x,y], [x, patch_size+y] , [patch_size+x, y], [patch_size+x, patch_size+y]]) # coordinates of patch P_a
        pts2 = np.zeros_like(pts1)

        # randomly shift coordinates of patch P_a to get coordinates of patch P_b
        for i,pt in enumerate(pts1):
            pts2[i][0] = pt[0] + np.random.randint(-pixel_shift_limit, pixel_shift_limit)
            pts2[i][1] = pt[1] + np.random.randint(-pixel_shift_limit, pixel_shift_limit)

        # find H inverse of usin patch coordinates of P_a, P_b
        H_inv = np.linalg.inv(cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))) 
        
        imageB = cv2.warpPerspective(image, H_inv, (w,h))

        Patch_a = image[y:y+patch_size, x:x+patch_size]
        Patch_b = imageB[y:y+patch_size, x:x+patch_size]
        H4 = (pts2 - pts1).astype(np.float32) 

        return Patch_a, Patch_b, H4, imageB, np.dstack((pts1,pts2))
    else:
        return None, None, None, None, None
    
    



# ########################################### SPECIFY THE DATA PATHs ###########################################
def main():
    """
    Generates data for the supervised and unsupervised homography networks.
    
    attempts: list with len(attempts) number of random crops inside a given image.
    options: list to generate Train/Validation/Test data 
    
    Generated Data:
    PA: all patches Pa from image A
    PB: all patches Pb from image B
    H4.csv: H4 homography matrices shaped 1x8 ordered[dx1,dx2,dx3,dx4,dy1,dy2,dy3,dy4]
    pointsList.npy: numpy array of corner coordinates of Pa and Pb with respect to images A and B. shaped (numSamples,4,2,2) 
    ImageFileNames.csv: meta file that contains image file names ordered with corresponding H4 and pointsList files.
    
    """
    
    attempts = ['a','b','c','d','e'] # 5 alphabets, for 5 random patches per image
    options = ['Train','Val', 'Test']

    for option in options:
        noneCounter=0

        if option == 'Train':
            print("Generating Train data ......")
            path = '../Data/Train/'
            savePath = '../Data/Train_synthetic/'
            imCount = 5001 # Train folder had 5000 images
            
        elif option == 'Val':
            print("Generating Validation data ......")
            path = '../Data/Val/'
            savePath = '../Data/Val_synthetic/'
            imCount = 1000 # Val folder had 999 images

        else:
            print("Generating Test data ......")
            path = '../Data/Phase2/'    # the released test folder with 1000 images was named as Phase2
            savePath = '../Data/Test_synthetic/'
            imCount = 1001 #Val folder had 1000 images
            
        if(not (os.path.isdir(savePath))):
            print(savePath, "  was not present, creating the folder...")
            os.makedirs(savePath)
        
        H4_list = []
        image_name_list = [] 
        pointsList = []
        print("Begin Data Generation .... ")
        
        for a in attempts:
            print("Doing attempt:  ",a)
            for i in range(1,imCount):

                #random_ind = np.random.choice(range(1, 5000), replace= False)
                imageA = cv2.imread(path + str(i) + '.jpg')
                imageA = cv2.resize(imageA, (320,240), interpolation = cv2.INTER_AREA)

                Patch_a, Patch_b, H4, _, points = getPatches(imageA, patch_size = 128, pixel_shift_limit = 32, border_margin = 42) 
                
                if ((Patch_a is None)&(Patch_b is None)&(H4 is None)):
                    print("encountered None return.. ignoring Image..")
                    noneCounter+=1
                else:
                    if(not (os.path.isdir(savePath +'PA/'))):
                        print(" Subdirectories inside  ", savePath, " were not present.. creating the folders...")
                        os.makedirs(savePath +'PA/')
                        os.makedirs(savePath +'PB/')
                        os.makedirs(savePath +'IA/')
 
                    pathA = savePath +'PA/' + str(i) +a+ '.jpg'
                    pathB = savePath +'PB/' + str(i) +a+ '.jpg'
                    impathA = savePath +'IA/' + str(i) +a+ '.jpg'
#                     impathB = savePath +'IB/' + str(i) +a+ '.jpg'

                    cv2.imwrite(pathA, Patch_a)
                    cv2.imwrite(pathB, Patch_b)
                    cv2.imwrite(impathA, imageA)
#                     cv2.imwrite(impathB, imageB)

                    H4_list.append(np.hstack((H4[:,0] , H4[:,1])))
                    pointsList.append(points)
                    image_name_list.append(str(i) +a+ '.jpg')
            
            
        print("done")
        print("No. of labels: ", len(H4_list),"No. of images: ", len(image_name_list), "No. of points: ", np.array(pointsList).shape,  "No. of patches generated: ",(i-noneCounter))

        df = pd.DataFrame(H4_list)
        df.to_csv(savePath+"H4.csv", index=False)
        print("saved H4 data in:  ", savePath)

        np.save(savePath+"pointsList.npy", np.array(pointsList))
        print("saved points data in:  ", savePath)
        
        df = pd.DataFrame(image_name_list)#sakshi
        df.to_csv(savePath+"ImageFileNames.csv", index=False)
        print("saved ImageFiles list  in:  ", savePath)


if __name__ == '__main__':
    main()