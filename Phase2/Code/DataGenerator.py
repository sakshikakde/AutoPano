import cv2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from skimage.feature import peak_local_max
import pandas as pd
import os


def getPatches(image, patch_size = 128, pixel_shift_limit = 20, border_margin = 30):
    
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
    is_Train_options = [True,False]

    for s in is_Train_options:
        noneCounter=0
        is_Train = s

        if is_Train  == True:
            print("Generating Train data ......")
            path = '/home/gokul/CMSC733/hgokul_p1/Phase2/Data/Train/'
            savePath = '/home/gokul/CMSC733/hgokul_p1/Phase2/Data/Train_dummy/'
            imCount = 5001
        else:
            print("Generating Test data ......")
            path = '/home/gokul/CMSC733/hgokul_p1/Phase2/Data/Val/'   
            # savePath = '../Data/'
            savePath = '/home/gokul/CMSC733/hgokul_p1/Phase2/Data/Val_dummy/'
            imCount = 1000

        H4_list = []
        image_name_list = [] 
        pointsList = []
        print("Begin Data Generation .... ")

        for i in range(1,imCount):
            
            #random_ind = np.random.choice(range(1, 5000), replace= False)
            imageA = cv2.imread(path + str(i) + '.jpg')
            imageA = cv2.resize(imageA, (320,240), interpolation = cv2.INTER_AREA)

            Patch_a, Patch_b, H4, imageB, points = getPatches(imageA, patch_size = 128, pixel_shift_limit = 32, border_margin = 42) # make sure border_margin > pixelshift_limit
            if ((Patch_a is None)&(Patch_b is None)&(H4 is None)):
                print("encountered None return")
                noneCounter+=1
            else: 
                pathA = savePath +'PA/' + str(i) + '.jpg'
                pathB = savePath +'PB/' + str(i) + '.jpg'
                impathA = savePath +'IA/' + str(i) + '.jpg'
                impathB = savePath +'IB/' + str(i) + '.jpg'

                cv2.imwrite(pathA, Patch_a)
                cv2.imwrite(pathB, Patch_b)
                cv2.imwrite(impathA, imageA)
                cv2.imwrite(impathB, imageB)

                H4_list.append(np.hstack((H4[:,0] , H4[:,1])))
                pointsList.append(points)
                image_name_list.append(str(i) + '.jpg')
            
            
    print("done")
    print("No. of labels: ", len(H4_list),"No. of patches generated: ",(i-noneCounter))

    df = pd.DataFrame(H4_list)
    df.to_csv(savePath+"H4.csv", index=False)
    
    np.save(savePath+"pointsList.npy", np.array(pointsList))

    df = pd.DataFrame(image_name_list)#sakshi
    df.to_csv(savePath+"ImageFileNames.csv", index=False)
    


if __name__ == '__main__':
    main()