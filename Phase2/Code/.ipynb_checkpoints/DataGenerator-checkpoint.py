import cv2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from skimage.feature import peak_local_max
import pandas as pd
import os


def getPatches(image, patch_size = 128, pixel_shift_limit = 20, border_margin = 50):
    
    h,w = image.shape[:2]
    minSize = patch_size+ 2*border_margin+1 # minimuum size of the image to be maintained.

    if ((w > minSize) & (h > minSize)):
        if(len(image.shape)==3):
            gray_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image

        gray_image = cv2.normalize(gray_image.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
        # pixel_shift_limit =  amount of random shift that the corner points can go through
        # make sure border_margin > pixel_shift_limit
         #  leave some margin space along borders 
        end_margin = patch_size + border_margin # to get right/bottom most pixel within image frame
        h,w = gray_image.shape

    
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
        
        gray_imageB = cv2.warpPerspective(gray_image, H_inv, (w,h))
        Patch_a = gray_image[y:y+patch_size, x:x+patch_size]
        Patch_b = gray_imageB[y:y+patch_size, x:x+patch_size] 
        Patch_a = (Patch_a*255).astype(np.uint8)
        Patch_b = (Patch_b*255).astype(np.uint8)
        
        H4 = (pts2 - pts1).astype(np.float32) 

        return Patch_a, Patch_b, H4, np.dstack((pts1,pts2))
    else:
        return None, None, None, None



# ########################################### SPECIFY THE DATA PATHs ###########################################

def main():
    noneCounter=0
    # path = '../Data/Train/'
#     path = '/home/sakshi/courses/CMSC733/sakshi_p1/Phase2/Data/Train/'
    path = '/home/gokul/CMSC733/hgokul_p1/Phase2/Data/Train/'

    # savePath = '../Data/'
    savePath = '/home/gokul/CMSC733/hgokul_p1/Phase2/Data/Train_synthetic/'

    H4_list = []
    image_name_list = [] #sakshi

    print("Begin Data Generation .... ")

    for i in range(1,5001):
        #random_ind = np.random.choice(range(1, 5000), replace= False)
        image = plt.imread( path + str(i) + '.jpg')
        Patch_a, Patch_b, H4,_ = getPatches(image, patch_size = 128, pixel_shift_limit = 30, border_margin = 40) # make sure border_margin > pixelshift_limit
        if ((Patch_a is None)&(Patch_b is None)&(H4 is None)):
            print("encountered None return")
            noneCounter+=1
        else: 
            pathA = savePath +'PA/' + str(i) + '.jpg'
            pathB = savePath +'PB/' + str(i) + '.jpg'
            cv2.imwrite(pathA, Patch_a)
            cv2.imwrite(pathB, Patch_b)
            H4_list.append(np.hstack((H4[:,0] , H4[:,1])))
            image_name_list.append(str(i) + '.jpg')

    print("done")
    print("No. of labels: ", len(H4_list),"No. of patches generated: ",(i-noneCounter))

    df = pd.DataFrame(H4_list)
    df.to_csv(savePath+"H4.csv", index=False)

    df = pd.DataFrame(image_name_list)#sakshi
    df.to_csv(savePath+"ImageFileNames.csv", index=False)#sakshi
    


if __name__ == '__main__':
    main()