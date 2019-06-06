import math
import numpy as np
import cv2
import os
import logging
logging.basicConfig(level=logging.DEBUG,
                    filename='/home/jingkunchen/experiment/kde/brain_health_kde.txt',
                    filemode='a',
                    format='')

def flattenImage(image):
    result = np.array([])
    b,g,r = cv2.split(image)
    r_arr = np.array(r).reshape(32*32)
    g_arr = np.array(g).reshape(32*32)
    b_arr = np.array(b).reshape(32*32)
    image_arr = np.concatenate((r_arr, g_arr, b_arr))
    result = np.concatenate((result, image_arr)).astype(np.int32)
    return result.reshape(3072)
    
def gaussianClaculate(image1, image2):
    loss = image1 - image2
    squareimage1 = np.power(image1, 2)
    squareimage2 = np.power(image2, 2)
    squareloss = np.power(loss, 2)
    sumimage = np.sum(squareimage1) + np.sum(squareimage2)
    sumloss = np.sum(squareloss)
    return math.exp(sumloss/(sumimage*(-2)))

def main():
    trainpath = "/home/jingkunchen/data/brain/healthtrain/"
    trainfiles= os.listdir(trainpath)
    testpath = "/home/jingkunchen/data/brain/healthtest/"
    testfiles= os.listdir(testpath)
    trainlist = []
    testlist = []
    for file in trainfiles:
        trainlist.append(trainpath + "/" + file)
    for file in testfiles:
        testlist.append(testpath+ "/" + file)
    trainimage = []
    testimage = []
    count = 0 
    
    for i in trainlist:
        trainimage.append(cv2.imread(i).flatten())
    for i in testlist: 
        testimage.append(cv2.imread(i).flatten())

    resultlist = []
    
    for i in testimage:
        total = 0
        for j in trainimage:
            total = total + gaussianClaculate(j,i)
        logging.debug(str(total/len(trainimage)))
        
if __name__ == "__main__":
    main()