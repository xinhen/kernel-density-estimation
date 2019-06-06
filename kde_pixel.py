from PIL import Image
from matplotlib import pyplot as plt
from scipy.spatial import distance
import numpy as np
import glob
import os
import math

Trainpath = "/Users/chenjingkun/Documents/data/2bak/healthtrain"
Testpath = "/Users/chenjingkun/Documents/data/2bak/healthtest"

class DistanceMethod:
    def chiSquared_distance(self, p, q):
        tmp = (np.array(p)-np.array(q))**2/(np.array(p)+np.array(q))
        return tmp
    def euclidean_distance(self, p, q):
        return distance.euclidean(p,q)

def grayChannelImageHistogram(path):
    im = Image.open(path)
    histogram = np.zeros(256)
    pixels = list(im.getdata())
    
    pix = np.array(pixels)
    for idx in range (0, len(pix)):
        histogram[pix[idx]] += 1
    return pix

def rgbChannelImageHistogram(path):
    im = Image.open(path)
    im_vals1 = np.zeros(256)
    im_vals2 = np.zeros(256)
    im_vals3 = np.zeros(256)

    r,g,b = im.split()

    pixels_r = list(r.getdata())
    pixels_g = list(g.getdata())
    pixels_b = list(b.getdata())
    pix_r = np.array(pixels_r)
    pix_g = np.array(pixels_g)
    pix_b = np.array(pixels_b)
    for idx in range (0, len(pix_r)):
        im_vals1[pix_r[idx]] += 1
        im_vals2[pix_g[idx]] += 1
        im_vals3[pix_b[idx]] += 1
    histogram = list(im_vals1) + list(im_vals2) + list(im_vals3)
    return histogram



def grayChannelImagepixel(path):
    im = Image.open(path)
    
    pixels = list(im.getdata())
    pix = np.array(pixels)
    return pix

def rgbChannelImagepixel(path):
    im = Image.open(path)
    im_vals1 = np.zeros(256)
    im_vals2 = np.zeros(256)
    im_vals3 = np.zeros(256)

    r,g,b = im.split()

    pixels_r = list(r.getdata())
    pixels_g = list(g.getdata())
    pixels_b = list(b.getdata())
    pix_r = np.array(pixels_r)
    pix_g = np.array(pixels_g)
    pix_b = np.array(pixels_b)
    
    pixel = list(pix_r) + list(pix_g) + list(pix_b)
    print(pixel)
    return pixel

def getImageList():
    trainfiles= os.listdir(Trainpath)
    testfiles= os.listdir(Testpath)
    trainfilelist = []
    testfilelist = []
    for file in trainfiles:
        trainfilelist.append(Trainpath + "/" + file)
    for file in testfiles:
        testfilelist.append(Testpath+ "/" + file)    
    return trainfilelist, testfilelist

def formula2(A,distance_score):
    return math.exp((-1)*distance_score/A)

def main():
    trainimagelist, testimagelist = getImageList()
    print(trainimagelist[0])
    print(testimagelist)
    im = Image.open(trainimagelist[0])
    
    histgramlist1 = []
    histgramlist2 = []
    distance_mothod = DistanceMethod()
    if im.mode == "L":
        im.close()
        for i in trainimagelist:
            histgramlist1.append(grayChannelImagepixel(i))
        for j in testimagelist:
            histgramlist2.append(grayChannelImagepixel(j))
        print("len1:",len(histgramlist1))
        print("len2:",len(histgramlist2))
    elif im.mode == "RGB":
        im.close()
        for i in trainimagelist:
            tmp = rgbChannelImagepixel(i)
            histgramlist1.append(tmp)
        for j in testimagelist:
            tmp = rgbChannelImagepixel(j)
            histgramlist2.append(tmp)
        print("len1:",len(histgramlist1))
        print("len2:",len(histgramlist2))
    else:
        return False
    count = 0
    distance_total = 0
    for i in xrange(len(histgramlist1)):
        j = i
        while j<(len(histgramlist1)-1):
            j = j + 1   
            count = count + 1
            tmp = distance_mothod.euclidean_distance(histgramlist1[i],histgramlist1[j])
            distance_total = distance_total + tmp
    A =distance_total/count
    print("A:",A)
    score_list = []
    for i in histgramlist2:
        score = 0
        for j in histgramlist1:
            distance_score = distance_mothod.euclidean_distance(i,j)
            #print("distance_score:",distance_score)
            single_score = formula2(A,distance_score)
            score = score + single_score
        final_score = score/len(histgramlist1)
        score_list.append(final_score)
        print(final_score)
    
if __name__ == '__main__':
    main()
