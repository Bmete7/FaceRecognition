# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 19:54:52 2018

@author: BurakBey
"""
import numpy

from model import create_model
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Layer
import tensorflow as tf

nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')

nn4_small2_pretrained.summary()


import numpy as np
import os.path

class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file) 
    
def load_metadata(path):
    metadata = []
    for i in os.listdir(path):
        for f in os.listdir(os.path.join(path, i)):
            metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)

metadata = load_metadata('images')


import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from align import AlignDlib

#%matplotlib inline

def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]

alignment = AlignDlib('models/landmarks.dat')



def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), 
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)


embedded = np.zeros((metadata.shape[0], 128))


dict = {}
for i, m in enumerate(metadata):
    dict[str(metadata[i]).split("\\")[2]] = i

for i, m in enumerate(metadata):
    img = load_image(str(metadata[i]))
    img = align_image(img)
    try:
        if(img.all() != None):
            img = (img / 255.).astype(np.float32)
            embedded[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
            dict[str(metadata[i])] = i
    except:
        print('no face here')
    
    
def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

def show_pair(idx1, idx2):
    plt.figure(figsize=(8,3))
    plt.suptitle(f'Difference Measure = {distance(embedded[idx1], embedded[idx2]):.2f}')
    plt.subplot(121)
    plt.imshow(load_image(metadata[idx1].image_path()))
    plt.subplot(122)
    plt.imshow(load_image(metadata[idx2].image_path()));    


def check_pair(idx1, idx2):
    dif = distance(embedded[idx1], embedded[idx2])
    if(dif<= 0.95):
        #print('Same Person')
        return 0
    else:
        #print('Different person')
        return 1
 


def readFile():
    true_positive = 0 
    true_negative = 0 
    false_positive = 0
    false_negative = 0
    firstLine = True
    times = 0
    iters = 0
    singles= 0
    iterTemp = 0
    with open('pairs.txt') as f:
        for line in f:
            if firstLine:
   
                times = int(line.split("\t")[0])         
                iters = int(line.split("\t")[1])
                iterTemp = int(line.split("\t")[1])
                firstLine = False
   
                continue
                    
            else:
                
                if (times == 0):
                    break
                if(iters == 0 and singles == 1):
                    
                    a1 = (line.split("\t")[0])
                    a2 = (line.split("\t")[1])
                    a3 = (line.split("\t")[2])
                    a3= (a3.split("\n")[0])
                    iters=iterTemp
                    singles = 0
                    times -= 1
                    if(int(a2) >= 100):
                        strImage = '_0' + str(a2)
                    elif(int(a2) >= 10):
                        strImage = '_00' + str(a2)
                    else:
                        strImage = '_000' + str(a2)
                    if(int(a3) >= 100):
                        strImage2 = '_0' + str(a3)
                    elif(int(a3) >= 10):
                        strImage2 = '_00' + str(a3)
                    else:
                        strImage2 = '_000' + str(a3)
                            
                        
                    path = a1 + strImage + '.jpg'
                    path2 = a1 + strImage2 + '.jpg'
 
                elif(iters == 0 and singles == 0):
                    
                    a1= (line.split("\t")[0])
                    a2= (line.split("\t")[1])
                    
                    a3= (line.split("\t")[2])
                    a4= (line.split("\t")[3])
                    a4= (a4.split("\n")[0])
                    singles = 1
                    iters = iterTemp
                    
                    if(int(a2) >= 100):
                        strImage = '_0' + str(a2)
                    elif(int(a2) >= 10):
                        strImage = '_00' + str(a2)
                    else:
                        strImage = '_000' + str(a2)
                    
                    if(int(a4) >= 100):
                        strImage2 = '_0' + str(a4)
                    elif(int(a4) >= 10):
                        strImage2 = '_00' + str(a4)
                    else:
                        strImage2 = '_000' + str(a4)
                            
                        
                    path = a1 + strImage + '.jpg'
                    path2 = a3 + strImage2 + '.jpg'

                else:
                
                    if(singles == 1):
                        a1= (line.split("\t")[0])
                        a2= (line.split("\t")[1])
                        a3= (line.split("\t")[2])
                        a4= (line.split("\t")[3])
                        
                        a4= (a4.split("\n")[0])
                        
                        if(int(a2) >= 100):
                            strImage = '_0' + str(a2)
                        elif(int(a2) >= 10):
                            strImage = '_00' + str(a2)
                        else:
                            strImage = '_000' + str(a2)
                        if(int(a4) >= 100):
                            strImage2 = '_0' + str(a4)
                        elif(int(a4) >= 10):
                            strImage2 = '_00' + str(a4)
                        else:
                            strImage2 = '_000' + str(a4)
                                
                            
                        path = a1 + strImage + '.jpg'
                        path2 = a3 + strImage2 + '.jpg'
  
                    else:
                        a1= (line.split("\t")[0])
                        a2= (line.split("\t")[1])
                        a3= (line.split("\t")[2])
                        a3= (a3.split("\n")[0])
                        if(int(a2) >= 100):
                            strImage = '_0' + str(a2)
                        elif(int(a2) >= 10):
                            strImage = '_00' + str(a2)
                        else:
                            strImage = '_000' + str(a2)
                        if(int(a3) >= 100):
                            strImage2 = '_0' + str(a3)
                        elif(int(a3) >= 10):
                            strImage2 = '_00' + str(a3)
                        else:
                            strImage2 = '_000' + str(a3)
                            
                        path = a1 + strImage + '.jpg'
                        path2 = a1 + strImage2 + '.jpg'

                iters -= 1
                
                ret_value = check_pair(dict[path], dict[path2])
                if(singles == ret_value):
                    if(singles==0):
                        true_positive += 1
                        
                    elif(singles==1):
                        true_negative +=1
                        
                else:
                    if(singles==0):
                        false_negative += 1
                        
                    elif(singles==1):
                        false_positive += 1

                
    print('True Positive: ' + str(true_positive))
    print('True Negative: ' + str(true_negative))
    print('false Positive: ' + str(false_positive))
    print('False Negative: ' + str(false_negative))
    print((true_positive+true_negative) / (true_negative+true_positive+false_positive+false_negative))
readFile()


    