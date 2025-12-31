from operator import index
from random import sample
from unittest.mock import inplace

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import cv2
import random

from contourpy.array import remove_nan
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense, BatchNormalization, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


##### STEP 1 - INITIALIZE DATA

def getName(filePath):
    return filePath.split('\\')[-1]

def importDataInfo(path):
    columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names=columns)
    # print(data.head())
    # print(data['Center'][0])
    # print(getName(data['Center'][0]))
    data['Center'] = data['Center'].apply(getName)
    # print(data.head())
    print('Total Images Imported: ', data.shape[0])
    return data

##### STEP 2 - VISUALIZATION AND BALANCE DATA

def balanceData(data, display = True):
    nBins = 41
    # samplesPerBin = 1500 # MapEZ
    samplesPerBin = 25354 # MapDiff
    hist, bins = np.histogram(data['Steering'], nBins)
    # print(bins)
    if display:
        center = (bins[:-1] + bins[1:])*0.5
        # print(center)
        plt.bar(center, hist, width=0.06)
        plt.plot((-1,1),(samplesPerBin,samplesPerBin))
        plt.show()
    removeIndexList = []
    for j in range(nBins):
        binDataList = []
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j+1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeIndexList.extend(binDataList)
    print('Remove Images: ', len(removeIndexList))
    data.drop(data.index[removeIndexList], inplace = True)
    print('Remaining Images: ', len(data))

    if display:
        hist, _ = np.histogram(data['Steering'], nBins)
        plt.bar(center, hist, width=0.06)
        plt.plot((-1,1),(samplesPerBin,samplesPerBin))
        plt.show()

    return data

##### STEP 3 - PREPROCESSING

def loadData(path, data):
    imagesPath = []
    steering = []

    for i in range (len(data)):
        indexedData = data.iloc[i]
        # print(indexedData)
        imagesPath.append(os.path.join(path, 'IMG', indexedData.iloc[0]))
        # print(os.path.join(path, 'IMG', indexedData.iloc[0]))
        steering.append(float(indexedData.iloc[3]))
    imagesPath = np.asarray(imagesPath)
    steering = np.asarray(steering)
    return imagesPath, steering

##### STEP 4 - SPLIT DATA TRAINING AND VALIDATION

##### STEP 5 -
def augmentImage(imgPath, steering):
    img = mpimg.imread(imgPath)
    # print(np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand())
    # PAN
    if np.random.rand() <0.5:
        pan = iaa.Affine(translate_percent={'x':(-0.1,0.1), 'y':(-0.1,0.1)})
        img = pan.augment_image(img)
    # ZOOM
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    # BRIGHTNESS
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.2, 1.2))
        img = brightness.augment_image(img)
    # FLIP
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering
    # MOTION BLUR
    if np.random.rand() < 0.5:
        blur = iaa.MotionBlur(k=5)
        img = blur.augment_image(img)
    # SHADOW
    if np.random.rand() < 0.5:
        shadow = iaa.AddToBrightness((-30, 30))
        img = shadow.augment_image(img)

    return img, steering
# imgRe, st = augmentImage('testEzCenter.jpg', 0)
# plt.imshow(imgRe)
# plt.show()

##### STEP 6 = PRE-PROCESSING
def preProcessing(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img

# imgRe = preProcessing(mpimg.imread('testEzCenter.jpg'))
# plt.imshow(imgRe)
# plt.show()

def batchGen(imagesPath, steeringList, batchSize, trainFlag):
    while True:
        while True:
            imgBatch = []
            steeringBatch = []

            for i in range(batchSize):
                index = random.randint(0, len(imagesPath) - 1)
                if trainFlag:
                    img, steering = augmentImage(imagesPath[index], steeringList[index])
                else:
                    img = mpimg.imread(imagesPath[index])
                    steering = steeringList[index]
                img = preProcessing(img)
                imgBatch.append(img)
                steeringBatch.append(steering)
            yield (np.asarray(imgBatch), np.asarray(steeringBatch))

##### STEP 8 - NVIDIA'S MODEL

def nvidiaModel():
    model = Sequential()

    model.add(Convolution2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.compile(Adam(learning_rate=0.0003), loss='mse')
    return model