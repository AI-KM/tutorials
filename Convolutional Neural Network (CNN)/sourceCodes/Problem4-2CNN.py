# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 11:56:43 2019
    7291 training observations and 2007 test observations,  as follows:
             0    1   2   3   4   5   6   7   8   9 Total
    Train 1194 1005 731 658 652 556 664 645 542 644 7291
     Test  359  264 198 166 200 160 170 147 166 177 2007
FORMAT-
1) digit:256 pixels
    
@author: LAM Ho Sang
25Nov2019 ver 1.0 - first workable version
26Nov2019 ver 2.0 - add CNN version
26Nov2019 ver 2.1 - touch-up to improve recognition accuracy further
"""

import pandas as pd; import numpy as np
import tensorflow as tf ; from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model, to_categorical # plot_model must installed pydot, X.2.
from sklearn.model_selection import KFold # for 3-folds split, A4
import matplotlib.pyplot as plt #; import cv2 as cv
import time

NB_EPOCHS=80 # number of repeating training cycles
BATCH=32 # SGD size per batch
NB_NEURON=128 #16 &8filters were too small with ~6x%
DROPOUT_RATE1=0.25 # 
DROPOUT_RATE2=0.15 # if 0 => no dropout regularisation. 

NB_CLASSES=10
COLUMNS=16; ROWS=16

NB_FILTER =64 # 8 is too small
STRIDES_1ST=(1,1) # pooling filter pixel move step, x= move-x-pixels per step.
STRIDES_2ND=(1,1) # 2,2 degraded a lot in performance

""" Non-common tuning network hyper-parameters """
KERNAL_SIZE =(3,3) # convolution filter kernel size 3x3.
POOL_SIZE =(2,2) # size of pooling area for max pooling. Should be smaller than Kernal size
CHANNELS = 1 # image channel is 1 for gray images, 3 for RGB

def imagesLabel(data, arrayIndex): # process the images indexed by arrayIndex
    dataset=data.iloc[arrayIndex] #get the actual images data from the indexs
    return imagesPreprocesses(dataset) # return reshaped, normalised images and corresponding labels

def imagesPreprocesses(dataset): # return reshaped, normalised images and corresponding labels
    labels=dataset["Digits"] # get all digits labels
    dataset=dataset.drop(columns=["Digits"], axis=1, inplace=False) # delete first label columns
    numberOfImages=len(dataset.index); images=np.ones( (numberOfImages*ROWS*COLUMNS), dtype=float) # init image buffer
    print("No of Image got=",numberOfImages, "Image buffer size=", len(images))
    images=np.array(dataset.values) #get all images pixels into numpy tensor
    
    images=images/images.max() # make sure data normalised, by the max pixel value in the images
    return images, labels


def createModeCNN(NB_FILTER, KERNAL_SIZE, STRIDES_1ST, STRIDES_2ND, POOL_SIZE, rows, columns, CHANNELS): # 2x(Convolute and Pooling layers) + 2x Dense Layers model
    model = tf.keras.Sequential() #Keras's sequential network graph model
    model.add(layers.Conv2D(NB_FILTER, kernel_size=KERNAL_SIZE, padding='same', # https://www.tensorflow.org/api_docs/python/tf/keras/backend/conv2d
                        input_shape=(rows, columns, CHANNELS), 
                        data_format="channels_last", strides=STRIDES_1ST, activation='relu')) 
    model.add(layers.MaxPooling2D(pool_size=POOL_SIZE))
    model.add(layers.Dropout(DROPOUT_RATE1)) # 0.25 => 0.15. Smaller dropout improve prediction accuracy.
#    model.add(layers.SpatialDropout2D(DROPOUT_RATE1)) #A12
    
    model.add(layers.Conv2D(NB_FILTER*2, kernel_size=KERNAL_SIZE, strides=STRIDES_2ND, activation='relu')) #; model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=POOL_SIZE))
    model.add(layers.Dropout(DROPOUT_RATE2)) # 0.25 => 0.15
#    model.add(layers.SpatialDropout2D(DROPOUT_RATE2)) #A12

    model.add(layers.Conv2D(NB_FILTER*2, kernel_size=KERNAL_SIZE, strides=STRIDES_2ND, activation='relu')) #; model.add(layers.Activation('relu'))
#    model.add(layers.MaxPooling2D(pool_size=POOL_SIZE)) # 3rd layer does'nt need pooling reduction at all.
    model.add(layers.Dropout(DROPOUT_RATE2)) # 0.25 => 0.15

    model.add(layers.Flatten())
    model.add(layers.Dense(NB_NEURON, activation='relu'))
    model.add(layers.Dropout(DROPOUT_RATE1)) # 0.5 => 0.25. Smaller dropout improve accuracy but don't add resource.

    model.add(layers.Dense(NB_CLASSES, activation='softmax')) # 10=Number of classes to be classified, O/P layer=softmax

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) # sgd=StochasticGradientDescent 
    return model


#Prepare Train & Test Datas
trainDatas=pd.read_csv("train.txt", sep=' ', header=None); trainDatas.dropna(axis='columns',  inplace=True)
trainDatas.rename(columns={0: "Digits"}, inplace=True)# ; trainDatas.drop(columns=[257], axis=1, inplace=True) # delete last NaN columns
#trainDatas1_5=trainDatas.loc[trainDatas["Digits"].isin([1,5]), : ] #pd.["Digits"].isin([1,5]) ret turth table, get the data according to the truth table.

testDatas=pd.read_csv("test.txt", sep=' ', header=None); testDatas.dropna(axis='columns',  inplace=True)
testDatas.rename(columns={0: "Digits"}, inplace=True)# ; testDatas.drop(columns=[257], axis=1, inplace=True) # delete last NaN columns
#testDatas1_5=testDatas.loc[testDatas["Digits"].isin([1,5]), : ] #pd.["Digits"].isin([1,5]) ret turth table, get the data according to the truth table.

traintestDatas = pd.concat([trainDatas, testDatas]) # add together for 3 folds

# prepare cross validation
noOfFolds=3 # 3 is 3-fold
kfold=KFold(n_splits=noOfFolds, random_state=43, shuffle=True) #instantise from the class KFold with random shuffle
one, two, three = kfold.split(traintestDatas) # splits into 3 sets, A5

i=1 #init as first fold, 1, for starting the loop
for folds in [one, two, three]:
    trainImage, trainLabels=imagesLabel(traintestDatas, folds[0]) #[0] is train datas 
    testImage, testLabels=imagesLabel(traintestDatas, folds[1]) #[1] is test datas
#debug1=trainImage.reshape(trainImage.shape[0], ROWS, COLUMNS) #for viewing images only    
    
    # convert to categorical matrix to fit Tensorflow API, A.11.
    noOfClasses=NB_CLASSES # =10 for problem 4 onwards
    trainLabels.replace([1,5], [0,1], inplace=True) #2-classes is [0,1], cannot [1,5], API requirement, A9
    testLabels.replace([1,5], [0,1], inplace=True)
    trainLabels = to_categorical(trainLabels, noOfClasses) #A7, creat labels as a matrix table. 1 for the right class
    testLabels = to_categorical(testLabels, noOfClasses) #
     
#    modelA1 = createModelA(ROWS*COLUMNS, noOfClasses) # rebuild mode everytime
        ### Few lines for tensorflow CNN API only ###
    modelA1 = createModeCNN(NB_FILTER, KERNAL_SIZE, STRIDES_1ST, STRIDES_2ND, POOL_SIZE, COLUMNS, ROWS, CHANNELS) #CNN ver, A11
    trainImage=trainImage.reshape(trainImage.shape[0], ROWS, COLUMNS, CHANNELS) # TF need channel tensor even 1channel only
    testImage=testImage.reshape(testImage.shape[0], ROWS, COLUMNS, CHANNELS) # TF need channel tensor even 1channel only

        #############################################
        
    filename="modelA1CNN-P4Fold"+str(i)
    modelA1.summary(); plot_model(modelA1, to_file=filename+".png", show_shapes=True, show_layer_names=True) 
    start=time.time(); fitHistory=modelA1.fit(trainImage, trainLabels, batch_size=BATCH, epochs=NB_EPOCHS, verbose=0)  # fit dataset, A6
    stop=time.time()-start; print("Training time:", stop,"s")  # realtime used in model.fit
    scoreTest = modelA1.evaluate(testImage, testLabels, batch_size=BATCH, verbose=0) # evaluate fitted model's performance. Verbose:0,1,2=silent,progress bar,one line per epoch. 
    probability = modelA1.predict(testImage, batch_size=BATCH, verbose=0) # use it to predict testing data
    plt.plot(fitHistory.history['acc']); plt.title('Train Accuracy & Loss'); plt.ylabel('Accuracy'); plt.xlabel('Epoch') #; plt.show()
    plt.plot(fitHistory.history['loss']); plt.ylabel('Acc & Loss'); plt.legend(['Acc', 'Loss'], loc='upper left'); plt.show()


    modelA1.save(filename+".h5"); print("Saved network model as "+ filename+".h5") # Save entire model to a HDF5 file  # A.8
    print("\nTest set Accurracy =", scoreTest[1])    
            
    scoreTrain = modelA1.evaluate(trainImage, trainLabels, batch_size=BATCH, verbose=0) # evaluate fitted model's performance. Verbose:0,1,2=silent,progress bar,one line per epoch. 
    print("Train sets Accurracy =", scoreTrain[1])
   
    # For full out sample error
    trainTestImage=np.vstack((trainImage,testImage))
    trainTestLabels=np.vstack((trainLabels,testLabels))
    scoreTrainTest = modelA1.evaluate(trainTestImage, trainTestLabels, batch_size=BATCH, verbose=0) # evaluate fitted model's performance. Verbose:0,1,2=silent,progress bar,one line per epoch. 
    print("Test&Train sets Accurracy =", scoreTrainTest[1])
    print("Above is using fold ",i," datas", "\n"); i+=1 # go to next fold
    
print("Parameters are:", "NB_EPOCHS=", NB_EPOCHS, "NB_FILTER=",NB_FILTER, 
                   "NB_NEURON=",NB_NEURON, "FIT BATCH=",BATCH, "STRIDES_1ST=",STRIDES_1ST,"STRIDES_2ND=", STRIDES_2ND,"KERNAL_SIZE=",KERNAL_SIZE,
                   "POOL_SIZE=",POOL_SIZE, "NB_CLASSES=",NB_CLASSES)

"""
Reference A
    1) reshape() - https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.reshape.html#numpy.ndarray.reshape
    2) relu - Activation functions available: https://keras.io/activations/
    3) keras.Sequential() - https://keras.io/getting-started/sequential-model-guide/
    4) Keras Tutorial: https://www.pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/
    5) k-fold validation(k=3) - https://machinelearningmastery.com/k-fold-cross-validation/
    6) Keras sequential fit method: https://keras.io/models/sequential/#fit  .Verbose:0,1,2=silent,progress bar,one line per epoch. 
    7) Keras .to_categorical(): https://keras.io/utils/#to_categorical. 2 classes is [0,1], cannot [1,5]
    8) model.save() - https://www.tensorflow.org/tutorials/keras/save_and_load
    9) pd.replace() - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html)
    10) softmax - https://keras.io/getting-started/sequential-model-guide/
    11) CNN, Convolutional Neural Network - https://keras.io/layers/convolutional/
    12) Spatial Dropout - https://keras.io/layers/core/#spatialdropout2d
"""

