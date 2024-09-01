# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 11:56:43 2019
    7291 training observations and 2007 test observations,  as follows:
             0    1   2   3   4   5   6   7   8   9 Total
    Train 1194 1005 731 658 652 556 664 645 542 644 7291
     Test  359  264 198 166 200 160 170 147 166 177 2007
FORMAT-
digit:feature1:feature2.   Feature 1 is the intensity and feature 2 is the symmetry feature.

@author: Admin
25Nov2019 ver 1.0 - first workable version
"""

import pandas as pd; import numpy as np
import tensorflow as tf ; from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model, to_categorical # plot_model must installed pydot, X.2.
# import sklearn.model_selection as sk m 
from sklearn.model_selection import KFold # for 3-folds split, A4
import matplotlib.pyplot as plt #; import cv2 as cv
#import hw4Library as h4 #homework 4's shared libraries

NB_EPOCHS=20 # number of repeating training cycles
BATCH=16 # SGD size per batch
NB_NEURON=2
DROPOUT_RATE1=0.5 #0 => no dropout regularisation. Some REGULARISATION is necessary for more reasonable result.


def createModelA(intensitySymmetry, noOfClasses): # 1 hidden layer NN
    model = tf.keras.Sequential() #Keras's sequential network graph model

    model.add(layers.Dense(NB_NEURON, input_shape=(intensitySymmetry.shape[1],), activation='softsign')) #use softsign to approx sign instead, A.2
    model.add(layers.Dropout(DROPOUT_RATE1)) # DROPOUT_RATE1=0 => no dropout regularisation

#    model.add(layers.Flatten()) # must flatten if not linearized image datas
    model.add(layers.Dense(noOfClasses, activation='softsign')) #output noOFClasses

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) # sgd=StochasticGradientDescent better than MSE, use it
    return model

def featuresLabel(data, arrayIndex): # process the images indexed by arrayIndex
    dataset=data.iloc[arrayIndex] #get the actual images data from the indexs

    return featuresPreprocesses(dataset) # return reshaped, normalised images and corresponding labels

def featuresPreprocesses(dataset): # return reshaped, normalised images and corresponding labels
    labels=dataset["Digits"] # get all digits labels
    features=dataset[["Intensity","Symmetry"]] #get only features
    features["Intensity"]=features["Intensity"]/features["Intensity"].max() #normalize intensities values by it's max value
    features["Symmetry"]=features["Symmetry"]/features["Symmetry"].max() #normalize symmetry values by it's max value
    
    numberOfImages=len(features.index)
    featuresBufffer=np.ones( numberOfImages*(len(features.columns)) , dtype=float) # init data buffer
    print("No of Image got=",numberOfImages, "Image buffer size=", len(featuresBufffer))
    featuresBufffer=np.array(features.values) #converts all features into numpy tensor
    
    return featuresBufffer, labels

#Prepare Train & Test Datas
trainDatas=pd.read_csv("featuresTrain.txt", sep=' ', header=None); trainDatas.dropna(axis='columns',  inplace=True)
trainDatas.rename(columns={3: "Digits", 6: "Intensity", 8:"Symmetry"}, inplace=True)
trainDatas1_5=trainDatas.loc[trainDatas["Digits"].isin([1,5]), : ] #pd.["Digits"].isin([1,5]) ret turth table, get the data according to the truth table.

testDatas=pd.read_csv("featuresTest.txt", sep=' ', header=None); testDatas.dropna(axis='columns',  inplace=True)
testDatas.rename(columns={3: "Digits", 6: "Intensity", 8:"Symmetry"}, inplace=True)
testDatas1_5=testDatas.loc[testDatas["Digits"].isin([1,5]), : ] #pd.["Digits"].isin([1,5]) ret turth table, get the data according to the truth table.
trainTestDatas1_5 = pd.concat([trainDatas1_5, testDatas1_5]) # add together for 3 folds

# prepare cross validation
noOfFolds=3 # 3 is 3-fold
kfold=KFold(n_splits=noOfFolds, random_state=43, shuffle=False) #instantise from the class KFold with random shuffle
one, two, three = kfold.split(trainTestDatas1_5) # splits into 3 sets, A5

i=1 #init as first fold, 1, for starting the loop
for folds in [one, two, three]:
    trainFeatures, trainLabels=featuresLabel(trainTestDatas1_5, folds[0]) #[0] is train datas 
    testFeatures, testLabels=featuresLabel(trainTestDatas1_5, folds[1]) #[1] is test datas
    
    # convert to categorical matrix to fit Tensorflow API, A.11.
    noOfClasses=2
    trainLabels.replace([1,5], [0,1], inplace=True) #2-classes is [0,1], cannot [1,5], API requirement, A9
    testLabels.replace([1,5], [0,1], inplace=True)
    trainLabels = to_categorical(trainLabels, noOfClasses) #A7, creat labels as a matrix table. 1 for the right class
    testLabels = to_categorical(testLabels, noOfClasses) #
    
    modelA1 = createModelA(trainFeatures, noOfClasses) # rebuild mode everytime
    filename="modelA1-P2Fold"+str(i)
    modelA1.summary(); plot_model(modelA1, to_file=filename+".png", show_shapes=True, show_layer_names=True) 
    
    fitHistory=modelA1.fit(trainFeatures, trainLabels, batch_size=BATCH, epochs=NB_EPOCHS, verbose=1)  # fit dataset, A6
    scoreTest = modelA1.evaluate(testFeatures, testLabels, batch_size=BATCH, verbose=0) # evaluate fitted model's performance. Verbose:0,1,2=silent,progress bar,one line per epoch. 
    probability = modelA1.predict(testFeatures, batch_size=BATCH, verbose=0) # use it to predict testing data
    plt.plot(fitHistory.history['acc']); plt.title('Train Accuracy'); plt.ylabel('Accuracy'); plt.xlabel('Epoch'); plt.show()
    modelA1.save(filename+".h5"); print("Saved network model as "+ filename+".h5") # Save entire model to a HDF5 file  # A.8
    print("\nTest set Accurracy by [2:",str(NB_NEURON),":2]=", scoreTest[1])    
    
    scoreTrain = modelA1.evaluate(trainFeatures, trainLabels, batch_size=BATCH, verbose=0) # evaluate fitted model's performance. Verbose:0,1,2=silent,progress bar,one line per epoch. 
    print("Train sets Accurracy =", scoreTrain[1])
  
    
    # For full out sample error
    trainTestFeatures=np.vstack((trainFeatures,testFeatures))
    trainTestLabels=np.vstack((trainLabels,testLabels))
    scoreTrainTest = modelA1.evaluate(trainTestFeatures, trainTestLabels, batch_size=BATCH, verbose=0) # evaluate fitted model's performance. Verbose:0,1,2=silent,progress bar,one line per epoch. 
    print("Test&Train sets Accurracy by [2:",str(NB_NEURON),":2]=", scoreTrainTest[1])
    print("Using fold ",i," datas"); i+=1 # go to next fold

#    cv.waitKey(1000)  # mSeconds, 0=forever

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
"""

