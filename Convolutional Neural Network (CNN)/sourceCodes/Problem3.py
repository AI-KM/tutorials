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
26Nov2019 ver 1.0 - improve plots
"""

import pandas as pd; import numpy as np
import tensorflow as tf ; from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model, to_categorical # plot_model must installed pydot, X.2.
from sklearn.model_selection import KFold # for 3-folds split, A4
import matplotlib.pyplot as plt #; import cv2 as cv


NB_EPOCHS=20 # number of repeating training cycles
BATCH=32 # SGD size per batch
NB_NEURON=3 # 6 or 3
DROPOUT_RATE1=0.5 # at least 0.5, otherwise not a very stable

COLUMNS=16; ROWS=16

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

def createModelA(pixels, noOfClasses): # 1 hidden layer NN
    model = tf.keras.Sequential() #Keras's sequential network graph model

    model.add(layers.Dense(NB_NEURON, input_shape=(pixels,), activation='softsign')) #use softsign to approx sign instead, A.2
    model.add(layers.Dropout(DROPOUT_RATE1)) # DROPOUT_RATE1=0 => no dropout regularisation

#    model.add(layers.Dense(noOfClasses, activation='softmax')) #output noOFClasses, A10 'softmax' is much better than sign
    model.add(layers.Dense(noOfClasses, activation='softsign'))

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) # sgd=StochasticGradientDescent better than MSE, use it
    return model

#Prepare Train & Test Datas
trainDatas=pd.read_csv("train.txt", sep=' ', header=None); trainDatas.dropna(axis='columns',  inplace=True)
trainDatas.rename(columns={0: "Digits"}, inplace=True)# ; trainDatas.drop(columns=[257], axis=1, inplace=True) # delete last NaN columns
trainDatas1_5=trainDatas.loc[trainDatas["Digits"].isin([1,5]), : ] #pd.["Digits"].isin([1,5]) ret turth table, get the data according to the truth table.

testDatas=pd.read_csv("test.txt", sep=' ', header=None); testDatas.dropna(axis='columns',  inplace=True)
testDatas.rename(columns={0: "Digits"}, inplace=True)# ; testDatas.drop(columns=[257], axis=1, inplace=True) # delete last NaN columns
testDatas1_5=testDatas.loc[testDatas["Digits"].isin([1,5]), : ] #pd.["Digits"].isin([1,5]) ret turth table, get the data according to the truth table.

trainTestDatas1_5 = pd.concat([trainDatas1_5, testDatas1_5]) # add together for 3 folds

# prepare cross validation
noOfFolds=3 # 3 is 3-fold
kfold=KFold(n_splits=noOfFolds, random_state=43, shuffle=False) #instantise from the class KFold with random shuffle
one, two, three = kfold.split(trainTestDatas1_5) # splits into 3 sets, A5

i=1 #init as first fold, 1, for starting the loop
for folds in [one, two, three]:
    trainImage, trainLabels=imagesLabel(trainTestDatas1_5, folds[0]) #[0] is train datas 
    testImage, testLabels=imagesLabel(trainTestDatas1_5, folds[1]) #[1] is test datas
#debug1=trainImages.reshape(len(one[0]), COLUMNS,ROWS) #for viewing images only    
    
    # convert to categorical matrix to fit Tensorflow API, A.11.
    noOfClasses=2
    trainLabels.replace([1,5], [0,1], inplace=True) #2-classes is [0,1], cannot [1,5], API requirement, A9
    testLabels.replace([1,5], [0,1], inplace=True)
    trainLabels = to_categorical(trainLabels, noOfClasses) #A7, creat labels as a matrix table. 1 for the right class
    testLabels = to_categorical(testLabels, noOfClasses) #
    
    modelA1 = createModelA(ROWS*COLUMNS, noOfClasses) # rebuild mode everytime
    filename="modelA1-N"+str(NB_NEURON)+"-P3Fold"+str(i)
    modelA1.summary(); plot_model(modelA1, to_file=filename+".png", show_shapes=True, show_layer_names=True) 
    
    fitHistory=modelA1.fit(trainImage, trainLabels, batch_size=BATCH, epochs=NB_EPOCHS, verbose=1)  # fit dataset, A6
    scoreTest = modelA1.evaluate(testImage, testLabels, batch_size=BATCH, verbose=0) # evaluate fitted model's performance. Verbose:0,1,2=silent,progress bar,one line per epoch. 
    probability = modelA1.predict(testImage, batch_size=BATCH, verbose=0) # use it to predict testing data
    plt.plot(fitHistory.history['acc']); plt.title('Train Accuracy & Loss'); plt.ylabel('Accuracy'); plt.xlabel('Epoch') #; plt.show()
    plt.plot(fitHistory.history['loss']); plt.ylabel('Acc & Loss'); plt.legend(['Acc', 'Loss'], loc='upper left'); plt.show()

    modelA1.save(filename+".h5"); print("Saved network model as "+ filename+".h5") # Save entire model to a HDF5 file  # A.8
    print("\nTest set Accurracy by [256:",str(NB_NEURON),":2]=", scoreTest[1])    
    
    scoreTrain = modelA1.evaluate(trainImage, trainLabels, batch_size=BATCH, verbose=0) # evaluate fitted model's performance. Verbose:0,1,2=silent,progress bar,one line per epoch. 
    print("Train sets Accurracy =", scoreTrain[1])
    
    # For full out sample error
    trainTestImage=np.vstack((trainImage,testImage))
    trainTestLabels=np.vstack((trainLabels,testLabels))
    scoreTrainTest = modelA1.evaluate(trainTestImage, trainTestLabels, batch_size=BATCH, verbose=0) # evaluate fitted model's performance. Verbose:0,1,2=silent,progress bar,one line per epoch. 
    print("Test&Train sets Accurracy by [256:",str(NB_NEURON),":2]=", scoreTrainTest[1], "\n")
    print("Using fold ",i," datas"); i+=1 # go to next fold
    
#    cv.waitKey(5000)  # mSeconds, 0=forever

print("Parameters are:", "DROPOUT_RATE", DROPOUT_RATE1, "NB_EPOCHS=", NB_EPOCHS, "NB_NEURON=",NB_NEURON, "FIT BATCH=",BATCH, )


"""
Reference A
    1) reshape() - https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.reshape.html#numpy.ndarray.reshape
    2) Activation functions available: https://keras.io/activations/
    3) keras.Sequential() - https://keras.io/getting-started/sequential-model-guide/
    4) Keras Tutorial: https://www.pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/
    5) k-fold validation(k=3) - https://machinelearningmastery.com/k-fold-cross-validation/
    6) Keras sequential fit method: https://keras.io/models/sequential/#fit  .Verbose:0,1,2=silent,progress bar,one line per epoch. 
    7) Keras .to_categorical(): https://keras.io/utils/#to_categorical. 2 classes is [0,1], cannot [1,5]
    8) model.save() - https://www.tensorflow.org/tutorials/keras/save_and_load
    9) pd.replace() - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html)
    10) softmax - https://keras.io/getting-started/sequential-model-guide/
"""

