# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 17:07:48 2019


@author: Admin
"""
import pandas as pd; import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical # plot_model must installed pydot, X.2.

# THESE TWO MUST same as those in Problem4-2CNN.py
COLUMNS=16; ROWS=16
CHANNELS = 1; NB_CLASSES=10; BATCH=32
def imagesPreprocesses(dataset): # return reshaped, normalised images and corresponding labels
    labels=dataset["Digits"] # get all digits labels
    dataset=dataset.drop(columns=["Digits"], axis=1, inplace=False) # delete first label columns
    numberOfImages=len(dataset.index); images=np.ones( (numberOfImages*ROWS*COLUMNS), dtype=float) # init image buffer
    print("No of Image got=",numberOfImages, "Image buffer size=", len(images))
    images=np.array(dataset.values) #get all images pixels into numpy tensor
    
    images=images/images.max() # make sure data normalised, by the max pixel value in the images
    return images, labels


print("System settings for just in case mismatch problem checking-")
print("tf ver= 1.14.0  ,Keras ver= 2.2.4-tf")
print("Python: 3.7.4 (default, Aug  9 2019, 18:34:13) [MSC v.1915 64 bit (AMD64)] Ver: sys.version_info(major=3, minor=7, micro=4, releaselevel='final', serial=0)\n" )


print("\nDefault verify images file Name is verify.txt !!" ); print("Default NeuralNet Model Name is modelA1CNN-P4F3-995.h5 !!" )

imageFileName=input("Press ENTER to continue or Input your verify image file name xxx.txt :")
if len(imageFileName)==0: imageFileName="verify.txt"

modelFileName=input("Press ENTER to continue or Input other TF Keras Model File name xxx.h5 :")
if len(modelFileName)==0: modelFileName="modelA1CNN-P4F3-995.h5" # default model file name

verifyDatas=pd.read_csv(imageFileName, sep=' ', header=None); verifyDatas.dropna(axis='columns',  inplace=True)
verifyDatas.rename(columns={0: "Digits"}, inplace=True)
noOfClasses=NB_CLASSES

verifyImages, verifyLabels=imagesPreprocesses(verifyDatas)
verifyImages=verifyImages.reshape(verifyImages.shape[0], ROWS, COLUMNS, CHANNELS)
verifyLabels.replace([1,5], [0,1], inplace=True) 
verifyLabels = to_categorical(verifyLabels, noOfClasses)

modelImported = tf.keras.models.load_model("modelA1CNN-P4F3-995.h5")
results=modelImported.predict_classes(verifyImages, verbose=2)
print("Predicted Results are", results)

scoreVerify= modelImported.evaluate(verifyImages, verifyLabels, batch_size=BATCH, verbose=1) # evaluate fitted model's performance. Verbose:0,1,2=silent,progress bar,one line per epoch. 
print("Verify set Accurracy =", scoreVerify[1]*100, "%")