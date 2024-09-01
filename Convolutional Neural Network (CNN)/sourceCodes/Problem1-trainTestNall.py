# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 11:56:43 2019
    7291 training observations and 2007 test observations,  as follows:
             0    1   2   3   4   5   6   7   8   9 Total
    Train 1194 1005 731 658 652 556 664 645 542 644 7291
     Test  359  264 198 166 200 160 170 147 166 177 2007
FORMAT-
digit:feature1:feature2.   Feature 1 is the intensity and feature 2 is the symmetry feature.

@author: LAM Ho Sang
24Nov2019 ver 1.0 - first workable version
"""

import pandas as pd
import seaborn as sns


#Prepare Train & Test Datas
trainDatas=pd.read_csv("featuresTrain.txt", sep=' ', header=None); trainDatas.dropna(axis='columns',  inplace=True)
trainDatas.rename(columns={3: "Digits", 6: "Intensity", 8:"Symmetry"}, inplace=True)
#trainDatas["Digits"] = trainDatas["Digits"].astype(str)

testDatas=pd.read_csv("featuresTest.txt", sep=' ', header=None); testDatas.dropna(axis='columns',  inplace=True)
testDatas.rename(columns={3: "Digits", 6: "Intensity", 8:"Symmetry"}, inplace=True)
#testDatas["Digits"] = testDatas["Digits"].astype(str)

#Plot and save the scatter plots
trainPlots = sns.pairplot(trainDatas, height=5, hue="Digits", kind='scatter') #Paired scatter plots, Ref A1
testPlots = sns.pairplot(testDatas, height=5, hue="Digits", kind='scatter') #Paired scatter plots, Ref A1
trainPlots.savefig('P1-trainplots.png')
testPlots.savefig('P1-testplots.png')

""" Combined two for more throughful analysis, prepare-plot-save """
trainTestDatas = pd.concat([trainDatas, trainDatas]) 
trainTestPlots = sns.pairplot(trainTestDatas, height=5, hue="Digits", kind='scatter') #Paired scatter plots, Ref A1
trainTestPlots.savefig('P1-trainTestplots.png')

""" Look at 1&5 ONLY """
#temp=trainDatas["Digits"].isin([1,5]) # return a truth table for row with digits 1&5
trainDatas1_5=trainDatas.loc[trainDatas["Digits"].isin([1,5]), : ] #pd.["Digits"].isin([1,5]) ret turth table, get the data according to the truth table.
trainTestDatas1_5=trainTestDatas.loc[trainTestDatas["Digits"].isin([1,5]), : ] #pd.["Digits"].isin([1,5]) ret turth table, get the data according to the truth table.

trainDatas1_5Plots = sns.pairplot(trainDatas1_5, height=5, hue="Digits", kind='scatter') #Paired scatter plots, Ref A1
trainTestDatas1_5Plots = sns.pairplot(trainTestDatas1_5, height=5, hue="Digits", kind='scatter') #Paired scatter plots, Ref A1

trainDatas1_5Plots.savefig('P1-trainDatas1_5plots.png')
trainTestDatas1_5Plots.savefig('P1-trainTestDatas1_5plots.png')



"""
Reference A
1) pairplots - 
    https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166
    https://seaborn.pydata.org/generated/seaborn.pairplot.html
"""

