# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:58:36 2019
Filename:   PLA.py
Input:      train.txt
Output:     Plots
Reference are at end of this file
    
Main:   Take “train.txt” as input and then output the learned weight using PLA.
        Output a weight vector <w0,w1,w2> and a plot contains all training data points and the line (represented by <w0,w1,w2>) in black color. 
        Blue circle represents positive labeled data points and red cross refers to negative labels.
 
Usage: >PLA or PLA train.txt 
       train.txt is generated from DataEmit.py 

IDE:    Ref2 1-2) in DataEmit.py. It is Spyder IDE under Anaconda with Python 3.7 in Win10

Version 3.0: This version also work under Jupyter Notebook

Note: weights update is made to be adaptive for better-convergence/less-oscillate but disabled by "#" on 21Sep2019

@author: LAM Ho Sang, 91870012
"""
import numpy as np # for faster vector processing
import pandas as pd # for cleaner table processing
import matplotlib.pyplot as plt # for plotting
import sys # for command line arguments handling
# from sklearn.utils import shuffle  # for future shuffling of the inputXs-Labels list, e.g. list_1, list_2 = shuffle(list_1, list_2)

class PerceptronX(object):
    def __init__(self, noOfWeights, epochs=2000, learningRate=1.0): # make training-cycles & learn-speed tunable
        self.epochs = epochs
        self.learningRate=learningRate # reserve for future usage only!!
        self.weights = np.zeros(noOfWeights + 1); print("INITIAL WEIGHTS=",self.weights)   
        
#        np.random.seed(); temp=np.random.uniform(-0.05,0.05, (noOfWeights + 1)) # for future better weigth initialisation
#        self.weights +=temp; print("INITIAL WEIGHTS=",self.weights) # after randomize, each run will difference slightly with SAME inputs
        
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights)
        if summation > 0:
          activation = 1 # implement sign()
        else:
          activation = -1 # implement sign()
        return activation

    def train(self, xInputs, yOutputs):
        i=0; cumErrors=-1
        while (i < self.epochs) & (cumErrors!=0):
            cumErrors=0 #accumlated errors
            print("\n           Epoch[",i,"] starting........")
            zipList=zip(xInputs, yOutputs)
            for input, output in zipList: # form 2-tuple, (input,output) by zip. Ref 1A
                input = np.append(1, input) #stack '1' on top for (1)x(w0)
                prediction = self.predict(input)
                error= output - prediction
                if error !=0:
                    cumErrors+=1
#                    lrate=self.learningRate*((self.epochs-i/2)/self.epochs) # for adaptive learning rate
                    lrate=self.learningRate
                    self.weights = self.weights + lrate*np.multiply(output, input)
                    print("- Weight updated are",self.weights, end='\r') #print at same line
            print("\n- Accmulated Error=",cumErrors, ", after Epoch:", i); i+=1
                

""" Main Program Start Here """
noOfArguments=len(sys.argv) # argv[0]=path../PLA.py, [1]=<w0,..>, [2]=m n
if noOfArguments==1:
    filename="train.txt"
elif noOfArguments==2:
    filename=sys.argv[1] # get train.txt
    print ('Number of arguments (included DataEmit):',noOfArguments , ' arguments.')
    print ('Argument List:', sys.argv )
else:
    print("Input arguments mismatch error. Use default FILE train.txt !!!"); filename="train.txt"
    

input("PLA started and reading "+ filename+ " .... and press ENTER to continue")
dataset=pd.read_csv("train.txt", sep=' ', header=None, names =['x1', 'x2','Label'])               
f = lambda x: 1 if x=="+" else -1; dataset['Label'] = dataset['Label'].map(f) # convert "+,-" to 1,-1

trainingInputs = np.array( dataset[ ['x1', 'x2'] ].values, dtype="float32" ) # turn to np array vector
trainingLabels = np.array( dataset[ ['Label'] ].values, dtype="float32" )

PerceptronX = PerceptronX(trainingInputs.shape[1]) # trainingInputs.shape[1] 2nd dim of train datas
print("Trainings....")
PerceptronX.train(trainingInputs, trainingLabels)
print("\nWeights after all Epochs done!", PerceptronX.weights)


pos=dataset[dataset.Label==1]; neg=dataset[dataset.Label==-1]
posX=pos["x1"].values; posY=pos["x2"].values; m=len(posX)
negX=neg["x1"].values; negY=neg["x2"].values; n=len(negX)

SIZE=0.5
plt.figure(figsize=(5,4),dpi=150)
plt.title("Data Points and the line")
plt.xlabel('x1'); plt.ylabel('x2')
plt.scatter(posX,posY, label='+ve Data Points', s=SIZE); plt.scatter(negX,negY, label='-ve Data Points', s=SIZE)

x1 = np.arange(-50,50,1)
x2 = -(PerceptronX.weights[0]+PerceptronX.weights[1]*x1)/PerceptronX.weights[2] # 5,2,3
plt.plot(x1,x2, label='Line wo+w1*x1+w2*x2')
plt.legend() # Place a legend on the axes
filename="PLAplot"+str(int(PerceptronX.weights[0]))+ "_"+ str(int(PerceptronX.weights[1]))+ "_"+ str(int(PerceptronX.weights[2]))+ "   "+str(m)+ "_"+ str(n)
plt.savefig(filename)   

           
"""
Ref 1)
    A.https://www.programiz.com/python-programming/methods/built-in/zip
"""