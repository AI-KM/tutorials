# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 18:35:18 2019

@author: hoson
"""

import numpy as np # for faster vector processing
import pandas as pd # for cleaner table processing
import matplotlib.pyplot as plt # for plotting
import sys # for command line arguments handling

# All constants are defined here
POSITIVE="+"; NEGATIVE="-" 

# All functions are defined here
def userGuideEcho():
    print("This Python generate data points x=<x1,x2> with sign (w1x1+w2x2+w0)>0 or (w1x1+w2x2+w0)<0")
    print("1: Negative sign label is -, positive sign label is + . Where w1,w2,w3 are weights")
    print("2: m is the number of points with label “+”, n is the number of points with label “-“.")
    print("== As an option you can just run DataEmit.py without command line arguments, it will guide you through the process ! ==")

def getUserInputs(displayString): # return float and length
    consoleString=input(displayString)
    vars=consoleString.split(","); length=len(vars)
    return vars, length

def genRandomNo(noOfNumbers):
    np.random.seed()
    f=np.random.uniform(-50,50, int(noOfNumbers))  # reset the seed everytime to make o/p more random
    return f

def calculateX2(x1, w, SIGN):
    x2 = -(w[0]+w[1]*x1)/w[2]  # where w1x1+w2x2+w0
    
    np.random.seed(); shift=np.random.uniform(0.1,10, x1.shape)  # reset the seed everytime to make o/p more random
    if SIGN == POSITIVE:
        x2+=shift # add x2 by a +ve number to make sure > 0
    else: x2-=shift # substract x2 by a +ve number to make sure < 0
    return x2

   

""" Main Program Start Here """
userGuideEcho()
# Check and echo command line arguments to user
noOfArguments=len(sys.argv) # Ref, 1C. argv[0]=path../DataEmit, [1]=<w0,..>, [2]=m n
if noOfArguments !=4:
    vars,length=getUserInputs("\nInput weights as w0,w1,w2(e.g. 0,-1,1/5,2,3), then press ENTER: "); w=[]; i=0
    while i < length: #put all user inputs in w[0],w[1]....
        w.append(float(vars[i])); i+=1
    
    vars,length=getUserInputs("Input number of +ve and -ve labels as m,n(e.g. 5,4/200,200), then press ENTER: ")
    mn4PosNegLabels=[]; i=0
    while i < length: 
        mn4PosNegLabels.append(int(vars[i])); i+=1
        
elif noOfArguments==4:
    print ('Number of arguments (included DataEmit):',noOfArguments , ' arguments.')
    print ('Argument List:', sys.argv )
    weightString=sys.argv[1]; weightString=weightString.replace("<", ""); weightString=weightString.replace(">", "")
    w=[int(x) for x in weightString.split(",")]
    mn4PosNegLabels=[int(sys.argv[2]), int(sys.argv[3])]


print("Your input weights are: ",w)
print("Your input m, n are: ",mn4PosNegLabels)
input('Check inputs and press ENTER to continue...')

posX1=genRandomNo(mn4PosNegLabels[0])
posX2=calculateX2(posX1, w, POSITIVE)

result = pd.DataFrame({'x1':posX1.tolist(), 'x2':posX2.tolist()}); result['Label']="+"
negX1=genRandomNo(mn4PosNegLabels[1])
negX2=calculateX2(negX1, w, NEGATIVE)
temp = pd.DataFrame({'x1':negX1.tolist(), 'x2':negX2.tolist()}); temp['Label']="-"

result=result.append(temp, ignore_index=False); print(result) # put everything in one table only
result.to_csv("train.txt", sep=' ', index=False, header=False)

# For easy of reference with different file names
filename="train "+ str(w[0])+ "_"+ str(w[1])+ "_"+ str(w[2])+ "   "+str(mn4PosNegLabels[0])+ "_"+ str(mn4PosNegLabels[1])+".xlsx"
result.to_excel(filename); print("Outputs save into  train.txt & .xlsx for ref")


SIZE=0.5
plt.figure(figsize=(5,4),dpi=150)
plt.title("Data Points and the line")
plt.xlabel('x1'); plt.ylabel('x2')
plt.scatter(posX1,posX2, label='+ve Data Points', s=SIZE); plt.scatter(negX1,negX2, label='-ve Data Points', s=SIZE)

x1 = np.arange(-50,50,1)
x2 = -(w[0]+w[1]*x1)/w[2] # 5,2,3
plt.plot(x1,x2, label='Line wo+w1*x1+w2*x2')
plt.legend() # Place a legend on the axes
filename="DataEmitPLOT"+ str(int(w[0]))+ "_"+ str(int(w[1]))+ "_"+ str(int(w[2]))+"   "+str(int(mn4PosNegLabels[0]))+ "_"+ str(int(mn4PosNegLabels[1]))
plt.savefig(filename) 


"""
Ref 1)
    A. https://www.anaconda.com/distribution/
    B. https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html
    C: https://stackoverflow.com/questions/2626026/python-sys-argv-lists-and-indexes
       In Spyder, set Run > Configure > command line option
    D: List comprehension: https://www.datacamp.com/community/tutorials/python-list-comprehension?utm_source=adwords_ppc&utm_campaignid=1455363063&utm_adgroupid=65083631748&utm_device=c&utm_keyword=&utm_matchtype=b&utm_network=g&utm_adpostion=1t1&utm_creative=332602034358&utm_targetid=dsa-486527602543&utm_loc_interest_ms=&utm_loc_physical_ms=9069536&gclid=EAIaIQobChMIxffiyMne5AIV0aqWCh0_qgizEAAYASAAEgKjZ_D_BwE
    
    
"""


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