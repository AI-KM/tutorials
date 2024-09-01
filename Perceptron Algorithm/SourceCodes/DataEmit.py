# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:58:36 2019

Filename:   DataEmit.py
Input:      <w0,w1,w2> m n
Output:     train.xlsx & .txt
Reference are at end of this file
    
Main:   Generate data points x=<x1,x2> with sign (w1x1+w2x2+w0)>0 or (w1x1+w2x2+w0)<0. 
        Negative sign label is “-“, positive sign label is “+”. 
 
Usage: >DataEmit or DataEmit <w0,w1,w2> m n
        where <w0,w1,w2> species the line. m is the number of points with label “+”, n is the number of points with label “-“. 
        w0,w1,w2 are separated by just “,”, no extra space is allowed. 

IDE:   Read readme.txt .  It is Spyder IDE under Anaconda with Python 3.7 in Win10. 

Version 3.0: This version also work under Jupyter Notebook

@author: LAM Ho Sang, 91870012

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