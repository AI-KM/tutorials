# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 22:10:02 2019

@author: Admin
"""
import pandas as pd

verifyDatas=pd.read_csv("test.txt", sep=' ', header=None)
temp=pd.read_csv("train.txt", sep=' ', header=None)
temp = pd.concat([verifyDatas, temp])
temp=temp.sample(frac=0.05)


temp.to_csv("verify.txt", sep=' ', index=False, header=False)