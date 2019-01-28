import scipy.io as sio 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
testindex = sio.loadmat('../data/IS16/testIndexCorsican.mat')
trainindex = sio.loadmat('../data/IS16/trainIndexCorsican.mat')

testindex = testindex['imageNumber'][0]
trainindex= trainindex['imageNumber'][0]

allindex1 = [0]*( max(max(trainindex),max(testindex))+1)
allindex2 = [0]*( max(max(trainindex),max(testindex))+1)
for v in testindex:
    allindex1[v]=1
for v in trainindex:
    allindex2[v]=1

try:
    os.mkdir("../out/IS16")
except:
    pass

plt.title("visulization of train set and test set description")
plt.xlabel("sample ID")
plt.ylabel("selected / not")
plt.plot(range(len(allindex1)),allindex1,'b',linewidth=1,label='test sample ID')
plt.plot(range(len(allindex2)),allindex2,'r',linewidth=0.05,label='train sample ID')
plt.legend()
plt.savefig("../out/IS16/traintestset.png")
