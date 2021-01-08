import wave
import numpy as np
from spectrum import linear_prediction as lpd
import time
import random
import tensorflow as tf
from audiolazy import lpc
import scipy.signal as spsig
import os
from tqdm import tqdm
import pickle
import datetime
from scipy import signal
import math
import argparse
import scipy.io as sio
from utils.lsf_utils import *
import numpy as np
import ssi_cnn as scnn

try:
    os.mkdir('../log/')
except:
    pass
expcount=10
f = open("../log/19_0_04_search_p2.txt",'a')

path = '../out/test_lsf/lsf_hamming_16kHZ.pkl'

classification = 100
multi_task = True
IS16 = True
AE = True
conv3d=False
name = ''
performance = []
for i in range(expcount):
    perf = scnn.keras_train(conv3d=conv3d,
                    classification=classification,
                    multi_task=multi_task,
                    IS16=IS16,
                    AE = AE,
                    name=name,path = path)
    print("is16sd:%.3f" % perf)
    performance.append(perf)
    m,h = confidenceinterval(performance)
f.write("%d,%s,%s,%s,%s,%.5f,%5f \n" % (classification,str(multi_task),str(IS16),str(AE),name,m,h) )





# try:
#     os.mkdir('../log/')
# except:
#     pass
# expcount=1
# f = open("../log/20_05_03_search_11.025khz.txt",'a')
# path = '../out/test_lsf/lsf_hamming_11.025kHZ.pkl'

# classification = 100
# multi_task = True
# IS16 = True
# AE = True
# conv3d=False
# name = ''
# performance = []
# for i in range(expcount):
#     perf = scnn.keras_train(conv3d=conv3d,
#                     classification=classification,
#                     multi_task=multi_task,
#                     IS16=IS16,
#                     AE = AE,
#                     name=name,path = path)
#     print("is16sd:%.3f" % perf)
#     performance.append(perf)
#     m,h = confidenceinterval(performance)
# f.write("%d,%s,%s,%s,%s,%.5f,%5f \n" % (classification,str(multi_task),str(IS16),str(AE),name,m,h) )


#
# classification = 100
# multi_task = True
# IS16 = True
# AE = False
# conv3d=False
# name = ''
# performance = []
# for i in range(expcount):
#     perf = scnn.keras_train(conv3d=conv3d,
#                     classification=classification,
#                     multi_task=multi_task,
#                     IS16=IS16,
#                     AE = AE,
#                     name=name,path = path)
#     print("is16sd:%.3f" % perf)
#     performance.append(perf)
#     m,h = confidenceinterval(performance)
# f.write("%d,%s,%s,%s,%s,%.5f,%5f \n" % (classification,str(multi_task),str(IS16),str(AE),name,m,h) )
#
# classification = 100
# multi_task = False
# IS16 = True
# AE = False
# conv3d=False
# name = ''
# performance = []
# for i in range(expcount):
#     perf = scnn.keras_train(conv3d=conv3d,
#                     classification=classification,
#                     multi_task=multi_task,
#                     IS16=IS16,
#                     AE = AE,
#                     name=name,path = path,
#                            )
#     print("is16sd:%.3f" % perf)
#     performance.append(perf)
#     m,h = confidenceinterval(performance)
# f.write("%d,%s,%s,%s,%s,%.5f,%5f \n" % (classification,str(multi_task),str(IS16),str(AE),name,m,h) )
#
#
# classification = 0
# multi_task = True
# IS16 = True
# AE = False
# conv3d=False
# name = ''
# performance = []
# for i in range(expcount):
#     perf = scnn.keras_train(conv3d=conv3d,
#                     classification=classification,
#                     multi_task=multi_task,
#                     IS16=IS16,
#                     AE = AE,
#                     name=name,path = path,
#                     optimizer='adam' )
#     print("is16sd:%.3f" % perf)
#     performance.append(perf)
#     m,h = confidenceinterval(performance)
# f.write("%d,%s,%s,%s,%s,%.5f,%5f,%s \n" % (classification,str(multi_task),str(IS16),str(AE),name,m,h,'adam') )
#
# classification = 0
# multi_task = False
# IS16 = True
# AE = False
# conv3d=False
# name = ''
# performance = []
# for i in range(expcount):
#     perf = scnn.keras_train(conv3d=conv3d,
#                     classification=classification,
#                     multi_task=multi_task,
#                     IS16=IS16,
#                     AE = AE,
#                     name=name,path = path,
#                     optimizer='adam' )
#     print("is16sd:%.3f" % perf)
#     performance.append(perf)
#     m,h = confidenceinterval(performance)
# f.write("%d,%s,%s,%s,%s,%.5f,%5f,%s \n" % (classification,str(multi_task),str(IS16),str(AE),name,m,h,'adam') )
#
# f.close()
#
#
#

# classification = 100
# multi_task = True
# IS16 = True
# AE = False
# conv3d=False
# name = ''
# performance = []
# for i in range(expcount):
#     perf = scnn.keras_train(conv3d=conv3d,
#                     classification=classification,
#                     multi_task=multi_task,
#                     IS16=IS16,
#                     AE = AE,
#                     name=name,path = path)
#     print("is16sd:%.3f" % perf)
#     performance.append(perf)
#     m,h = confidenceinterval(performance)
# f.write("%d,%s,%s,%s,%s,%.5f,%5f \n" % (classification,str(multi_task),str(IS16),str(AE),name,m,h) )
#
# classification = 100
# multi_task = False
# IS16 = True
# AE = False
# conv3d=False
# name = ''
# performance = []
# for i in range(expcount):
#     perf = scnn.keras_train(conv3d=conv3d,
#                     classification=classification,
#                     multi_task=multi_task,
#                     IS16=IS16,
#                     AE = AE,
#                     name=name,path = path)
#     print("is16sd:%.3f" % perf)
#     performance.append(perf)
#     m,h = confidenceinterval(performance)
# f.write("%d,%s,%s,%s,%s,%.5f,%5f \n" % (classification,str(multi_task),str(IS16),str(AE),name,m,h) )
#
#
# classification = 0
# multi_task = True
# IS16 = True
# AE = False
# conv3d=False
# name = ''
# performance = []
# for i in range(expcount):
#     perf = scnn.keras_train(conv3d=conv3d,
#                     classification=classification,
#                     multi_task=multi_task,
#                     IS16=IS16,
#                     AE = AE,
#                     name=name,path = path)
#     print("is16sd:%.3f" % perf)
#     performance.append(perf)
#     m,h = confidenceinterval(performance)
# f.write("%d,%s,%s,%s,%s,%.5f,%5f \n" % (classification,str(multi_task),str(IS16),str(AE),name,m,h) )
#
# classification = 0
# multi_task = False
# IS16 = True
# AE = False
# conv3d=False
# name = ''
# performance = []
# for i in range(expcount):
#     perf = scnn.keras_train(conv3d=conv3d,
#                     classification=classification,
#                     multi_task=multi_task,
#                     IS16=IS16,
#                     AE = AE,
#                     name=name,path = path)
#     print("is16sd:%.3f" % perf)
#     performance.append(perf)
#     m,h = confidenceinterval(performance)
# f.write("%d,%s,%s,%s,%s,%.5f,%5f \n" % (classification,str(multi_task),str(IS16),str(AE),name,m,h) )
#
# f.close()

# f = open("../log/19_03_05_search.txt",'a')
# path = '../out/test_lsf/lsf_hamming_16kHZ.pkl'
# expcount = 1
# classification = 100
# multi_task = True
# IS16 = True
# AE = True
# conv3d=False
# name = ''
# performance = []
# for i in range(expcount):
#     perf = scnn.keras_train(conv3d=conv3d,
#                     classification=classification,
#                     multi_task=multi_task,
#                     IS16=IS16,
#                     AE = AE,
#                     name=name,path = path,optimizer='adam')
#     print("is16sd:%.3f" % perf)
#     performance.append(perf)
#     m,h = confidenceinterval(performance)
# f.write("%d,%s,%s,%s,%s,%.5f,%5f,adam \n" % (classification,str(multi_task),str(IS16),str(AE),name,m,h) )

# expcount = 1
# classification = 100
# multi_task = True
# IS16 = True
# AE = True
# conv3d=False
# name = ''
# performance = []
# for i in range(expcount):
#     perf = scnn.keras_train(conv3d=conv3d,
#                     classification=classification,
#                     multi_task=multi_task,
#                     IS16=IS16,
#                     AE = AE,
#                     name=name,path = path,optimizer='RMSprop')
#     print("is16sd:%.3f" % perf)
#     performance.append(perf)
#     m,h = confidenceinterval(performance)
# f.write("%d,%s,%s,%s,%s,%.5f,%5f,rmsprop \n" % (classification,str(multi_task),str(IS16),str(AE),name,m,h) )
