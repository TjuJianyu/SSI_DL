from ssi_cnn import *
import wave
import numpy as np
from spectrum import linear_prediction as lpd
import time
import random
#import soundfile as sf
import tensorflow as tf
#import cv2
from audiolazy import lpc
from matplotlib import pyplot as plt
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
import pandas as pd

def keras_train_pred_freq(name = 'lipstongue2Fzero_',egg=False,audioonly=True,\
                          path="../out/test_lsf/lsf_hamming_16kHZ.pkl",classification=0, \
                          IS16=True,multi_task=True, optimizer='adam', lr=0.0001, AE=False,epochs=30,order=1,stepbystep=False,\
                         inverse_target = False 
                         ):
    print(optimizer)
    print(multi_task)
    EXP_NAME = name
    try:
        os.mkdir("../out/%s" % EXP_NAME)
    except:
        pass
    # load data
    train_tongue, train_lips, _, test_tongue, test_lips, _ = load_dataset(path, IS16, False)
    print(train_tongue.shape)
    print(train_lips.shape)

    data_dir = '../out/'

    if egg:
        f = open(data_dir+'traineggfeat_fit.pkl','rb')
    else:
        print('from song')
        f = open(data_dir+'trainsongfeat_fit.pkl','rb')
    trainegg = pickle.load(f)
    f.close()

    if egg:
        f = open(data_dir+'testeggfeat_fit.pkl','rb')
    else:
        print('from song')
        f = open(data_dir + 'testsongfeat_fit.pkl', 'rb')

    testegg = pickle.load(f)
    f.close()

    trainegg = trainegg[:,[0,1,3]]
    testegg = testegg[:,[0,1,3]]

    if audioonly:

        tmpdata = pd.DataFrame(trainegg)
        tmpdata = tmpdata[tmpdata[0] > 0.6]
        tmpdata = tmpdata[tmpdata[0] < 1.9]
        tmpdata = tmpdata[tmpdata[1] > 60]
        tmpdata = tmpdata[tmpdata[1] < 173]
        train_tongue = train_tongue[tmpdata.index.tolist()]
        train_lips   = train_lips[tmpdata.index.tolist()]
        trainegg = trainegg [tmpdata.index.tolist()]

        tmpdata = pd.DataFrame(testegg)
        tmpdata = tmpdata[tmpdata[0] > 0.6]
        tmpdata = tmpdata[tmpdata[0] < 1.9]
        tmpdata = tmpdata[tmpdata[1] > 60]
        tmpdata = tmpdata[tmpdata[1] < 173]
        test_tongue = test_tongue[tmpdata.index.tolist()]
        test_lips = test_lips[tmpdata.index.tolist()]
        testegg = testegg[tmpdata.index.tolist()]

        mean = trainegg.mean(axis=0)
        std = trainegg.std(axis=0)
        print(mean,std)
        print(trainegg.shape)
        print(testegg.shape)
    else:

        unsineval = [0.6 - (1.9 - 0.6) / (classification - 1), 60. - (173. - 60.) / (classification - 1),
                     -1. - (1. - (-1)) / (classification - 1)]
        trainegg[trainegg[:, 0] < 0.6] = unsineval
        trainegg[trainegg[:, 0] > 1.9] = unsineval
        trainegg[trainegg[:, 1] < 60] = unsineval
        trainegg[trainegg[:, 1] > 173] = unsineval
        trainegg[trainegg[:, 2] < -1] = unsineval
        trainegg[trainegg[:, 2] > 1] = unsineval

        testegg[testegg[:, 0] < 0.6] = unsineval
        testegg[testegg[:, 0] > 1.9] = unsineval
        testegg[testegg[:, 1] < 60] = unsineval
        testegg[testegg[:, 1] > 173] = unsineval
        testegg[testegg[:, 2] < -1] = unsineval
        testegg[testegg[:, 2] > 1] = unsineval
    
    trainegg = trainegg[:,[1]]
    testegg = testegg[:,[1]]
    # target preprocessing
    train_ys, test_ys = target_preprocessing(trainegg, testegg, classification,order=order)

    if inverse_target:
        print('train & test target min')
        print(np.array(train_ys).min())
        print(np.array(test_ys).min())
        train_ys = 16000 / np.array(train_ys)
        test_ys = 16000/ np.array(test_ys) 
        
    # load model
    model = cnn_model_keras(conv3d=False, classification=classification, multi_task=multi_task, AE=AE,order=order)
    print(model.summary())

    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr=lr)
    else:
        optimizer = tf.keras.optimizers.RMSprop(lr=lr)

    # compile
    model_compile(model, optimizer, classification, multi_task, AE=AE,order=order)

    timenow = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    model_path = '../out/%s/bst_model.h5' % (EXP_NAME)

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, \
                                                          save_weights_only=True)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, mode='min')

    f_record = open("record.csv", "a")

    if multi_task:

        if AE:
            train_ys.append(train_tongue)
            train_ys.append(train_lips)

        if stepbystep:
            f = open("../out/%s/predict_index.pkl" % (EXP_NAME), 'wb')
            pickle.dump(tmpdata.index.tolist(), f)
            f.close()
            for i in range(epochs):
                hist = model.fit([train_tongue, train_lips], train_ys,
                                 validation_split=0.05, \
                                 verbose=2, \
                                 batch_size=512, epochs=i+1, initial_epoch=i, shuffle=True)

                predict = model.predict([test_tongue, test_lips], batch_size=512)
                print(i)
                f = open("../out/%s/predict_epoch%d.pkl" % (EXP_NAME,i), 'wb')
                pickle.dump(predict, f)
                f.close()







        else:
            hist = model.fit([train_tongue, train_lips], train_ys,#[trainegg[:,0],trainegg[:,1],trainegg[:,2]], \
                             validation_split=0.05, \
                             verbose=2, \
                             batch_size=512, epochs=epochs, shuffle=True, \
                             callbacks=[earlystop, model_checkpoint])


            model.load_weights(model_path)
            predict = model.predict([test_tongue, test_lips], batch_size=512)
            f = open("../out/%s/predict.pkl" % (EXP_NAME), 'wb')
            pickle.dump(predict, f)
            f.close()
            f = open("../out/%s/predict_index.pkl" % (EXP_NAME), 'wb')
            pickle.dump(tmpdata.index.tolist(), f)
            f.close()



    else:
        predict = []
        bst_epoch = 0
        trn_loss = 0
        val_loss = 0
        initmodelpath = '../out/%s/init_model.h5' % (EXP_NAME)
        tf.keras.models.save_model(model, initmodelpath)
        for i in range(order):
            model = tf.keras.models.load_model(initmodelpath)
            model_path = '../out/%s/bst_lsf%d_model.h5' % (EXP_NAME, i)
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, \
                                                                  save_weights_only=True)
            print(train_ys[i])
            
            hist = model.fit([train_tongue, train_lips], train_ys[i],
                             validation_split=0.05,
                             batch_size=512, epochs=epochs, shuffle=True, \
                             callbacks=[earlystop, model_checkpoint])
            
        for i in range(order):

            model.load_weights('../out/%s/bst_lsf%d_model.h5' % (EXP_NAME,i))
            pred = model.predict([test_tongue, test_lips], batch_size=512)
            f = open('../out/%s/lsf%d.pkl' % (EXP_NAME,i), 'wb')
            pickle.dump(pred,f)
            f.close()
            f = open("../out/%s/predict_index.pkl" % (EXP_NAME), 'wb')
            pickle.dump(tmpdata.index.tolist(), f)
            f.close()
            from scipy.stats import pearsonr

            from sklearn.metrics import mean_squared_error
            print(test_ys[i].shape, pred.shape)
            nmse = mean_squared_error(test_ys[i].flatten(), pred.flatten()) / mean_squared_error(test_ys[i].flatten(), 
                                                                                          [0]*len(test_ys[i].flatten()))
            pcc = pearsonr(test_ys[i].flatten(), pred.flatten())
            
            print('nmse',nmse, 'pcc', pcc)
            print(train_ys[i].flatten().min(), pred.flatten().min())
            nmse = mean_squared_error(16000/test_ys[i].flatten(), 16000/pred.flatten()) / mean_squared_error(16000/test_ys[i].flatten(), [0]*len(test_ys[i].flatten()))
            pcc = pearsonr(16000/test_ys[i].flatten(), 16000/pred.flatten())
            
            print('nmse',nmse, 'pcc', pcc)
            
            
            
            
            
            
            

        # sd, is16sd, predict_lsf = measure(test_lsf, predict, train_lsf, classification,order=order)
        # f = open("../out/%s/predict_is16sd%.3f.pkl" % (EXP_NAME, is16sd), 'wb')
        # pickle.dump(predict_lsf, f)
        # f.close()
    # res = '%s,%s,%s,%s,%s,%d,%.6f,%.6f,%s,%s \n' % (timenow, 'multi', EXP_NAME, str(classification   ), str(is16sd), bst_epoch + 1, trn_loss, val_loss, str(IS16), name)
    # f_record.write(res)
    # f_record.close()

    return is16sd

if __name__ == "__main__":

    #keras_train_pred_freq(epochs=200,classification=0,AE=False,name='lipstongue2Fzero_init',stepbystep=True,multi_task=True)

    #keras_train_pred_freq(epochs=200, classification=0, AE=True, name='lipstongue2Fzero_AE', stepbystep=True, multi_task=True)

    keras_train_pred_freq(epochs=400, classification=0, AE=False, name='lipstongue2Fzero_traineach2', stepbystep=False,
                          multi_task=False)
#     keras_train_pred_freq(epochs=400, classification=0, AE=False, name='lipstongue2Fzero_traineach_invserse', stepbystep=False,
#                           multi_task=False, inverse_target = True )