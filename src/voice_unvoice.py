from ssi_cnn import *
import wave
import numpy as np
from spectrum import linear_prediction as lpd
import time
import random
import soundfile as sf
import tensorflow as tf
import cv2
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
from sklearn.metrics import log_loss
def keras_train_pred_uv(name = 'lipstongue2uv_',egg=False,\
                          path="../out/test_lsf/lsf_hamming_16kHZ.pkl",classification=2, \
                          IS16=True,multi_task=True, optimizer='adam', lr=0.0001, AE=True,epochs=30,order=1,stepbystep=False):

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

    train_y = np.array([1]*len(trainegg))
    test_y = np.array([1]*len(testegg))


    tmpdata = pd.DataFrame(trainegg)
    tmpdata = tmpdata[tmpdata[0] > 0.6]
    tmpdata = tmpdata[tmpdata[0] < 1.9]
    tmpdata = tmpdata[tmpdata[1] > 60]
    tmpdata = tmpdata[tmpdata[1] < 173]
    train_y[tmpdata.index.tolist()] = 0

    tmpdata = pd.DataFrame(testegg)
    tmpdata = tmpdata[tmpdata[0] > 0.6]
    tmpdata = tmpdata[tmpdata[0] < 1.9]
    tmpdata = tmpdata[tmpdata[1] > 60]
    tmpdata = tmpdata[tmpdata[1] < 173]
    test_y[tmpdata.index.tolist()] = 0


    # target preprocessing
    train_ys, test_ys = target_preprocessing(train_y[:,np.newaxis], test_y[:,np.newaxis], classification,order=order)

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
            logloss = log_loss(test_y,predict[:,1])
            print(logloss)


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
            hist = model.fit([train_tongue, train_lips], train_ys[i],
                             validation_split=0.05,
                             batch_size=512, epochs=epochs, shuffle=True, \
                             callbacks=[earlystop, model_checkpoint])
            bst_epoch_inter = np.argmax(hist.history['val_loss'])
            trn_loss += hist.history['loss'][bst_epoch_inter]
            val_loss += hist.history['val_loss'][bst_epoch_inter]
            bst_epoch += (bst_epoch_inter + 1)

            model.load_weights(model_path)
            predict_iter = model.predict([test_tongue, test_lips], batch_size=512)
            predict.append(predict_iter)

        bst_epoch -= 1


        sd, is16sd, predict_lsf = measure(test_lsf, predict, train_lsf, classification,order=order)
        f = open("../out/%s/predict_is16sd%.3f.pkl" % (EXP_NAME, is16sd), 'wb')
        pickle.dump(predict_lsf, f)
        f.close()
    res = '%s,%s,%s,%s,%s,%d,%.6f,%.6f,%s,%s \n' % (timenow, 'multi', EXP_NAME, str(classification   ), str(is16sd), bst_epoch + 1, trn_loss, val_loss, str(IS16), name)
    f_record.write(res)
    f_record.close()

    return is16sd

if __name__ == "__main__":

    #keras_train_pred_freq(epochs=200,classification=0,AE=False,name='lipstongue2Fzero_init',stepbystep=True,multi_task=True)
    keras_train_pred_uv(epochs=200, AE=False, name='lipstongue2uv',
                          multi_task=True,order=1)
