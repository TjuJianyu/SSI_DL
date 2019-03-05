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

EXP_NAME = str(random.randint(0, 65536))


def nowtime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def mean_option_score(SD):
    mos = 3.56 - 0.8 * SD + 0.04 * (SD ** 2)
    dmos = mos - (3.56 - 0.8 * 1 + 0.04 * (1 ** 2))
    return mos, dmos


def spectral_distortion(lsf_true, lsf_pred, N, n0, n1):
    SD = []
    IS16SD = []
    print(len(lsf_true))
    print(len(lsf_pred))
    for frameid in range(len(lsf_true)):
        lpc_true = lpd.lsf2poly(lsf_true[frameid])
        lpc_pred = lpd.lsf2poly(lsf_pred[frameid])

        _, freqResponse_true = signal.freqz(b=1, a=lpc_true, worN=N)
        _, freqResponse_pred = signal.freqz(b=1, a=lpc_pred, worN=N)

        freq_th = freqResponse_true[n0 - 1:n1]
        freq_pred = freqResponse_pred[n0 - 1:n1]

        absoluteRadio = (freq_th.real ** 2 + freq_th.imag ** 2) ** 0.5 / (
                    freq_pred.real ** 2 + freq_pred.imag ** 2) ** 0.5

        logValue = np.log10(absoluteRadio ** 2)
        bigsum = ((10 * logValue) ** 2).sum()
        sd = math.sqrt(1.0 / (n1 - n0)) * bigsum
        IS16sd = math.sqrt(1.0 / (n1 - n0) * bigsum)
        SD.append(sd)
        IS16SD.append(IS16sd)

    return SD, IS16SD, sum(SD) * 1.0 / len(SD), sum(IS16SD) * 1.0 / len(IS16SD)


def cnn_model_keras(AE=False, conv3d=False, classification=0, multi_task=True, lw1=1e-7, dropout=0.2,order=12):
    if conv3d:
        tongue_inputs = tf.keras.Input(shape=(8, 48, 48, 1), name='tongue')
        lips_inputs = tf.keras.Input(shape=(8, 48, 48, 1), name='lips')

        layer = tf.keras.layers.Conv3D(filters=16, kernel_size=(5, 5, 4),
                                       kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                       padding='same', activation=tf.nn.relu)(lips_inputs)
        layer = tf.keras.layers.Conv3D(filters=16, kernel_size=(5, 5, 4),
                                       kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                       padding='same', activation=tf.nn.relu)(layer)
        layer = tf.keras.layers.MaxPool3D(pool_size=(3, 3, 3), padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization(axis=1)(layer)
        layer = tf.keras.layers.Dropout(dropout)(layer)

        layer = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3),
                                       kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                       padding='same', activation=tf.nn.relu)(layer)
        layer = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3),
                                       kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                       padding='same', activation=tf.nn.relu)(layer)
        layer = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization(axis=1)(layer)
        layer = tf.keras.layers.Dropout(dropout)(layer)

        # layer = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),kernel_regularizer=tf.keras.regularizers.l1(lw1),\
        #                               padding='same',activation=tf.nn.relu)(layer)
        # layer = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same')(layer)
        # layer = tf.keras.layers.BatchNormalization(axis=1)(layer)

        # layer = tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),kernel_regularizer=tf.keras.regularizers.l1(lw1),\
        #                               padding='same',activation=tf.nn.relu)(layer)
        # layer = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same')(layer)
        # layer = tf.keras.layers.BatchNormalization(axis=1)(layer)

        lips = tf.keras.layers.Flatten()(layer)

        layer = tf.keras.layers.Conv3D(filters=16, kernel_size=(5, 5, 4),
                                       kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                       padding='same', activation=tf.nn.relu)(tongue_inputs)
        layer = tf.keras.layers.Conv3D(filters=16, kernel_size=(5, 5, 4),
                                       kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                       padding='same', activation=tf.nn.relu)(layer)
        layer = tf.keras.layers.MaxPool3D(pool_size=(3, 3, 3), padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization(axis=1)(layer)
        layer = tf.keras.layers.Dropout(dropout)(layer)

        layer = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3),
                                       kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                       padding='same', activation=tf.nn.relu)(layer)
        layer = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3),
                                       kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                       padding='same', activation=tf.nn.relu)(layer)
        layer = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization(axis=1)(layer)
        layer = tf.keras.layers.Dropout(dropout)(layer)

        # layer = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),kernel_regularizer=tf.keras.regularizers.l1(lw1),\
        #                               padding='same',activation=tf.nn.relu)(layer)
        # layer = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same')(layer)
        # layer = tf.keras.layers.BatchNormalization(axis=1)(layer)

        # layer = tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),kernel_regularizer=tf.keras.regularizers.l1(lw1),\
        #                               padding='same',activation=tf.nn.relu)(layer)

        # layer = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same')(layer)

        # layer = tf.keras.layers.BatchNormalization(axis=1)(layer)
        tongue = tf.keras.layers.Flatten()(layer)

        concat_lip_tog = tf.keras.layers.concatenate([lips, tongue])
        # concat_lip_tog = tf.keras.layers.add([lips,tongue])
        concat_lip_tog = tf.keras.layers.Dense(1000, activation=tf.nn.relu, \
                                               kernel_regularizer=tf.keras.regularizers.l1(1e-6), \
                                               name='fc')(concat_lip_tog)


    else:
        tongue_inputs = tf.keras.Input(shape=(48, 48, 1), name='tongue')
        # tongue_intpus = tf.keras.Dropout(dropout)(tongue_inputs)
        lips_inputs = tf.keras.Input(shape=(48, 48, 1), name='lips')

        layer = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                       padding='same', activation=tf.nn.relu)(lips_inputs)
        layer = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                       padding='same', activation=tf.nn.relu)(layer)
        layer = tf.keras.layers.MaxPool2D(pool_size=(3, 3), padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization(axis=1)(layer)
        layer = tf.keras.layers.Dropout(dropout)(layer)

        layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                       padding='same', activation=tf.nn.relu)(layer)
        layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                       padding='same', activation=tf.nn.relu)(layer)
        lips_layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(layer)

        delayer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                         kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                         padding='same', activation=tf.nn.relu)(lips_layer)
        delayer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                         kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                         padding='same', activation=tf.nn.relu)(delayer)
        delayer = tf.keras.layers.Dropout(dropout)(delayer)
        delayer = tf.keras.layers.UpSampling2D((2, 2))(delayer)
        delayer = tf.keras.layers.BatchNormalization(axis=1)(delayer)

        delayer = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5),
                                         kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                         padding='same', activation=tf.nn.relu)(delayer)
        delayer = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5),
                                         kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                         padding='same', activation=tf.nn.relu)(delayer)
        delayer = tf.keras.layers.Dropout(dropout)(delayer)
        delayer = tf.keras.layers.UpSampling2D((3, 3))(delayer)
        delayer = tf.keras.layers.BatchNormalization(axis=1)(delayer)

        lips_decode = tf.keras.layers.Conv2D(filters=1, kernel_size=(5, 5),
                                             kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                             padding='same', activation=None, name='lips_decode')(delayer)

        lips_layer = tf.keras.layers.BatchNormalization(axis=1)(lips_layer)
        lips_layer = tf.keras.layers.Dropout(dropout)(lips_layer)

        # layer = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),kernel_regularizer=tf.keras.regularizers.l1(lw1),\
        #                               padding='same',activation=tf.nn.relu)(layer)
        # layer = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same')(layer)
        # layer = tf.keras.layers.BatchNormalization(axis=1)(layer)

        # layer = tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),kernel_regularizer=tf.keras.regularizers.l1(lw1),\
        #                               padding='same',activation=tf.nn.relu)(layer)
        # layer = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same')(layer)
        # layer = tf.keras.layers.BatchNormalization(axis=1)(layer)

        lips = tf.keras.layers.Flatten()(lips_layer)

        layer = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                       padding='same', activation=tf.nn.relu)(tongue_inputs)
        layer = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                       padding='same', activation=tf.nn.relu)(layer)
        layer = tf.keras.layers.MaxPool2D(pool_size=(3, 3), padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization(axis=1)(layer)
        layer = tf.keras.layers.Dropout(dropout)(layer)

        layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                       padding='same', activation=tf.nn.relu)(layer)
        layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                       padding='same', activation=tf.nn.relu)(layer)
        tongue_layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(layer)

        # layer = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),kernel_regularizer=tf.keras.regularizers.l1(lw1),\
        #                               padding='same',activation=tf.nn.relu)(layer)
        # layer = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same')(layer)
        # layer = tf.keras.layers.BatchNormalization(axis=1)(layer)

        # layer = tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),kernel_regularizer=tf.keras.regularizers.l1(lw1),\
        #                               padding='same',activation=tf.nn.relu)(layer)

        # layer = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same')(layer)

        # delayer = tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),kernel_regularizer=tf.keras.regularizers.l1(lw1),\
        #                               padding='same',activation=tf.nn.relu)(layer)
        # delayer = tf.keras.layers.UpSampling2D((2,2))(delayer)
        # delayer = tf.keras.layers.BatchNormalization(axis=1)(delayer)

        # delayer = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),kernel_regularizer=tf.keras.regularizers.l1(lw1),\
        #                               padding='same',activation=tf.nn.relu)(layer)
        # delayer = tf.keras.layers.UpSampling2D((2,2))(delayer)
        # delayer = tf.keras.layers.BatchNormalization(axis=1)(delayer)

        delayer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                         kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                         padding='same', activation=tf.nn.relu)(tongue_layer)
        delayer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                         kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                         padding='same', activation=tf.nn.relu)(delayer)
        delayer = tf.keras.layers.Dropout(dropout)(delayer)
        delayer = tf.keras.layers.UpSampling2D((2, 2))(delayer)
        delayer = tf.keras.layers.BatchNormalization(axis=1)(delayer)

        delayer = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5),
                                         kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                         padding='same', activation=tf.nn.relu)(delayer)
        delayer = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5),
                                         kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                         padding='same', activation=tf.nn.relu)(delayer)
        delayer = tf.keras.layers.Dropout(dropout)(delayer)
        delayer = tf.keras.layers.UpSampling2D((3, 3))(delayer)
        delayer = tf.keras.layers.BatchNormalization(axis=1)(delayer)

        tongue_decode = tf.keras.layers.Conv2D(filters=1, kernel_size=(5, 5), \
                                               kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                               padding='same', activation=None, name='tongue_decode')(delayer)

        tongue_layer = tf.keras.layers.BatchNormalization(axis=1)(tongue_layer)
        tongue_layer = tf.keras.layers.Dropout(dropout)(tongue_layer)
        tongue = tf.keras.layers.Flatten()(tongue_layer)

        concat_lip_tog = tf.keras.layers.concatenate([lips, tongue])
        # concat_lip_tog = tf.keras.layers.add([lips,tongue])
        concat_lip_tog = tf.keras.layers.Dense(1000, activation=tf.nn.relu, \
                                               kernel_regularizer=tf.keras.regularizers.l1(1e-6), \
                                               name='fc')(concat_lip_tog)

    if classification > 1:
        if multi_task:
            pred = []
            for i in range(order):
                sub_pred = concat_lip_tog
                # sub_pred = tf.keras.layers.Dense(1000, activation = tf.nn.relu,  kernel_regularizer=tf.keras.regularizers.l1(lw1),\
                #                                 name='neck_each%i' % i)(concat_lip_tog)
                # sub_pred = tf.keras.layers.Dense(1000, activation = tf.nn.relu,  kernel_regularizer=tf.keras.regularizers.l1(lw1),\
                #                                 name='neck2_each%i' % i)(concat_lip_tog)

                sub_pred = tf.keras.layers.Dense(classification, \
                                                 activation=tf.nn.softmax, \
                                                 name='pred_lsf%d' % i)(sub_pred)
                pred.append(sub_pred)

        else:
            sub_pred = tf.keras.layers.Dense(1000, activation=tf.nn.relu, \
                                             name='neck')(concat_lip_tog)
            pred = tf.keras.layers.Dense(classification, \
                                         activation=tf.nn.softmax)(concat_lip_tog)


    else:
        if multi_task:
            pred = []
            for i in range(order):
                sub_pred = concat_lip_tog
                # sub_pred = tf.keras.layers.Dense(1000, activation = tf.nn.relu, \
                #                                 name='neck_%d' % i)(concat_lip_tog)
                sub_pred = tf.keras.layers.Dense(1, \
                                                 activation=None, \
                                                 name='pred_lsf%d' % i)(sub_pred)
                pred.append(sub_pred)


        else:
            sub_pred = tf.keras.layers.Dense(1000, activation=tf.nn.relu, \
                                             name='neck')(concat_lip_tog)
            pred = tf.keras.layers.Dense(1, activation=None)(concat_lip_tog)

    if AE:
        model = tf.keras.Model(inputs=[tongue_inputs, lips_inputs], \
                               outputs=pred + [tongue_decode, lips_decode])

    else:
        model = tf.keras.Model(inputs=[tongue_inputs, lips_inputs], \
                               outputs=pred)
    return model


def target_preprocessing(train_lsf, test_lsf, classification, order=12):
    train_ys = []
    test_ys = []
    if classification > 0:
        for i in range(order):
            tr_i_max = train_lsf[:, i].max()
            tr_i_min = train_lsf[:, i].min()
            cl_train_lsf = (classification * ((train_lsf[:, i] - tr_i_min) / (tr_i_max - tr_i_min))).astype(int)
            cl_train_lsf[cl_train_lsf >= classification] = classification - 1
            cl_train_lsf = tf.keras.utils.to_categorical(cl_train_lsf, classification)
            cl_test_lsf = (classification * ((test_lsf[:, i] - tr_i_min) / (tr_i_max - tr_i_min))).astype(int)
            cl_test_lsf[cl_test_lsf >= classification] = classification - 1
            cl_test_lsf[cl_test_lsf < 0] = 0
            cl_test_lsf = tf.keras.utils.to_categorical(cl_test_lsf, classification)
            train_ys.append(cl_train_lsf)
            test_ys.append(cl_test_lsf)
    else:
        for i in range(order):
            train_ys.append(train_lsf[:, i][:, np.newaxis])
            test_ys.append(test_lsf[:, i][:, np.newaxis])
    return train_ys, test_ys


def model_compile(model, optimizer, classification, multi_task=True, AE=False,order=12):
    if classification > 0:
        if multi_task:
            loss = {'pred_lsf%d' % i: "categorical_crossentropy" for i in range(order)}
            loss_weights = {'pred_lsf%d' % i: 1 for i in range(order)}
            metrics = {'pred_lsf%d' % i: "categorical_crossentropy" for i in range(order)}

            if AE:
                loss['tongue_decode'] = "mse"
                loss['lips_decode'] = "mse"
                loss_weights['tongue_decode'] = 1 / 100
                loss_weights['lips_decode'] = 1 / 100
                # metrics['tongue_decode'] = 'mse'
                # metrics['lips_decode'] = 'mse'

            model.compile(optimizer=optimizer,
                          loss=loss,
                          loss_weights=loss_weights,
                          metrics=metrics)
        else:

            model.compile(optimizer=optimizer, loss='categorical_crossentropy', \
                          metrics=["categorical_crossentropy"])
    else:
        if multi_task:

            loss = {'pred_lsf%d' % i: "mse" for i in range(order)}
            model.compile(optimizer=optimizer,
                          loss=loss,
                          metrics=["mse"])
        else:
            model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])


def measure(ytrue, ypred, train_lsf, classification,order=12):
    predict = ypred
    test_lsf = ytrue
    if classification > 0:
        predict_lsf = []
        for i in range(order):
            predict_iter = predict[i].argmax(axis=1)
            tr_i_max = train_lsf[:, i].max()
            tr_i_min = train_lsf[:, i].min()
            predict_n = ((predict_iter + 0.5) / classification) * (tr_i_max - tr_i_min) + tr_i_min
            predict_lsf.append(predict_n)
        predict_lsf = np.array(predict_lsf)
        predict_lsf = predict_lsf.transpose()
        try:
            _, _, sd, is16sd = spectral_distortion(predict_lsf, test_lsf, 512, 6, 200)
            print("sd %.3f is16sd %.3f" % (sd, is16sd))
        except:
            sd = 0
            is16sd = 0
    else:
        predict = [val.flatten() for val in predict]
        predict = np.array(predict)
        predict_lsf = predict.transpose()
        predict_lsf[predict_lsf > 3.1415926] = 3.1415926
        predict_lsf[predict_lsf < 0] = 0
        try:
            _, _, sd, is16sd = spectral_distortion(predict_lsf, test_lsf, 512, 6, 200)
            print("sd %.3f is16sd %.3f" % (sd, is16sd))
        except:
            sd = 0
            is16sd = 0
    return sd, is16sd, predict_lsf
    #return 0,0,predict_lsf

def data_preprocessing(train_tongue, train_lips, train_lsf, test_tongue, test_lips, test_lsf, steps=8):
    data = []
    for x in [train_tongue, train_lips, test_tongue, test_lips]:
        subdata = []
        for i in tqdm(range(len(x))):
            if (i - int(steps / 2) >= 0) and (i + steps - int(steps / 2) <= len(x)):
                subdata.append(x[i - int(steps / 2):i + steps - int(steps / 2)])
        data.append(np.array(subdata))
    for x in [train_lsf, test_lsf]:
        subdata = []
        for i in range(len(x)):
            if (i - int(steps / 2) >= 0) and (i + steps - int(steps / 2) <= len(x)):
                subdata.append(x[i])
        data.append(np.array(subdata))

    return data[0], data[1], data[4], data[2], data[3], data[5]


def keras_train(path="../out/test_lsf/lsf_hamming_ds4.pkl", \
                conv3d=False, classification=0, order=12, multi_task=True, IS16=False, fakeIS16=False, \
                optimizer='adam', lr=0.0001, AE=False, name='',epochs=40):
    print(optimizer)
    print(multi_task)
    try:
        os.mkdir("../out/%s" % EXP_NAME)
    except:
        pass
    # load data
    train_tongue, train_lips, train_lsf, test_tongue, test_lips, test_lsf = load_dataset(path, IS16, fakeIS16)
    print(train_tongue.shape)
    print(train_lips.shape)

    # data preprocessing
    if conv3d:
        train_tongue, train_lips, train_lsf, test_tongue, test_lips, test_lsf = \
            data_preprocessing(train_tongue, train_lips, train_lsf, test_tongue, test_lips, test_lsf)
    print(train_tongue.shape)
    print(train_lips.shape)

    # target preprocessing
    train_ys, test_ys = target_preprocessing(train_lsf, test_lsf, classification,order=order)

    # load model
    model = cnn_model_keras(conv3d=conv3d, classification=classification, multi_task=multi_task, AE=AE,order=order)
    print(model.summary())


    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr=0.0001)
    else:
        optimizer = tf.keras.optimizers.RMSprop(lr=0.0001)

    # compile
    model_compile(model, optimizer, classification, multi_task, AE=AE,order=order)

    timenow = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    model_path = '../out/%s/bst_model.h5' % (EXP_NAME)

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, \
                                                          save_weights_only=True)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')

    f_record = open("record.csv", "a")

    if multi_task:

        if AE:
            train_ys.append(train_tongue)
            train_ys.append(train_lips)

        hist = model.fit([train_tongue, train_lips], train_ys, \
                         validation_split=0.05, \
                         verbose=2, \
                         batch_size=512, epochs=epochs, shuffle=True, \
                         callbacks=[earlystop, model_checkpoint])
        bst_epoch = np.argmax(hist.history['val_loss'])
        trn_loss = hist.history['loss'][bst_epoch]
        val_loss = hist.history['val_loss'][bst_epoch]

        model.load_weights(model_path)
        predict = model.predict([test_tongue, test_lips], batch_size=512)

        sd, is16sd, predict_lsf = measure(test_lsf, predict, train_lsf, classification,order=order)
        f = open("../out/%s/predict_is16sd%.3f.pkl" % (EXP_NAME, is16sd), 'wb')
        pickle.dump(predict_lsf, f)
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
            hist = model.fit([train_tongue, train_lips], train_ys[i],
                             validation_split=0.05,
                             batch_size=512, epochs=40, shuffle=True, \
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
    res = '%s,%s,%s,%d,%s,%d,%.6f,%.6f,%s,%s \n' % \
          (timenow, 'multi', EXP_NAME, classification, str(is16sd), bst_epoch + 1, trn_loss, val_loss, str(IS16), name)
    f_record.write(res)
    f_record.close()

    return is16sd


def parse_args():
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--conv3d', type=int,
                        default=0, help='use 3d cnn model')
    parser.add_argument('--classification', type=int,
                        default=100, help='count of classification target, 0 for regression')
    parser.add_argument('--multi_task', type=int,
                        default=1, help='12 lsf target as multi task or not')
    parser.add_argument('--is16', type=int,
                        default=1, help='use test dataset as is16')
    parser.add_argument('--name', type=str,
                        default='', help='name of experiment')

    parser.add_argument('--AE', type=int,
                        default=0, help='use autoencoder or not')

    parser.add_argument('--gpu', type=int,
                        default=0, help='which gpu to use')

    return parser.parse_args()

def endtoend_model_keras(AE=False, conv3d=False, classification=0, multi_task=True, lw1=1e-7, dropout=0.2,order=12):

    tongue_inputs = tf.keras.Input(shape=(48, 48, 1), name='tongue')
    lips_inputs = tf.keras.Input(shape=(48, 48, 1), name='lips')

    layer = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                   padding='same', activation=tf.nn.relu)(lips_inputs)
    layer = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                   padding='same', activation=tf.nn.relu)(layer)
    layer = tf.keras.layers.MaxPool2D(pool_size=(3, 3), padding='same')(layer)
    layer = tf.keras.layers.BatchNormalization(axis=1)(layer)
    layer = tf.keras.layers.Dropout(dropout)(layer)

    layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                   padding='same', activation=tf.nn.relu)(layer)
    layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                   padding='same', activation=tf.nn.relu)(layer)
    lips_layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(layer)
    lips_layer = tf.keras.layers.BatchNormalization(axis=1)(lips_layer)
    lips_layer = tf.keras.layers.Dropout(dropout)(lips_layer)
    lips = tf.keras.layers.Flatten()(lips_layer)


    layer = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                   padding='same', activation=tf.nn.relu)(tongue_inputs)
    layer = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                   padding='same', activation=tf.nn.relu)(layer)
    layer = tf.keras.layers.MaxPool2D(pool_size=(3, 3), padding='same')(layer)
    layer = tf.keras.layers.BatchNormalization(axis=1)(layer)
    layer = tf.keras.layers.Dropout(dropout)(layer)

    layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                   padding='same', activation=tf.nn.relu)(layer)
    layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                   padding='same', activation=tf.nn.relu)(layer)
    tongue_layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(layer)
    tongue_layer = tf.keras.layers.BatchNormalization(axis=1)(tongue_layer)
    tongue_layer = tf.keras.layers.Dropout(dropout)(tongue_layer)
    tongue = tf.keras.layers.Flatten()(tongue_layer)
    concat_lip_tog = tf.keras.layers.concatenate([lips, tongue])
    concat_lip_tog = tf.keras.layers.Dense(66,activation=None)(concat_lip_tog)
    concat_lip_tog = tf.keras.layers.Reshape((66,1))(concat_lip_tog)
    recon = tf.keras.layers.Conv1D(filters=32,kernel_size=5,padding='same',activation=tf.nn.relu)(concat_lip_tog)
    recon = tf.keras.layers.UpSampling1D(2)(recon)
    recon = tf.keras.layers.Conv1D(filters=16, kernel_size=10, padding='same', activation=tf.nn.relu)(recon)
    recon = tf.keras.layers.UpSampling1D(2)(recon)
    recon = tf.keras.layers.Conv1D(filters=1, kernel_size=10, padding='same', activation=None)(recon)

    model = tf.keras.Model(inputs=[tongue_inputs,lips_inputs],outputs=recon)

    return model

def endtoend_train():
    size = 16000 * 1.0 / 60
    wave_data = []
    for i in range(5):
        f = open('../out/song%d.wav' % (i + 1), 'rb')
        data, samplerate = sf.read(f, dtype='int16')
        # print(data)
        wave_data.extend(data.tolist()[:-int(size * 10)])
    wave_data = np.array(wave_data, dtype='float32')
    mean = wave_data.mean()
    std = wave_data.std()
    wave_data -= mean
    wave_data /= std
    y = []
    for i in range(int(len(wave_data) / size)):
        instance = wave_data[int(i * size):int((i) * size) + int(size)]
        y.append(instance)

    y = np.array(y, dtype='float')
    y = y[:,:264,np.newaxis]

    train_tongue, train_lips,_, test_tongue, test_lips, _ = load_dataset(IS16=True)
    train_ys = np.concatenate([y[:25000],y[30000:]])
    test_ys = np.array(y[25000:30000])
    model = endtoend_model_keras()
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  loss='mse',
                  metrics=['mse'])
    print(model.summary())
    if not os.path.isdir('../out/%s' % EXP_NAME):
        os.mkdir('../out/%s' % EXP_NAME)
    timenow = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    model_path = '../out/%s/bst_model.h5' % (EXP_NAME)

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, \
                                                          save_weights_only=True)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')

    hist = model.fit([train_tongue, train_lips], train_ys, \
                     validation_split=0.05, \
                     verbose=2, \
                     batch_size=512, epochs=1, shuffle=True, \
                     callbacks=[earlystop, model_checkpoint])
    bst_epoch = np.argmax(hist.history['val_loss'])
    trn_loss = hist.history['loss'][bst_epoch]
    val_loss = hist.history['val_loss'][bst_epoch]

    model.load_weights(model_path)
    predict = model.predict([test_tongue, test_lips], batch_size=512)
    print()
    predict = predict.reshape((1,len(predict)*264))[0]
    print(predict.shape)
    plt.plot(range(len(predict)),predict)

    plt.show()


if __name__ == "__main__":
    pass
    #endtoend_train()
    # args = parse_args()
    #
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # keras_train(conv3d=True if args.conv3d == 1 else False,
    #             classification=args.classification,
    #             multi_task=True if args.multi_task == 1 else False,
    #             IS16=True if args.is16 == 1 else False,
    #             AE=True if args.AE == 1 else False,
    #             # fakeIS16=True if args.fakeis16==1 else False,
    #             name=args.name
    #             )


