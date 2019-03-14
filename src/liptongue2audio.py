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
def endtoend_model_keras(lw1=1e-7, dropout=0.2):

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
        f = open('../out/egg%d.wav' % (i + 1), 'rb')
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

