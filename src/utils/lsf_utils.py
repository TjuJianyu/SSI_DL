import wave
import numpy as np
from spectrum import linear_prediction as lpd
import time
import random
# from tensorflow.keras.regularizers import l1,l2,l1_l2i
import tensorflow as tf
import cv2
from audiolazy import lpc
# from matplotlib import pyplot as plt
import scipy.signal as spsig
import os
from tqdm import tqdm
import pickle
import datetime
from scipy import signal
import math
import argparse
import scipy.io as sio


def wavereader(f_path):
    f = wave.open(f_path, "rb")
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = f.readframes(nframes)
    f.close()
    wave_data = np.fromstring(str_data, dtype=np.short)
    return wave_data, nchannels, sampwidth, framerate, nframes


def musicdata_wavereader(normalized=True):
    songs = ['184341', '190633', '192504', '193153', '193452']
    song_name = "MICRO_RecFile_1_20140523_%s_Micro_EGG_Sound_Capture_monoOutput1.wav"
    wave_dataset = []
    totalnframes = 0

    zerocount = 0
    for song in songs:
        wave_data, nchannels, sampwidth, framerate, nframes \
            = wavereader('../../data/Songs_Audio/%s' % (song_name % song))
        print(nframes)
        print(nframes * 1.0 / 44100 * 60)
        print(int(nframes / 44100) * 60)

        i = nframes - 1
        count = 0
        while wave_data[i] == 0:
            count += 1
            i -= 1
        zerocount += count
        print("#zero end", count)
        # eliminate the last 10 * 1/60 s
        wave_data = wave_data[:-int(10 * framerate * 1 / 60)]
        nframes -= int(10 * framerate * 1 / 60)
        wave_dataset += wave_data.tolist()
        totalnframes += nframes
        print("_ed wave %s: nchannels %d, sampwidth %d, framerate %d, nframes %d" % (song, nchannels, sampwidth, framerate, nframes))
    # print(zerocount)
    wave_dataset = np.array(wave_dataset)
    if normalized:
        wave_dataset = 2 * (wave_dataset - (-32767)) / (32767 - (-32767)) - 1

    # print(wave_dataset[:10])
    print("total # frames %d" % totalnframes)

    return wave_dataset, nchannels, sampwidth, framerate, totalnframes


def load_dataset(path="../out/test_lsf/lsf_hamming_ds4.pkl", IS16=False, fakeIS16=False, reshape=False):
    print('loading...')
    if IS16:
        f = open(path, "rb")
        lsf = pickle.load(f)[0]
        lsf = lsf.tolist()
        train_lsf = lsf[:25000]
        train_lsf.extend(lsf[30000:])
        train_lsf = np.array(train_lsf)
        test_lsf = np.array(lsf[25000:30000])
        f.close()
        if os.path.isfile('../out/IS16_train_test_tongue_lips.pkl'):
            f = open('../out/IS16_train_test_tongue_lips.pkl','rb')
            train_lips,test_lips,train_tongue,test_tongue = pickle.load(f)
            f.close()
        else:

            lips = []
            tongue = []
            tong_fdir = "../out/resize_tongue/%s.tif"
            lips_fdir = "../out/resize_lips/%s.tif"
            for i in tqdm(range(1, 68147)):

                ran = str(i)
                for i in range(6 - len(ran)):
                    ran = '0' + ran
                #print(lips_fdir % ran)
                lframe_0 = cv2.imread(lips_fdir % ran, 0)
                #print(lframe_0)
                lframe_0.resize(48, 48)
                lframe_0 = np.array(lframe_0)
                #print(lframe_0)

                lips.append(lframe_0.reshape((lframe_0.shape[0], lframe_0.shape[1], 1)))

                tframe_0 = cv2.imread(tong_fdir % ran, 0)
                tframe_0.resize(48, 48)
                tframe_0 = np.array(tframe_0)
                tongue.append(tframe_0.reshape((tframe_0.shape[0], tframe_0.shape[1], 1)))

            train_lips = lips[:25000]
            train_lips.extend(lips[30000:])
            train_tongue = tongue[:25000]
            train_tongue.extend(tongue[30000:])
            train_lips = np.array(train_lips)
            train_tongue = np.array(train_tongue)
            test_lips = np.array(lips[25000:30000])
            test_tongue = np.array(tongue[25000:30000])

            f = open('../out/IS16_train_test_tongue_lips.pkl','wb')
            pickle.dump([train_lips,test_lips,train_tongue,test_tongue],f)
            f.close()
    # elif os.path.exists("../out/train_tongue_01.pkl") and \
    # os.path.exists("../out/train_lips_01.pkl") and \
    # os.path.exists("../out/train_lsf_01.pkl"):
    #    f1 = open("../out/train_tongue_01.pkl","rb")
    #    f2 = open("../out/train_lips_01.pkl","rb")
    #    f3 = open("../out/train_lsf_01.pkl","rb")
    #    f4 = open("../out/test_tongue_01.pkl","rb")
    #    f5 = open("../out/test_lips_01.pkl","rb")
    #    f6 = open("../out/test_lsf_01.pkl","rb")
    #    train_tongue = pickle.load(f1)
    #    train_lips = pickle.load(f2)
    #    train_lsf = pickle.load(f3)
    #    test_tongue = pickle.load(f4)
    #    test_lips = pickle.load(f5)
    #    test_lsf = pickle.load(f6)

    else:
        f = open(path, "rb")
        lsf = pickle.load(f)
        lips = []
        tongue = []
        tong_fdir = "../out/resize_tongue/%s.tif"
        lips_fdir = "../out/resize_lips/%s.tif"
        for i in tqdm(range(1, 68147)):

            ran = str(i)
            for i in range(6 - len(ran)):
                ran = '0' + ran
            lframe_0 = cv2.imread(lips_fdir % ran, 0)
            lframe_0.resize(48, 48)
            lframe_0 = np.array(lframe_0)
            lips.append(lframe_0.reshape((lframe_0.shape[0], lframe_0.shape[1], 1)))

            tframe_0 = cv2.imread(tong_fdir % ran, 0)
            tframe_0.resize(48, 48)
            tframe_0 = np.array(tframe_0)
            tongue.append(tframe_0.reshape((tframe_0.shape[0], tframe_0.shape[1], 1)))
        testrate = 0.1
        length = np.array([10670146, 7050413, 13645126, 5311110, 13410518])
        length = (length / 44100 * 60)
        test = length * testrate
        left = length.cumsum() - test
        right = length.cumsum()
        left = left.astype(int)
        right = right.astype(int)
        test_lips, train_lips, test_tongue, train_tongue, test_lsf, train_lsf = [], [], [], [], [], []
        for i in range(len(left)):
            test_lips.extend(lips[left[i]:right[i]])
            test_tongue.extend(tongue[left[i]:right[i]])
            test_lsf.extend(lsf[left[i]:right[i]])

        for i in range(len(left)):
            if i == 0:
                start = 0
            else:
                start = right[i - 1]
            train_lips.extend(lips[start:left[i]])
            train_tongue.extend(tongue[start:left[i]])
            train_lsf.extend(lsf[start:left[i]])
        train_lips = np.array(train_lips, dtype=int)
        train_tongue = np.array(train_tongue, dtype=int)
        train_lsf = np.array(train_lsf)
        test_lips = np.array(test_lips, dtype=int)
        test_tongue = np.array(test_tongue, dtype=int)
        test_lsf = np.array(test_lsf)
        # print(test_lsf)
        # f1 = open("../out/train_tongue_01.pkl","wb")
        # f2 = open("../out/train_lips_01.pkl","wb")
        # f3 = open("../out/train_lsf_01.pkl","wb")
        # f4 = open("../out/test_tongue_01.pkl","wb")
        # f5 = open("../out/test_lips_01.pkl","wb")
        # f6 = open("../out/test_lsf_01.pkl","wb")
        # pickle.dump(train_tongue,f1)
        # pickle.dump(train_lips,f2)
        # pickle.dump(train_lsf,f3)
        # pickle.dump(test_tongue,f4)
        # pickle.dump(test_lips,f5)
        # pickle.dump(test_lsf,f6)
        # f1.close()
        # f2.close()
        # f3.close()
        # f4.close()
        # f5.close()
        # f6.close()

    return train_tongue, train_lips, train_lsf, test_tongue, test_lips, test_lsf

def u_law(x,u=255):
    val = np.array(x)
    return np.sign(val) * np.log1p(u*np.absolute(val)) / np.log1p(u)


def simple_audio2lsf(audio_data, order):
    lpcfilter = lpc.nautocor(audio_data, order)
    reproduce = np.array(list(lpcfilter(frame)))
    lpcfilter = list(lpcfilter)[0]
    lsf = lpd.poly2lsf([lpcfilter[i] for i in range(order + 1)])
    return lsf


import wave
import audioop
import sys
import os
from scipy.stats import sem, t
from scipy import mean


def confidenceinterval(data, confidence=0.95):
    n = len(data)
    m = mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    return m, h


def downsampleWav(src, dst, inrate=44100, outrate=16000, inchannels=1, outchannels=1):
    if not os.path.exists(src):
        print('Source not found!')
        return False

    if not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))

    try:
        s_read = wave.open(src, 'r')
        s_write = wave.open(dst, 'w')
    except:
        print('Failed to open files!')
        return False

    n_frames = s_read.getnframes()
    data = s_read.readframes(n_frames)

    try:
        converted = audioop.ratecv(data, 2, inchannels, inrate, outrate, None)
        if outchannels == 1 & inchannels != 1:
            converted[0] = audioop.tomono(converted[0], 2, 1, 0)
    except:
        print('Failed to downsample wav')
        return False

    try:
        s_write.setparams((outchannels, 2, outrate, 0, 'NONE', 'Uncompressed'))
        s_write.writeframes(converted[0])
    except:
        print('Failed to write wav')
        return False

    try:
        s_read.close()
        s_write.close()
    except:
        print('Failed to close wav files')
        return False

    return True


def audio2lsf(audio_data, fps, order, hamming=False, \
              usefilter=True, frame_length=1 / 60, useGPU=False, bins=-1):
    if usefilter:
        downsample_audio_data = spsig.lfilter([1, -0.95], 1, audio_data)

    exact_frame_length = fps * frame_length
    signal_length = len(downsample_audio_data)

    lsf = []
    error = 0
    reproduce = []

    downsample_audio_data = np.hstack([np.array([0] * int(exact_frame_length / 2)) \
                                          , downsample_audio_data, np.array([0] * int(exact_frame_length / 2))])

    for i in tqdm(range(int(signal_length / exact_frame_length))):
        start_index = int(i * exact_frame_length) - int(exact_frame_length / 2) + int(exact_frame_length / 2)
        stop_index = int((i + 1) * exact_frame_length) + int(exact_frame_length / 2) + int(exact_frame_length / 2)
        frame = downsample_audio_data[start_index:stop_index] * spsig.hamming(stop_index - start_index)
        lpcfilter = lpc.nautocor(frame, order)

        reproduce_iter = list(lpcfilter(frame))
        reproduce.extend(reproduce_iter)

        err = ((frame - np.array(reproduce_iter)) ** 2).mean()
        error += err
        lpcfilter = list(lpcfilter)[0]
        lsfPframe = lpd.poly2lsf([lpcfilter[i] for i in range(order + 1)])
        if bins > 0:
            lsfPframe
        lsf.append(lsfPframe)
    error = error / i
    print("error MSE:%.6f" % (error))

    return np.array(lsf), np.array(reproduce), error
if __name__ == "__main__":
    #audio2lsf()
    downsampleWav()