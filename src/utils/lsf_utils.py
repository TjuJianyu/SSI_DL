import wave
import numpy as np
from spectrum import linear_prediction as lpd
import time
import random
import tensorflow as tf
import cv2
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
import soundfile as sf
import heapq
from scipy import optimize
import wave
import audioop
import sys
import os
from scipy.stats import sem, t
from scipy import mean

pi = 3.1415926
def test_func(x, a, b,c,d):
    return a*np.sin((2*pi)/b * x+c*pi)  +d
def test_func2(a, b, d):
    def test_func(x, c):
        return a * np.sin((2 * pi) / b * x + c * pi) + d

    return test_func


def to_square_wave(data, n=10, context=0.25):
    softmax = sum(heapq.nlargest(n, data[int(len(data) * context):-int(len(data) * context)])) * 1.0 / n
    softmin = sum(heapq.nsmallest(n, data[int(len(data) * context):-int(len(data) * context)])) * 1.0 / n
    median = (softmax + softmin) / 2
    square_data = np.copy(data)
    square_data[data >= median] = 1
    square_data[data < median] = -1

    return square_data, softmax, softmin, median

def wave_cyc(square_wave, real_wave, threshold=10):
    data = np.copy(square_wave)

    start = label = data[0]
    stack = []
    labels = []
    count = 0
    for val in data:
        if val == label:
            count += 1
        else:
            stack.append(count)
            labels.append(label)
            label = val
            count = 1
    stack.append(count)
    labels.append(label)

    total_index = 0
    size = len(stack)
    for i in range(size):
        val = stack[i]
        if i == 0:
            total_index += val
            continue
        if val < threshold:
            for j in range(total_index, total_index + val):
                data[j] = labels[i - 1]
                labels[i] = labels[i - 1]
        total_index += val

    start = label = data[0]
    stack = []
    labels = []
    count = 0
    for val in data:
        if val == label:
            count += 1
        else:
            stack.append(count)
            labels.append(label)
            label = val
            count = 1
    stack.append(count)
    labels.append(label)

    bound = []
    start = 0
    for i in range(len(stack)):
        val = stack[i]
        bound.append(real_wave[start:start + val].min() if labels[i] == -1 else real_wave[start:start + val].max())
        data[start:start + val] = min(real_wave[start:start + val]) if labels[i] == -1 else max(
            real_wave[start:start + val])
        start += val

    if len(stack) >= 4:
        tmp_stack = stack[1:len(stack) - 1 if (len(stack) - 2) % 2 == 0 else len(stack) - 2]
        cyc = sum(tmp_stack) * 1.0 / (len(tmp_stack) / 2)
    else:
        cyc = sum(stack) * 1.0 / (len(stack) / 2)

    return data, bound, stack, cyc, (bound[int(len(bound) / 2) - 1], bound[int(len(bound) / 2)])

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

def loadeggsong(data_dir = '../out/',size = 16000 * 1.0/60):

    wave_data = []
    egg_data = []
    for i in range(5):
        f = open(data_dir + 'song%d.wav' % (i + 1), 'rb')
        data, samplerate = sf.read(f, dtype='int16')
        wave_data.extend(data.tolist()[:-int(size * 10)])
        f = open(data_dir + 'egg%d.wav' % (i + 1), 'rb')
        data, samplerate = sf.read(f, dtype='int16')
        egg_data.extend(data.tolist()[:-int(size * 10)])

    wave_data = np.array(wave_data, dtype='float32')
    mean = wave_data.mean()
    std = wave_data.std()
    wave_data -= mean
    wave_data /= std

    egg_data = np.array(egg_data, dtype='float32')
    mean = egg_data.mean()
    std = egg_data.std()
    egg_data -= mean
    egg_data /= std
    egg_data = np.concatenate([[0] * 30, egg_data[:-30]])

    data = []
    ydata = []
    for i in range(int(len(wave_data) / size)):
        instance = wave_data[int(i * size):int((i) * size) + int(size)]
        ydata.append(instance)
        egginstance = egg_data[int(i * size):int((i) * size) + int(size)]
        data.append(egginstance)

    data = np.array(data, dtype='float')
    ydata = np.array(ydata, dtype='float')

    # data = data.tolist()
    # (len(data))
    cnn = True

    trainegg = np.concatenate([data[:25000], data[30000:]])
    testegg = np.array(data[25000:30000])
    trainsong = np.concatenate([ydata[:25000], ydata[30000:]])
    testsong = np.array(ydata[25000:30000])

    return trainegg,testegg,trainsong,testsong

def load_dataset(path="../out/test_lsf/lsf_hamming_16kHZ.pkl", IS16=False, fakeIS16=False, reshape=False):
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

def confidenceinterval(data, confidence=0.95):
    n = len(data)
    m = mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    return m, h

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


def vocoder(lsf, activation, fs=16000, hamming=True, frame_length=1 / 60, overlap_percent=50):
    exact_frame_length = fs * frame_length
    signal_length = len(activation)
    print(signal_length)
    downsample_audio_data = np.hstack([np.array([0] * int(exact_frame_length / 2)) , activation, np.array([0] * int(exact_frame_length / 2))])
    # reproduced = [0]*len(downsample_audio_data)
    reproduced = []
    for i in tqdm(range(int(signal_length / exact_frame_length))):
        start_index = int(i * exact_frame_length) - int(exact_frame_length / 2) + int(exact_frame_length / 2)
        stop_index = int((i + 1) * exact_frame_length) + int(exact_frame_length / 2) + int(exact_frame_length / 2)
        if hamming:
            frame = downsample_audio_data[start_index:stop_index] * spsig.hamming(stop_index - start_index)
        else:
            frame = downsample_audio_data[start_index:stop_index]
        lsf_iter = lsf[i]
        # print(lsf_iter)
        lpcval = lpd.lsf2poly(lsf_iter)
        # print(lpcval)
        lpcfilter = audiolazy.ZFilter(lpcval.tolist())
        reproduced_iter = list(lpcfilter(frame))

        reproduced.extend(reproduced_iter[int(exact_frame_length / 2):-int(exact_frame_length / 2)])
    return np.array(reproduced)


# len(lsf)
def audio2lsf(audio_data, fps, order, \
              usefilter=True, frame_length=1 / 60,  bins=-1):
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
    pass
